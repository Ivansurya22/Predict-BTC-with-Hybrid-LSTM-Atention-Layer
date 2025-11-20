import numpy as np
import pandas as pd
import pyqtgraph as pg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QFileDialog,
)
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import sys

# Import models
ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

try:
    from models.models import (
        LSTMAttentionEnhanced,
        LSTMAttention,
        SimpleLSTM,
        FocalLoss,
        LabelSmoothingLoss
    )
except ImportError:
    print("Warning: Model classes not found. Make sure models.py is in models/ folder")
    LSTMAttentionEnhanced = None


# ============================================================================
# DATASET WITH MASK
# ============================================================================
class BTCSequenceDataset(Dataset):
    """Dataset that returns sequences with attention masks"""
    def __init__(self, X, y, masks=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

        if masks is not None:
            self.masks = torch.FloatTensor(masks)
        else:
            self.masks = torch.ones(X.shape[0], X.shape[1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx]


# ============================================================================
# COMBINED LOSS
# ============================================================================
class CombinedLoss(nn.Module):
    """Combined Focal Loss + Label Smoothing"""
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, num_classes=3):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, inputs, targets):
        # Focal loss
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = (alpha_t * focal_term * ce_loss).mean()
        else:
            focal_loss = (focal_term * ce_loss).mean()

        # Label smoothing
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        smooth_loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

        # Combine: 70% focal + 30% label smoothing
        return 0.7 * focal_loss + 0.3 * smooth_loss


# ============================================================================
# TRAINING WORKER THREAD
# ============================================================================
class TrainingWorker(QThread):
    """Worker thread for model training"""
    progress = Signal(int)  # Progress percentage
    epoch_update = Signal(int, dict)  # Epoch number, metrics dict
    log_message = Signal(str)
    finished = Signal(dict)  # Final results
    error = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.should_stop = False

    def stop(self):
        """Stop training"""
        self.should_stop = True

    def run(self):
        try:
            self.log_message.emit("🔧 Initializing training environment...")
            self.progress.emit(5)

            # Load data
            self.log_message.emit("📥 Loading training data...")
            train_loader, val_loader, test_loader, input_size, class_weights = self.load_data()

            if train_loader is None:
                self.error.emit("Failed to load data. Check data paths.")
                return

            self.progress.emit(15)

            # Initialize model
            self.log_message.emit(f"🤖 Initializing {self.config['model_type']} model...")
            model = self.create_model(input_size)

            if model is None:
                self.error.emit("Failed to create model.")
                return

            self.progress.emit(20)

            # Setup training
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_message.emit(f"🖥️  Using device: {device}")
            if torch.cuda.is_available():
                self.log_message.emit(f"   GPU: {torch.cuda.get_device_name(0)}")

            model = model.to(device)

            # Setup loss function with class weights
            if class_weights is not None:
                weights = torch.FloatTensor([
                    class_weights[0] * 1.2,
                    class_weights[1] * 1.0,
                    class_weights[2] * 1.2
                ]).to(device)
                self.log_message.emit(f"📊 Class weights: DOWN={weights[0]:.2f}, HOLD={weights[1]:.2f}, UP={weights[2]:.2f}")
            else:
                weights = torch.FloatTensor([1.2, 1.0, 1.2]).to(device)
                self.log_message.emit(f"📊 Default weights: DOWN=1.2, HOLD=1.0, UP=1.2")

            # Loss function
            criterion = CombinedLoss(alpha=weights, gamma=2.0, smoothing=0.1)
            self.log_message.emit("🎯 Loss: Combined (70% Focal + 30% Label Smoothing)")

            # Optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=0.03,
                betas=(0.9, 0.999)
            )
            self.log_message.emit(f"🔧 Optimizer: AdamW (lr={self.config['learning_rate']}, weight_decay=0.03)")

            # Scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                threshold=0.001
            )
            self.log_message.emit("📉 Scheduler: ReduceLROnPlateau (patience=7, factor=0.5)")

            self.log_message.emit("\n" + "="*60)
            self.log_message.emit("🚀 Starting training...")
            self.log_message.emit("="*60)

            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'val_f1': [],
                'val_f1_per_class': [],
                'learning_rate': []
            }

            best_val_f1 = 0.0
            patience_counter = 0
            best_checkpoint = None

            # Training loop
            for epoch in range(1, self.config['epochs'] + 1):
                if self.should_stop:
                    self.log_message.emit("⚠️ Training stopped by user")
                    break

                # Train
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, criterion, optimizer, device
                )

                # Validate
                val_loss, val_acc, val_f1, val_f1_per_class, val_preds, val_targets = self.validate(
                    model, val_loader, criterion, device
                )

                # Update scheduler
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_f1)

                # Save history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)
                history['val_f1_per_class'].append(val_f1_per_class)
                history['learning_rate'].append(current_lr)

                # Progress
                progress_pct = 20 + int(((epoch) / self.config['epochs']) * 70)
                self.progress.emit(progress_pct)

                # Send update
                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_f1_per_class': val_f1_per_class,
                    'lr': current_lr,
                    'val_preds': val_preds,
                    'val_targets': val_targets
                }
                self.epoch_update.emit(epoch, metrics)

                # Save best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0

                    best_checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_f1': val_f1,
                        'val_f1_per_class': val_f1_per_class
                    }

                    if epoch % 5 == 0 or epoch == 1:
                        self.log_message.emit(f"  ✅ Best model saved (F1: {val_f1:.3f})")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= self.config['patience']:
                    self.log_message.emit(f"\n⏹️  Early stopping at epoch {epoch}")
                    self.log_message.emit(f"   Best F1: {best_val_f1:.3f}")
                    break

            self.progress.emit(90)

            # Load best model for evaluation
            if best_checkpoint:
                model.load_state_dict(best_checkpoint['model_state_dict'])

            # Final evaluation
            self.log_message.emit("\n" + "="*60)
            self.log_message.emit("📊 Evaluating on test set...")
            self.log_message.emit("="*60)
            test_metrics = self.evaluate_model(model, test_loader, device)

            self.progress.emit(95)

            # Save final model
            self.log_message.emit("\n💾 Saving final model...")
            self.save_final_model(model, optimizer, history, test_metrics, best_checkpoint)

            self.progress.emit(100)

            # Prepare results
            results = {
                'history': history,
                'test_metrics': test_metrics,
                'best_val_f1': best_val_f1,
                'config': self.config
            }

            self.log_message.emit("\n✅ Training completed!")
            self.finished.emit(results)

        except Exception as e:
            import traceback
            error_details = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_details)

    def load_data(self):
        """Load training data"""
        try:
            # Load preprocessed data
            data_dir = ROOT_DIR / 'preprocessed_data_lstm_1h'

            X_train = np.load(data_dir / 'X_train.npy')
            y_train = np.load(data_dir / 'y_train.npy')
            masks_train = np.load(data_dir / 'masks_train.npy')

            X_val = np.load(data_dir / 'X_val.npy')
            y_val = np.load(data_dir / 'y_val.npy')
            masks_val = np.load(data_dir / 'masks_val.npy')

            X_test = np.load(data_dir / 'X_test.npy')
            y_test = np.load(data_dir / 'y_test.npy')
            masks_test = np.load(data_dir / 'masks_test.npy')

            class_weights = joblib.load(data_dir / 'class_weights.pkl')

            self.log_message.emit(f"✓ Train: {len(X_train):,} samples")
            self.log_message.emit(f"✓ Val:   {len(X_val):,} samples")
            self.log_message.emit(f"✓ Test:  {len(X_test):,} samples")
            self.log_message.emit(f"✓ Shape: {X_train.shape}")

            # Show target distribution
            self.log_message.emit("\n📊 Target Distribution:")
            for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
                dist = np.bincount(y_split, minlength=3)
                total = len(y_split)
                self.log_message.emit(
                    f"   {split_name:5s} → DOWN: {dist[0]:,} ({dist[0]/total*100:.1f}%) | "
                    f"HOLD: {dist[1]:,} ({dist[1]/total*100:.1f}%) | "
                    f"UP: {dist[2]:,} ({dist[2]/total*100:.1f}%)"
                )

            # Create datasets
            train_dataset = BTCSequenceDataset(X_train, y_train, masks_train)
            val_dataset = BTCSequenceDataset(X_val, y_val, masks_val)
            test_dataset = BTCSequenceDataset(X_test, y_test, masks_test)

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

            input_size = X_train.shape[2]

            return train_loader, val_loader, test_loader, input_size, class_weights

        except Exception as e:
            self.log_message.emit(f"❌ Error loading data: {str(e)}")
            return None, None, None, None, None

    def create_model(self, input_size):
        """Create model based on config"""
        try:
            model_type = self.config['model_type']

            if model_type == 'enhanced':
                model = LSTMAttentionEnhanced(
                    input_size=input_size,
                    hidden_size=self.config['hidden_size'],
                    num_layers=self.config['num_layers'],
                    num_heads=self.config['num_heads'],
                    dropout=self.config['dropout'],
                    num_classes=3,
                    bidirectional=True
                )
            elif model_type == 'standard':
                model = LSTMAttention(
                    input_size=input_size,
                    hidden_size=self.config['hidden_size'],
                    num_layers=self.config['num_layers'],
                    dropout=self.config['dropout'],
                    num_classes=3,
                    bidirectional=True
                )
            else:  # simple
                model = SimpleLSTM(
                    input_size=input_size,
                    hidden_size=self.config['hidden_size'],
                    num_layers=self.config['num_layers'],
                    dropout=self.config['dropout'],
                    num_classes=3
                )

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.log_message.emit(f"✓ Model created: {model_type}")
            self.log_message.emit(f"✓ Total parameters: {total_params:,}")
            self.log_message.emit(f"✓ Trainable parameters: {trainable_params:,}")

            return model

        except Exception as e:
            self.log_message.emit(f"❌ Error creating model: {str(e)}")
            return None

    def train_epoch(self, model, dataloader, criterion, optimizer, device):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for X_batch, y_batch, masks_batch in dataloader:
            if self.should_stop:
                break

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            masks_batch = masks_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch, mask=masks_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            # Predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds) * 100

        return avg_loss, accuracy

    def validate(self, model, dataloader, criterion, device):
        """Validate model"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch, masks_batch in dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                masks_batch = masks_batch.to(device)

                outputs = model(X_batch, mask=masks_batch)
                loss = criterion(outputs, y_batch)

                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds) * 100

        # Calculate F1 scores
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        _, _, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        return avg_loss, accuracy, f1, f1_per_class, all_preds, all_labels

    def evaluate_model(self, model, dataloader, device):
        """Evaluate model and return detailed metrics"""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch, masks_batch in dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                masks_batch = masks_batch.to(device)

                outputs = model(X_batch, mask=masks_batch)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        weighted_f1 = np.average(f1, weights=support)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Log results
        self.log_message.emit(f"\n✅ OVERALL METRICS:")
        self.log_message.emit(f"   Accuracy:    {accuracy:.2f}%")
        self.log_message.emit(f"   Weighted F1: {weighted_f1:.3f}")

        self.log_message.emit(f"\n📊 PER-CLASS METRICS:")
        class_names = ['DOWN', 'HOLD', 'UP']
        for i, name in enumerate(class_names):
            self.log_message.emit(
                f"   {name:4s} → Precision: {precision[i]:.3f} | "
                f"Recall: {recall[i]:.3f} | F1: {f1[i]:.3f} | Support: {support[i]:,}"
            )

        self.log_message.emit(f"\n🔢 CONFUSION MATRIX:")
        self.log_message.emit(f"{'Actual':>8} | {'DOWN':>8} {'HOLD':>8} {'UP':>8}")
        self.log_message.emit(f"{'-'*40}")
        for i, name in enumerate(class_names):
            row_total = cm[i].sum()
            self.log_message.emit(
                f"{name:>8} | {cm[i][0]:>8,} {cm[i][1]:>8,} {cm[i][2]:>8,}  ({row_total:,})"
            )

        # Prediction distribution
        self.log_message.emit(f"\n🎯 PREDICTION DISTRIBUTION:")
        pred_dist = np.bincount(all_preds, minlength=3)
        target_dist = np.bincount(all_labels, minlength=3)
        self.log_message.emit(f"{'Class':>6} | {'Predicted':>10} {'Actual':>10} {'Diff':>10}")
        self.log_message.emit(f"{'-'*40}")
        for i, name in enumerate(class_names):
            pred_pct = pred_dist[i] / len(all_preds) * 100
            target_pct = target_dist[i] / len(all_labels) * 100
            diff = pred_pct - target_pct
            self.log_message.emit(
                f"{name:>6} | {pred_dist[i]:>6,} ({pred_pct:>4.1f}%) | "
                f"{target_dist[i]:>6,} ({target_pct:>4.1f}%) | {diff:>+5.1f}%"
            )

        metrics = {
            'predictions': all_preds,
            'targets': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'confusion_matrix': cm,
            'support': support
        }

        return metrics

    def save_final_model(self, model, optimizer, history, test_metrics, checkpoint):
        """Save final model and training artifacts"""
        try:
            models_dir = ROOT_DIR / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)

            # Save best model
            best_path = models_dir / 'lstm_balanced_best.pth'
            torch.save(checkpoint, best_path)
            self.log_message.emit(f"✓ Best model saved: {best_path}")

            # Save full model for inference
            full_path = models_dir / 'lstm_balanced_full.pth'
            torch.save({
                'model_type': self.config['model_type'],
                'model_state_dict': model.state_dict(),
                'input_size': self.config.get('input_size', 50),
                'hidden_size': self.config['hidden_size'],
                'num_layers': self.config['num_layers'],
                'num_heads': self.config.get('num_heads', 4),
                'dropout': self.config['dropout'],
                'config': self.config
            }, full_path)
            self.log_message.emit(f"✓ Full model saved: {full_path}")

            # Save history
            history_path = models_dir / 'training_history_balanced.pkl'
            joblib.dump(history, history_path)
            self.log_message.emit(f"✓ History saved: {history_path}")

            # Save test results
            results_path = models_dir / 'test_results_balanced.pkl'
            joblib.dump(test_metrics, results_path)
            self.log_message.emit(f"✓ Test results saved: {results_path}")

        except Exception as e:
            self.log_message.emit(f"⚠️  Error saving artifacts: {str(e)}")


# ============================================================================
# ENHANCED MODEL MANAGEMENT TAB
# ============================================================================
class ModelManagementTab(QWidget):
    """Enhanced Model training and management tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.worker = None
        self.training_results = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("🤖 Model Training & Management")
        title.setObjectName("headerLabel")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2962FF;")
        layout.addWidget(title)

        # Training Configuration Section
        config_group = QGroupBox("⚙️ Training Configuration")
        config_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        config_layout = QGridLayout()

        # Model Type
        config_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_type = QComboBox()
        self.model_type.addItems(['Enhanced LSTM+Attention', 'Standard LSTM+Attention', 'Simple LSTM'])
        self.model_type.setCurrentIndex(0)
        config_layout.addWidget(self.model_type, 0, 1)

        config_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(32, 512)
        self.batch_size.setValue(256)
        self.batch_size.setSingleStep(32)
        config_layout.addWidget(self.batch_size, 0, 3)

        # Row 1
        config_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.00001, 0.01)
        self.learning_rate.setValue(0.0003)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setSingleStep(0.0001)
        config_layout.addWidget(self.learning_rate, 1, 1)

        config_layout.addWidget(QLabel("Hidden Size:"), 1, 2)
        self.hidden_size = QSpinBox()
        self.hidden_size.setRange(64, 512)
        self.hidden_size.setValue(128)
        self.hidden_size.setSingleStep(32)
        config_layout.addWidget(self.hidden_size, 1, 3)

        # Row 2
        config_layout.addWidget(QLabel("Num Layers:"), 2, 0)
        self.num_layers = QSpinBox()
        self.num_layers.setRange(1, 6)
        self.num_layers.setValue(2)
        config_layout.addWidget(self.num_layers, 2, 1)

        config_layout.addWidget(QLabel("Num Attention Heads:"), 2, 2)
        self.num_heads = QSpinBox()
        self.num_heads.setRange(2, 16)
        self.num_heads.setValue(4)
        config_layout.addWidget(self.num_heads, 2, 3)

        # Row 3
        config_layout.addWidget(QLabel("Dropout:"), 3, 0)
        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0.1, 0.7)
        self.dropout.setValue(0.3)
        self.dropout.setDecimals(2)
        self.dropout.setSingleStep(0.05)
        config_layout.addWidget(self.dropout, 3, 1)

        config_layout.addWidget(QLabel("Epochs:"), 3, 2)
        self.epochs = QSpinBox()
        self.epochs.setRange(10, 500)
        self.epochs.setValue(100)
        self.epochs.setSingleStep(10)
        config_layout.addWidget(self.epochs, 3, 3)

        # Row 4
        config_layout.addWidget(QLabel("Early Stop Patience:"), 4, 0)
        self.patience = QSpinBox()
        self.patience.setRange(5, 50)
        self.patience.setValue(15)
        config_layout.addWidget(self.patience, 4, 1)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Model Info Section
        info_group = QGroupBox("ℹ️ Model Information")
        info_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "<b>Architecture:</b> LSTM with Multi-Head Attention + Positional Encoding<br>"
            "<b>Loss:</b> Combined (70% Focal Loss + 30% Label Smoothing)<br>"
            "<b>Optimizer:</b> AdamW (weight_decay=0.03)<br>"
            "<b>Scheduler:</b> ReduceLROnPlateau (patience=7, factor=0.5)<br>"
            "<b>Class Weights:</b> Moderate boost (1.2x DOWN/UP, 1.0x HOLD)"
        )
        info_text.setStyleSheet("color: #D1D4DC; font-size: 11px; padding: 5px;")
        info_layout.addWidget(info_text)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Actions Section
        actions_group = QGroupBox("🎯 Actions")
        actions_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        actions_layout = QHBoxLayout()

        self.train_btn = QPushButton("🏋️ Start Training")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #2962FF;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1E53E5;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888888;
            }
        """)
        self.train_btn.clicked.connect(self.train_model)
        actions_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("⏹️ Stop Training")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #EF5350;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #E53935;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888888;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_training)
        actions_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("💾 Export Results")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #26A69A;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #00897B;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888888;
            }
        """)
        self.save_btn.clicked.connect(self.export_results)
        actions_layout.addWidget(self.save_btn)

        actions_layout.addStretch()
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        # Progress Section
        progress_group = QGroupBox("📊 Training Progress")
        progress_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        progress_layout = QVBoxLayout()

        self.training_progress = QProgressBar()
        self.training_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2A2E39;
                border-radius: 5px;
                text-align: center;
                background-color: #1E222D;
                height: 30px;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #2962FF, stop:1 #1E53E5);
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.training_progress)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(200)
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E222D;
                color: #D1D4DC;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 10px;
            }
        """)
        self.status_text.setPlainText(
            "Ready to train. Configure hyperparameters and click 'Start Training'.\n"
            "Make sure preprocessed data exists in 'preprocessed_data_lstm_1h' folder."
        )
        progress_layout.addWidget(self.status_text)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Results Section - Split into two columns
        results_splitter = QSplitter(Qt.Horizontal)

        # Left: Metrics
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(5, 5, 5, 5)

        metrics_title = QLabel("📈 Model Performance")
        metrics_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2962FF;")
        metrics_layout.addWidget(metrics_title)

        # Metrics grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(8)

        self.model_metric_labels = {}
        metrics_list = [
            ("Current Epoch", "epoch", "#FFB74D"),
            ("Train Accuracy", "train_acc", "#26A69A"),
            ("Val Accuracy", "val_acc", "#2962FF"),
            ("Test Accuracy", "test_acc", "#AB47BC"),
            ("Train Loss", "train_loss", "#26A69A"),
            ("Val Loss", "val_loss", "#2962FF"),
            ("Val F1 Score", "val_f1", "#2962FF"),
            ("Test F1 Score", "test_f1", "#AB47BC"),
            ("Precision", "precision", "#AB47BC"),
            ("Recall", "recall", "#AB47BC"),
            ("Learning Rate", "lr", "#FFB74D"),
            ("Best Val F1", "best_f1", "#26A69A"),
        ]

        for i, (label, key, color) in enumerate(metrics_list):
            label_widget = QLabel(f"{label}:")
            label_widget.setStyleSheet("font-size: 11px; color: #D1D4DC;")
            value_widget = QLabel("--")
            value_widget.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 12px;")
            self.model_metric_labels[key] = value_widget

            metrics_grid.addWidget(label_widget, i, 0, Qt.AlignLeft)
            metrics_grid.addWidget(value_widget, i, 1, Qt.AlignRight)

        metrics_layout.addLayout(metrics_grid)
        metrics_layout.addStretch()

        # Right: Loss Curve
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        chart_title = QLabel("📉 Training Curves")
        chart_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2962FF;")
        chart_layout.addWidget(chart_title)

        self.loss_plot = pg.PlotWidget()
        self.loss_plot.setBackground("#131722")
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self.loss_plot.setLabel("left", "Loss", color="#D1D4DC", size="11pt")
        self.loss_plot.setLabel("bottom", "Epoch", color="#D1D4DC", size="11pt")
        self.loss_plot.addLegend(offset=(10, 10))
        self.loss_plot.getAxis('left').setPen(pg.mkPen(color='#D1D4DC'))
        self.loss_plot.getAxis('bottom').setPen(pg.mkPen(color='#D1D4DC'))
        self.loss_plot.getAxis('left').setTextPen(pg.mkPen(color='#D1D4DC'))
        self.loss_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#D1D4DC'))

        chart_layout.addWidget(self.loss_plot)

        results_splitter.addWidget(metrics_widget)
        results_splitter.addWidget(chart_widget)
        results_splitter.setSizes([400, 800])

        layout.addWidget(results_splitter, stretch=1)

    def train_model(self):
        """Start model training"""
        if LSTMAttentionEnhanced is None:
            QMessageBox.warning(
                self,
                "Model Not Available",
                "Model classes not found. Make sure models.py is in models/ folder"
            )
            return

        # Check if data exists
        data_dir = ROOT_DIR / 'preprocessed_data_lstm_1h'
        if not data_dir.exists():
            QMessageBox.warning(
                self,
                "Data Not Found",
                f"Preprocessed data not found at:\n{data_dir}\n\n"
                "Please run preprocessing first from the Data tab."
            )
            return

        # Map model type
        model_type_map = {
            'Enhanced LSTM+Attention': 'enhanced',
            'Standard LSTM+Attention': 'standard',
            'Simple LSTM': 'simple'
        }

        # Prepare configuration
        config = {
            'model_type': model_type_map[self.model_type.currentText()],
            'batch_size': self.batch_size.value(),
            'learning_rate': self.learning_rate.value(),
            'hidden_size': self.hidden_size.value(),
            'num_layers': self.num_layers.value(),
            'num_heads': self.num_heads.value(),
            'dropout': self.dropout.value(),
            'epochs': self.epochs.value(),
            'patience': self.patience.value(),
        }

        # Confirm start
        reply = QMessageBox.question(
            self,
            "Start Training",
            f"<b>Start training with the following configuration?</b><br><br>"
            f"Model: <b>{self.model_type.currentText()}</b><br>"
            f"Batch Size: {config['batch_size']}<br>"
            f"Learning Rate: {config['learning_rate']}<br>"
            f"Hidden Size: {config['hidden_size']}<br>"
            f"Layers: {config['num_layers']}<br>"
            f"Epochs: {config['epochs']}<br>"
            f"Patience: {config['patience']}<br><br>"
            f"<i>This may take a while depending on your hardware.</i>",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Update UI
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.training_progress.setValue(0)
        self.loss_plot.clear()

        # Reset metrics
        for key in self.model_metric_labels:
            self.model_metric_labels[key].setText("--")

        # Clear status
        status = f"""╔══════════════════════════════════════════════════════════════╗
║              🚀 TRAINING STARTED                            ║
╚══════════════════════════════════════════════════════════════╝

📋 MODEL CONFIGURATION:
   Architecture: {self.model_type.currentText()}
   Hidden Size:  {config['hidden_size']}
   Num Layers:   {config['num_layers']}
   Attention:    {config['num_heads']} heads
   Dropout:      {config['dropout']}
   Bidirectional: Yes

⚙️  TRAINING SETTINGS:
   Learning Rate: {config['learning_rate']}
   Batch Size:    {config['batch_size']}
   Max Epochs:    {config['epochs']}
   Patience:      {config['patience']}

🎯 OPTIMIZATION:
   Optimizer:     AdamW (weight_decay=0.03)
   Loss:          Combined (70% Focal + 30% Label Smoothing)
   Scheduler:     ReduceLROnPlateau
   Class Weights: Moderate (1.2x DOWN/UP, 1.0x HOLD)

════════════════════════════════════════════════════════════════
"""
        self.status_text.setPlainText(status)

        # Create and start worker
        self.worker = TrainingWorker(config)
        self.worker.progress.connect(self.update_progress)
        self.worker.epoch_update.connect(self.update_epoch)
        self.worker.log_message.connect(self.append_log)
        self.worker.finished.connect(self.training_finished)
        self.worker.error.connect(self.training_error)
        self.worker.start()

    def stop_training(self):
        """Stop training"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Stop Training",
                "Are you sure you want to stop training?\n\n"
                "Progress will be lost if you haven't saved the best model yet.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.append_log("\n⚠️  Stopping training...")

    def update_progress(self, value):
        """Update progress bar"""
        self.training_progress.setValue(value)

    def append_log(self, message):
        """Append log message"""
        current_text = self.status_text.toPlainText()
        self.status_text.setPlainText(current_text + "\n" + message)
        # Scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_epoch(self, epoch, metrics):
        """Update metrics and plot after each epoch"""
        # Update metrics display
        self.model_metric_labels['epoch'].setText(f"{epoch}")
        self.model_metric_labels['train_acc'].setText(f"{metrics['train_acc']:.2f}%")
        self.model_metric_labels['val_acc'].setText(f"{metrics['val_acc']:.2f}%")
        self.model_metric_labels['train_loss'].setText(f"{metrics['train_loss']:.4f}")
        self.model_metric_labels['val_loss'].setText(f"{metrics['val_loss']:.4f}")
        self.model_metric_labels['val_f1'].setText(f"{metrics['val_f1']:.3f}")
        self.model_metric_labels['lr'].setText(f"{metrics['lr']:.6f}")

        # Log epoch (every 5 epochs or first epoch)
        if epoch % 5 == 0 or epoch == 1:
            val_f1_per_class = metrics['val_f1_per_class']
            val_preds = metrics['val_preds']
            val_targets = metrics['val_targets']

            log_msg = f"""
{'─'*60}
Epoch {epoch:3d} | Train Loss: {metrics['train_loss']:.4f} | Val Loss: {metrics['val_loss']:.4f}
          | Train Acc:  {metrics['train_acc']:.2f}% | Val Acc:  {metrics['val_acc']:.2f}%
          | Val F1: {metrics['val_f1']:.3f} | LR: {metrics['lr']:.6f}
  F1/class → DOWN: {val_f1_per_class[0]:.3f} | HOLD: {val_f1_per_class[1]:.3f} | UP: {val_f1_per_class[2]:.3f}"""

            # Prediction distribution
            pred_dist = np.bincount(val_preds, minlength=3)
            pred_pct = pred_dist / len(val_preds) * 100
            target_dist = np.bincount(val_targets, minlength=3)
            target_pct = target_dist / len(val_targets) * 100

            log_msg += f"\n  Predictions → DOWN: {pred_pct[0]:.1f}% | HOLD: {pred_pct[1]:.1f}% | UP: {pred_pct[2]:.1f}%"
            log_msg += f"\n  Actual      → DOWN: {target_pct[0]:.1f}% | HOLD: {target_pct[1]:.1f}% | UP: {target_pct[2]:.1f}%"

            # Check for bias
            max_imbalance = max(abs(pred_pct[i] - target_pct[i]) for i in range(3))
            if max_imbalance > 15:
                log_msg += f"\n  ⚠️  Prediction imbalance: {max_imbalance:.1f}% deviation"

            self.append_log(log_msg)

    def training_finished(self, results):
        """Handle training completion"""
        self.training_results = results
        history = results['history']
        test_metrics = results['test_metrics']

        # Update final metrics
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        final_val_f1 = history['val_f1'][-1]

        self.model_metric_labels['train_acc'].setText(f"{final_train_acc:.2f}%")
        self.model_metric_labels['val_acc'].setText(f"{final_val_acc:.2f}%")
        self.model_metric_labels['test_acc'].setText(f"{test_metrics['accuracy']:.2f}%")
        self.model_metric_labels['train_loss'].setText(f"{history['train_loss'][-1]:.4f}")
        self.model_metric_labels['val_loss'].setText(f"{history['val_loss'][-1]:.4f}")
        self.model_metric_labels['val_f1'].setText(f"{final_val_f1:.3f}")
        self.model_metric_labels['test_f1'].setText(f"{test_metrics['weighted_f1']:.3f}")
        self.model_metric_labels['precision'].setText(f"{test_metrics['precision'].mean():.2f}%")
        self.model_metric_labels['recall'].setText(f"{test_metrics['recall'].mean():.2f}%")
        self.model_metric_labels['best_f1'].setText(f"{results['best_val_f1']:.3f}")

        # Plot training curves
        self.loss_plot.clear()
        epochs = np.arange(1, len(history['train_loss']) + 1)

        # Plot losses
        self.loss_plot.plot(
            epochs,
            history['train_loss'],
            pen=pg.mkPen("#26A69A", width=2.5),
            name="Train Loss"
        )
        self.loss_plot.plot(
            epochs,
            history['val_loss'],
            pen=pg.mkPen("#EF5350", width=2.5),
            name="Val Loss"
        )

        # Add confusion matrix and final results to log
        cm = test_metrics['confusion_matrix']
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']

        final_text = f"""
{'═'*60}
✅ TRAINING COMPLETED SUCCESSFULLY!
{'═'*60}

🏆 FINAL TEST SET PERFORMANCE:
   Accuracy:    {test_metrics['accuracy']:.2f}%
   Weighted F1: {test_metrics['weighted_f1']:.3f}

📊 PER-CLASS METRICS:
   Class    Precision  Recall    F1-Score  Support
   ───────────────────────────────────────────────
   DOWN     {precision[0]:6.2f}%   {recall[0]:6.2f}%  {f1[0]:6.2f}%   {test_metrics['support'][0]:>6,}
   HOLD     {precision[1]:6.2f}%   {recall[1]:6.2f}%  {f1[1]:6.2f}%   {test_metrics['support'][1]:>6,}
   UP       {precision[2]:6.2f}%   {recall[2]:6.2f}%  {f1[2]:6.2f}%   {test_metrics['support'][2]:>6,}

🔢 CONFUSION MATRIX:
                    Predicted
              DOWN      HOLD        UP
   Actual  ─────────────────────────────
   DOWN    │ {cm[0,0]:>6,}    {cm[0,1]:>6,}    {cm[0,2]:>6,}
   HOLD    │ {cm[1,0]:>6,}    {cm[1,1]:>6,}    {cm[1,2]:>6,}
   UP      │ {cm[2,0]:>6,}    {cm[2,1]:>6,}    {cm[2,2]:>6,}

📈 PREDICTION DISTRIBUTION:
"""
        pred_dist = np.bincount(test_metrics['predictions'], minlength=3)
        target_dist = np.bincount(test_metrics['targets'], minlength=3)
        total = len(test_metrics['predictions'])

        for i, name in enumerate(['DOWN', 'HOLD', 'UP']):
            pred_pct = pred_dist[i] / total * 100
            target_pct = target_dist[i] / total * 100
            diff = pred_pct - target_pct
            final_text += f"   {name:4s}: Pred {pred_pct:5.1f}% | Actual {target_pct:5.1f}% | Diff {diff:+5.1f}%\n"

        # Balance evaluation
        max_diff = max(abs(pred_dist[i]/total - target_dist[i]/total) * 100 for i in range(3))
        if max_diff < 10:
            final_text += f"\n✅ Excellent balance! (max deviation: {max_diff:.1f}%)"
        elif max_diff < 15:
            final_text += f"\n👍 Good balance (max deviation: {max_diff:.1f}%)"
        else:
            final_text += f"\n⚠️  Moderate imbalance (max deviation: {max_diff:.1f}%)"

        # Performance evaluation
        if test_metrics['weighted_f1'] >= 0.45:
            final_text += f"\n\n🎉 EXCELLENT! Model ready for deployment"
        elif test_metrics['weighted_f1'] >= 0.35:
            final_text += f"\n\n👍 GOOD! Model is usable"
        else:
            final_text += f"\n\n⚠️  NEEDS IMPROVEMENT"

        final_text += f"""

💾 SAVED FILES:
   • models/lstm_balanced_best.pth
   • models/lstm_balanced_full.pth
   • models/training_history_balanced.pkl
   • models/test_results_balanced.pkl

{'═'*60}
"""
        self.append_log(final_text)

        # Enable buttons
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)

        # Show completion dialog
        QMessageBox.information(
            self,
            "✅ Training Complete",
            f"<b>Model training completed successfully!</b><br><br>"
            f"<b>Final Test Performance:</b><br>"
            f"• Accuracy: {test_metrics['accuracy']:.2f}%<br>"
            f"• F1 Score: {test_metrics['weighted_f1']:.3f}<br>"
            f"• DOWN F1: {f1[0]:.2f}%<br>"
            f"• HOLD F1: {f1[1]:.2f}%<br>"
            f"• UP F1: {f1[2]:.2f}%<br><br>"
            f"<i>Model saved to models/ directory</i>"
        )

    def training_error(self, error_msg):
        """Handle training error"""
        self.append_log(f"\n\n{'═'*60}")
        self.append_log(f"❌ ERROR OCCURRED!")
        self.append_log(f"{'═'*60}")
        self.append_log(f"\n{error_msg}")

        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        QMessageBox.critical(
            self,
            "Training Error",
            f"An error occurred during training:\n\n{error_msg[:500]}\n\n"
            f"Check the log for full details."
        )

    def export_results(self):
        """Export training results"""
        if self.training_results is None:
            QMessageBox.warning(
                self,
                "No Results",
                "No training results to export."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Training Results",
            str(ROOT_DIR / "training_results.txt"),
            "Text Files (*.txt)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.status_text.toPlainText())

                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"✅ Training results exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export results:\n\n{str(e)}"
                )
