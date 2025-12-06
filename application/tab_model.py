import numpy as np
import pandas as pd
import pyqtgraph as pg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PySide6.QtCore import Qt, QThread, Signal, QTimer
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
    QTabWidget,
)
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import sys
from datetime import datetime
import sqlite3

# Setup paths - Get the project root directory
if '__file__' in globals():
    ROOT_DIR = Path(__file__).resolve().parent.parent
else:
    ROOT_DIR = Path.cwd()

# Add project root and models dir to path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

models_dir = ROOT_DIR / 'models'
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

# Import models
MODELS_AVAILABLE = False
LSTMAttentionEnhanced = None

try:
    from models.models import LSTMAttentionEnhanced
    MODELS_AVAILABLE = True
    print(f"✓ Models imported successfully")
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("models", ROOT_DIR / "models" / "models.py")
        if spec and spec.loader:
            models_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(models_module)
            LSTMAttentionEnhanced = models_module.LSTMAttentionEnhanced
            MODELS_AVAILABLE = True
            print(f"✓ Models imported successfully (fallback)")
        else:
            print(f"✗ Could not load models.py")
    except Exception as e:
        print(f"✗ Warning: Could not import models: {e}")


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
# IMPROVED FOCAL LOSS
# ============================================================================
class ImprovedFocalLoss(nn.Module):
    """Enhanced Focal Loss with class balancing"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# COMBINED LOSS
# ============================================================================
class CombinedLoss(nn.Module):
    """Combined Focal Loss + Label Smoothing for better generalization"""
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, num_classes=3):
        super(CombinedLoss, self).__init__()
        self.focal = ImprovedFocalLoss(alpha=alpha, gamma=gamma)
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)

        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        smooth_loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
        return 0.7 * focal_loss + 0.3 * smooth_loss


# ============================================================================
# TRAINING WORKER THREAD
# ============================================================================
class TrainingWorker(QThread):
    """Worker thread for model training - ALIGNED with train.py"""
    progress = Signal(int)
    epoch_update = Signal(int, dict)
    log_message = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.should_stop = False

    def stop(self):
        self.should_stop = True
        self.log_message.emit("⚠️  Stop signal received, finishing current epoch...")

    def run(self):
        try:
            self.log_message.emit("🔧 Initializing training environment...")
            self.progress.emit(5)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_message.emit(f"🖥️  Using device: {device}")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.log_message.emit(f"   GPU: {gpu_name}")
                self.log_message.emit(f"   Memory: {gpu_memory:.1f} GB")

            self.progress.emit(10)

            # Load data
            self.log_message.emit("\n📥 Loading training data...")
            result = self.load_data()
            if result is None:
                self.error.emit("Failed to load data. Check if preprocessed data exists.")
                return

            train_loader, val_loader, test_loader, input_size, class_weights = result
            self.progress.emit(20)

            # Create model
            self.log_message.emit(f"\n🤖 Building Enhanced LSTM + Multi-Head Attention model...")
            model = self.create_model(input_size)
            if model is None:
                self.error.emit("Failed to create model.")
                return

            model = model.to(device)
            self.progress.emit(25)

            # Setup optimizer and loss (ALIGNED with train.py)
            self.log_message.emit("\n⚙️  Setting up training components...")

            if class_weights is not None:
                weights = torch.FloatTensor([
                    class_weights[0] * 1.2,
                    class_weights[1] * 1.0,
                    class_weights[2] * 1.2
                ]).to(device)
                self.log_message.emit(f"   📊 Adjusted weights: DOWN={weights[0]:.2f}, HOLD={weights[1]:.2f}, UP={weights[2]:.2f}")
            else:
                weights = torch.FloatTensor([1.2, 1.0, 1.2]).to(device)
                self.log_message.emit(f"   📊 Default weights: DOWN=1.2, HOLD=1.0, UP=1.2")

            criterion = CombinedLoss(alpha=weights, gamma=2.0, smoothing=0.1)
            self.log_message.emit(f"   🎯 Loss: Combined (70% Focal + 30% Label Smoothing)")

            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=0.03,
                betas=(0.9, 0.999)
            )
            self.log_message.emit(f"   🔧 Optimizer: AdamW (lr={self.config['learning_rate']:.6f}, wd=0.03)")

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                threshold=0.001
            )
            self.log_message.emit(f"   📉 Scheduler: ReduceLROnPlateau (patience=7, factor=0.5)")

            history = {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [],
                'val_f1': [], 'val_f1_per_class': [],
                'learning_rate': []
            }

            best_val_f1 = 0.0
            patience_counter = 0
            best_checkpoint = None

            self.log_message.emit("\n" + "="*60)
            self.log_message.emit("🚀 Starting training...")
            self.log_message.emit("="*60)

            # Training loop
            for epoch in range(1, self.config['epochs'] + 1):
                if self.should_stop:
                    self.log_message.emit(f"\n⏹️  Training stopped by user at epoch {epoch}")
                    break

                train_loss, train_acc = self.train_epoch(
                    model, train_loader, criterion, optimizer, device
                )

                val_loss, val_acc, val_f1, val_f1_per_class, val_preds, val_targets = self.validate(
                    model, val_loader, criterion, device
                )

                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_f1)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)
                history['val_f1_per_class'].append(val_f1_per_class)
                history['learning_rate'].append(current_lr)

                progress_pct = 25 + int((epoch / self.config['epochs']) * 60)
                self.progress.emit(min(progress_pct, 85))

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
                        'val_f1_per_class': val_f1_per_class,
                        'input_size': input_size
                    }

                    models_dir = ROOT_DIR / 'models'
                    models_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(best_checkpoint, models_dir / 'lstm_best.pth')

                    if epoch % 5 == 0 or epoch == 1:
                        self.log_message.emit(f"  ✅ Best model saved (F1: {val_f1:.3f})")
                else:
                    patience_counter += 1

                if patience_counter >= self.config['patience']:
                    self.log_message.emit(f"\n⏹️  Early stopping at epoch {epoch}")
                    self.log_message.emit(f"   Best Val F1: {best_val_f1:.3f}")
                    break

            self.progress.emit(90)

            if best_checkpoint:
                model.load_state_dict(best_checkpoint['model_state_dict'])
                self.log_message.emit("\n🔥 Loading best model for final evaluation...")

            self.log_message.emit("\n" + "="*60)
            self.log_message.emit("📊 Evaluating on test set...")
            self.log_message.emit("="*60)
            test_metrics = self.evaluate_model(model, test_loader, device)

            self.progress.emit(95)

            self.log_message.emit("\n💾 Saving final model and artifacts...")
            self.save_final_model(model, history, test_metrics, best_checkpoint, input_size)

            self.progress.emit(100)

            results = {
                'history': history,
                'test_metrics': test_metrics,
                'best_val_f1': best_val_f1,
                'best_checkpoint': best_checkpoint,
                'config': self.config
            }

            self.log_message.emit("\n✅ Training completed successfully!")
            self.finished.emit(results)

        except Exception as e:
            import traceback
            error_details = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.log_message.emit(f"\n❌ ERROR: {str(e)}")
            self.error.emit(error_details)

    def load_data(self):
        """Load preprocessed training data"""
        try:
            data_dir = ROOT_DIR / 'preprocessed_data_lstm_1h'

            if not data_dir.exists():
                self.log_message.emit(f"❌ Data directory not found: {data_dir}")
                return None

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

            self.log_message.emit(f"   ✓ Train: {len(X_train):,} samples")
            self.log_message.emit(f"   ✓ Val:   {len(X_val):,} samples")
            self.log_message.emit(f"   ✓ Test:  {len(X_test):,} samples")
            self.log_message.emit(f"   ✓ Shape: {X_train.shape}")

            self.log_message.emit("\n📊 Target Distribution:")
            for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
                dist = np.bincount(y_split, minlength=3)
                total = len(y_split)
                self.log_message.emit(
                    f"   {split_name:5s} → DOWN: {dist[0]:,} ({dist[0]/total*100:.1f}%) | "
                    f"HOLD: {dist[1]:,} ({dist[1]/total*100:.1f}%) | "
                    f"UP: {dist[2]:,} ({dist[2]/total*100:.1f}%)"
                )

            train_dataset = BTCSequenceDataset(X_train, y_train, masks_train)
            val_dataset = BTCSequenceDataset(X_val, y_val, masks_val)
            test_dataset = BTCSequenceDataset(X_test, y_test, masks_test)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

            input_size = X_train.shape[2]
            return train_loader, val_loader, test_loader, input_size, class_weights

        except Exception as e:
            self.log_message.emit(f"❌ Error loading data: {str(e)}")
            return None

    def create_model(self, input_size):
        """Create model based on configuration"""
        try:
            model = LSTMAttentionEnhanced(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                num_heads=self.config['num_heads'],
                dropout=self.config['dropout'],
                num_classes=3,
                bidirectional=True
            )

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.log_message.emit(f"   ✓ Model: Enhanced LSTM + Multi-Head Attention")
            self.log_message.emit(f"   ✓ Total parameters: {total_params:,}")
            self.log_message.emit(f"   ✓ Trainable parameters: {trainable_params:,}")

            return model

        except Exception as e:
            self.log_message.emit(f"❌ Error creating model: {str(e)}")
            return None

    def train_epoch(self, model, dataloader, criterion, optimizer, device):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for inputs, targets, masks in dataloader:
            if self.should_stop:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, mask=masks)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * accuracy_score(all_targets, all_preds)

        return epoch_loss, epoch_acc

    def validate(self, model, dataloader, criterion, device):
        """Validate model"""
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets, masks in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                masks = masks.to(device)

                outputs = model(inputs, mask=masks)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * accuracy_score(all_targets, all_preds)

        _, _, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )

        _, _, f1_per_class, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )

        return epoch_loss, epoch_acc, f1, f1_per_class, all_preds, all_targets

    def evaluate_model(self, model, dataloader, device):
        """Comprehensive model evaluation"""
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets, masks in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                masks = masks.to(device)

                outputs = model(inputs, mask=masks)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        accuracy = accuracy_score(all_targets, all_preds) * 100
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        weighted_f1 = np.average(f1, weights=support)
        cm = confusion_matrix(all_targets, all_preds)

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
            self.log_message.emit(
                f"{name:>8} | {cm[i][0]:>8,} {cm[i][1]:>8,} {cm[i][2]:>8,}"
            )

        self.log_message.emit(f"\n🎯 PREDICTION DISTRIBUTION:")
        pred_dist = np.bincount(all_preds, minlength=3)
        target_dist = np.bincount(all_targets, minlength=3)
        self.log_message.emit(f"{'Class':>6} | {'Predicted':>10} {'Actual':>10} {'Diff':>10}")
        self.log_message.emit(f"{'-'*40}")
        for i, name in enumerate(class_names):
            pred_pct = pred_dist[i] / len(all_preds) * 100
            target_pct = target_dist[i] / len(all_targets) * 100
            diff = pred_pct - target_pct
            self.log_message.emit(
                f"{name:>6} | {pred_dist[i]:>6,} ({pred_pct:>4.1f}%) | "
                f"{target_dist[i]:>6,} ({target_pct:>4.1f}%) | {diff:>+5.1f}%"
            )

        metrics = {
            'predictions': all_preds,
            'targets': all_targets,
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

    def save_final_model(self, model, history, test_metrics, checkpoint, input_size):
        """Save final model and training artifacts - GUI VERSION (tidak overwrite model asli)"""
        try:
            models_dir = ROOT_DIR / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)

            # Save dengan prefix 'gui_' untuk tidak overwrite model asli
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save best checkpoint dengan timestamp
            best_path = models_dir / f'gui_lstm_best_{timestamp}.pth'
            torch.save(checkpoint, best_path)
            self.log_message.emit(f"   ✓ Best model: {best_path}")

            # Save full model untuk inference (dengan timestamp)
            full_path = models_dir / f'gui_lstm_full_{timestamp}.pth'
            torch.save({
                'model_type': 'enhanced',
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'hidden_size': self.config['hidden_size'],
                'num_layers': self.config['num_layers'],
                'num_heads': self.config['num_heads'],
                'dropout': self.config['dropout']
            }, full_path)
            self.log_message.emit(f"   ✓ Full model: {full_path}")

            # Save training history (dengan timestamp)
            history_path = models_dir / f'gui_training_history_{timestamp}.pkl'
            joblib.dump(history, history_path)
            self.log_message.emit(f"   ✓ History: {history_path}")

            # Save test results (dengan timestamp)
            results_path = models_dir / f'gui_test_results_{timestamp}.pkl'
            joblib.dump(test_metrics, results_path)
            self.log_message.emit(f"   ✓ Test results: {results_path}")

            self.log_message.emit(f"\n⚠️  NOTE: GUI training saved with 'gui_' prefix and timestamp")
            self.log_message.emit(f"   Original model files (lstm_*.pth) are NOT modified")

        except Exception as e:
            self.log_message.emit(f"⚠️  Error saving artifacts: {str(e)}")


# ============================================================================
# PREDICTION WORKER THREAD
# ============================================================================
class PredictionWorker(QThread):
    """Worker thread for making predictions - ALIGNED with predict.py"""
    log_message = Signal(str)
    prediction_result = Signal(dict)
    error = Signal(str)

    def __init__(self, model_path, db_path):
        super().__init__()
        self.model_path = model_path
        self.db_path = db_path

    def run(self):
        try:
            self.log_message.emit("="*70)
            self.log_message.emit("🔮 BTC PRICE PREDICTION - LIVE")
            self.log_message.emit("="*70)
            self.log_message.emit(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Load model
            self.log_message.emit("\n📦 Loading model...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            checkpoint = torch.load(str(self.model_path), map_location=device)

            model = LSTMAttentionEnhanced(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                num_heads=checkpoint.get('num_heads', 4),
                dropout=checkpoint['dropout'],
                num_classes=3,
                bidirectional=True
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            self.log_message.emit("   ✓ Model loaded successfully")

            # Load preprocessing artifacts
            data_dir = ROOT_DIR / 'preprocessed_data_lstm_1h'
            scaler_data = joblib.load(data_dir / 'scalers.pkl')
            scaler = scaler_data['robust'] if isinstance(scaler_data, dict) else scaler_data
            feature_cols = joblib.load(data_dir / 'feature_cols.pkl')
            sequence_length = joblib.load(data_dir / 'sequence_length.pkl')

            self.log_message.emit(f"   ✓ Sequence length: {sequence_length}")

            # Fetch latest data
            self.log_message.emit("\n📥 Fetching latest data from database...")
            df = self.fetch_latest_data(sequence_length + 168)

            if df is None or len(df) < sequence_length:
                raise ValueError(f"Insufficient data: need {sequence_length}, got {len(df) if df is not None else 0}")

            # Preprocess
            self.log_message.emit("\n⚙️  Preprocessing data...")
            X_seq, mask, close_price = self.preprocess_data(df, feature_cols, scaler, sequence_length)

            # Predict
            self.log_message.emit("\n🔮 Making prediction...")
            X_tensor = torch.FloatTensor(X_seq).to(device)
            mask_tensor = torch.FloatTensor(mask).to(device)

            with torch.no_grad():
                outputs = model(X_tensor, mask=mask_tensor)
                probs = torch.softmax(outputs, dim=1)

            probs_np = probs.cpu().numpy()[0]
            pred_class = np.argmax(probs_np)
            class_names = ['DOWN', 'HOLD', 'UP']
            pred_label = class_names[pred_class]
            confidence = probs_np[pred_class]

            result = {
                'prediction': pred_label,
                'confidence': confidence,
                'prob_down': probs_np[0],
                'prob_hold': probs_np[1],
                'prob_up': probs_np[2],
                'close_price': close_price,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Log results
            self.log_message.emit(f"\n{'='*70}")
            self.log_message.emit(f"📊 PREDICTION RESULT")
            self.log_message.emit(f"{'='*70}")
            self.log_message.emit(f"\n💰 Current Price: ${close_price:,.2f}")
            self.log_message.emit(f"\n🎯 Prediction: {pred_label}")
            self.log_message.emit(f"   Confidence: {confidence:.1%}")
            self.log_message.emit(f"\n📈 Probabilities:")
            self.log_message.emit(f"   📉 DOWN: {probs_np[0]:.1%}")
            self.log_message.emit(f"   ➖ HOLD: {probs_np[1]:.1%}")
            self.log_message.emit(f"   📈 UP:   {probs_np[2]:.1%}")

            # Signal strength
            if confidence >= 0.65:
                strength = "🔥 STRONG"
            elif confidence >= 0.50:
                strength = "⚡ MODERATE"
            elif confidence >= 0.40:
                strength = "⚠️  WEAK"
            else:
                strength = "❌ VERY WEAK"

            self.log_message.emit(f"\n{strength} Signal Strength")
            self.log_message.emit(f"{'='*70}")

            # Save to database
            self.save_prediction(result)

            self.prediction_result.emit(result)

        except Exception as e:
            import traceback
            error_details = f"{str(e)}\n\n{traceback.format_exc()}"
            self.log_message.emit(f"\n❌ ERROR: {str(e)}")
            self.error.emit(error_details)

    def fetch_latest_data(self, limit):
        """Fetch latest data from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))

            # Load OHLCV
            query_ohlcv = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM btc_1h
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            df_ohlcv = pd.read_sql_query(query_ohlcv, conn)

            # Load Technical Indicators
            query_indicators = f"""
                SELECT * FROM smc_btc_1h_technical_indicators
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            df_indicators = pd.read_sql_query(query_indicators, conn)

            # Load Optimized Features
            query_optimized = f"""
                SELECT * FROM smc_btc_1h_optimized_features
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            df_optimized = pd.read_sql_query(query_optimized, conn)

            conn.close()

            # Reverse to chronological order
            df_ohlcv = df_ohlcv.iloc[::-1].reset_index(drop=True)
            df_indicators = df_indicators.iloc[::-1].reset_index(drop=True)
            df_optimized = df_optimized.iloc[::-1].reset_index(drop=True)

            # Convert timestamps
            df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp']).dt.tz_localize(None)
            df_indicators['timestamp'] = pd.to_datetime(df_indicators['timestamp']).dt.tz_localize(None)
            df_optimized['timestamp'] = pd.to_datetime(df_optimized['timestamp']).dt.tz_localize(None)

            # Merge
            df = pd.merge(df_ohlcv, df_indicators, on='timestamp', how='inner')
            df = pd.merge(df, df_optimized, on='timestamp', how='inner')
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Handle NaN (ALIGNED with preprocessing)
            nan_counts_before = df.isna().sum().sum()
            if nan_counts_before > 0:
                df = df.ffill(limit=2)
                df = df.bfill(limit=2)
                rows_before = len(df)
                df = df.dropna()
                rows_dropped = rows_before - len(df)
                self.log_message.emit(f"   ✓ Cleaned {nan_counts_before:,} NaN values ({rows_dropped:,} rows dropped)")

            self.log_message.emit(f"   ✓ Loaded: {len(df)} candles")
            self.log_message.emit(f"   ✓ Latest price: ${df['close'].iloc[-1]:,.2f}")

            return df

        except Exception as e:
            self.log_message.emit(f"❌ Error fetching data: {str(e)}")
            return None

    def preprocess_data(self, df, feature_cols, scaler, sequence_length):
        """Preprocess data for prediction"""
        if len(df) > sequence_length:
            df = df.iloc[-sequence_length:].reset_index(drop=True)

        close_price = df['close'].iloc[-1]

        # Validate features
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing {len(missing_cols)} required features")

        # Select and scale features
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        X_scaled = np.clip(X_scaled, -10, 10)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequence
        X_seq = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1])
        mask = np.ones((1, X_scaled.shape[0]))

        self.log_message.emit(f"   ✓ Sequence shape: {X_seq.shape}")

        return X_seq, mask, close_price

    def save_prediction(self, result):
        """Save prediction to database - GUI VERSION (tabel terpisah)"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Gunakan tabel terpisah untuk GUI predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gui_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    prob_down REAL NOT NULL,
                    prob_hold REAL NOT NULL,
                    prob_up REAL NOT NULL,
                    close_price REAL NOT NULL,
                    model_type TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                INSERT INTO gui_predictions (
                    timestamp, prediction, confidence,
                    prob_down, prob_hold, prob_up,
                    close_price, model_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['timestamp'],
                result['prediction'],
                float(result['confidence']),
                float(result['prob_down']),
                float(result['prob_hold']),
                float(result['prob_up']),
                float(result['close_price']),
                'enhanced_gui'
            ))

            conn.commit()
            conn.close()

            self.log_message.emit("\n💾 Prediction saved to gui_predictions table")

        except Exception as e:
            self.log_message.emit(f"\n⚠️  Warning: Could not save to database: {str(e)}")


# ============================================================================
# MODEL MANAGEMENT TAB
# ============================================================================
class ModelManagementTab(QWidget):
    """Model training and management tab - ALIGNED with train.py and predict.py"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.worker = None
        self.prediction_worker = None
        self.training_results = None
        self.setup_ui()

    def setup_ui(self):
        """Setup UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("🤖 Model Training & Management")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2962FF;")
        layout.addWidget(title)

        # Check if models are available
        if not MODELS_AVAILABLE:
            warning = QLabel(
                "⚠️ Warning: Model classes not found!\n"
                "Make sure models.py is in the models/ folder."
            )
            warning.setStyleSheet(
                "background-color: #3d2900; color: #FFB74D; "
                "padding: 15px; border-radius: 5px; font-weight: bold;"
            )
            layout.addWidget(warning)

        # Create tab widget for Training and Prediction
        tab_widget = QTabWidget()
        tab_widget.addTab(self.create_training_tab(), "🚀 Training")
        tab_widget.addTab(self.create_prediction_tab(), "🔮 Prediction")
        layout.addWidget(tab_widget)

    def create_training_tab(self):
        """Create training tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Training Configuration
        config_group = self.create_config_section()
        layout.addWidget(config_group)

        # Model Info
        info_group = self.create_info_section()
        layout.addWidget(info_group)

        # Actions
        actions_group = self.create_actions_section()
        layout.addWidget(actions_group)

        # Progress
        progress_group = self.create_progress_section()
        layout.addWidget(progress_group)

        # Results
        results_splitter = self.create_results_section()
        layout.addWidget(results_splitter, stretch=1)

        return widget

    def create_prediction_tab(self):
        """Create prediction tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Model selection
        model_group = QGroupBox("🎯 Model Selection")
        model_layout = QHBoxLayout()

        model_layout.addWidget(QLabel("Select Model:"))

        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(400)
        self.model_selector.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                background-color: #1E222D;
                color: #D1D4DC;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #2962FF;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #1E222D;
                color: #D1D4DC;
                selection-background-color: #2962FF;
                border: 2px solid #2A2E39;
            }
        """)
        model_layout.addWidget(self.model_selector, stretch=1)

        self.refresh_models_btn = QPushButton("🔄 Refresh")
        self.refresh_models_btn.setStyleSheet("""
            QPushButton {
                background-color: #26A69A;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00897B;
            }
        """)
        self.refresh_models_btn.clicked.connect(self.refresh_model_list)
        model_layout.addWidget(self.refresh_models_btn)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Prediction controls
        controls_group = QGroupBox("🔮 Prediction Controls")
        controls_layout = QHBoxLayout()

        self.predict_btn = QPushButton("🚀 Run Prediction")
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #2962FF;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1E53E5;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888888;
            }
        """)
        self.predict_btn.clicked.connect(self.run_prediction)
        controls_layout.addWidget(self.predict_btn)

        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Prediction result display
        result_group = QGroupBox("📊 Prediction Result")
        result_layout = QVBoxLayout()

        # Model info label
        self.model_info_label = QLabel("Model: Not selected")
        self.model_info_label.setStyleSheet(
            "color: #FFB74D; font-size: 11px; font-style: italic; padding: 5px;"
        )
        result_layout.addWidget(self.model_info_label)

        # Metrics grid
        metrics_grid = QGridLayout()
        self.pred_metric_labels = {}

        pred_metrics = [
            ("Timestamp", "timestamp", "#FFB74D"),
            ("Current Price", "price", "#26A69A"),
            ("Prediction", "prediction", "#2962FF"),
            ("Confidence", "confidence", "#AB47BC"),
            ("DOWN Probability", "prob_down", "#EF5350"),
            ("HOLD Probability", "prob_hold", "#FFB74D"),
            ("UP Probability", "prob_up", "#26A69A"),
        ]

        for i, (label, key, color) in enumerate(pred_metrics):
            label_widget = QLabel(f"{label}:")
            label_widget.setStyleSheet("font-size: 12px; color: #D1D4DC;")
            value_widget = QLabel("--")
            value_widget.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 13px;")
            self.pred_metric_labels[key] = value_widget

            metrics_grid.addWidget(label_widget, i, 0, Qt.AlignLeft)
            metrics_grid.addWidget(value_widget, i, 1, Qt.AlignRight)

        result_layout.addLayout(metrics_grid)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # Prediction log
        log_group = QGroupBox("📝 Prediction Log")
        log_layout = QVBoxLayout()

        self.prediction_log = QTextEdit()
        self.prediction_log.setReadOnly(True)
        self.prediction_log.setMinimumHeight(400)
        self.prediction_log.setStyleSheet("""
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
        self.prediction_log.setPlainText(
            "🔮 Ready to make predictions!\n\n"
            "Requirements:\n"
            "• Select a trained model from dropdown above\n"
            "• Database with latest data (data/btc_ohlcv.db)\n"
            "• Preprocessed data artifacts\n\n"
            "Model Types:\n"
            "• Production: Official model from train.py\n"
            "• GUI Training: Models trained in GUI (with timestamp)\n\n"
            "NOTE:\n"
            "• Predictions are saved to 'gui_predictions' table\n"
            "• Original 'predictions' table is NOT modified\n\n"
            "Steps:\n"
            "1. Click 'Refresh' to load available models\n"
            "2. Select model from dropdown\n"
            "3. Click 'Run Prediction' to start"
        )

        log_layout.addWidget(self.prediction_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Initialize model list
        self.refresh_model_list()

        return widget

    def create_config_section(self):
        """Create configuration section"""
        group = QGroupBox("⚙️ Training Configuration")
        group.setStyleSheet("""
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

        grid = QGridLayout()

        # Row 0
        grid.addWidget(QLabel("Batch Size:"), 0, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(32, 512)
        self.batch_size.setValue(256)
        self.batch_size.setSingleStep(32)
        grid.addWidget(self.batch_size, 0, 1)

        grid.addWidget(QLabel("Learning Rate:"), 0, 2)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.00001, 0.01)
        self.learning_rate.setValue(0.0003)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setSingleStep(0.0001)
        grid.addWidget(self.learning_rate, 0, 3)

        # Row 1
        grid.addWidget(QLabel("Hidden Size:"), 1, 0)
        self.hidden_size = QSpinBox()
        self.hidden_size.setRange(64, 512)
        self.hidden_size.setValue(128)
        self.hidden_size.setSingleStep(32)
        grid.addWidget(self.hidden_size, 1, 1)

        grid.addWidget(QLabel("Num Layers:"), 1, 2)
        self.num_layers = QSpinBox()
        self.num_layers.setRange(1, 6)
        self.num_layers.setValue(2)
        grid.addWidget(self.num_layers, 1, 3)

        # Row 2
        grid.addWidget(QLabel("Attention Heads:"), 2, 0)
        self.num_heads = QSpinBox()
        self.num_heads.setRange(2, 16)
        self.num_heads.setValue(4)
        grid.addWidget(self.num_heads, 2, 1)

        grid.addWidget(QLabel("Dropout:"), 2, 2)
        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0.1, 0.7)
        self.dropout.setValue(0.3)
        self.dropout.setDecimals(2)
        self.dropout.setSingleStep(0.05)
        grid.addWidget(self.dropout, 2, 3)

        # Row 3
        grid.addWidget(QLabel("Max Epochs:"), 3, 0)
        self.epochs = QSpinBox()
        self.epochs.setRange(10, 500)
        self.epochs.setValue(100)
        self.epochs.setSingleStep(10)
        grid.addWidget(self.epochs, 3, 1)

        grid.addWidget(QLabel("Early Stop Patience:"), 3, 2)
        self.patience = QSpinBox()
        self.patience.setRange(5, 50)
        self.patience.setValue(15)
        grid.addWidget(self.patience, 3, 3)

        group.setLayout(grid)
        return group

    def create_info_section(self):
        """Create model info section"""
        group = QGroupBox("ℹ️ Model Information")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QVBoxLayout()

        info_text = QLabel(
            "<b>Architecture:</b> Bidirectional LSTM + Multi-Head Attention + Positional Encoding<br>"
            "<b>Loss Function:</b> Combined (70% Focal Loss + 30% Label Smoothing)<br>"
            "<b>Optimizer:</b> AdamW (weight_decay=0.03, gradient clipping=1.0)<br>"
            "<b>Scheduler:</b> ReduceLROnPlateau (patience=7, factor=0.5)<br>"
            "<b>Class Weights:</b> Moderate boost (1.2x DOWN/UP, 1.0x HOLD)<br>"
            "<b>Features:</b> Residual connections, Layer Normalization, Mask support"
        )
        info_text.setStyleSheet("color: #D1D4DC; font-size: 11px; padding: 5px;")
        info_text.setWordWrap(True)
        layout.addWidget(info_text)

        group.setLayout(layout)
        return group

    def create_actions_section(self):
        """Create actions section"""
        group = QGroupBox("🎯 Actions")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QHBoxLayout()

        # Train button
        self.train_btn = QPushButton("🚀 Start Training")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #2962FF;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1E53E5;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888888;
            }
        """)
        self.train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.train_btn)

        # Stop button
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
        layout.addWidget(self.stop_btn)

        # Export button
        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #26A69A;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00897B;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888888;
            }
        """)
        self.export_btn.clicked.connect(self.export_results)
        layout.addWidget(self.export_btn)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_progress_section(self):
        """Create progress section - DIPERBESAR"""
        group = QGroupBox("📊 Training Progress")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QVBoxLayout()

        # Progress bar - DIPERBESAR
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2A2E39;
                border-radius: 5px;
                text-align: center;
                background-color: #1E222D;
                height: 35px;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #2962FF, stop:1 #1E53E5);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Status log - DIPERBESAR SIGNIFIKAN
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMinimumHeight(550)  # Diperbesar dari 450 ke 550
        self.status_log.setStyleSheet("""
            QTextEdit {
                background-color: #1E222D;
                color: #D1D4DC;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                padding: 12px;
                line-height: 1.4;
            }
        """)
        self.status_log.setPlainText(
            "🎯 Ready to train!\n\n"
            "Configuration:\n"
            "• Ensure preprocessed data exists in 'preprocessed_data_lstm_1h' folder\n"
            "• Adjust hyperparameters above\n"
            "• Click 'Start Training' to begin\n\n"
            "NOTE - GUI Training:\n"
            "• Models saved with 'gui_' prefix and timestamp\n"
            "• Original models (lstm_*.pth) are NOT modified\n"
            "• Safe to experiment without affecting production models\n\n"
            "The training process will:\n"
            "1. Load and validate preprocessed data\n"
            "2. Build the model architecture\n"
            "3. Train with early stopping\n"
            "4. Evaluate on test set\n"
            "5. Save model as gui_lstm_best_TIMESTAMP.pth"
        )
        layout.addWidget(self.status_log)

        group.setLayout(layout)
        return group

    def create_results_section(self):
        """Create results section with metrics and charts"""
        splitter = QSplitter(Qt.Horizontal)

        # Left: Metrics
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(5, 5, 5, 5)

        metrics_title = QLabel("📈 Performance Metrics")
        metrics_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2962FF;")
        metrics_layout.addWidget(metrics_title)

        # Metrics grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(8)

        self.metric_labels = {}
        metrics_list = [
            ("Current Epoch", "epoch", "#FFB74D"),
            ("Train Accuracy", "train_acc", "#26A69A"),
            ("Val Accuracy", "val_acc", "#2962FF"),
            ("Test Accuracy", "test_acc", "#AB47BC"),
            ("Train Loss", "train_loss", "#26A69A"),
            ("Val Loss", "val_loss", "#2962FF"),
            ("Val F1 Score", "val_f1", "#2962FF"),
            ("Test F1 Score", "test_f1", "#AB47BC"),
            ("Precision (avg)", "precision", "#AB47BC"),
            ("Recall (avg)", "recall", "#AB47BC"),
            ("Learning Rate", "lr", "#FFB74D"),
            ("Best Val F1", "best_f1", "#26A69A"),
        ]

        for i, (label, key, color) in enumerate(metrics_list):
            label_widget = QLabel(f"{label}:")
            label_widget.setStyleSheet("font-size: 11px; color: #D1D4DC;")
            value_widget = QLabel("--")
            value_widget.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 12px;")
            self.metric_labels[key] = value_widget

            metrics_grid.addWidget(label_widget, i, 0, Qt.AlignLeft)
            metrics_grid.addWidget(value_widget, i, 1, Qt.AlignRight)

        metrics_layout.addLayout(metrics_grid)
        metrics_layout.addStretch()

        # Right: Charts
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        chart_title = QLabel("📉 Training Curves")
        chart_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2962FF;")
        chart_layout.addWidget(chart_title)

        # Loss plot
        self.loss_plot = pg.PlotWidget()
        self.loss_plot.setBackground("#131722")
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self.loss_plot.setLabel("left", "Loss", color="#D1D4DC", size="11pt")
        self.loss_plot.setLabel("bottom", "Epoch", color="#D1D4DC", size="11pt")
        self.loss_plot.addLegend(offset=(10, 10))

        for axis in ['left', 'bottom']:
            self.loss_plot.getAxis(axis).setPen(pg.mkPen(color='#D1D4DC'))
            self.loss_plot.getAxis(axis).setTextPen(pg.mkPen(color='#D1D4DC'))

        chart_layout.addWidget(self.loss_plot)

        splitter.addWidget(metrics_widget)
        splitter.addWidget(chart_widget)
        splitter.setSizes([350, 850])  # Chart lebih besar

        return splitter

    def start_training(self):
        """Start model training"""
        if not MODELS_AVAILABLE:
            QMessageBox.critical(
                self,
                "Models Not Available",
                f"<b>Model classes not found!</b><br><br>"
                f"Expected location:<br>"
                f"<code>{ROOT_DIR / 'models' / 'models.py'}</code><br><br>"
                f"Please ensure:<br>"
                f"• File <b>models.py</b> exists in <b>models/</b> folder<br>"
                f"• File contains <b>LSTMAttentionEnhanced</b> class<br>"
                f"• No syntax errors in models.py"
            )
            return

        data_dir = ROOT_DIR / 'preprocessed_data_lstm_1h'
        if not data_dir.exists():
            QMessageBox.warning(
                self,
                "Data Not Found",
                f"<b>Preprocessed data not found!</b><br><br>"
                f"Expected location:<br>"
                f"<code>{data_dir}</code><br><br>"
                f"Please run preprocessing first from the Data tab."
            )
            return

        config = {
            'batch_size': self.batch_size.value(),
            'learning_rate': self.learning_rate.value(),
            'hidden_size': self.hidden_size.value(),
            'num_layers': self.num_layers.value(),
            'num_heads': self.num_heads.value(),
            'dropout': self.dropout.value(),
            'epochs': self.epochs.value(),
            'patience': self.patience.value(),
        }

        reply = QMessageBox.question(
            self,
            "Start Training",
            f"<b>Start training with the following configuration?</b><br><br>"
            f"<b>Model:</b> Enhanced LSTM + Multi-Head Attention<br><br>"
            f"<b>Architecture:</b><br>"
            f"  • Hidden Size: {config['hidden_size']}<br>"
            f"  • Layers: {config['num_layers']}<br>"
            f"  • Attention Heads: {config['num_heads']}<br>"
            f"  • Dropout: {config['dropout']}<br>"
            f"  • Bidirectional: Yes<br><br>"
            f"<b>Training:</b><br>"
            f"  • Batch Size: {config['batch_size']}<br>"
            f"  • Learning Rate: {config['learning_rate']}<br>"
            f"  • Max Epochs: {config['epochs']}<br>"
            f"  • Patience: {config['patience']}<br><br>"
            f"<i>Training may take a while depending on your hardware.</i>",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.loss_plot.clear()

        for key in self.metric_labels:
            self.metric_labels[key].setText("--")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = f"""╔══════════════════════════════════════════════════════════════╗
║              🚀 TRAINING STARTED                            ║
║              {timestamp}                          ║
╚══════════════════════════════════════════════════════════════╝

📋 MODEL CONFIGURATION:
   Architecture: Enhanced LSTM + Multi-Head Attention
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
   Scheduler:     ReduceLROnPlateau (patience=7, factor=0.5)
   Class Weights: Moderate (1.2x DOWN/UP, 1.0x HOLD)
   Grad Clip:     1.0

════════════════════════════════════════════════════════════════
"""
        self.status_log.setPlainText(status)

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
                "The best model so far will be saved.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.append_log("\n⚠️  Stop requested, finishing current epoch...")

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def append_log(self, message):
        """Append log message"""
        self.status_log.append(message)
        scrollbar = self.status_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_epoch(self, epoch, metrics):
        """Update metrics and plots after each epoch"""
        self.metric_labels['epoch'].setText(f"{epoch}")
        self.metric_labels['train_acc'].setText(f"{metrics['train_acc']:.2f}%")
        self.metric_labels['val_acc'].setText(f"{metrics['val_acc']:.2f}%")
        self.metric_labels['train_loss'].setText(f"{metrics['train_loss']:.4f}")
        self.metric_labels['val_loss'].setText(f"{metrics['val_loss']:.4f}")
        self.metric_labels['val_f1'].setText(f"{metrics['val_f1']:.3f}")
        self.metric_labels['lr'].setText(f"{metrics['lr']:.6f}")

        if epoch % 5 == 0 or epoch == 1:
            val_f1_per_class = metrics['val_f1_per_class']
            val_preds = metrics['val_preds']
            val_targets = metrics['val_targets']

            log_msg = f"""
{'─'*60}
📍 Epoch {epoch:3d}
   Train → Loss: {metrics['train_loss']:.4f} | Acc: {metrics['train_acc']:.2f}%
   Val   → Loss: {metrics['val_loss']:.4f} | Acc: {metrics['val_acc']:.2f}% | F1: {metrics['val_f1']:.3f}
   F1/class → DOWN: {val_f1_per_class[0]:.3f} | HOLD: {val_f1_per_class[1]:.3f} | UP: {val_f1_per_class[2]:.3f}
   LR: {metrics['lr']:.6f}"""

            pred_dist = np.bincount(val_preds, minlength=3)
            pred_pct = pred_dist / len(val_preds) * 100
            target_dist = np.bincount(val_targets, minlength=3)
            target_pct = target_dist / len(val_targets) * 100

            log_msg += f"\n   Predictions → DOWN: {pred_pct[0]:.1f}% | HOLD: {pred_pct[1]:.1f}% | UP: {pred_pct[2]:.1f}%"
            log_msg += f"\n   Actual      → DOWN: {target_pct[0]:.1f}% | HOLD: {target_pct[1]:.1f}% | UP: {target_pct[2]:.1f}%"

            max_imbalance = max(abs(pred_pct[i] - target_pct[i]) for i in range(3))
            if max_imbalance > 15:
                log_msg += f"\n   ⚠️  Imbalance: {max_imbalance:.1f}% deviation"

            self.append_log(log_msg)

    def training_finished(self, results):
        """Handle training completion"""
        self.training_results = results
        history = results['history']
        test_metrics = results['test_metrics']

        final_epoch = len(history['train_loss'])
        self.metric_labels['epoch'].setText(f"{final_epoch}")
        self.metric_labels['train_acc'].setText(f"{history['train_acc'][-1]:.2f}%")
        self.metric_labels['val_acc'].setText(f"{history['val_acc'][-1]:.2f}%")
        self.metric_labels['test_acc'].setText(f"{test_metrics['accuracy']:.2f}%")
        self.metric_labels['train_loss'].setText(f"{history['train_loss'][-1]:.4f}")
        self.metric_labels['val_loss'].setText(f"{history['val_loss'][-1]:.4f}")
        self.metric_labels['val_f1'].setText(f"{history['val_f1'][-1]:.3f}")
        self.metric_labels['test_f1'].setText(f"{test_metrics['weighted_f1']:.3f}")
        self.metric_labels['precision'].setText(f"{test_metrics['precision'].mean():.2f}%")
        self.metric_labels['recall'].setText(f"{test_metrics['recall'].mean():.2f}%")
        self.metric_labels['best_f1'].setText(f"{results['best_val_f1']:.3f}")

        self.loss_plot.clear()
        epochs = np.arange(1, len(history['train_loss']) + 1)

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

        cm = test_metrics['confusion_matrix']
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']

        final_report = f"""
{'═'*60}
✅ TRAINING COMPLETED!
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
   ─────────────────────────────
   DOWN │ {cm[0,0]:>6,}    {cm[0,1]:>6,}    {cm[0,2]:>6,}
   HOLD │ {cm[1,0]:>6,}    {cm[1,1]:>6,}    {cm[1,2]:>6,}
   UP   │ {cm[2,0]:>6,}    {cm[2,1]:>6,}    {cm[2,2]:>6,}

📈 PREDICTION DISTRIBUTION:
"""
        pred_dist = np.bincount(test_metrics['predictions'], minlength=3)
        target_dist = np.bincount(test_metrics['targets'], minlength=3)
        total = len(test_metrics['predictions'])

        for i, name in enumerate(['DOWN', 'HOLD', 'UP']):
            pred_pct = pred_dist[i] / total * 100
            target_pct = target_dist[i] / total * 100
            diff = pred_pct - target_pct
            final_report += f"   {name:4s}: Pred {pred_pct:5.1f}% | Actual {target_pct:5.1f}% | Diff {diff:+5.1f}%\n"

        max_diff = max(abs(pred_dist[i]/total - target_dist[i]/total) * 100 for i in range(3))
        if max_diff < 10:
            final_report += f"\n✅ Excellent balance! (max deviation: {max_diff:.1f}%)"
        elif max_diff < 15:
            final_report += f"\n👍 Good balance (max deviation: {max_diff:.1f}%)"
        else:
            final_report += f"\n⚠️  Moderate imbalance (max deviation: {max_diff:.1f}%)"

        if test_metrics['weighted_f1'] >= 0.45:
            final_report += "\n\n🎉 EXCELLENT! Model ready for deployment"
        elif test_metrics['weighted_f1'] >= 0.35:
            final_report += "\n\n👍 GOOD! Model is usable"
        else:
            final_report += "\n\n⚠️  NEEDS IMPROVEMENT - Consider tuning hyperparameters"

        final_report += f"""

💾 SAVED FILES (GUI Version - Safe):
   • models/gui_lstm_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth
   • models/gui_lstm_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth
   • models/gui_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl
   • models/gui_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl

⚠️  NOTE: Original model files (lstm_*.pth) are NOT modified

{'═'*60}
"""
        self.append_log(final_report)

        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(True)

        QMessageBox.information(
            self,
            "✅ Training Complete",
            f"<b>Model training completed successfully!</b><br><br>"
            f"<b>Final Test Performance:</b><br>"
            f"• Accuracy: <b>{test_metrics['accuracy']:.2f}%</b><br>"
            f"• F1 Score: <b>{test_metrics['weighted_f1']:.3f}</b><br><br>"
            f"<b>Per-Class F1:</b><br>"
            f"• DOWN: {f1[0]:.2f}%<br>"
            f"• HOLD: {f1[1]:.2f}%<br>"
            f"• UP: {f1[2]:.2f}%<br><br>"
            f"<i>Model artifacts saved with 'gui_' prefix<br>"
            f"Original models are NOT modified</i>"
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
        """Export training results to file"""
        if self.training_results is None:
            QMessageBox.warning(
                self,
                "No Results",
                "No training results available to export.\n"
                "Complete a training session first."
            )
            return

        default_path = str(ROOT_DIR / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Training Results",
            default_path,
            "Text Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.status_log.toPlainText())

            QMessageBox.information(
                self,
                "Export Complete",
                f"✅ Training results exported successfully!\n\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results:\n\n{str(e)}"
            )

    def refresh_model_list(self):
        """Refresh available models list"""
        self.model_selector.clear()

        models_dir = ROOT_DIR / 'models'
        if not models_dir.exists():
            self.model_selector.addItem("⚠️ No models directory found")
            self.predict_btn.setEnabled(False)
            return

        available_models = []

        # Find production model
        prod_model = models_dir / 'lstm_full.pth'
        if prod_model.exists():
            try:
                # Get file modification time
                mod_time = datetime.fromtimestamp(prod_model.stat().st_mtime)
                size_mb = prod_model.stat().st_size / (1024 * 1024)

                available_models.append({
                    'path': prod_model,
                    'name': f"🏭 Production Model (lstm_full.pth)",
                    'display': f"🏭 Production | Modified: {mod_time.strftime('%Y-%m-%d %H:%M')} | Size: {size_mb:.2f} MB",
                    'type': 'production',
                    'time': mod_time
                })
            except Exception as e:
                print(f"Error reading production model: {e}")

        # Find GUI models
        gui_models = sorted(
            models_dir.glob('gui_lstm_full_*.pth'),
            key=lambda x: x.stat().st_mtime,
            reverse=True  # Newest first
        )

        for model_path in gui_models:
            try:
                # Extract timestamp from filename
                filename = model_path.stem
                timestamp_str = filename.replace('gui_lstm_full_', '')

                # Parse timestamp
                try:
                    model_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    time_display = model_time.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    time_display = timestamp_str

                size_mb = model_path.stat().st_size / (1024 * 1024)

                available_models.append({
                    'path': model_path,
                    'name': model_path.name,
                    'display': f"🧪 GUI Training | {time_display} | Size: {size_mb:.2f} MB",
                    'type': 'gui',
                    'time': model_time if 'model_time' in locals() else datetime.now()
                })
            except Exception as e:
                print(f"Error reading GUI model {model_path.name}: {e}")

        if not available_models:
            self.model_selector.addItem("⚠️ No models found - Train a model first")
            self.predict_btn.setEnabled(False)
            self.model_info_label.setText("Model: No models available")
            return

        # Add models to dropdown
        for model_info in available_models:
            self.model_selector.addItem(model_info['display'], model_info)

        self.predict_btn.setEnabled(True)

        # Update info label
        if self.model_selector.count() > 0:
            current_model = self.model_selector.currentData()
            if current_model:
                self.model_info_label.setText(f"Model: {current_model['name']}")

        # Connect change event
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)

    def on_model_selected(self, index):
        """Handle model selection change"""
        if index >= 0:
            model_info = self.model_selector.currentData()
            if model_info:
                self.model_info_label.setText(f"Model: {model_info['name']}")

                # Show model type info
                if model_info['type'] == 'production':
                    self.prediction_log.append(
                        f"\n✅ Selected: Production Model\n"
                        f"   Path: {model_info['path']}\n"
                        f"   Modified: {model_info['time'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                else:
                    self.prediction_log.append(
                        f"\n✅ Selected: GUI Training Model\n"
                        f"   Path: {model_info['path']}\n"
                        f"   Trained: {model_info['time'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )

    def run_prediction(self):
        """Run prediction with selected model"""
        # Get selected model
        model_info = self.model_selector.currentData()

        if not model_info:
            QMessageBox.warning(
                self,
                "No Model Selected",
                "Please select a model from the dropdown first."
            )
            return

        model_path = model_info['path']

        if not model_path.exists():
            QMessageBox.warning(
                self,
                "Model Not Found",
                f"<b>Selected model not found!</b><br><br>"
                f"Path: {model_path}<br><br>"
                f"Please select another model or train a new one."
            )
            return

        db_path = ROOT_DIR / 'data' / 'btc_ohlcv.db'
        if not db_path.exists():
            QMessageBox.warning(
                self,
                "Database Not Found",
                f"<b>Database not found!</b><br><br>"
                f"Expected location:<br>"
                f"<code>{db_path}</code><br><br>"
                f"Please ensure the database exists with latest data."
            )
            return

        self.predict_btn.setEnabled(False)
        self.prediction_log.clear()

        # Show which model is being used
        self.prediction_log.append(
            f"{'='*70}\n"
            f"🔮 STARTING PREDICTION\n"
            f"{'='*70}\n"
            f"Using Model: {model_info['name']}\n"
            f"Model Type: {model_info['type'].upper()}\n"
            f"Path: {model_path}\n"
            f"{'='*70}\n"
        )

        self.prediction_worker = PredictionWorker(model_path, db_path)
        self.prediction_worker.log_message.connect(self.append_prediction_log)
        self.prediction_worker.prediction_result.connect(self.show_prediction_result)
        self.prediction_worker.error.connect(self.prediction_error)
        self.prediction_worker.start()

    def append_prediction_log(self, message):
        """Append prediction log message"""
        self.prediction_log.append(message)
        scrollbar = self.prediction_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def show_prediction_result(self, result):
        """Show prediction result"""
        self.pred_metric_labels['timestamp'].setText(result['timestamp'])
        self.pred_metric_labels['price'].setText(f"${result['close_price']:,.2f}")
        self.pred_metric_labels['prediction'].setText(result['prediction'])
        self.pred_metric_labels['confidence'].setText(f"{result['confidence']:.1%}")
        self.pred_metric_labels['prob_down'].setText(f"{result['prob_down']:.1%}")
        self.pred_metric_labels['prob_hold'].setText(f"{result['prob_hold']:.1%}")
        self.pred_metric_labels['prob_up'].setText(f"{result['prob_up']:.1%}")

        self.predict_btn.setEnabled(True)

        QMessageBox.information(
            self,
            "✅ Prediction Complete",
            f"<b>Prediction completed successfully!</b><br><br>"
            f"<b>Result:</b> {result['prediction']}<br>"
            f"<b>Confidence:</b> {result['confidence']:.1%}<br><br>"
            f"<b>Probabilities:</b><br>"
            f"• DOWN: {result['prob_down']:.1%}<br>"
            f"• HOLD: {result['prob_hold']:.1%}<br>"
            f"• UP: {result['prob_up']:.1%}<br><br>"
            f"<b>Current Price:</b> ${result['close_price']:,.2f}<br><br>"
            f"<i>Saved to gui_predictions table (original predictions table not modified)</i>"
        )

    def prediction_error(self, error_msg):
        """Handle prediction error"""
        self.predict_btn.setEnabled(True)

        QMessageBox.critical(
            self,
            "Prediction Error",
            f"An error occurred during prediction:\n\n{error_msg[:500]}\n\n"
            f"Check the log for full details."
        )
