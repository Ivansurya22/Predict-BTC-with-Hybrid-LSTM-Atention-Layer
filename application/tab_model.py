import numpy as np
import pandas as pd
import pyqtgraph as pg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QSpinBox,
    QSplitter, QVBoxLayout, QWidget, QMessageBox, QFileDialog,
    QTabWidget, QTextEdit
)
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import sys
from datetime import datetime
import gc

# Setup paths - SESUAIKAN DENGAN STRUKTUR FOLDER
import os
ROOT_DIR = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path.cwd()
MODELS_DIR = ROOT_DIR / 'models'
DATA_DIR = ROOT_DIR / 'preprocessed_data_multi_lstm_1h'

try:
    # Coba import dari models.py di folder models
    sys.path.insert(0, str(ROOT_DIR / 'models'))
    from model import MultiInputBidirectionalLSTMAttention
    MODELS_AVAILABLE = True
except ImportError:
    try:
        # Coba import dari folder yang sama
        from models import MultiInputBidirectionalLSTMAttention
        MODELS_AVAILABLE = True
    except ImportError:
        print("Warning: Could not import MultiInputBidirectionalLSTMAttention")
        print("Creating a dummy model class for GUI...")
        MODELS_AVAILABLE = False

        # Dummy model untuk GUI testing
        class MultiInputBidirectionalLSTMAttention(nn.Module):
            def __init__(self, input_sizes, hidden_size=128, num_layers=2, num_heads=4, dropout=0.3, num_classes=3):
                super().__init__()
                self.dummy = nn.Linear(1, num_classes)

            def forward(self, inputs):
                batch_size = list(inputs.values())[0].shape[0]
                return self.dummy(torch.ones(batch_size, 1))


# ============================================================================
# DATASET
# ============================================================================
class BTCMultiInputDataset(Dataset):
    """Multi-input dataset for training"""
    def __init__(self, sequences_dict, targets):
        self.input_branches = list(sequences_dict.keys())
        self.sequences = {
            branch: torch.FloatTensor(seq)
            for branch, seq in sequences_dict.items()
        }
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        inputs = {branch: seq[idx] for branch, seq in self.sequences.items()}
        return inputs, self.targets[idx]


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class ImprovedFocalLoss(nn.Module):
    """Enhanced Focal Loss with class balancing"""
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super().__init__()
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

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class CombinedLoss(nn.Module):
    """Combined Focal Loss + Label Smoothing"""
    def __init__(self, alpha=None, gamma=2.5, smoothing=0.05, num_classes=3):
        super().__init__()
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
# TRAINING WORKER
# ============================================================================
class TrainingWorker(QThread):
    """Worker thread for model training"""
    progress = Signal(int)
    epoch_update = Signal(int, dict)
    progress_message = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, config, save_path=None):
        super().__init__()
        self.config = config
        self.save_path = Path(save_path) if save_path else MODELS_DIR
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def run(self):
        try:
            self.progress_message.emit("Initializing training...")
            self.progress.emit(5)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.progress_message.emit(f"Using device: {device}")

            # Load data
            self.progress_message.emit("Loading data...")
            result = self.load_data()
            if result is None:
                self.error.emit("Failed to load data")
                return

            train_loader, val_loader, test_loader, input_sizes, class_weights, feature_groups = result
            self.progress.emit(20)

            # Create model
            self.progress_message.emit("Building model...")
            model = MultiInputBidirectionalLSTMAttention(
                input_sizes=input_sizes,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                num_heads=self.config['num_heads'],
                dropout=self.config['dropout'],
                num_classes=3
            ).to(device)

            total_params = sum(p.numel() for p in model.parameters())
            self.progress_message.emit(f"Model built with {total_params:,} parameters")
            self.progress.emit(25)

            # Setup training
            if class_weights is not None:
                weights = torch.FloatTensor([
                    class_weights[0] * 1.5,
                    class_weights[1] * 1.0,
                    class_weights[2] * 1.5
                ]).to(device)
            else:
                weights = torch.FloatTensor([1.0, 2.5, 1.0]).to(device)

            criterion = CombinedLoss(alpha=weights, gamma=3.0, smoothing=0.02)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=0.03,
                betas=(0.9, 0.999)
            )
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )

            history = {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [],
                'val_f1': [], 'val_f1_per_class': [],
                'learning_rate': []
            }
            best_val_f1 = 0.0
            patience_counter = 0
            best_checkpoint = None

            self.progress_message.emit("Training started...")

            # Training loop
            for epoch in range(1, self.config['epochs'] + 1):
                if self.should_stop:
                    self.progress_message.emit(f"Training stopped at epoch {epoch}")
                    break

                # Train
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, criterion, optimizer, device
                )

                # Validate
                val_loss, val_acc, val_f1, val_f1_per_class = self.validate(
                    model, val_loader, criterion, device
                )

                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

                # Update history
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)
                history['val_f1_per_class'].append(val_f1_per_class)
                history['learning_rate'].append(current_lr)

                # Progress
                progress_pct = 25 + int((epoch / self.config['epochs']) * 60)
                self.progress.emit(min(progress_pct, 85))

                # Emit metrics
                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_f1_per_class': val_f1_per_class,
                    'lr': current_lr
                }
                self.epoch_update.emit(epoch, metrics)

                # Update progress message
                self.progress_message.emit(
                    f"Epoch {epoch}/{self.config['epochs']} - "
                    f"Loss: {train_loss:.4f}/{val_loss:.4f} - "
                    f"Acc: {train_acc:.1f}%/{val_acc:.1f}% - "
                    f"F1: {val_f1:.3f}"
                )

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
                        'val_f1_per_class': val_f1_per_class,
                        'input_sizes': input_sizes,
                        'feature_groups': feature_groups
                    }
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= self.config['patience']:
                    self.progress_message.emit(f"Early stopping triggered at epoch {epoch}")
                    break

                # Cleanup
                if epoch % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            self.progress.emit(90)

            # Evaluate on test set
            if best_checkpoint:
                model.load_state_dict(best_checkpoint['model_state_dict'])
                self.progress_message.emit("Evaluating on test set...")
                test_metrics = self.evaluate(model, test_loader, device)
            else:
                test_metrics = None

            # Save model
            if self.save_path and best_checkpoint:
                self.save_model(best_checkpoint, history, test_metrics)

            self.progress.emit(100)

            results = {
                'history': history,
                'test_metrics': test_metrics,
                'best_val_f1': best_val_f1,
                'best_checkpoint': best_checkpoint
            }

            self.progress_message.emit("Training completed successfully!")
            self.finished.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())

    def load_data(self):
        """Load preprocessed multi-input data"""
        try:
            if not DATA_DIR.exists():
                return None

            metadata_path = DATA_DIR / 'metadata.pkl'
            if not metadata_path.exists():
                return None

            metadata = joblib.load(metadata_path)
            feature_groups = joblib.load(DATA_DIR / 'feature_groups.pkl')
            class_weights = joblib.load(DATA_DIR / 'class_weights.pkl')

            sequences_train = {}
            sequences_val = {}
            sequences_test = {}

            for group_name in feature_groups.keys():
                train_path = DATA_DIR / f'X_train_{group_name}.npz'
                val_path = DATA_DIR / f'X_val_{group_name}.npz'
                test_path = DATA_DIR / f'X_test_{group_name}.npz'

                if train_path.exists():
                    sequences_train[group_name] = np.load(train_path)['data']
                if val_path.exists():
                    sequences_val[group_name] = np.load(val_path)['data']
                if test_path.exists():
                    sequences_test[group_name] = np.load(test_path)['data']

            y_train = np.load(DATA_DIR / 'y_train.npz')['data']
            y_val = np.load(DATA_DIR / 'y_val.npz')['data']
            y_test = np.load(DATA_DIR / 'y_test.npz')['data']

            train_dataset = BTCMultiInputDataset(sequences_train, y_train)
            val_dataset = BTCMultiInputDataset(sequences_val, y_val)
            test_dataset = BTCMultiInputDataset(sequences_test, y_test)

            train_loader = DataLoader(
                train_dataset, batch_size=self.config['batch_size'],
                shuffle=True, num_workers=0, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.config['batch_size'],
                shuffle=False, num_workers=0, pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.config['batch_size'],
                shuffle=False, num_workers=0, pin_memory=True
            )

            input_sizes = {k: len(v) for k, v in feature_groups.items()}
            return train_loader, val_loader, test_loader, input_sizes, class_weights, feature_groups

        except Exception as e:
            return None

    def train_epoch(self, model, dataloader, criterion, optimizer, device):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for inputs, targets in dataloader:
            if self.should_stop:
                break

            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
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
            for inputs, targets in dataloader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                targets = targets.to(device)

                outputs = model(inputs)
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

        return epoch_loss, epoch_acc, f1, f1_per_class

    def evaluate(self, model, dataloader, device):
        """Evaluate model on test set"""
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                targets = targets.to(device)

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        weighted_f1 = np.average(f1, weights=support)
        cm = confusion_matrix(all_targets, all_preds)

        return {
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs,
            'accuracy': accuracy * 100,
            'weighted_f1': weighted_f1,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'support': support
        }

    def save_model(self, checkpoint, history, test_metrics):
        """Save model and artifacts"""
        try:
            save_dir = self.save_path
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save best checkpoint
            best_path = save_dir / f'multi_input_lstm_best_{timestamp}.pth'
            torch.save(checkpoint, best_path)

            # Save full model
            full_path = save_dir / f'multi_input_lstm_full_{timestamp}.pth'
            torch.save({
                'model_state_dict': checkpoint['model_state_dict'],
                'input_sizes': checkpoint['input_sizes'],
                'feature_groups': checkpoint['feature_groups'],
                'hidden_size': self.config['hidden_size'],
                'num_layers': self.config['num_layers'],
                'num_heads': self.config['num_heads'],
                'dropout': self.config['dropout']
            }, full_path)

            # Save history
            history_path = save_dir / f'training_history_{timestamp}.pkl'
            joblib.dump(history, history_path)

            # Save test results
            if test_metrics:
                results_path = save_dir / f'test_results_{timestamp}.pkl'
                joblib.dump(test_metrics, results_path)

        except Exception as e:
            pass


# ============================================================================
# TRAINING GUI WIDGET
# ============================================================================
class TrainingGUI(QWidget):
    """Training GUI widget for tab integration"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.training_results = None
        self.save_path = None
        self.setup_ui()

    def setup_ui(self):
        """Setup UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("Multi-Input LSTM Training")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2962FF;")
        layout.addWidget(title)

        # Check model availability
        if not MODELS_AVAILABLE:
            warning = QLabel(
                "WARNING: MultiInputBidirectionalLSTMAttention not found!\n"
                "Ensure models.py is in the models folder.\n"
                "Using dummy model for GUI testing."
            )
            warning.setStyleSheet(
                "background-color: #3d2900; color: #FFB74D; "
                "padding: 15px; border-radius: 5px; font-weight: bold;"
            )
            layout.addWidget(warning)

        # Configuration
        config_group = self.create_config_section()
        layout.addWidget(config_group)

        # Actions
        actions_group = self.create_actions_section()
        layout.addWidget(actions_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
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
                background-color: #2962FF;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Progress message
        self.progress_label = QLabel("Ready to train")
        self.progress_label.setStyleSheet(
            "color: #D1D4DC; font-size: 11px; padding: 5px;"
        )
        layout.addWidget(self.progress_label)

        # Results (Metrics + Charts)
        results_splitter = self.create_results_section()
        layout.addWidget(results_splitter, stretch=1)

    def create_config_section(self):
        """Create configuration section"""
        group = QGroupBox("Training Configuration")
        grid = QGridLayout()

        # Style for labels - SAMA SEPERTI START DATE
        label_style = """
            QLabel {
                color: #D1D4DC;
                font-size: 14px;
            }
        """

        # Style for input widgets
        input_style = """
            QSpinBox, QDoubleSpinBox {
                background-color: #1E222D;
                color: #D1D4DC;
                border: 2px solid #2A2E39;
                border-radius: 4px;
                padding: 5px;
                font-size: 14px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #2962FF;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border: 2px solid #3D4758;
            }
        """

        # Batch Size
        batch_label = QLabel("Batch Size:")
        batch_label.setStyleSheet(label_style)
        grid.addWidget(batch_label, 0, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(32, 512)
        self.batch_size.setValue(64)
        self.batch_size.setSingleStep(32)
        self.batch_size.setStyleSheet(input_style)
        grid.addWidget(self.batch_size, 0, 1)

        # Learning Rate
        lr_label = QLabel("Learning Rate:")
        lr_label.setStyleSheet(label_style)
        grid.addWidget(lr_label, 0, 2)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.00001, 0.01)
        self.learning_rate.setValue(0.0001)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setSingleStep(0.0001)
        self.learning_rate.setStyleSheet(input_style)
        grid.addWidget(self.learning_rate, 0, 3)

        # Hidden Size
        hidden_label = QLabel("Hidden Size:")
        hidden_label.setStyleSheet(label_style)
        grid.addWidget(hidden_label, 1, 0)
        self.hidden_size = QSpinBox()
        self.hidden_size.setRange(64, 512)
        self.hidden_size.setValue(128)
        self.hidden_size.setSingleStep(32)
        self.hidden_size.setStyleSheet(input_style)
        grid.addWidget(self.hidden_size, 1, 1)

        # Num Layers
        layers_label = QLabel("Num Layers:")
        layers_label.setStyleSheet(label_style)
        grid.addWidget(layers_label, 1, 2)
        self.num_layers = QSpinBox()
        self.num_layers.setRange(1, 6)
        self.num_layers.setValue(3)
        self.num_layers.setStyleSheet(input_style)
        grid.addWidget(self.num_layers, 1, 3)

        # Attention Heads
        heads_label = QLabel("Attention Heads:")
        heads_label.setStyleSheet(label_style)
        grid.addWidget(heads_label, 2, 0)
        self.num_heads = QSpinBox()
        self.num_heads.setRange(2, 16)
        self.num_heads.setValue(4)
        self.num_heads.setStyleSheet(input_style)
        grid.addWidget(self.num_heads, 2, 1)

        # Dropout
        dropout_label = QLabel("Dropout:")
        dropout_label.setStyleSheet(label_style)
        grid.addWidget(dropout_label, 2, 2)
        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0.1, 0.7)
        self.dropout.setValue(0.25)
        self.dropout.setDecimals(2)
        self.dropout.setSingleStep(0.05)
        self.dropout.setStyleSheet(input_style)
        grid.addWidget(self.dropout, 2, 3)

        # Epochs
        epochs_label = QLabel("Max Epochs:")
        epochs_label.setStyleSheet(label_style)
        grid.addWidget(epochs_label, 3, 0)
        self.epochs = QSpinBox()
        self.epochs.setRange(10, 500)
        self.epochs.setValue(150)
        self.epochs.setSingleStep(10)
        self.epochs.setStyleSheet(input_style)
        grid.addWidget(self.epochs, 3, 1)

        # Patience
        patience_label = QLabel("Early Stop Patience:")
        patience_label.setStyleSheet(label_style)
        grid.addWidget(patience_label, 3, 2)
        self.patience = QSpinBox()
        self.patience.setRange(5, 50)
        self.patience.setValue(20)
        self.patience.setStyleSheet(input_style)
        grid.addWidget(self.patience, 3, 3)

        group.setLayout(grid)
        return group

    def create_actions_section(self):
        """Create actions section"""
        group = QGroupBox("Actions")
        layout = QHBoxLayout()

        # Train button
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #2962FF;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #1E53E5; }
            QPushButton:disabled { background-color: #4A4A4A; }
        """)
        self.train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.train_btn)

        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #EF5350;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #E53935; }
            QPushButton:disabled { background-color: #4A4A4A; }
        """)
        self.stop_btn.clicked.connect(self.stop_training)
        layout.addWidget(self.stop_btn)

        # Save location button
        self.save_location_btn = QPushButton("Set Save Location")
        self.save_location_btn.setStyleSheet("""
            QPushButton {
                background-color: #26A69A;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #00897B; }
        """)
        self.save_location_btn.clicked.connect(self.choose_save_location)
        layout.addWidget(self.save_location_btn)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_results_section(self):
        """Create results section"""
        splitter = QSplitter(Qt.Horizontal)

        # Left: Metrics
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        metrics_title = QLabel("Training Metrics")
        metrics_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        left_layout.addWidget(metrics_title)

        # Metrics
        metrics_grid = QGridLayout()
        self.metric_labels = {}

        metrics_list = [
            ("Epoch", "epoch", "#FFB74D"),
            ("Train Acc", "train_acc", "#26A69A"),
            ("Val Acc", "val_acc", "#2962FF"),
            ("Test Acc", "test_acc", "#AB47BC"),
            ("Train Loss", "train_loss", "#26A69A"),
            ("Val Loss", "val_loss", "#2962FF"),
            ("Val F1", "val_f1", "#2962FF"),
            ("Test F1", "test_f1", "#AB47BC"),
            ("LR", "lr", "#FFB74D"),
            ("Best F1", "best_f1", "#26A69A"),
        ]

        for i, (label, key, color) in enumerate(metrics_list):
            label_widget = QLabel(f"{label}:")
            label_widget.setStyleSheet("font-size: 12px;")
            value_widget = QLabel("--")
            value_widget.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 12px;")
            self.metric_labels[key] = value_widget

            metrics_grid.addWidget(label_widget, i, 0)
            metrics_grid.addWidget(value_widget, i, 1)

        left_layout.addLayout(metrics_grid)
        left_layout.addStretch()

        # Right: Charts
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Loss plot
        loss_title = QLabel("Training Loss")
        loss_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        right_layout.addWidget(loss_title)

        self.loss_plot = pg.PlotWidget()
        self.loss_plot.setBackground("#131722")
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self.loss_plot.setLabel("left", "Loss", color="white", **{'font-size': '12pt'})
        self.loss_plot.setLabel("bottom", "Epoch", color="white", **{'font-size': '12pt'})
        legend = self.loss_plot.addLegend(offset=(10, 10))
        legend.setLabelTextColor("#D1D4DC")
        self.loss_plot.getAxis('left').setTextPen('white')
        self.loss_plot.getAxis('bottom').setTextPen('white')
        self.loss_plot.getAxis('left').setPen('#2A2E39')
        self.loss_plot.getAxis('bottom').setPen('#2A2E39')
        right_layout.addWidget(self.loss_plot)

        # Accuracy plot
        acc_title = QLabel("Accuracy")
        acc_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        right_layout.addWidget(acc_title)

        self.acc_plot = pg.PlotWidget()
        self.acc_plot.setBackground("#131722")
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        self.acc_plot.setLabel("left", "Accuracy (%)", color="white", **{'font-size': '12pt'})
        self.acc_plot.setLabel("bottom", "Epoch", color="white", **{'font-size': '12pt'})
        legend = self.acc_plot.addLegend(offset=(10, 10))
        legend.setLabelTextColor("#D1D4DC")
        self.acc_plot.getAxis('left').setTextPen('white')
        self.acc_plot.getAxis('bottom').setTextPen('white')
        self.acc_plot.getAxis('left').setPen('#2A2E39')
        self.acc_plot.getAxis('bottom').setPen('#2A2E39')
        right_layout.addWidget(self.acc_plot)

        # F1 plot
        f1_title = QLabel("F1 Score")
        f1_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        right_layout.addWidget(f1_title)

        self.f1_plot = pg.PlotWidget()
        self.f1_plot.setBackground("#131722")
        self.f1_plot.showGrid(x=True, y=True, alpha=0.3)
        self.f1_plot.setLabel("left", "F1 Score", color="white", **{'font-size': '12pt'})
        self.f1_plot.setLabel("bottom", "Epoch", color="white", **{'font-size': '12pt'})
        legend = self.f1_plot.addLegend(offset=(10, 10))
        legend.setLabelTextColor("#D1D4DC")
        self.f1_plot.getAxis('left').setTextPen('white')
        self.f1_plot.getAxis('bottom').setTextPen('white')
        self.f1_plot.getAxis('left').setPen('#2A2E39')
        self.f1_plot.getAxis('bottom').setPen('#2A2E39')
        right_layout.addWidget(self.f1_plot)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([280, 1120])

        return splitter

    def choose_save_location(self):
        """Choose save location for models"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Save Location",
            str(MODELS_DIR)
        )
        if folder:
            self.save_path = Path(folder)
            QMessageBox.information(
                self,
                "Save Location Set",
                f"Models will be saved to:\n\n{self.save_path}"
            )

    def start_training(self):
        """Start training"""
        if not MODELS_AVAILABLE and not isinstance(MultiInputBidirectionalLSTMAttention, type):
            QMessageBox.warning(
                self,
                "Model Not Available",
                "MultiInputBidirectionalLSTMAttention class not found!\n"
                "Using dummy model for GUI testing only."
            )

        if not DATA_DIR.exists():
            QMessageBox.warning(
                self,
                "Data Not Found",
                f"Preprocessed data not found at:\n{DATA_DIR}\n\n"
                f"Please run preprocessing first."
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
            f"<b>Start training with configuration?</b><br><br>"
            f"<b>Architecture:</b><br>"
            f"Hidden: {config['hidden_size']}, Layers: {config['num_layers']}<br>"
            f"Heads: {config['num_heads']}, Dropout: {config['dropout']}<br><br>"
            f"<b>Training:</b><br>"
            f"Batch: {config['batch_size']}, LR: {config['learning_rate']}<br>"
            f"Epochs: {config['epochs']}, Patience: {config['patience']}<br><br>"
            f"<b>Save to:</b> {self.save_path or MODELS_DIR}<br>",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Reset UI
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Initializing: 0%")
        self.progress_label.setText("Ready to train")
        self.loss_plot.clear()
        self.acc_plot.clear()
        self.f1_plot.clear()

        for key in self.metric_labels:
            self.metric_labels[key].setText("--")

        # Store config for progress calculation
        self.config = config

        # Start worker
        save_location = self.save_path or MODELS_DIR
        self.worker = TrainingWorker(config, save_location)
        self.worker.progress.connect(self.update_progress)
        self.worker.epoch_update.connect(self.update_epoch)
        self.worker.progress_message.connect(self.update_progress_message)
        self.worker.finished.connect(self.training_finished)
        self.worker.error.connect(self.training_error)
        self.worker.start()

    def stop_training(self):
        """Stop training"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Stop Training",
                "Stop training?\nBest model will be saved.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.stop()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

        # Update format based on progress stage
        if value < 20:
            self.progress_bar.setFormat(f"Loading Data: {value}%")
        elif value < 25:
            self.progress_bar.setFormat(f"Building Model: {value}%")
        elif value < 90:
            self.progress_bar.setFormat(f"Training: {value}%")
        elif value < 100:
            self.progress_bar.setFormat(f"Evaluating: {value}%")
        else:
            self.progress_bar.setFormat(f"Complete: {value}%")

    def update_progress_message(self, message):
        """Update progress message"""
        self.progress_label.setText(message)

    def update_epoch(self, epoch, metrics):
        """Update metrics and plots"""
        # Update metrics
        self.metric_labels['epoch'].setText(f"{epoch}")
        self.metric_labels['train_acc'].setText(f"{metrics['train_acc']:.2f}%")
        self.metric_labels['val_acc'].setText(f"{metrics['val_acc']:.2f}%")
        self.metric_labels['train_loss'].setText(f"{metrics['train_loss']:.4f}")
        self.metric_labels['val_loss'].setText(f"{metrics['val_loss']:.4f}")
        self.metric_labels['val_f1'].setText(f"{metrics['val_f1']:.3f}")
        self.metric_labels['lr'].setText(f"{metrics['lr']:.6f}")

        # Update progress bar with epoch progress
        epoch_progress = int((epoch / self.config['epochs']) * 100)
        progress_text = f"Training: {epoch_progress}% (Epoch {epoch}/{self.config['epochs']})"
        self.progress_bar.setFormat(progress_text)

        # Update plots
        self.update_plots(epoch, metrics)

    def update_plots(self, epoch, metrics):
        """Update training plots"""
        # Loss plot
        if epoch == 1:
            self.loss_plot.plot([epoch], [metrics['train_loss']],
                               pen=pg.mkPen('#26A69A', width=3),
                               symbol='o', symbolBrush='#26A69A',
                               symbolPen='#26A69A', symbolSize=7, name='Train Loss')
            self.loss_plot.plot([epoch], [metrics['val_loss']],
                               pen=pg.mkPen('#EF5350', width=3),
                               symbol='o', symbolBrush='#EF5350',
                               symbolPen='#EF5350', symbolSize=7, name='Val Loss')
        else:
            self.loss_plot.plot([epoch], [metrics['train_loss']],
                               pen=None, symbol='o', symbolBrush='#26A69A',
                               symbolPen='#26A69A', symbolSize=7)
            self.loss_plot.plot([epoch], [metrics['val_loss']],
                               pen=None, symbol='o', symbolBrush='#EF5350',
                               symbolPen='#EF5350', symbolSize=7)

        # Accuracy plot
        if epoch == 1:
            self.acc_plot.plot([epoch], [metrics['train_acc']],
                              pen=pg.mkPen('#26A69A', width=3),
                              symbol='o', symbolBrush='#26A69A',
                              symbolPen='#26A69A', symbolSize=7, name='Train Acc')
            self.acc_plot.plot([epoch], [metrics['val_acc']],
                              pen=pg.mkPen('#2962FF', width=3),
                              symbol='o', symbolBrush='#2962FF',
                              symbolPen='#2962FF', symbolSize=7, name='Val Acc')
        else:
            self.acc_plot.plot([epoch], [metrics['train_acc']],
                              pen=None, symbol='o', symbolBrush='#26A69A',
                              symbolPen='#26A69A', symbolSize=7)
            self.acc_plot.plot([epoch], [metrics['val_acc']],
                              pen=None, symbol='o', symbolBrush='#2962FF',
                              symbolPen='#2962FF', symbolSize=7)

        # F1 plot
        if epoch == 1:
            self.f1_plot.plot([epoch], [metrics['val_f1']],
                             pen=pg.mkPen('#AB47BC', width=3),
                             symbol='o', symbolBrush='#AB47BC',
                             symbolPen='#AB47BC', symbolSize=7, name='Val F1')
        else:
            self.f1_plot.plot([epoch], [metrics['val_f1']],
                             pen=None, symbol='o', symbolBrush='#AB47BC',
                             symbolPen='#AB47BC', symbolSize=7)

    def training_finished(self, results):
        """Handle training completion"""
        self.training_results = results
        history = results['history']
        test_metrics = results['test_metrics']

        # Update final metrics
        final_epoch = len(history['train_loss'])
        self.metric_labels['epoch'].setText(f"{final_epoch}")
        self.metric_labels['train_acc'].setText(f"{history['train_acc'][-1]:.2f}%")
        self.metric_labels['val_acc'].setText(f"{history['val_acc'][-1]:.2f}%")
        self.metric_labels['train_loss'].setText(f"{history['train_loss'][-1]:.4f}")
        self.metric_labels['val_loss'].setText(f"{history['val_loss'][-1]:.4f}")
        self.metric_labels['val_f1'].setText(f"{history['val_f1'][-1]:.3f}")
        self.metric_labels['best_f1'].setText(f"{results['best_val_f1']:.3f}")

        if test_metrics:
            self.metric_labels['test_acc'].setText(f"{test_metrics['accuracy']:.2f}%")
            self.metric_labels['test_f1'].setText(f"{test_metrics['weighted_f1']:.3f}")

        # Update progress bar to complete
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Training Complete: 100%")
        self.progress_label.setText("Training completed successfully!")

        # Plot final results with connecting lines
        if history['train_loss']:
            epochs = np.arange(1, len(history['train_loss']) + 1)

            # Loss plot
            self.loss_plot.clear()
            self.loss_plot.plot(
                epochs, history['train_loss'],
                pen=pg.mkPen('#26A69A', width=3),
                symbol='o', symbolBrush='#26A69A',
                symbolPen='#26A69A', symbolSize=6, name="Train"
            )
            self.loss_plot.plot(
                epochs, history['val_loss'],
                pen=pg.mkPen('#EF5350', width=3),
                symbol='o', symbolBrush='#EF5350',
                symbolPen='#EF5350', symbolSize=6, name="Val"
            )

            # Accuracy plot
            self.acc_plot.clear()
            self.acc_plot.plot(
                epochs, history['train_acc'],
                pen=pg.mkPen('#26A69A', width=3),
                symbol='o', symbolBrush='#26A69A',
                symbolPen='#26A69A', symbolSize=6, name="Train"
            )
            self.acc_plot.plot(
                epochs, history['val_acc'],
                pen=pg.mkPen('#2962FF', width=3),
                symbol='o', symbolBrush='#2962FF',
                symbolPen='#2962FF', symbolSize=6, name="Val"
            )

            # F1 plot
            self.f1_plot.clear()
            self.f1_plot.plot(
                epochs, history['val_f1'],
                pen=pg.mkPen('#AB47BC', width=3),
                symbol='o', symbolBrush='#AB47BC',
                symbolPen='#AB47BC', symbolSize=6, name="Val F1"
            )

        # Re-enable UI
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # Show completion message
        if test_metrics:
            QMessageBox.information(
                self,
                "Training Complete",
                f"<b>Training completed successfully!</b><br><br>"
                f"<b>Test Performance:</b><br>"
                f"Accuracy: <b>{test_metrics['accuracy']:.2f}%</b><br>"
                f"F1 Score: <b>{test_metrics['weighted_f1']:.3f}</b><br><br>"
                f"<b>Model saved to:</b><br>{self.save_path or MODELS_DIR}"
            )

    def training_error(self, error_msg):
        """Handle training error"""
        self.progress_label.setText("Training error occurred!")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        QMessageBox.critical(
            self,
            "Training Error",
            f"Training error:\n\n{error_msg[:500]}"
        )
