import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import models
from models import LSTMAttentionEnhanced, LSTMAttention, SimpleLSTM


# ==================== COMBINED LOSS ====================
class CombinedLoss(nn.Module):
    """
    Combined Focal Loss + Label Smoothing for better generalization
    """
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, num_classes=3):
        super(CombinedLoss, self).__init__()
        self.focal = ImprovedFocalLoss(alpha=alpha, gamma=gamma)
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, inputs, targets):
        # Focal loss component
        focal_loss = self.focal(inputs, targets)

        # Label smoothing component
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        smooth_loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

        # Combine: 70% focal + 30% label smoothing
        return 0.7 * focal_loss + 0.3 * smooth_loss


# ==================== IMPROVED FOCAL LOSS ====================
class ImprovedFocalLoss(nn.Module):
    """
    Enhanced Focal Loss with class balancing
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Focal term
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weights if provided
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


# ==================== DATASET WITH MASK ====================
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


# ==================== IMPROVED TRAINER ====================
class ImprovedTrainer:
    """
    Optimized trainer aligned with balanced preprocessing
    """
    def __init__(self, model, device, class_weights=None, learning_rate=0.0003):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

        # ADJUSTED class weights for BALANCED data from preprocessing
        if class_weights is not None:
            # Since preprocessing already balanced the data (~33% each),
            # we use MODERATE weights to avoid over-correction
            weights = torch.FloatTensor([
                class_weights[0] * 1.2,  # DOWN - slight boost
                class_weights[1] * 1.0,  # HOLD - baseline (was over-boosted before)
                class_weights[2] * 1.2   # UP - slight boost
            ]).to(device)
            print(f"   📊 Adjusted weights: DOWN={weights[0]:.2f}, HOLD={weights[1]:.2f}, UP={weights[2]:.2f}")
        else:
            weights = torch.FloatTensor([1.2, 1.0, 1.2]).to(device)
            print(f"   📊 Default weights: DOWN=1.2, HOLD=1.0, UP=1.2")

        # Use Combined Loss (Focal + Label Smoothing)
        self.criterion = CombinedLoss(alpha=weights, gamma=2.0, smoothing=0.1)
        print(f"   🎯 Loss: Combined (70% Focal + 30% Label Smoothing)")

        # AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.03,      # Moderate weight decay
            betas=(0.9, 0.999)
        )

        # ReduceLROnPlateau scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            threshold=0.001
        )
        print(f"   📉 Scheduler: ReduceLROnPlateau (patience=7, factor=0.5)")

        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_f1': [], 'learning_rate': [],
            'val_f1_per_class': []
        }

        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for inputs, targets, masks in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, mask=masks)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * accuracy_score(all_targets, all_preds)

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(inputs, mask=masks)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * accuracy_score(all_targets, all_preds)

        # Calculate F1 scores
        _, _, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )

        _, _, f1_per_class, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )

        return epoch_loss, epoch_acc, f1, f1_per_class, all_preds, all_targets

    def fit(self, train_loader, val_loader, epochs=100, patience=15,
            save_dir='models', save_name='lstm_balanced'):
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"🚀 Training {save_name}")
        print(f"{'='*80}")
        print(f"Max epochs: {epochs} | Patience: {patience}")
        print(f"Initial LR: {self.learning_rate}")

        best_checkpoint = None

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_f1, val_f1_per_class, val_preds, val_targets = self.validate(val_loader)

            # Update LR based on F1
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_f1)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['val_f1_per_class'].append(val_f1_per_class)
            self.history['learning_rate'].append(current_lr)

            # Print progress every 5 epochs or on first epoch
            if epoch % 5 == 0 or epoch == 1:
                print(f"\n{'─'*80}")
                print(f"Epoch {epoch:3d}/{epochs}")
                print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
                print(f"  Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.1f}% | F1: {val_f1:.3f}")
                print(f"  F1/class → DOWN: {val_f1_per_class[0]:.3f} | HOLD: {val_f1_per_class[1]:.3f} | UP: {val_f1_per_class[2]:.3f}")
                print(f"  LR: {current_lr:.6f}")

                # Prediction distribution
                pred_dist = np.bincount(val_preds, minlength=3)
                pred_pct = pred_dist / len(val_preds) * 100
                target_dist = np.bincount(val_targets, minlength=3)
                target_pct = target_dist / len(val_targets) * 100

                print(f"  Predictions → DOWN: {pred_pct[0]:.1f}% | HOLD: {pred_pct[1]:.1f}% | UP: {pred_pct[2]:.1f}%")
                print(f"  Actual      → DOWN: {target_pct[0]:.1f}% | HOLD: {target_pct[1]:.1f}% | UP: {target_pct[2]:.1f}%")

                # Check for bias
                max_imbalance = max(abs(pred_pct[i] - target_pct[i]) for i in range(3))
                if max_imbalance > 15:
                    print(f"  ⚠️  Prediction imbalance detected: {max_imbalance:.1f}% deviation")

            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0

                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_f1_per_class': val_f1_per_class
                }
                torch.save(best_checkpoint, f'{save_dir}/{save_name}_best.pth')

                if epoch % 5 == 0 or epoch == 1:
                    print(f"  ✅ Best model saved (F1: {val_f1:.3f})")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n{'─'*80}")
                print(f"⏹️  Early stopping at epoch {epoch}")
                print(f"   Best F1: {self.best_val_f1:.3f}")
                break

        print(f"\n{'='*80}")
        print(f"✅ Training completed!")
        print(f"Best F1: {self.best_val_f1:.3f}")
        print(f"{'='*80}")

        return self.history, best_checkpoint


# ==================== EVALUATION ====================
def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    print(f"\n{'='*80}")
    print(f"📊 Evaluating on test set...")
    print(f"{'='*80}")

    with torch.no_grad():
        for inputs, targets, masks in test_loader:
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

    # Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )
    weighted_f1 = np.average(f1, weights=support)

    print(f"\n✅ OVERALL METRICS:")
    print(f"   Accuracy:    {accuracy*100:.2f}%")
    print(f"   Weighted F1: {weighted_f1:.3f}")

    # Per-class metrics
    print(f"\n📊 PER-CLASS METRICS:")
    class_names = ['DOWN', 'HOLD', 'UP']
    for i, name in enumerate(class_names):
        print(f"   {name:4s} → Precision: {precision[i]:.3f} | Recall: {recall[i]:.3f} | F1: {f1[i]:.3f} | Support: {support[i]:,}")

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print(f"\n🔢 CONFUSION MATRIX:")
    print(f"{'Actual':>8} | {'DOWN':>8} {'HOLD':>8} {'UP':>8}")
    print(f"{'-'*40}")
    for i, name in enumerate(class_names):
        row_total = cm[i].sum()
        print(f"{name:>8} | {cm[i][0]:>8,} {cm[i][1]:>8,} {cm[i][2]:>8,}  ({row_total:,})")

    # Prediction distribution
    print(f"\n🎯 PREDICTION DISTRIBUTION:")
    pred_dist = np.bincount(all_preds, minlength=3)
    target_dist = np.bincount(all_targets, minlength=3)
    print(f"{'Class':>6} | {'Predicted':>10} {'Actual':>10} {'Diff':>10}")
    print(f"{'-'*40}")
    for i, name in enumerate(class_names):
        pred_pct = pred_dist[i] / len(all_preds) * 100
        target_pct = target_dist[i] / len(all_targets) * 100
        diff = pred_pct - target_pct
        print(f"{name:>6} | {pred_dist[i]:>6,} ({pred_pct:>4.1f}%) | {target_dist[i]:>6,} ({target_pct:>4.1f}%) | {diff:>+5.1f}%")

    # Confidence analysis
    print(f"\n🎲 CONFIDENCE ANALYSIS:")
    max_probs = all_probs.max(axis=1)
    print(f"   Mean confidence: {max_probs.mean():.3f}")
    print(f"   Std confidence:  {max_probs.std():.3f}")

    # Per-class confidence
    for i, name in enumerate(class_names):
        class_mask = all_preds == i
        if class_mask.sum() > 0:
            class_conf = max_probs[class_mask].mean()
            print(f"   {name} avg confidence: {class_conf:.3f}")

    return {
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'f1': f1,
        'confusion_matrix': cm
    }


# ==================== MAIN ====================
def main():
    print(f"\n{'='*80}")
    print("🎯 BTC LSTM + Attention Training (BALANCED & OPTIMIZED)")
    print(f"{'='*80}")

    # Configuration
    DATA_DIR = 'preprocessed_data_lstm_1h'
    MODEL_DIR = 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Optimized Hyperparameters for BALANCED data
    BATCH_SIZE = 256
    LEARNING_RATE = 0.0003
    HIDDEN_SIZE = 128           # Optimal size
    NUM_LAYERS = 2
    NUM_HEADS = 4
    DROPOUT = 0.3               # Moderate dropout
    EPOCHS = 100
    PATIENCE = 15

    MODEL_TYPE = 'enhanced'

    print(f"\n⚙️  OPTIMIZED CONFIGURATION:")
    print(f"   Model:        Enhanced LSTM + Multi-Head Attention")
    print(f"   Loss:         Combined (70% Focal + 30% Label Smoothing)")
    print(f"   Class boost:  Moderate (1.2x for DOWN/UP, 1.0x for HOLD)")
    print(f"   Batch size:   {BATCH_SIZE}")
    print(f"   Learning rate:{LEARNING_RATE} (with ReduceLROnPlateau)")
    print(f"   Hidden size:  {HIDDEN_SIZE}")
    print(f"   LSTM layers:  {NUM_LAYERS}")
    print(f"   Attention:    {NUM_HEADS} heads")
    print(f"   Dropout:      {DROPOUT}")
    print(f"   Weight decay: 0.03")
    print(f"   Max epochs:   {EPOCHS}")
    print(f"   Patience:     {PATIENCE}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print(f"\n📂 Loading data from {DATA_DIR}/...")
    X_train = np.load(f'{DATA_DIR}/X_train.npy')
    y_train = np.load(f'{DATA_DIR}/y_train.npy')
    masks_train = np.load(f'{DATA_DIR}/masks_train.npy')

    X_val = np.load(f'{DATA_DIR}/X_val.npy')
    y_val = np.load(f'{DATA_DIR}/y_val.npy')
    masks_val = np.load(f'{DATA_DIR}/masks_val.npy')

    X_test = np.load(f'{DATA_DIR}/X_test.npy')
    y_test = np.load(f'{DATA_DIR}/y_test.npy')
    masks_test = np.load(f'{DATA_DIR}/masks_test.npy')

    class_weights = joblib.load(f'{DATA_DIR}/class_weights.pkl')

    print(f"   ✓ Train: {X_train.shape[0]:,} samples")
    print(f"   ✓ Val:   {X_val.shape[0]:,} samples")
    print(f"   ✓ Test:  {X_test.shape[0]:,} samples")
    print(f"   ✓ Shape: {X_train.shape}")

    # Show target distribution
    print(f"\n📊 Target Distribution:")
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        dist = np.bincount(y_split, minlength=3)
        total = len(y_split)
        print(f"   {split_name:5s} → DOWN: {dist[0]:,} ({dist[0]/total*100:.1f}%) | "
              f"HOLD: {dist[1]:,} ({dist[1]/total*100:.1f}%) | "
              f"UP: {dist[2]:,} ({dist[2]/total*100:.1f}%)")

    # Create dataloaders
    print(f"\n🔄 Creating DataLoaders...")
    train_dataset = BTCSequenceDataset(X_train, y_train, masks_train)
    val_dataset = BTCSequenceDataset(X_val, y_val, masks_val)
    test_dataset = BTCSequenceDataset(X_test, y_test, masks_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=4, pin_memory=True)

    input_size = X_train.shape[2]

    # Build model
    print(f"\n🏗️  Building Enhanced LSTM + Attention model...")
    model = LSTMAttentionEnhanced(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        num_classes=3,
        bidirectional=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Train
    trainer = ImprovedTrainer(
        model=model,
        device=device,
        class_weights=class_weights,
        learning_rate=LEARNING_RATE
    )

    history, checkpoint = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        patience=PATIENCE,
        save_dir=MODEL_DIR,
        save_name='lstm_balanced'
    )

    # Evaluate on test set
    print(f"\n🔥 Loading best model for final evaluation...")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_results = evaluate_model(model, test_loader, device)

    # Save artifacts
    print(f"\n💾 Saving artifacts...")
    joblib.dump(history, f'{MODEL_DIR}/training_history_balanced.pkl')
    joblib.dump(test_results, f'{MODEL_DIR}/test_results_balanced.pkl')

    # Save full model for inference
    torch.save({
        'model_type': MODEL_TYPE,
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'num_heads': NUM_HEADS,
        'dropout': DROPOUT
    }, f'{MODEL_DIR}/lstm_balanced_full.pth')

    # Final summary
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   Test Accuracy:  {test_results['accuracy']*100:.2f}%")
    print(f"   Test F1:        {test_results['weighted_f1']:.3f}")
    print(f"   Per-class F1:")
    print(f"     DOWN: {test_results['f1'][0]:.3f}")
    print(f"     HOLD: {test_results['f1'][1]:.3f}")
    print(f"     UP:   {test_results['f1'][2]:.3f}")

    # Prediction distribution check
    pred_dist = np.bincount(test_results['predictions'], minlength=3)
    pred_pct = pred_dist / len(test_results['predictions']) * 100
    target_dist = np.bincount(test_results['targets'], minlength=3)
    target_pct = target_dist / len(test_results['targets']) * 100

    print(f"\n📊 Test Set Distribution:")
    print(f"   Predictions: DOWN {pred_pct[0]:.1f}% | HOLD {pred_pct[1]:.1f}% | UP {pred_pct[2]:.1f}%")
    print(f"   Actual:      DOWN {target_pct[0]:.1f}% | HOLD {target_pct[1]:.1f}% | UP {target_pct[2]:.1f}%")

    # Balance check
    max_diff = max(abs(pred_pct[i] - target_pct[i]) for i in range(3))
    if max_diff < 10:
        print(f"\n✅ Excellent balance! (max deviation: {max_diff:.1f}%)")
    elif max_diff < 15:
        print(f"\n👍 Good balance (max deviation: {max_diff:.1f}%)")
    else:
        print(f"\n⚠️  Moderate imbalance (max deviation: {max_diff:.1f}%)")

    # Performance evaluation
    if test_results['weighted_f1'] >= 0.45:
        print(f"\n🎉 EXCELLENT! Model ready for deployment")
    elif test_results['weighted_f1'] >= 0.35:
        print(f"\n👍 GOOD! Model is usable")
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT")
        print(f"   Try: Increase sequence length or adjust thresholds")

    print(f"\n📁 Saved files:")
    print(f"   Model:   {MODEL_DIR}/lstm_balanced_best.pth")
    print(f"   Full:    {MODEL_DIR}/lstm_balanced_full.pth")
    print(f"   History: {MODEL_DIR}/training_history_balanced.pkl")
    print(f"   Results: {MODEL_DIR}/test_results_balanced.pkl")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
