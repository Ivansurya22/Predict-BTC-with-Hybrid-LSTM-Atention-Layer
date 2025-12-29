import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import gc

from models import MultiInputBidirectionalLSTMAttention


class ImprovedFocalLoss(nn.Module):
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


class BTCMultiInputDataset(Dataset):
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


class MultiInputTrainer:
    def __init__(self, model, device, class_weights=None, learning_rate=0.0001):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

        if class_weights is not None:
            weights = torch.FloatTensor([
                class_weights[0] * 1.5,
                class_weights[1] * 1.0,
                class_weights[2] * 1.5
            ]).to(device)
        else:
            weights = torch.FloatTensor([1.0, 2.5, 1.0]).to(device)

        self.criterion = CombinedLoss(alpha=weights, gamma=3.0, smoothing=0.02)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.03,
            betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

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

        for inputs, targets in train_loader:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
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
            for inputs, targets in val_loader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * accuracy_score(all_targets, all_preds)

        _, _, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        _, _, f1_per_class, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )

        return epoch_loss, epoch_acc, f1, f1_per_class, all_preds, all_targets

    def fit(self, train_loader, val_loader, epochs=200, patience=25,
            save_dir='models', save_name='multi_input_lstm_optimized'):
        os.makedirs(save_dir, exist_ok=True)

        best_checkpoint = None

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1, val_f1_per_class, val_preds, val_targets = self.validate(val_loader)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['val_f1_per_class'].append(val_f1_per_class)
            self.history['learning_rate'].append(current_lr)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                      f"Acc: {train_acc:.1f}%/{val_acc:.1f}% | "
                      f"F1: {val_f1:.3f} | "
                      f"LR: {current_lr:.6f}")

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
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (Best F1: {self.best_val_f1:.3f})")
                break

            if epoch % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return self.history, best_checkpoint


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
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

    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy*100:.2f}% | Weighted F1: {weighted_f1:.3f}")

    class_names = ['DOWN', 'HOLD', 'UP']
    print(f"\nPer-class metrics:")
    for i, name in enumerate(class_names):
        print(f"  {name}: P={precision[i]:.3f} R={recall[i]:.3f} F1={f1[i]:.3f} N={support[i]:,}")

    cm = confusion_matrix(all_targets, all_preds)
    print(f"\nConfusion Matrix:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {cm[i].tolist()}")

    return {
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'f1': f1,
        'confusion_matrix': cm
    }


def load_multi_input_data(data_dir):
    metadata = joblib.load(f'{data_dir}/metadata.pkl')
    feature_groups = joblib.load(f'{data_dir}/feature_groups.pkl')
    class_weights = joblib.load(f'{data_dir}/class_weights.pkl')

    sequences_train = {}
    sequences_val = {}
    sequences_test = {}

    for group_name in feature_groups.keys():
        sequences_train[group_name] = np.load(f'{data_dir}/X_train_{group_name}.npz')['data']
        sequences_val[group_name] = np.load(f'{data_dir}/X_val_{group_name}.npz')['data']
        sequences_test[group_name] = np.load(f'{data_dir}/X_test_{group_name}.npz')['data']

    y_train = np.load(f'{data_dir}/y_train.npz')['data']
    y_val = np.load(f'{data_dir}/y_val.npz')['data']
    y_test = np.load(f'{data_dir}/y_test.npz')['data']

    return (sequences_train, y_train), (sequences_val, y_val), (sequences_test, y_test), feature_groups, class_weights, metadata


def main():
    print("BTC Multi-Input LSTM Training (6 Regime Features)")

    DATA_DIR = 'preprocessed_data_multi_lstm_1h'
    MODEL_DIR = 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    NUM_HEADS = 4
    DROPOUT = 0.25
    EPOCHS = 150
    PATIENCE = 20

    print(f"\nConfig: Batch={BATCH_SIZE} LR={LEARNING_RATE} Hidden={HIDDEN_SIZE} "
          f"Layers={NUM_LAYERS} Heads={NUM_HEADS} Dropout={DROPOUT}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_data, val_data, test_data, feature_groups, class_weights, metadata = load_multi_input_data(DATA_DIR)
    sequences_train, y_train = train_data
    sequences_val, y_val = val_data
    sequences_test, y_test = test_data

    print(f"\nDataset: Train={len(y_train):,} Val={len(y_val):,} Test={len(y_test):,}")
    print(f"Features: {metadata['total_features']} total ({metadata['feature_groups']})")

    train_dataset = BTCMultiInputDataset(sequences_train, y_train)
    val_dataset = BTCMultiInputDataset(sequences_val, y_val)
    test_dataset = BTCMultiInputDataset(sequences_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=2, pin_memory=True)

    input_sizes = {k: len(v) for k, v in feature_groups.items()}
    print(f"\nInput sizes: {input_sizes}")

    model = MultiInputBidirectionalLSTMAttention(
        input_sizes=input_sizes,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        num_classes=3
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    trainer = MultiInputTrainer(
        model=model,
        device=device,
        class_weights=class_weights,
        learning_rate=LEARNING_RATE
    )

    print(f"\nTraining for {EPOCHS} epochs (patience={PATIENCE})...")
    history, checkpoint = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        patience=PATIENCE,
        save_dir=MODEL_DIR,
        save_name='multi_input_lstm_optimized'
    )

    print(f"\nEvaluating best model...")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_results = evaluate_model(model, test_loader, device)

    joblib.dump(history, f'{MODEL_DIR}/multi_input_lstm_optimized_history.pkl')
    joblib.dump(test_results, f'{MODEL_DIR}/multi_input_lstm_optimized_results.pkl')

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_sizes': input_sizes,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'num_heads': NUM_HEADS,
        'dropout': DROPOUT,
        'feature_groups': feature_groups
    }, f'{MODEL_DIR}/multi_input_lstm_optimized_full.pth')

    print(f"\nTraining completed!")
    print(f"Best model saved: {MODEL_DIR}/multi_input_lstm_optimized_best.pth")
    print(f"Final Test F1: {test_results['weighted_f1']:.3f}")


if __name__ == "__main__":
    main()
