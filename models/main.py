import numpy as np
import joblib
import torch

def validate_pipeline(data_dir='preprocessed_data_lstm_1h'):
    """
    Validate that preprocessing output matches model/training expectations
    """
    print("="*60)
    print("🔍 VALIDATING PREPROCESSING → TRAINING PIPELINE")
    print("="*60)

    # Load preprocessed data
    print("\n📂 Loading preprocessed data...")
    X_train = np.load(f'{data_dir}/X_train.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    masks_train = np.load(f'{data_dir}/masks_train.npy')

    X_val = np.load(f'{data_dir}/X_val.npy')
    y_val = np.load(f'{data_dir}/y_val.npy')
    masks_val = np.load(f'{data_dir}/masks_val.npy')

    X_test = np.load(f'{data_dir}/X_test.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    masks_test = np.load(f'{data_dir}/masks_test.npy')

    feature_cols = joblib.load(f'{data_dir}/feature_cols.pkl')
    class_weights = joblib.load(f'{data_dir}/class_weights.pkl')
    sequence_length = joblib.load(f'{data_dir}/sequence_length.pkl')

    # Validation 1: Shapes
    print("\n✅ CHECK 1: Shapes")
    print(f"   X_train:  {X_train.shape}")
    print(f"   y_train:  {y_train.shape}")
    print(f"   masks:    {masks_train.shape}")

    assert X_train.shape[0] == y_train.shape[0] == masks_train.shape[0], "❌ Sample count mismatch!"
    assert X_train.shape[1] == sequence_length, f"❌ Sequence length mismatch! Expected {sequence_length}"
    assert X_train.shape[2] == len(feature_cols), f"❌ Feature count mismatch! Expected {len(feature_cols)}"
    print("   ✅ All shapes valid")

    # Validation 2: Target Classes
    print("\n✅ CHECK 2: Target Classes")
    unique_train = np.unique(y_train)
    unique_val = np.unique(y_val)
    unique_test = np.unique(y_test)

    print(f"   Train classes: {unique_train}")
    print(f"   Val classes:   {unique_val}")
    print(f"   Test classes:  {unique_test}")

    assert set(unique_train) == {0, 1, 2}, "❌ Train classes invalid!"
    assert all(c in [0, 1, 2] for c in unique_val), "❌ Val classes invalid!"
    assert all(c in [0, 1, 2] for c in unique_test), "❌ Test classes invalid!"
    print("   ✅ All targets in [0, 1, 2]")

    # Validation 3: Data Range (after scaling)
    print("\n✅ CHECK 3: Data Range (after RobustScaler)")
    print(f"   X_train min: {X_train.min():.4f}")
    print(f"   X_train max: {X_train.max():.4f}")
    print(f"   X_train mean: {X_train.mean():.4f}")
    print(f"   X_train std: {X_train.std():.4f}")

    # Check for NaN/Inf
    assert not np.isnan(X_train).any(), "❌ NaN found in X_train!"
    assert not np.isinf(X_train).any(), "❌ Inf found in X_train!"
    print("   ✅ No NaN/Inf values")

    # Validation 4: Masks
    print("\n✅ CHECK 4: Attention Masks")
    print(f"   Mask unique values: {np.unique(masks_train)}")
    print(f"   Mask mean: {masks_train.mean():.4f}")

    assert set(np.unique(masks_train)) <= {0.0, 1.0}, "❌ Masks should be 0 or 1!"
    print("   ✅ Masks valid (binary)")

    # Validation 5: Class Weights
    print("\n✅ CHECK 5: Class Weights")
    print(f"   Class weights: {class_weights}")

    assert len(class_weights) == 3, "❌ Should have 3 class weights!"
    assert all(k in class_weights for k in [0, 1, 2]), "❌ Missing class weight keys!"
    print("   ✅ Class weights valid")

    # Validation 6: Feature Columns
    print("\n✅ CHECK 6: Feature Columns")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Sample features:")
    for i, col in enumerate(feature_cols[:5]):
        print(f"      {i+1}. {col}")
    print(f"      ...")

    # Expected features (rough check)
    expected_types = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'atr']
    found = sum(1 for col in feature_cols if any(t in col.lower() for t in expected_types))
    print(f"   Found {found} expected feature types")
    assert found > 0, "❌ No expected features found!"
    print("   ✅ Feature columns look valid")

    # Validation 7: Class Distribution
    print("\n✅ CHECK 7: Class Distribution")

    for split_name, y_data in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        dist = np.bincount(y_data, minlength=3)
        total = len(y_data)
        print(f"   {split_name}:")
        print(f"      DOWN: {dist[0]:,} ({dist[0]/total*100:.1f}%)")
        print(f"      HOLD: {dist[1]:,} ({dist[1]/total*100:.1f}%)")
        print(f"      UP:   {dist[2]:,} ({dist[2]/total*100:.1f}%)")

        # Check for severe imbalance
        min_pct = dist.min() / total * 100
        max_pct = dist.max() / total * 100
        if min_pct < 20 or max_pct > 50:
            print(f"      ⚠️  Warning: Imbalanced ({min_pct:.1f}% - {max_pct:.1f}%)")

    print("   ✅ Distribution checked")

    # Validation 8: Model Compatibility
    print("\n✅ CHECK 8: Model Compatibility")
    from models import LSTMAttentionEnhanced

    try:
        model = LSTMAttentionEnhanced(
            input_size=X_train.shape[2],
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            dropout=0.3,
            num_classes=3,
            bidirectional=True
        )

        # Test forward pass
        sample = torch.FloatTensor(X_train[:2])
        sample_mask = torch.FloatTensor(masks_train[:2])
        output = model(sample, mask=sample_mask)

        assert output.shape == (2, 3), f"❌ Output shape invalid! Got {output.shape}"
        print(f"   Model input:  {sample.shape}")
        print(f"   Model output: {output.shape}")
        print("   ✅ Model compatible")
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        raise

    # Final Summary
    print("\n" + "="*60)
    print("✅ ALL VALIDATIONS PASSED!")
    print("="*60)
    print("\n📊 Summary:")
    print(f"   Total samples: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]:,}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Classes: 3 (DOWN/HOLD/UP)")
    print(f"   Class weights: {class_weights}")
    print("\n✅ Ready for training!")
    print("   Run: python train.py")
    print("="*60)

if __name__ == "__main__":
    validate_pipeline()
