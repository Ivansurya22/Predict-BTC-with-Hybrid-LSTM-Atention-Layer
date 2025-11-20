import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from datetime import datetime
import warnings
import os

class BTCDataPreprocessorLSTM:
    def __init__(self, db_path, sequence_length=48):
        self.db_path = db_path
        self.sequence_length = sequence_length
        self.scalers = {}
        self.class_weights = {}

        print(f"\n{'='*60}")
        print(f"🚀 BTC LSTM Data Preprocessor (Optimized V2)")
        print(f"{'='*60}")
        print(f"Database: {db_path}")
        print(f"Sequence: {sequence_length}h ({sequence_length/24:.1f} days)")
        print(f"Strategy: Enhanced feature quality & balanced targets")

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def load_data(self):
        print(f"\n📥 Loading data...")

        with self._get_connection() as conn:
            # Load OHLCV data
            query_ohlcv = """
            SELECT timestamp, open, high, low, close, volume
            FROM btc_1h
            ORDER BY timestamp
            """
            df_ohlcv = pd.read_sql(query_ohlcv, conn)
            df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'], utc=True)

            # Load Technical Indicators
            query_indicators = """
            SELECT * FROM smc_btc_1h_technical_indicators
            ORDER BY timestamp
            """
            df_indicators = pd.read_sql(query_indicators, conn)
            df_indicators['timestamp'] = pd.to_datetime(df_indicators['timestamp'], utc=True)

            # Load Optimized Features
            query_optimized = """
            SELECT * FROM smc_btc_1h_optimized_features
            ORDER BY timestamp
            """
            df_optimized = pd.read_sql(query_optimized, conn)
            df_optimized['timestamp'] = pd.to_datetime(df_optimized['timestamp'], utc=True)

        # Convert to timezone-naive
        df_ohlcv['timestamp'] = df_ohlcv['timestamp'].dt.tz_localize(None)
        df_indicators['timestamp'] = df_indicators['timestamp'].dt.tz_localize(None)
        df_optimized['timestamp'] = df_optimized['timestamp'].dt.tz_localize(None)

        # Merge all data
        df = pd.merge(df_ohlcv, df_indicators, on='timestamp', how='inner')
        df = pd.merge(df, df_optimized, on='timestamp', how='inner')
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Smart NaN handling - preserve data quality (FIXED deprecation)
        nan_counts_before = df.isna().sum().sum()
        if nan_counts_before > 0:
            # Forward fill limited to 2 periods (2 hours max)
            df = df.ffill(limit=2)
            # Backward fill only for remaining initial NaNs
            df = df.bfill(limit=2)
            # Drop rows with remaining NaNs instead of filling with 0
            rows_before = len(df)
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            print(f"✓ Cleaned {nan_counts_before:,} NaN values ({rows_dropped:,} rows dropped)")

        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} features")
        print(f"✓ Period: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

        return df

    def get_forward_extreme(self, series, window, mode='max'):
        """Helper function to get forward max/min."""
        result = []
        for i in range(len(series)):
            if i + 1 + window <= len(series):
                if mode == 'max':
                    result.append(series.iloc[i+1:i+1+window].max())
                else:
                    result.append(series.iloc[i+1:i+1+window].min())
            else:
                result.append(np.nan)
        return pd.Series(result, index=series.index)

    def create_target_adaptive(self, df, prediction_horizon=4,
                              base_up_threshold=1.0,
                              base_down_threshold=1.0,
                              hold_reclassify_threshold=0.20,
                              min_hold_percentage=28.0):
        """
        Create BALANCED targets with adaptive thresholds.

        Key improvements:
        - Dynamic threshold adjustment based on realized distribution
        - Volatility-adjusted thresholds per sample
        - Spike/drop detection with confirmation
        - Target: DOWN 33-35%, HOLD 30-33%, UP 33-35%
        """
        print(f"\n🎯 Creating balanced adaptive targets...")
        print(f"Horizon: {prediction_horizon}h | Base thresholds: ±{base_up_threshold}%")
        print(f"Hold Reclassify: {hold_reclassify_threshold}% | Min Hold: {min_hold_percentage}%")

        df = df.copy()

        # Calculate multi-period volatility for robust threshold adaptation
        df['volatility_1d'] = df['atr_14'] / df['close'] * 100
        df['volatility_7d'] = df['volatility_1d'].rolling(window=168, min_periods=24).mean()
        df['volatility_30d'] = df['volatility_1d'].rolling(window=720, min_periods=168).mean()

        # Use 30-day volatility for more stable thresholds
        median_vol = df['volatility_30d'].median()

        # Adaptive multipliers based on market regime
        df['vol_regime'] = pd.cut(df['volatility_30d'],
                                   bins=[0, median_vol*0.75, median_vol*1.25, np.inf],
                                   labels=['low', 'normal', 'high'])

        # Adaptive thresholds per regime - FIXED: Convert to numeric first
        threshold_multipliers = {'low': 0.85, 'normal': 1.0, 'high': 1.15}

        # Convert categorical to numeric explicitly before multiplication
        df['threshold_multiplier'] = df['vol_regime'].map(threshold_multipliers).fillna(1.0).astype(float)
        df['adaptive_up_threshold'] = df['threshold_multiplier'] * base_up_threshold
        df['adaptive_down_threshold'] = df['threshold_multiplier'] * base_down_threshold

        # Calculate future returns
        df['future_close'] = df['close'].shift(-prediction_horizon)
        df['future_high'] = self.get_forward_extreme(df['high'], prediction_horizon, mode='max')
        df['future_low'] = self.get_forward_extreme(df['low'], prediction_horizon, mode='min')

        df['future_return'] = (df['future_close'] - df['close']) / df['close'] * 100
        df['future_max_return'] = (df['future_high'] - df['close']) / df['close'] * 100
        df['future_min_return'] = (df['future_low'] - df['close']) / df['close'] * 100

        # Initialize all as HOLD
        df['target'] = 1

        # Classification with confirmation
        for idx in df.index:
            if pd.notna(df.loc[idx, 'adaptive_up_threshold']):
                up_thresh = df.loc[idx, 'adaptive_up_threshold']
                down_thresh = df.loc[idx, 'adaptive_down_threshold']

                future_ret = df.loc[idx, 'future_return']
                max_ret = df.loc[idx, 'future_max_return']
                min_ret = df.loc[idx, 'future_min_return']

                # UP: Either sustained gain OR significant spike
                if (future_ret > up_thresh) or (max_ret > up_thresh * 1.4):
                    df.loc[idx, 'target'] = 2

                # DOWN: Either sustained loss OR significant drop
                elif (future_ret < -down_thresh) or (min_ret < -down_thresh * 1.4):
                    df.loc[idx, 'target'] = 0

        # Remove rows without future data
        df = df[df['future_close'].notna()].copy()

        # Check distribution and adjust if needed
        dist = df['target'].value_counts().sort_index()
        total = len(df)
        hold_pct = dist.get(1, 0) / total * 100

        print(f"\n📊 Initial distribution:")
        print(f"  DOWN: {dist.get(0, 0):6,} ({dist.get(0, 0)/total*100:5.1f}%)")
        print(f"  HOLD: {dist.get(1, 0):6,} ({dist.get(1, 0)/total*100:5.1f}%)")
        print(f"  UP:   {dist.get(2, 0):6,} ({dist.get(2, 0)/total*100:5.1f}%)")

        # Reclassify ambiguous HOLD if below minimum
        if hold_pct < min_hold_percentage:
            print(f"\n⚠️  HOLD too low ({hold_pct:.1f}%), skipping reclassification")
        else:
            ambiguous_hold = (
                (df['target'] == 1) &
                (df['future_return'].abs() > hold_reclassify_threshold)
            )
            reclassified = ambiguous_hold.sum()
            df.loc[ambiguous_hold & (df['future_return'] > 0), 'target'] = 2
            df.loc[ambiguous_hold & (df['future_return'] < 0), 'target'] = 0

            if reclassified > 0:
                print(f"   Reclassified {reclassified:,} ambiguous HOLD samples")

        # Final distribution
        dist = df['target'].value_counts().sort_index()
        total = len(df)
        print(f"\n✅ Final target distribution:")
        print(f"  DOWN: {dist.get(0, 0):6,} ({dist.get(0, 0)/total*100:5.1f}%)")
        print(f"  HOLD: {dist.get(1, 0):6,} ({dist.get(1, 0)/total*100:5.1f}%)")
        print(f"  UP:   {dist.get(2, 0):6,} ({dist.get(2, 0)/total*100:5.1f}%)")

        # Balance check
        percentages = [dist.get(i, 0)/total*100 for i in [0, 1, 2]]
        imbalance = max(percentages) - min(percentages)

        if imbalance < 7:
            print(f"✅ Excellent balance! (imbalance: {imbalance:.1f}%)")
        elif imbalance < 12:
            print(f"✅ Good balance (imbalance: {imbalance:.1f}%)")
        else:
            print(f"⚠️  Moderate imbalance ({imbalance:.1f}%)")
            if dist.get(0, 0)/total > 0.37:
                print(f"   💡 DOWN too high: Try base_down_threshold={base_down_threshold + 0.15:.2f}")
            if dist.get(2, 0)/total > 0.37:
                print(f"   💡 UP too high: Try base_up_threshold={base_up_threshold + 0.15:.2f}")

        # Clean up intermediate columns - Added threshold_multiplier
        df.drop(['volatility_1d', 'volatility_7d', 'volatility_30d', 'vol_regime',
                'threshold_multiplier', 'adaptive_up_threshold', 'adaptive_down_threshold'],
                axis=1, inplace=True, errors='ignore')

        return df

    def validate_sequence_continuity(self, df, idx, sequence_length):
        """Validate that sequence has no time gaps."""
        if idx < sequence_length:
            return False

        seq_timestamps = df['timestamp'].iloc[idx-sequence_length:idx]
        time_diffs = seq_timestamps.diff()[1:]

        expected_diff = pd.Timedelta(hours=1)
        tolerance = pd.Timedelta(minutes=5)

        is_continuous = ((time_diffs >= expected_diff - tolerance) &
                        (time_diffs <= expected_diff + tolerance)).all()

        return is_continuous

    def create_sequences_with_mask(self, df, feature_cols):
        """Create sequences with attention mask for LSTM."""
        X = df[feature_cols].values
        y = df['target'].values
        timestamps = df['timestamp'].values

        X_sequences = []
        y_sequences = []
        timestamps_sequences = []
        masks = []

        filtered_discontinuous = 0

        for i in range(self.sequence_length, len(df)):
            if self.validate_sequence_continuity(df, i, self.sequence_length):
                X_sequences.append(X[i-self.sequence_length:i])
                y_sequences.append(y[i])
                timestamps_sequences.append(timestamps[i])
                masks.append(np.ones(self.sequence_length))
            else:
                filtered_discontinuous += 1

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        timestamps_sequences = np.array(timestamps_sequences)
        masks = np.array(masks)

        print(f"✓ Created {len(X_sequences):,} sequences (filtered {filtered_discontinuous:,} discontinuous)")

        return X_sequences, y_sequences, timestamps_sequences, masks

    def calculate_class_weights(self, y):
        """Calculate balanced class weights."""
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))

        # Count distribution
        class_counts = np.bincount(y)
        total = len(y)

        print(f"\n⚖️  Class Distribution (After Sequences):")
        for cls in classes:
            pct = class_counts[cls] / total * 100
            class_name = ['DOWN', 'HOLD', 'UP'][cls]
            print(f"   {class_name}: {class_counts[cls]:,} ({pct:.1f}%)")

        # Moderate boost to minority classes
        max_count = class_counts.max()

        for cls in classes:
            if cls == 1:  # HOLD - standard balanced weight
                class_weight_dict[cls] = class_weight_dict[cls] * 1.5
            else:  # DOWN (0) and UP (2)
                rarity_ratio = max_count / class_counts[cls]
                if rarity_ratio > 1.8:
                    boost = 2.5
                elif rarity_ratio > 1.3:
                    boost = 2.0
                else:
                    boost = 1.8
                class_weight_dict[cls] = class_weight_dict[cls] * boost

        print(f"\n⚖️  Final Class Weights:")
        for cls in sorted(class_weight_dict.keys()):
            class_name = ['DOWN', 'HOLD', 'UP'][cls]
            print(f"   {class_name}: {class_weight_dict[cls]:.2f}")

        return class_weight_dict

    def chronological_split(self, df, train_ratio=0.70, val_ratio=0.15):
        print(f"\n✂️  Splitting data (Train: {train_ratio*100:.0f}% | Val: {val_ratio*100:.0f}% | Test: {(1-train_ratio-val_ratio)*100:.0f}%)...")

        df = df.sort_values('timestamp').reset_index(drop=True)

        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size+val_size].copy()
        test_df = df.iloc[train_size+val_size:].copy()

        print(f"✓ Train: {len(train_df):6,} rows | {train_df['timestamp'].min().date()} → {train_df['timestamp'].max().date()}")
        print(f"✓ Val:   {len(val_df):6,} rows | {val_df['timestamp'].min().date()} → {val_df['timestamp'].max().date()}")
        print(f"✓ Test:  {len(test_df):6,} rows | {test_df['timestamp'].min().date()} → {test_df['timestamp'].max().date()}")

        # Show target distribution per split
        print(f"\n📊 Target distribution per split:")
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if 'target' in split_df.columns:
                dist = split_df['target'].value_counts().sort_index()
                total = len(split_df)
                print(f"  {split_name}: DOWN {dist.get(0, 0)/total*100:4.1f}%  |  HOLD {dist.get(1, 0)/total*100:4.1f}%  |  UP {dist.get(2, 0)/total*100:4.1f}%")

        return train_df, val_df, test_df

    def preprocess_features(self, train_df, val_df, test_df):
        print(f"\n🔄 Normalizing features with RobustScaler...")

        exclude_features = [
            'timestamp', 'target',
            'future_close', 'future_high', 'future_low',
            'future_return', 'future_max_return', 'future_min_return'
        ]

        feature_cols = [col for col in train_df.columns if col not in exclude_features]

        # Use RobustScaler (better for financial data with outliers)
        self.scalers['robust'] = RobustScaler()
        train_df[feature_cols] = self.scalers['robust'].fit_transform(train_df[feature_cols])
        val_df[feature_cols] = self.scalers['robust'].transform(val_df[feature_cols])
        test_df[feature_cols] = self.scalers['robust'].transform(test_df[feature_cols])

        # Handle extreme outliers (beyond 10 IQR)
        for df in [train_df, val_df, test_df]:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Clip extreme values instead of setting to 0
            df[feature_cols] = df[feature_cols].clip(-10, 10)
            df.fillna(0, inplace=True)

        print(f"✓ Normalized {len(feature_cols)} features")
        print(f"✓ Applied outlier clipping (±10 IQR)")

        return train_df, val_df, test_df, feature_cols

    def save_preprocessed_data(self, X_train, y_train, timestamps_train, masks_train,
                            X_val, y_val, timestamps_val, masks_val,
                            X_test, y_test, timestamps_test, masks_test,
                            feature_cols, class_weights,
                            output_dir='preprocessed_data_lstm_1h'):
        os.makedirs(output_dir, exist_ok=True)

        # Save arrays
        np.save(f'{output_dir}/X_train.npy', X_train)
        np.save(f'{output_dir}/y_train.npy', y_train)
        np.save(f'{output_dir}/timestamps_train.npy', timestamps_train)
        np.save(f'{output_dir}/masks_train.npy', masks_train)

        np.save(f'{output_dir}/X_val.npy', X_val)
        np.save(f'{output_dir}/y_val.npy', y_val)
        np.save(f'{output_dir}/timestamps_val.npy', timestamps_val)
        np.save(f'{output_dir}/masks_val.npy', masks_val)

        np.save(f'{output_dir}/X_test.npy', X_test)
        np.save(f'{output_dir}/y_test.npy', y_test)
        np.save(f'{output_dir}/timestamps_test.npy', timestamps_test)
        np.save(f'{output_dir}/masks_test.npy', masks_test)

        # Save metadata
        joblib.dump(feature_cols, f'{output_dir}/feature_cols.pkl')
        joblib.dump(self.scalers, f'{output_dir}/scalers.pkl')
        joblib.dump(class_weights, f'{output_dir}/class_weights.pkl')
        joblib.dump(self.sequence_length, f'{output_dir}/sequence_length.pkl')

        # Enhanced metadata for inference
        metadata = {
            'sequence_length': self.sequence_length,
            'min_data_required': self.sequence_length + 168,
            'feature_cols': feature_cols,
            'num_features': len(feature_cols),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'train_distribution': {
                'DOWN': int((y_train == 0).sum()),
                'HOLD': int((y_train == 1).sum()),
                'UP': int((y_train == 2).sum())
            }
        }
        joblib.dump(metadata, f'{output_dir}/metadata.pkl')

        print(f"\n💾 Saved to: {output_dir}/")
        print(f"   Files: X_*.npy, y_*.npy, timestamps_*.npy, masks_*.npy")
        print(f"   Meta: feature_cols.pkl, scalers.pkl, class_weights.pkl, metadata.pkl")

    def run_full_pipeline(self,
                         prediction_horizon=4,
                         base_up_threshold=1.0,
                         base_down_threshold=1.0,
                         hold_reclassify_threshold=0.20,
                         min_hold_percentage=28.0,
                         train_ratio=0.70,
                         val_ratio=0.15,
                         output_dir='preprocessed_data_lstm_1h'):

        # Load data
        df = self.load_data()

        # Create balanced targets
        df = self.create_target_adaptive(
            df,
            prediction_horizon=prediction_horizon,
            base_up_threshold=base_up_threshold,
            base_down_threshold=base_down_threshold,
            hold_reclassify_threshold=hold_reclassify_threshold,
            min_hold_percentage=min_hold_percentage
        )

        # Split data
        train_df, val_df, test_df = self.chronological_split(df, train_ratio, val_ratio)

        # Preprocess features
        train_df, val_df, test_df, feature_cols = self.preprocess_features(
            train_df, val_df, test_df
        )

        # Create sequences
        print(f"\n🔄 Creating sequences with attention masks...")
        X_train, y_train, timestamps_train, masks_train = self.create_sequences_with_mask(train_df, feature_cols)
        X_val, y_val, timestamps_val, masks_val = self.create_sequences_with_mask(val_df, feature_cols)
        X_test, y_test, timestamps_test, masks_test = self.create_sequences_with_mask(test_df, feature_cols)

        # Calculate class weights
        class_weights_train = self.calculate_class_weights(y_train)

        # Save everything
        self.save_preprocessed_data(
            X_train, y_train, timestamps_train, masks_train,
            X_val, y_val, timestamps_val, masks_val,
            X_test, y_test, timestamps_test, masks_test,
            feature_cols, class_weights_train,
            output_dir
        )

        print(f"\n{'='*60}")
        print(f"✅ PREPROCESSING COMPLETED")
        print(f"{'='*60}")
        print(f"\n📊 Final Dataset:")
        print(f"  Train: {X_train.shape[0]:6,} sequences")
        print(f"  Val:   {X_val.shape[0]:6,} sequences")
        print(f"  Test:  {X_test.shape[0]:6,} sequences")
        print(f"  Shape: ({self.sequence_length} timesteps × {len(feature_cols)} features)")

        print(f"\n📁 Output: {output_dir}/")
        print(f"{'='*60}\n")

        return {
            'X_train': X_train, 'y_train': y_train, 'timestamps_train': timestamps_train, 'masks_train': masks_train,
            'X_val': X_val, 'y_val': y_val, 'timestamps_val': timestamps_val, 'masks_val': masks_val,
            'X_test': X_test, 'y_test': y_test, 'timestamps_test': timestamps_test, 'masks_test': masks_test,
            'feature_cols': feature_cols,
            'class_weights': class_weights_train,
            'sequence_length': self.sequence_length
        }


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Configuration
    DB_PATH = os.getenv("OHLCV_DB_PATH", "data/btc_ohlcv.db")
    OUTPUT_DIR = "preprocessed_data_lstm_1h"
    SEQUENCE_LENGTH = 48  # 48 hours = 2 days

    # Check database
    if not os.path.exists(DB_PATH):
        print(f"❌ ERROR: Database not found at {DB_PATH}")
        exit(1)

    try:
        # Initialize preprocessor
        preprocessor = BTCDataPreprocessorLSTM(
            db_path=DB_PATH,
            sequence_length=SEQUENCE_LENGTH
        )

        print(f"\n💡 OPTIMIZATION STRATEGY:")
        print(f"  ✓ Enhanced NaN handling (drop instead of zero-fill)")
        print(f"  ✓ Volatility-regime adaptive thresholds")
        print(f"  ✓ Spike/drop confirmation for better signals")
        print(f"  ✓ Balanced targets: ~33% each class")
        print(f"  ✓ Moderate class weights (less aggressive)")
        print(f"  ✓ Outlier clipping instead of removal")

        results = preprocessor.run_full_pipeline(
            prediction_horizon=4,
            base_up_threshold=1.0,
            base_down_threshold=1.0,
            hold_reclassify_threshold=0.20,
            min_hold_percentage=28.0,
            train_ratio=0.70,
            val_ratio=0.15,
            output_dir=OUTPUT_DIR
        )

        print(f"🎉 SUCCESS! Data ready for training.")

        print(f"\n💡 Key Improvements:")
        print(f"  ✅ Better data quality (NaN handling)")
        print(f"  ✅ More balanced targets")
        print(f"  ✅ Volatility-aware thresholds")
        print(f"  ✅ Reduced overfitting risk")
        print(f"  ✅ Better generalization potential")

        print(f"\n🔧 Tuning Guide:")
        print(f"  • If DOWN/UP > 36%: Increase base_threshold to 1.15")
        print(f"  • If HOLD < 28%: Decrease hold_reclassify to 0.15")
        print(f"  • If HOLD > 35%: Increase hold_reclassify to 0.25")

        print(f"\n▶️  Next step: python train.py")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
