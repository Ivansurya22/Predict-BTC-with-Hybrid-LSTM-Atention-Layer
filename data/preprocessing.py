import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from datetime import datetime
import os
import gc

class BTCDataPreprocessorMultiInputLSTM:
    def __init__(self, db_path, sequence_length=48):
        self.db_path = db_path
        self.sequence_length = sequence_length
        self.scalers = {}
        self.class_weights = {}

        print(f"\n{'='*60}")
        print(f"BTC Multi-Input LSTM Preprocessor (OPTIMIZED)")
        print(f"{'='*60}")
        print(f"Database: {db_path}")
        print(f"Sequence: {sequence_length}h ({sequence_length/24:.1f} days)")
        print(f"Features: 6 Market Regime Features Only")

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def load_data(self):
        print(f"\nLoading data...")

        with self._get_connection() as conn:
            # Load OHLCV data
            query_ohlcv = """
            SELECT timestamp, open, high, low, close, volume
            FROM btc_1h
            ORDER BY timestamp
            """
            df_ohlcv = pd.read_sql(query_ohlcv, conn)
            df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'], utc=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_ohlcv[col] = df_ohlcv[col].astype('float32')

            # Load Technical Indicators
            query_indicators = """
            SELECT * FROM smc_btc_1h_technical_indicators
            ORDER BY timestamp
            """
            df_indicators = pd.read_sql(query_indicators, conn)
            df_indicators['timestamp'] = pd.to_datetime(df_indicators['timestamp'], utc=True)

            for col in df_indicators.columns:
                if col != 'timestamp':
                    df_indicators[col] = df_indicators[col].astype('float32')

            # Load Market Regimes (6 features only)
            query_regimes = """
            SELECT * FROM smc_btc_1h_market_regimes
            ORDER BY timestamp
            """
            df_regimes = pd.read_sql(query_regimes, conn)
            df_regimes['timestamp'] = pd.to_datetime(df_regimes['timestamp'], utc=True)

            for col in df_regimes.columns:
                if col != 'timestamp':
                    if df_regimes[col].dtype in ['int64', 'int32']:
                        df_regimes[col] = df_regimes[col].astype('int8')
                    else:
                        df_regimes[col] = df_regimes[col].astype('float32')

        # Convert to timezone-naive
        for df in [df_ohlcv, df_indicators, df_regimes]:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        # Merge all data
        df = pd.merge(df_ohlcv, df_indicators, on='timestamp', how='inner')
        del df_ohlcv, df_indicators
        gc.collect()

        df = pd.merge(df, df_regimes, on='timestamp', how='inner')
        del df_regimes
        gc.collect()

        df = df.sort_values('timestamp').reset_index(drop=True)

        # NaN cleaning
        print(f"Initial rows: {len(df):,}")

        critical_features = ['close', 'volume', 'ema_21', 'rsi_14']
        nan_count_before = df[critical_features].isna().sum().sum()

        if nan_count_before > 0:
            print(f"NaN in critical features: {nan_count_before:,}")
            df = df.dropna(subset=critical_features)
            print(f"After dropping NaN critical: {len(df):,} rows")

        df = df.ffill(limit=3)
        df = df.bfill(limit=3)

        rows_before = len(df)
        df = df.dropna()
        rows_dropped = rows_before - len(df)

        if rows_dropped > 0:
            print(f"Final cleanup: dropped {rows_dropped:,} rows")

        print(f"Final dataset: {len(df):,} rows, {len(df.columns)} features")
        print(f"Period: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        total_nan = df.isna().sum().sum()
        if total_nan > 0:
            print(f"Warning: {total_nan:,} NaN values remaining")
        else:
            print(f"No NaN values")

        return df

    def define_feature_groups(self, df):
        print(f"\nDefining feature groups...")

        exclude_cols = [
            'timestamp', 'target',
            'future_close', 'future_high', 'future_low',
            'future_return', 'future_max_return', 'future_min_return'
        ]

        all_feature_cols = [col for col in df.columns if col not in exclude_cols]

        feature_groups = {
            # Group 1: Price & Volume (5 features)
            'price_volume': [
                'open', 'high', 'low', 'close', 'volume'
            ],

            # Group 2: Technical Indicators (9 features)
            'technical': [
                'ema_21',
                'macd',
                'macd_signal',
                'macd_hist',
                'rsi_14',
                'atr_14',
                'adx_14',
                'bb_width',
                'obv'
            ],

            # Group 3: Market Regimes (6 features ONLY!)
            'regimes': [
                'hmm_regime_high',
                'hmm_regime_duration',
                'ema_alignment',
                'trend_strong_bull',
                'volume_percentile',
                'volume_trend'
            ]
        }

        # Filter existing columns only
        for group_name in feature_groups.keys():
            feature_groups[group_name] = [f for f in feature_groups[group_name] if f in all_feature_cols]

        print(f"\nFeature Groups:")
        print(f"  price_volume: {len(feature_groups['price_volume'])} features")
        print(f"  technical:    {len(feature_groups['technical'])} features")
        print(f"  regimes:      {len(feature_groups['regimes'])} features")

        total = sum(len(f) for f in feature_groups.values())
        print(f"  TOTAL:        {total} features")

        # Verify regimes has exactly 6 features
        if len(feature_groups['regimes']) != 6:
            print(f"\nWarning: Expected 6 regime features, got {len(feature_groups['regimes'])}")
            print(f"Found: {feature_groups['regimes']}")

        return feature_groups

    def get_forward_extreme(self, series, window, mode='max'):
        """Memory optimized forward extreme calculation."""
        result = np.full(len(series), np.nan, dtype='float32')

        for i in range(len(series) - window):
            if mode == 'max':
                result[i] = series[i+1:i+1+window].max()
            else:
                result[i] = series[i+1:i+1+window].min()

        return result

    def create_target_adaptive(self, df, prediction_horizon=4,
                              base_up_threshold=1.0,
                              base_down_threshold=1.0,
                              hold_reclassify_threshold=0.20,
                              min_hold_percentage=28.0):

        print(f"\nCreating targets...")
        print(f"Horizon: {prediction_horizon}h | Thresholds: ±{base_up_threshold}%")

        df = df.copy()

        df['volatility_1d'] = (df['atr_14'] / df['close'] * 100).astype('float32')
        df['volatility_7d'] = df['volatility_1d'].rolling(window=168, min_periods=24).mean().astype('float32')
        df['volatility_30d'] = df['volatility_1d'].rolling(window=720, min_periods=168).mean().astype('float32')

        median_vol = df['volatility_30d'].median()

        df['vol_regime'] = pd.cut(df['volatility_30d'],
                                   bins=[0, median_vol*0.75, median_vol*1.25, np.inf],
                                   labels=['low', 'normal', 'high'])

        threshold_multipliers = {'low': 0.85, 'normal': 1.0, 'high': 1.15}
        df['threshold_multiplier'] = df['vol_regime'].map(threshold_multipliers).fillna(1.0).astype('float32')
        df['adaptive_up_threshold'] = (df['threshold_multiplier'] * base_up_threshold).astype('float32')
        df['adaptive_down_threshold'] = (df['threshold_multiplier'] * base_down_threshold).astype('float32')

        df['future_close'] = df['close'].shift(-prediction_horizon).astype('float32')
        df['future_high'] = self.get_forward_extreme(df['high'].values, prediction_horizon, mode='max')
        df['future_low'] = self.get_forward_extreme(df['low'].values, prediction_horizon, mode='min')

        df['future_return'] = ((df['future_close'] - df['close']) / df['close'] * 100).astype('float32')
        df['future_max_return'] = ((df['future_high'] - df['close']) / df['close'] * 100).astype('float32')
        df['future_min_return'] = ((df['future_low'] - df['close']) / df['close'] * 100).astype('float32')

        df['target'] = np.ones(len(df), dtype='int8')

        up_mask = (df['future_return'] > df['adaptive_up_threshold']) | \
                  (df['future_max_return'] > df['adaptive_up_threshold'] * 1.4)
        down_mask = (df['future_return'] < -df['adaptive_down_threshold']) | \
                    (df['future_min_return'] < -df['adaptive_down_threshold'] * 1.4)

        df.loc[up_mask, 'target'] = 2
        df.loc[down_mask, 'target'] = 0

        df = df[df['future_close'].notna()].copy()

        dist = df['target'].value_counts().sort_index()
        total = len(df)
        hold_pct = dist.get(1, 0) / total * 100

        print(f"\nInitial distribution:")
        print(f"  DOWN: {dist.get(0, 0):6,} ({dist.get(0, 0)/total*100:5.1f}%)")
        print(f"  HOLD: {dist.get(1, 0):6,} ({dist.get(1, 0)/total*100:5.1f}%)")
        print(f"  UP:   {dist.get(2, 0):6,} ({dist.get(2, 0)/total*100:5.1f}%)")

        if hold_pct >= min_hold_percentage:
            ambiguous_hold = (df['target'] == 1) & (df['future_return'].abs() > hold_reclassify_threshold)
            reclassified = ambiguous_hold.sum()

            df.loc[ambiguous_hold & (df['future_return'] > 0), 'target'] = 2
            df.loc[ambiguous_hold & (df['future_return'] < 0), 'target'] = 0

            if reclassified > 0:
                print(f"  Reclassified {reclassified:,} ambiguous HOLD")

        dist = df['target'].value_counts().sort_index()
        print(f"\nFinal distribution:")
        print(f"  DOWN: {dist.get(0, 0):6,} ({dist.get(0, 0)/total*100:5.1f}%)")
        print(f"  HOLD: {dist.get(1, 0):6,} ({dist.get(1, 0)/total*100:5.1f}%)")
        print(f"  UP:   {dist.get(2, 0):6,} ({dist.get(2, 0)/total*100:5.1f}%)")

        df.drop(['volatility_1d', 'volatility_7d', 'volatility_30d', 'vol_regime',
                'threshold_multiplier', 'adaptive_up_threshold', 'adaptive_down_threshold'],
                axis=1, inplace=True, errors='ignore')

        gc.collect()
        return df

    def validate_sequence_continuity(self, timestamps, idx, sequence_length):
        """Optimized continuity check."""
        if idx < sequence_length:
            return False

        seq_timestamps = timestamps[idx-sequence_length:idx]
        time_diffs = np.diff(seq_timestamps)

        expected_diff = np.timedelta64(1, 'h')
        tolerance = np.timedelta64(5, 'm')

        return np.all((time_diffs >= expected_diff - tolerance) &
                     (time_diffs <= expected_diff + tolerance))

    def create_multi_input_sequences_chunked(self, df, feature_groups, chunk_size=1000):
        """Memory efficient sequence creation using chunking."""
        print(f"\nCreating sequences (chunked)...")

        y = df['target'].values.astype('int8')
        timestamps = df['timestamp'].values

        sequences = {group: [] for group in feature_groups.keys()}
        y_sequences = []
        timestamps_sequences = []

        filtered = 0
        created = 0

        for chunk_start in range(self.sequence_length, len(df), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df))

            for i in range(chunk_start, chunk_end):
                if self.validate_sequence_continuity(timestamps, i, self.sequence_length):
                    for group_name, feature_cols in feature_groups.items():
                        seq = df[feature_cols].values[i-self.sequence_length:i].astype('float32')
                        sequences[group_name].append(seq)

                    y_sequences.append(y[i])
                    timestamps_sequences.append(timestamps[i])
                    created += 1
                else:
                    filtered += 1

            if created % 5000 == 0 and created > 0:
                print(f"  Progress: {created:,} sequences created")
                gc.collect()

        for group_name in sequences.keys():
            sequences[group_name] = np.array(sequences[group_name], dtype='float32')

        y_sequences = np.array(y_sequences, dtype='int8')
        timestamps_sequences = np.array(timestamps_sequences)

        print(f"Created {len(y_sequences):,} sequences (filtered {filtered:,})")
        print(f"\nMulti-Input Shapes:")
        for group_name, seq_array in sequences.items():
            mem_mb = seq_array.nbytes / 1024**2
            print(f"  {group_name:15s}: {seq_array.shape} ({mem_mb:.1f} MB)")

        total_mem = sum(seq.nbytes for seq in sequences.values()) / 1024**2
        print(f"  {'TOTAL MEMORY':15s}: {total_mem:.1f} MB")

        gc.collect()
        return sequences, y_sequences, timestamps_sequences

    def calculate_class_weights(self, y):
        """Calculate balanced class weights."""
        if len(y) == 0:
            print(f"Warning: Empty array, default weights")
            return {0: 1.0, 1: 1.0, 2: 1.0}

        classes = np.unique(y)

        if len(classes) < 2:
            print(f"Warning: Only {len(classes)} class, default weights")
            return {int(c): 1.0 for c in classes}

        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip([int(c) for c in classes], class_weights))

        class_counts = np.bincount(y.astype(int))
        total = len(y)

        print(f"\nClass Distribution:")
        for cls in classes:
            cls_int = int(cls)
            pct = class_counts[cls_int] / total * 100
            class_name = ['DOWN', 'HOLD', 'UP'][cls_int]
            print(f"  {class_name}: {class_counts[cls_int]:,} ({pct:.1f}%)")

        max_count = class_counts.max()

        for cls in classes:
            cls_int = int(cls)
            if cls_int == 1:
                class_weight_dict[cls_int] = class_weight_dict[cls_int] * 1.5
            else:
                rarity_ratio = max_count / class_counts[cls_int]
                if rarity_ratio > 1.8:
                    boost = 2.5
                elif rarity_ratio > 1.3:
                    boost = 2.0
                else:
                    boost = 1.8
                class_weight_dict[cls_int] = class_weight_dict[cls_int] * boost

        print(f"\nFinal Class Weights:")
        for cls in sorted(class_weight_dict.keys()):
            class_name = ['DOWN', 'HOLD', 'UP'][cls]
            print(f"  {class_name}: {class_weight_dict[cls]:.2f}")

        return class_weight_dict

    def chronological_split(self, df, train_ratio=0.70, val_ratio=0.15):
        print(f"\nSplitting data (Train: {train_ratio*100:.0f}% | Val: {val_ratio*100:.0f}% | Test: {(1-train_ratio-val_ratio)*100:.0f}%)...")

        df = df.sort_values('timestamp').reset_index(drop=True)

        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size+val_size].copy()
        test_df = df.iloc[train_size+val_size:].copy()

        print(f"Train: {len(train_df):6,} rows | {train_df['timestamp'].min().date()} → {train_df['timestamp'].max().date()}")
        print(f"Val:   {len(val_df):6,} rows | {val_df['timestamp'].min().date()} → {val_df['timestamp'].max().date()}")
        print(f"Test:  {len(test_df):6,} rows | {test_df['timestamp'].min().date()} → {test_df['timestamp'].max().date()}")

        return train_df, val_df, test_df

    def preprocess_features_by_group(self, train_df, val_df, test_df, feature_groups):
        """Normalize with memory optimization."""
        print(f"\nNormalizing features...")

        for group_name, feature_cols in feature_groups.items():
            if not feature_cols:
                continue

            print(f"  {group_name}: {len(feature_cols)} features")

            scaler = RobustScaler()
            self.scalers[group_name] = scaler

            train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
            val_df[feature_cols] = scaler.transform(val_df[feature_cols])
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])

            for df in [train_df, val_df, test_df]:
                df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
                df[feature_cols] = df[feature_cols].clip(-10, 10)
                df[feature_cols] = df[feature_cols].fillna(0)

                for col in feature_cols:
                    df[col] = df[col].astype('float32')

        print(f"Normalized {len(feature_groups)} groups")
        gc.collect()

        return train_df, val_df, test_df

    def save_preprocessed_data(self,
                              sequences_train, y_train, timestamps_train,
                              sequences_val, y_val, timestamps_val,
                              sequences_test, y_test, timestamps_test,
                              feature_groups, class_weights,
                              output_dir='preprocessed_data_multi_lstm_1h'):
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving (compressed)...")

        for group_name in feature_groups.keys():
            np.savez_compressed(f'{output_dir}/X_train_{group_name}.npz', data=sequences_train[group_name])
            np.savez_compressed(f'{output_dir}/X_val_{group_name}.npz', data=sequences_val[group_name])
            np.savez_compressed(f'{output_dir}/X_test_{group_name}.npz', data=sequences_test[group_name])

        np.savez_compressed(f'{output_dir}/y_train.npz', data=y_train)
        np.savez_compressed(f'{output_dir}/y_val.npz', data=y_val)
        np.savez_compressed(f'{output_dir}/y_test.npz', data=y_test)

        np.savez_compressed(f'{output_dir}/timestamps_train.npz', data=timestamps_train)
        np.savez_compressed(f'{output_dir}/timestamps_val.npz', data=timestamps_val)
        np.savez_compressed(f'{output_dir}/timestamps_test.npz', data=timestamps_test)

        joblib.dump(feature_groups, f'{output_dir}/feature_groups.pkl')
        joblib.dump(self.scalers, f'{output_dir}/scalers.pkl')
        joblib.dump(class_weights, f'{output_dir}/class_weights.pkl')
        joblib.dump(self.sequence_length, f'{output_dir}/sequence_length.pkl')

        metadata = {
            'sequence_length': self.sequence_length,
            'feature_groups': {k: len(v) for k, v in feature_groups.items()},
            'total_features': sum(len(v) for v in feature_groups.values()),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'architecture': 'multi_input_bidirectional_lstm_attention',
            'dtype': 'float32',
            'compressed': True,
            'optimization': '6_market_regime_features_only',
            'regime_features': [
                'hmm_regime_high',
                'hmm_regime_duration',
                'ema_alignment',
                'trend_strong_bull',
                'volume_percentile',
                'volume_trend'
            ]
        }
        joblib.dump(metadata, f'{output_dir}/metadata.pkl')

        print(f"Saved to: {output_dir}/")

    def run_full_pipeline(self,
                         prediction_horizon=4,
                         base_up_threshold=1.0,
                         base_down_threshold=1.0,
                         hold_reclassify_threshold=0.20,
                         min_hold_percentage=28.0,
                         train_ratio=0.70,
                         val_ratio=0.15,
                         output_dir='preprocessed_data_multi_lstm_1h'):

        df = self.load_data()
        feature_groups = self.define_feature_groups(df)

        df = self.create_target_adaptive(
            df, prediction_horizon, base_up_threshold, base_down_threshold,
            hold_reclassify_threshold, min_hold_percentage
        )

        train_df, val_df, test_df = self.chronological_split(df, train_ratio, val_ratio)

        del df
        gc.collect()

        train_df, val_df, test_df = self.preprocess_features_by_group(
            train_df, val_df, test_df, feature_groups
        )

        sequences_train, y_train, timestamps_train = self.create_multi_input_sequences_chunked(
            train_df, feature_groups
        )
        del train_df
        gc.collect()

        sequences_val, y_val, timestamps_val = self.create_multi_input_sequences_chunked(
            val_df, feature_groups
        )
        del val_df
        gc.collect()

        sequences_test, y_test, timestamps_test = self.create_multi_input_sequences_chunked(
            test_df, feature_groups
        )
        del test_df
        gc.collect()

        class_weights_train = self.calculate_class_weights(y_train)

        self.save_preprocessed_data(
            sequences_train, y_train, timestamps_train,
            sequences_val, y_val, timestamps_val,
            sequences_test, y_test, timestamps_test,
            feature_groups, class_weights_train, output_dir
        )

        print(f"\n{'='*60}")
        print(f"COMPLETED")
        print(f"{'='*60}")
        print(f"\nDataset:")
        print(f"  Train: {len(y_train):6,} sequences")
        print(f"  Val:   {len(y_val):6,} sequences")
        print(f"  Test:  {len(y_test):6,} sequences")
        print(f"\nFeature Summary:")
        for group_name, features in feature_groups.items():
            print(f"  {group_name}: {len(features)} features")
        print(f"  TOTAL: {sum(len(v) for v in feature_groups.values())} features")
        print(f"\nOutput: {output_dir}/")
        print(f"{'='*60}\n")

        return {
            'sequences_train': sequences_train,
            'sequences_val': sequences_val,
            'sequences_test': sequences_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_groups': feature_groups,
            'class_weights': class_weights_train
        }


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    DB_PATH = os.getenv("OHLCV_DB_PATH", "data/btc_ohlcv.db")
    OUTPUT_DIR = "preprocessed_data_multi_lstm_1h"
    SEQUENCE_LENGTH = 48

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        exit(1)

    try:
        preprocessor = BTCDataPreprocessorMultiInputLSTM(
            db_path=DB_PATH,
            sequence_length=SEQUENCE_LENGTH
        )

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

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
