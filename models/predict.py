import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import sqlite3
from datetime import datetime
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Paths
ROOT_DIR = Path.cwd()
MODELS_DIR = ROOT_DIR / 'models'
sys.path.insert(0, str(MODELS_DIR))

from models import LSTMAttentionEnhanced, LSTMAttention, SimpleLSTM


class BTCPredictor:
    def __init__(self, model_path, scaler_path, feature_cols_path, sequence_length_path,
                 db_path='data/btc_ohlcv.db'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert to absolute paths
        self.model_path = ROOT_DIR / model_path if not Path(model_path).is_absolute() else Path(model_path)
        self.scaler_path = ROOT_DIR / scaler_path if not Path(scaler_path).is_absolute() else Path(scaler_path)
        self.feature_cols_path = ROOT_DIR / feature_cols_path if not Path(feature_cols_path).is_absolute() else Path(feature_cols_path)
        self.sequence_length_path = ROOT_DIR / sequence_length_path if not Path(sequence_length_path).is_absolute() else Path(sequence_length_path)
        self.db_path = ROOT_DIR / db_path if not Path(db_path).is_absolute() else Path(db_path)

        print(f"\n{'='*70}")
        print(f"🔧 Initializing BTC Predictor (ALIGNED VERSION)")
        print(f"{'='*70}")
        print(f"Device: {self.device}")

        # Load model
        print(f"\n📦 Loading model...")
        checkpoint = torch.load(str(self.model_path), map_location=self.device)
        self.model_type = checkpoint.get('model_type', 'enhanced')

        if self.model_type == 'enhanced':
            self.model = LSTMAttentionEnhanced(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                num_heads=checkpoint.get('num_heads', 4),
                dropout=checkpoint['dropout'],
                num_classes=3,
                bidirectional=True
            )
        elif self.model_type == 'standard':
            self.model = LSTMAttention(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                dropout=checkpoint['dropout'],
                num_classes=3,
                bidirectional=True
            )
        else:
            self.model = SimpleLSTM(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                dropout=checkpoint['dropout'],
                num_classes=3
            )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"   ✓ Model type: {self.model_type}")
        print(f"   ✓ Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Load preprocessing artifacts
        print(f"\n📦 Loading preprocessing artifacts...")
        scaler_data = joblib.load(str(self.scaler_path))
        self.scaler = scaler_data['robust'] if isinstance(scaler_data, dict) else scaler_data
        self.feature_cols = joblib.load(str(self.feature_cols_path))
        self.sequence_length = joblib.load(str(self.sequence_length_path))

        print(f"   ✓ Scaler: RobustScaler")
        print(f"   ✓ Features: {len(self.feature_cols)}")
        print(f"   ✓ Sequence length: {self.sequence_length}")

        # Calculate minimum data required (sequence + buffer for rolling calculations)
        self.min_data_required = self.sequence_length + 168  # +168 for rolling features

        self.class_names = ['DOWN', 'HOLD', 'UP']

        print(f"{'='*70}")

    def fetch_latest_data(self):
        """
        Fetch latest data from database
        ALIGNED: Same as preprocessing - loads optimized_features
        """
        print(f"\n{'='*70}")
        print(f"📥 Fetching Latest Data")
        print(f"{'='*70}")
        print(f"Sequence needed: {self.sequence_length}")
        print(f"Fetching (with buffer): {self.min_data_required}")

        conn = sqlite3.connect(str(self.db_path))

        try:
            # 1. Load OHLCV
            query_ohlcv = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM btc_1h
                ORDER BY timestamp DESC
                LIMIT {self.min_data_required}
            """
            df_ohlcv = pd.read_sql_query(query_ohlcv, conn)

            if len(df_ohlcv) < self.sequence_length:
                raise ValueError(f"Insufficient data: need {self.sequence_length}, got {len(df_ohlcv)}")

            # 2. Load Technical Indicators
            query_indicators = f"""
                SELECT * FROM smc_btc_1h_technical_indicators
                ORDER BY timestamp DESC
                LIMIT {self.min_data_required}
            """
            df_indicators = pd.read_sql_query(query_indicators, conn)

            # 3. Load Optimized Features (CRITICAL!)
            query_optimized = f"""
                SELECT * FROM smc_btc_1h_optimized_features
                ORDER BY timestamp DESC
                LIMIT {self.min_data_required}
            """
            df_optimized = pd.read_sql_query(query_optimized, conn)

        finally:
            conn.close()

        # Reverse to chronological order
        df_ohlcv = df_ohlcv.iloc[::-1].reset_index(drop=True)
        df_indicators = df_indicators.iloc[::-1].reset_index(drop=True)
        df_optimized = df_optimized.iloc[::-1].reset_index(drop=True)

        # Convert timestamps to timezone-naive
        df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp']).dt.tz_localize(None)
        df_indicators['timestamp'] = pd.to_datetime(df_indicators['timestamp']).dt.tz_localize(None)
        df_optimized['timestamp'] = pd.to_datetime(df_optimized['timestamp']).dt.tz_localize(None)

        # Merge all data (SAME AS PREPROCESSING)
        df = pd.merge(df_ohlcv, df_indicators, on='timestamp', how='inner')
        df = pd.merge(df, df_optimized, on='timestamp', how='inner')
        df = df.sort_values('timestamp').reset_index(drop=True)

        # ============================================================
        # FIX #1: NaN handling - NOW MATCHES PREPROCESSING EXACTLY
        # ============================================================
        nan_counts_before = df.isna().sum().sum()
        if nan_counts_before > 0:
            # Forward fill limited to 2 periods (SAME as preprocessing)
            df = df.ffill(limit=2)
            # Backward fill only for remaining initial NaNs
            df = df.bfill(limit=2)
            # Drop rows with remaining NaNs (SAME as preprocessing - NOT fillna(0))
            rows_before = len(df)
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            print(f"   ✓ Cleaned {nan_counts_before:,} NaN values ({rows_dropped:,} rows dropped)")

        # Validate sufficient data after cleaning
        if len(df) < self.sequence_length:
            raise ValueError(
                f"Insufficient clean data after NaN handling: "
                f"need {self.sequence_length}, got {len(df)}. "
                f"Increase fetch limit or check data quality."
            )

        print(f"   ✓ Loaded: {len(df)} candles")
        print(f"   ✓ Columns: {len(df.columns)} features")
        print(f"   ✓ Range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        print(f"   ✓ Latest price: ${df['close'].iloc[-1]:,.2f}")
        print(f"{'='*70}")

        return df

    def validate_features(self, df):
        """Validate that all required features exist"""
        missing_cols = [col for col in self.feature_cols if col not in df.columns]

        if missing_cols:
            print(f"\n❌ Missing features ({len(missing_cols)}):")
            for col in missing_cols[:10]:  # Show first 10
                print(f"   - {col}")
            if len(missing_cols) > 10:
                print(f"   ... and {len(missing_cols) - 10} more")

            print(f"\n📋 Available features ({len(df.columns)}):")
            print(f"   {', '.join(list(df.columns)[:20])}")
            if len(df.columns) > 20:
                print(f"   ... and {len(df.columns) - 20} more")

            raise ValueError(f"Missing {len(missing_cols)} required features")

        print(f"   ✓ All {len(self.feature_cols)} features present")

    # ============================================================
    # FIX #2: Added time gap validation (SAME as preprocessing)
    # ============================================================
    def validate_sequence_continuity(self, df):
        """
        Validate that sequence has no time gaps
        ALIGNED: Same validation as preprocessing
        """
        if len(df) < 2:
            return True  # Can't check gaps with less than 2 rows

        timestamps = df['timestamp']
        time_diffs = timestamps.diff()[1:]

        expected_diff = pd.Timedelta(hours=1)
        tolerance = pd.Timedelta(minutes=5)

        is_continuous = (
            (time_diffs >= expected_diff - tolerance) &
            (time_diffs <= expected_diff + tolerance)
        ).all()

        if not is_continuous:
            # Find gaps
            gaps = time_diffs[
                (time_diffs < expected_diff - tolerance) |
                (time_diffs > expected_diff + tolerance)
            ]
            gap_indices = gaps.index.tolist()

            print(f"\n⚠️  WARNING: Time gaps detected!")
            print(f"   Found {len(gaps)} gap(s) in sequence:")
            for idx in gap_indices[:3]:  # Show first 3 gaps
                prev_time = timestamps.iloc[idx-1]
                curr_time = timestamps.iloc[idx]
                gap_hours = (curr_time - prev_time).total_seconds() / 3600
                print(f"   - Gap at index {idx}: {prev_time} → {curr_time} ({gap_hours:.1f}h)")

            raise ValueError(
                f"Sequence has {len(gaps)} time gap(s). "
                f"Model requires continuous hourly data. "
                f"Check data collection or increase fetch limit."
            )

        print(f"   ✓ Sequence continuity validated (no gaps)")
        return True

    def preprocess_data(self, df):
        """
        Preprocess data for prediction
        ALIGNED: Same as preprocessing pipeline with all fixes
        """
        print(f"\n{'='*70}")
        print(f"⚙️  Preprocessing Data")
        print(f"{'='*70}")

        # Take only the last sequence_length rows
        if len(df) > self.sequence_length:
            df = df.iloc[-self.sequence_length:].reset_index(drop=True)
            print(f"   ✓ Using last {self.sequence_length} candles")

        # FIX #2: Validate sequence continuity (SAME as preprocessing)
        self.validate_sequence_continuity(df)

        # Store sequence info
        seq_start = df['timestamp'].iloc[0]
        seq_end = df['timestamp'].iloc[-1]
        latest_close = df['close'].iloc[-1]

        # Validate features
        self.validate_features(df)

        # Select features in EXACT order from training
        X = df[self.feature_cols].values
        print(f"   ✓ Features shape: {X.shape}")

        # Scale features (SAME AS TRAINING)
        X_scaled = self.scaler.transform(X)

        # Clip outliers (SAME AS PREPROCESSING)
        X_scaled = np.clip(X_scaled, -10, 10)
        print(f"   ✓ Features scaled and clipped")

        # Handle any remaining NaN/inf after scaling
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequence: (1, seq_len, features)
        X_seq = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1])
        mask = np.ones((1, X_scaled.shape[0]))

        print(f"   ✓ Sequence shape: {X_seq.shape}")
        print(f"   ✓ Mask shape: {mask.shape}")
        print(f"   ✓ Period: {seq_start} → {seq_end}")
        print(f"{'='*70}")

        return X_seq, mask, seq_start, seq_end, latest_close

    def predict(self, X_seq, mask):
        """Make prediction with confidence scores"""
        print(f"\n{'='*70}")
        print(f"🔮 Making Prediction")
        print(f"{'='*70}")

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        mask_tensor = torch.FloatTensor(mask).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor, mask=mask_tensor)
            probs = torch.softmax(outputs, dim=1)

        probs_np = probs.cpu().numpy()[0]
        pred_class = np.argmax(probs_np)
        pred_label = self.class_names[pred_class]
        confidence = probs_np[pred_class]

        print(f"   ✓ Raw outputs: {outputs.cpu().numpy()[0]}")
        print(f"   ✓ Probabilities: DOWN={probs_np[0]:.3f}, HOLD={probs_np[1]:.3f}, UP={probs_np[2]:.3f}")
        print(f"   ✓ Prediction: {pred_label} ({confidence:.1%})")
        print(f"{'='*70}")

        return {
            'prediction': pred_label,
            'confidence': confidence,
            'prob_down': probs_np[0],
            'prob_hold': probs_np[1],
            'prob_up': probs_np[2],
            'raw_outputs': outputs.cpu().numpy()[0].tolist()
        }

    def save_prediction(self, result, close_price, seq_start, seq_end):
        """Save prediction to database"""
        print(f"\n💾 Saving to database...")

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Convert timestamps to string
        if hasattr(seq_start, 'strftime'):
            seq_start = seq_start.strftime('%Y-%m-%d %H:%M:%S')
        if hasattr(seq_end, 'strftime'):
            seq_end = seq_end.strftime('%Y-%m-%d %H:%M:%S')

        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                prob_down REAL NOT NULL,
                prob_hold REAL NOT NULL,
                prob_up REAL NOT NULL,
                close_price REAL NOT NULL,
                sequence_start DATETIME NOT NULL,
                sequence_end DATETIME NOT NULL,
                sequence_length INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Insert prediction
        cursor.execute('''
            INSERT INTO predictions (
                timestamp, prediction, confidence,
                prob_down, prob_hold, prob_up,
                close_price, sequence_start, sequence_end,
                sequence_length, model_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            result['prediction'],
            float(result['confidence']),
            float(result['prob_down']),
            float(result['prob_hold']),
            float(result['prob_up']),
            float(close_price),
            seq_start,
            seq_end,
            int(self.sequence_length),
            self.model_type
        ))

        conn.commit()
        pred_id = cursor.lastrowid
        conn.close()

        print(f"   ✓ Saved (ID: {pred_id})")

    def interpret_prediction(self, result, close_price):
        """Interpret prediction with actionable insights"""
        print(f"\n{'='*70}")
        print(f"📊 PREDICTION RESULT")
        print(f"{'='*70}")

        print(f"\n💰 Current Price: ${close_price:,.2f}")
        print(f"\n🎯 Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.1%}")

        print(f"\n📈 Probabilities:")
        print(f"   📉 DOWN: {result['prob_down']:.1%}")
        print(f"   ➖ HOLD: {result['prob_hold']:.1%}")
        print(f"   📈 UP:   {result['prob_up']:.1%}")

        # Signal strength
        if result['confidence'] >= 0.65:
            strength = "🔥 STRONG"
            color = "🟢"
        elif result['confidence'] >= 0.50:
            strength = "⚡ MODERATE"
            color = "🟡"
        elif result['confidence'] >= 0.40:
            strength = "⚠️  WEAK"
            color = "🟠"
        else:
            strength = "❌ VERY WEAK"
            color = "🔴"

        print(f"\n{color} Signal Strength: {strength}")

        # Actionable insight
        print(f"\n💡 Interpretation:")
        if result['prediction'] == 'UP':
            if result['confidence'] >= 0.60:
                print(f"   Strong bullish signal. Consider long position.")
                print(f"   Expected move: Price may rise in next 4 hours.")
            elif result['confidence'] >= 0.45:
                print(f"   Moderate bullish signal. Wait for confirmation.")
            else:
                print(f"   Weak signal. High uncertainty - avoid trading.")

        elif result['prediction'] == 'DOWN':
            if result['confidence'] >= 0.60:
                print(f"   Strong bearish signal. Consider short position.")
                print(f"   Expected move: Price may fall in next 4 hours.")
            elif result['confidence'] >= 0.45:
                print(f"   Moderate bearish signal. Wait for confirmation.")
            else:
                print(f"   Weak signal. High uncertainty - avoid trading.")

        else:  # HOLD
            if result['confidence'] >= 0.50:
                print(f"   Range-bound market. Expect sideways movement.")
                print(f"   Strategy: Wait for clearer signal or range trade.")
            else:
                print(f"   Unclear market direction. Avoid trading.")

        # Risk warning
        if result['confidence'] < 0.50:
            print(f"\n⚠️  WARNING: Low confidence prediction!")
            print(f"   Model is uncertain. Do NOT take high-risk positions.")

        # Probability spread analysis
        prob_spread = max(result['prob_down'], result['prob_hold'], result['prob_up']) - \
                     min(result['prob_down'], result['prob_hold'], result['prob_up'])

        if prob_spread < 0.20:
            print(f"\n⚠️  CAUTION: Probabilities are close ({prob_spread:.1%} spread)")
            print(f"   Market indecision - consider waiting.")

        print(f"\n{'='*70}")

    def run(self, save_to_db=True, show_interpretation=True):
        """Main prediction pipeline"""
        print(f"\n{'='*70}")
        print(f"🚀 BTC PRICE PREDICTION - LIVE")
        print(f"{'='*70}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {self.model_type}")
        print(f"Sequence: {self.sequence_length} hours")
        print(f"Horizon: 4 hours ahead")
        print(f"{'='*70}")

        try:
            # 1. Fetch data
            df = self.fetch_latest_data()

            # 2. Preprocess
            X_seq, mask, seq_start, seq_end, close_price = self.preprocess_data(df)

            # 3. Predict
            result = self.predict(X_seq, mask)

            # 4. Interpret
            if show_interpretation:
                self.interpret_prediction(result, close_price)

            # 5. Save to database
            if save_to_db:
                self.save_prediction(result, close_price, seq_start, seq_end)

            print(f"\n✅ Prediction completed successfully!\n")

            return result

        except Exception as e:
            print(f"\n{'='*70}")
            print(f"❌ ERROR OCCURRED")
            print(f"{'='*70}")
            print(f"{str(e)}\n")
            import traceback
            traceback.print_exc()
            print(f"{'='*70}\n")
            return None


def view_history(db_path='data/btc_ohlcv.db', limit=20):
    """View prediction history with statistics"""
    db_path = ROOT_DIR / db_path if not Path(db_path).is_absolute() else Path(db_path)

    conn = sqlite3.connect(str(db_path))

    # Check if table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    if not cursor.fetchone():
        print("❌ No predictions found. Run prediction first.")
        conn.close()
        return

    df = pd.read_sql_query(f"""
        SELECT
            timestamp,
            prediction,
            confidence,
            prob_down,
            prob_hold,
            prob_up,
            close_price,
            model_type
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT {limit}
    """, conn)
    conn.close()

    if len(df) == 0:
        print("❌ No predictions found.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"\n{'='*70}")
    print(f"📜 PREDICTION HISTORY (Last {len(df)})")
    print(f"{'='*70}\n")

    # Format and display
    display_df = df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    display_df['close_price'] = display_df['close_price'].apply(lambda x: f"${x:,.0f}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)

    print(display_df[['timestamp', 'prediction', 'confidence', 'close_price', 'model_type']].to_string(index=False))

    print(f"\n{'='*70}")
    print(f"📊 STATISTICS")
    print(f"{'='*70}")

    # Prediction distribution
    print(f"\n🎯 Prediction Distribution:")
    pred_counts = df['prediction'].value_counts()
    for pred in ['DOWN', 'HOLD', 'UP']:
        count = pred_counts.get(pred, 0)
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"   {pred}: {count:3d} ({pct:5.1f}%)")

    # Confidence statistics
    print(f"\n📈 Confidence Statistics:")
    print(f"   Mean:   {df['confidence'].mean():.1%}")
    print(f"   Median: {df['confidence'].median():.1%}")
    print(f"   Min:    {df['confidence'].min():.1%}")
    print(f"   Max:    {df['confidence'].max():.1%}")

    # High confidence predictions
    high_conf = df[df['confidence'] >= 0.60]
    print(f"\n🔥 High Confidence (≥60%): {len(high_conf)} ({len(high_conf)/len(df)*100:.1f}%)")

    # Recent trend
    if len(df) >= 5:
        recent = df.head(5)
        print(f"\n📊 Recent Trend (Last 5):")
        trend_str = " → ".join([f"{p}" for p in recent['prediction'].values])
        print(f"   {trend_str}")

    print(f"\n{'='*70}\n")


def compare_predictions(db_path='data/btc_ohlcv.db', hours_back=24):
    """
    Compare predictions with actual price movements
    ALIGNED: Uses same thresholds as preprocessing (±1.0%)
    """
    db_path = ROOT_DIR / db_path if not Path(db_path).is_absolute() else Path(db_path)

    conn = sqlite3.connect(str(db_path))

    # Get predictions from last N hours
    cutoff_time = (datetime.now() - pd.Timedelta(hours=hours_back)).strftime('%Y-%m-%d %H:%M:%S')

    df_pred = pd.read_sql_query(f"""
        SELECT
            timestamp,
            prediction,
            confidence,
            close_price as pred_price,
            sequence_end
        FROM predictions
        WHERE timestamp >= '{cutoff_time}'
        ORDER BY timestamp
    """, conn)

    if len(df_pred) == 0:
        print(f"❌ No predictions found in last {hours_back} hours.")
        conn.close()
        return

    df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
    df_pred['sequence_end'] = pd.to_datetime(df_pred['sequence_end'])

    # Get actual prices (4 hours after each prediction)
    results = []
    for _, row in df_pred.iterrows():
        target_time = row['sequence_end'] + pd.Timedelta(hours=4)

        # Query actual price at target time (±30 min tolerance)
        query = f"""
            SELECT close, timestamp
            FROM btc_1h
            WHERE timestamp BETWEEN
                '{(target_time - pd.Timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')}'
                AND '{(target_time + pd.Timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')}'
            ORDER BY ABS(JULIANDAY(timestamp) - JULIANDAY('{target_time.strftime('%Y-%m-%d %H:%M:%S')}'))
            LIMIT 1
        """
        df_actual = pd.read_sql_query(query, conn)

        if len(df_actual) > 0:
            actual_price = df_actual['close'].iloc[0]
            price_change = (actual_price - row['pred_price']) / row['pred_price'] * 100

            # Determine actual outcome (ALIGNED with preprocessing thresholds: ±1.0%)
            if price_change > 1.0:
                actual_outcome = 'UP'
            elif price_change < -1.0:
                actual_outcome = 'DOWN'
            else:
                actual_outcome = 'HOLD'

            correct = (row['prediction'] == actual_outcome)

            results.append({
                'timestamp': row['timestamp'],
                'prediction': row['prediction'],
                'actual': actual_outcome,
                'correct': correct,
                'confidence': row['confidence'],
                'price_change': price_change,
                'pred_price': row['pred_price'],
                'actual_price': actual_price
            })

    conn.close()

    if len(results) == 0:
        print(f"❌ No predictions with actual outcomes found.")
        return

    df_results = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"🎯 PREDICTION ACCURACY ANALYSIS")
    print(f"{'='*70}")
    print(f"Period: Last {hours_back} hours")
    print(f"Predictions with outcomes: {len(df_results)}")
    print(f"Thresholds: UP >1.0%, DOWN <-1.0%, HOLD ±1.0%")

    # Overall accuracy
    accuracy = df_results['correct'].sum() / len(df_results) * 100
    print(f"\n📊 Overall Accuracy: {accuracy:.1f}%")

    # Per-class accuracy
    print(f"\n📈 Per-Class Accuracy:")
    for pred_class in ['DOWN', 'HOLD', 'UP']:
        class_df = df_results[df_results['prediction'] == pred_class]
        if len(class_df) > 0:
            class_acc = class_df['correct'].sum() / len(class_df) * 100
            print(f"   {pred_class}: {class_acc:5.1f}% ({class_df['correct'].sum()}/{len(class_df)})")

    # Confidence vs accuracy
    print(f"\n🎯 Confidence vs Accuracy:")
    high_conf = df_results[df_results['confidence'] >= 0.60]
    low_conf = df_results[df_results['confidence'] < 0.60]

    if len(high_conf) > 0:
        high_acc = high_conf['correct'].sum() / len(high_conf) * 100
        print(f"   High confidence (≥60%): {high_acc:5.1f}% ({len(high_conf)} predictions)")

    if len(low_conf) > 0:
        low_acc = low_conf['correct'].sum() / len(low_conf) * 100
        print(f"   Low confidence (<60%):  {low_acc:5.1f}% ({len(low_conf)} predictions)")

    # Show recent predictions
    print(f"\n📋 Recent Predictions:")
    print(f"{'Time':<17} {'Pred':<6} {'Actual':<6} {'Conf':<7} {'Price Δ':<9} {'Result'}")
    print(f"{'-'*70}")

    for _, row in df_results.head(10).iterrows():
        result_icon = '✅' if row['correct'] else '❌'
        print(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M'):<17} "
              f"{row['prediction']:<6} {row['actual']:<6} "
              f"{row['confidence']:>6.1%} {row['price_change']:>8.2f}% {result_icon}")

    print(f"\n{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='BTC Price Prediction with LSTM+Attention')
    parser.add_argument('--model', default='models/lstm_full.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--scaler', default='preprocessed_data_lstm_1h/scalers.pkl',
                       help='Path to scaler file')
    parser.add_argument('--features', default='preprocessed_data_lstm_1h/feature_cols.pkl',
                       help='Path to feature columns file')
    parser.add_argument('--sequence', default='preprocessed_data_lstm_1h/sequence_length.pkl',
                       help='Path to sequence length file')
    parser.add_argument('--db', default='data/btc_ohlcv.db',
                       help='Path to database')
    parser.add_argument('--history', action='store_true',
                       help='View prediction history')
    parser.add_argument('--compare', action='store_true',
                       help='Compare predictions with actual outcomes')
    parser.add_argument('--hours', type=int, default=24,
                       help='Hours to look back for comparison (default: 24)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save prediction to database')
    parser.add_argument('--limit', type=int, default=20,
                       help='Number of history records to show')

    args = parser.parse_args()

    # View history mode
    if args.history:
        view_history(args.db, args.limit)
        return

    # Compare predictions mode
    if args.compare:
        compare_predictions(args.db, args.hours)
        return

    # Check required files
    required_files = {
        'Model': args.model,
        'Scaler': args.scaler,
        'Features': args.features,
        'Sequence': args.sequence,
        'Database': args.db
    }

    missing = []
    for name, path in required_files.items():
        full_path = ROOT_DIR / path if not Path(path).is_absolute() else Path(path)
        if not full_path.exists():
            missing.append(f"{name}: {full_path}")

    if missing:
        print(f"\n{'='*70}")
        print("❌ MISSING REQUIRED FILES")
        print(f"{'='*70}")
        for m in missing:
            print(f"   - {m}")
        print(f"\n💡 Make sure you have:")
        print(f"   1. Trained the model (run train.py)")
        print(f"   2. Preprocessed data exists (run preprocessing.py)")
        print(f"   3. Database is accessible")
        print(f"{'='*70}\n")
        return

    # Run prediction
    try:
        predictor = BTCPredictor(
            model_path=args.model,
            scaler_path=args.scaler,
            feature_cols_path=args.features,
            sequence_length_path=args.sequence,
            db_path=args.db
        )

        result = predictor.run(save_to_db=not args.no_save)

        if result:
            print(f"{'='*70}")
            print(f"✅ SUCCESS")
            print(f"{'='*70}")
            print(f"\n💡 Quick Commands:")
            print(f"   View history:  python predict.py --history")
            print(f"   Compare:       python predict.py --compare --hours 48")
            print(f"   Help:          python predict.py --help")
            print(f"\n{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"❌ PREDICTION FAILED")
            print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ FATAL ERROR")
        print(f"{'='*70}")
        print(f"{str(e)}\n")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
