import os
import numpy as np
import pandas as pd
import joblib
import torch
import sqlite3
from datetime import datetime
from pathlib import Path

from models import MultiInputBidirectionalLSTMAttention

ROOT_DIR = Path.cwd()


class BTCMultiInputPredictor:
    def __init__(self, model_path, preprocessed_dir, db_path='data/btc_ohlcv.db', config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_path = ROOT_DIR / model_path if not Path(model_path).is_absolute() else Path(model_path)
        self.preprocessed_dir = ROOT_DIR / preprocessed_dir if not Path(preprocessed_dir).is_absolute() else Path(preprocessed_dir)
        self.db_path = ROOT_DIR / db_path if not Path(db_path).is_absolute() else Path(db_path)

        self.config = config or {
            'thresholds': {'strong': 0.60, 'moderate': 0.50, 'weak': 0.40},
            'min_confidence_to_save': 0.30,
            'price_change_thresholds': {'up': 1.0, 'down': -1.0}
        }

        print("Initializing Multi-Input BTC Predictor (6 Regime Features)")
        print(f"Device: {self.device}")

        checkpoint = torch.load(str(self.model_path), map_location=self.device)
        self.input_sizes = checkpoint['input_sizes']
        self.feature_groups = checkpoint['feature_groups']

        self.model = MultiInputBidirectionalLSTMAttention(
            input_sizes=self.input_sizes,
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            num_heads=checkpoint['num_heads'],
            dropout=checkpoint['dropout'],
            num_classes=3
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Input branches: {self.input_sizes}")

        self.scalers = joblib.load(str(self.preprocessed_dir / 'scalers.pkl'))
        self.sequence_length = joblib.load(str(self.preprocessed_dir / 'sequence_length.pkl'))
        metadata = joblib.load(str(self.preprocessed_dir / 'metadata.pkl'))

        print(f"Sequence length: {self.sequence_length}")
        print(f"Total features: {metadata['total_features']}")

        max_rolling_window = 720
        self.min_data_required = self.sequence_length + max_rolling_window + 10

        self.class_names = ['DOWN', 'HOLD', 'UP']

    def _standardize_timestamp(self, df, col='timestamp'):
        df[col] = pd.to_datetime(df[col])
        if df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)
        return df

    def fetch_latest_data(self):
        print(f"\nFetching latest {self.min_data_required} candles...")

        conn = sqlite3.connect(str(self.db_path))

        try:
            query_ohlcv = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM btc_1h
                ORDER BY timestamp DESC
                LIMIT {self.min_data_required}
            """
            df_ohlcv = pd.read_sql_query(query_ohlcv, conn)

            if len(df_ohlcv) < self.sequence_length:
                raise ValueError(f"Insufficient data: need {self.sequence_length}, got {len(df_ohlcv)}")

            query_indicators = f"""
                SELECT * FROM smc_btc_1h_technical_indicators
                ORDER BY timestamp DESC
                LIMIT {self.min_data_required}
            """
            df_indicators = pd.read_sql_query(query_indicators, conn)

            query_regimes = f"""
                SELECT * FROM smc_btc_1h_market_regimes
                ORDER BY timestamp DESC
                LIMIT {self.min_data_required}
            """
            df_regimes = pd.read_sql_query(query_regimes, conn)

        finally:
            conn.close()

        df_ohlcv = df_ohlcv.iloc[::-1].reset_index(drop=True)
        df_indicators = df_indicators.iloc[::-1].reset_index(drop=True)
        df_regimes = df_regimes.iloc[::-1].reset_index(drop=True)

        for df in [df_ohlcv, df_indicators, df_regimes]:
            df = self._standardize_timestamp(df)

        df = pd.merge(df_ohlcv, df_indicators, on='timestamp', how='inner')
        df = pd.merge(df, df_regimes, on='timestamp', how='inner')
        df = df.sort_values('timestamp').reset_index(drop=True)

        critical_features = ['close', 'volume', 'ema_21', 'rsi_14']
        df = df.dropna(subset=critical_features)
        df = df.ffill(limit=3).bfill(limit=3).dropna()

        if len(df) < self.sequence_length:
            raise ValueError(f"Insufficient clean data: need {self.sequence_length}, got {len(df)}")

        for col in df.columns:
            if col != 'timestamp' and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].astype('float32')

        print(f"Loaded: {len(df)} candles")
        print(f"Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        print(f"Latest price: ${df['close'].iloc[-1]:,.2f}")

        return df

    def validate_features(self, df):
        all_required_features = []
        for features in self.feature_groups.values():
            all_required_features.extend(features)

        missing_cols = [col for col in all_required_features if col not in df.columns]

        if missing_cols:
            print(f"\nMissing features: {missing_cols[:10]}")
            raise ValueError(f"Missing {len(missing_cols)} required features")

    def validate_sequence_continuity(self, df):
        if len(df) < 2:
            return True

        timestamps = df['timestamp']
        time_diffs = timestamps.diff()[1:]

        expected_diff = pd.Timedelta(hours=1)
        tolerance = pd.Timedelta(minutes=5)

        is_continuous = (
            (time_diffs >= expected_diff - tolerance) &
            (time_diffs <= expected_diff + tolerance)
        ).all()

        if not is_continuous:
            raise ValueError("Sequence has time gaps")

        return True

    def preprocess_data(self, df):
        print("\nPreprocessing...")

        if len(df) > self.sequence_length:
            df = df.iloc[-self.sequence_length:].reset_index(drop=True)

        self.validate_sequence_continuity(df)
        self.validate_features(df)

        seq_start = df['timestamp'].iloc[0]
        seq_end = df['timestamp'].iloc[-1]
        latest_close = df['close'].iloc[-1]

        sequences = {}

        for group_name, feature_cols in self.feature_groups.items():
            # FIX: Transform DataFrame directly (keep feature names)
            X_group = df[feature_cols].astype('float32')  # ← DataFrame, not array
            X_scaled = self.scalers[group_name].transform(X_group)  # ← No warning
            X_scaled = np.clip(X_scaled, -10, 10)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_seq = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1])
            sequences[group_name] = X_seq

        print(f"Sequence: {seq_start} to {seq_end}")

        return sequences, seq_start, seq_end, latest_close

    def predict(self, sequences):
        print("\nMaking prediction...")

        inputs = {
            name: torch.FloatTensor(seq).to(self.device)
            for name, seq in sequences.items()
        }

        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)

        probs_np = probs.cpu().numpy()[0]
        pred_class = np.argmax(probs_np)
        pred_label = self.class_names[pred_class]
        confidence = probs_np[pred_class]

        print(f"Probabilities: DOWN={probs_np[0]:.3f} HOLD={probs_np[1]:.3f} UP={probs_np[2]:.3f}")
        print(f"Prediction: {pred_label} ({confidence:.1%})")

        return {
            'prediction': pred_label,
            'confidence': confidence,
            'prob_down': probs_np[0],
            'prob_hold': probs_np[1],
            'prob_up': probs_np[2]
        }

    def save_prediction(self, result, close_price, seq_start, seq_end):
        if result['confidence'] < self.config['min_confidence_to_save']:
            print(f"Confidence {result['confidence']:.1%} below threshold, not saving")
            return None

        print("\nSaving to database...")

        if hasattr(seq_start, 'strftime'):
            seq_start = seq_start.strftime('%Y-%m-%d %H:%M:%S')
        if hasattr(seq_end, 'strftime'):
            seq_end = seq_end.strftime('%Y-%m-%d %H:%M:%S')

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            cursor = conn.cursor()

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
                'multi_input_lstm_optimized_6_regimes'
            ))

            conn.commit()
            pred_id = cursor.lastrowid
            conn.close()

            print(f"Saved (ID: {pred_id})")
            return pred_id

        except Exception as e:
            print(f"Save failed: {e}")
            return None

    def interpret_prediction(self, result, close_price):
        print(f"\n{'='*60}")
        print("PREDICTION RESULT")
        print(f"{'='*60}")

        print(f"\nCurrent Price: ${close_price:,.2f}")
        print(f"Prediction: {result['prediction']} ({result['confidence']:.1%})")

        print(f"\nProbabilities:")
        print(f"  DOWN: {result['prob_down']:.1%}")
        print(f"  HOLD: {result['prob_hold']:.1%}")
        print(f"  UP:   {result['prob_up']:.1%}")

        thresholds = self.config['thresholds']
        if result['confidence'] >= thresholds['strong']:
            strength = "STRONG"
        elif result['confidence'] >= thresholds['moderate']:
            strength = "MODERATE"
        elif result['confidence'] >= thresholds['weak']:
            strength = "WEAK"
        else:
            strength = "VERY WEAK"

        print(f"\nSignal Strength: {strength}")

        print(f"\nInterpretation:")
        if result['prediction'] == 'UP':
            if result['confidence'] >= thresholds['strong']:
                print("  Strong bullish signal")
            elif result['confidence'] >= thresholds['moderate']:
                print("  Moderate bullish signal - wait for confirmation")
            else:
                print("  Weak signal - avoid trading")
        elif result['prediction'] == 'DOWN':
            if result['confidence'] >= thresholds['strong']:
                print("  Strong bearish signal")
            elif result['confidence'] >= thresholds['moderate']:
                print("  Moderate bearish signal - wait for confirmation")
            else:
                print("  Weak signal - avoid trading")
        else:
            if result['confidence'] >= thresholds['moderate']:
                print("  Range-bound market")
            else:
                print("  Unclear direction - avoid trading")

        if result['confidence'] < thresholds['moderate']:
            print("\nWARNING: Low confidence prediction")

        print(f"{'='*60}")

    def run(self, save_to_db=True, show_interpretation=True):
        print(f"\n{'='*60}")
        print("BTC MULTI-INPUT PREDICTION (6 Regime Features)")
        print(f"{'='*60}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sequence: {self.sequence_length} hours | Horizon: 4 hours")
        print(f"{'='*60}")

        try:
            df = self.fetch_latest_data()
            sequences, seq_start, seq_end, close_price = self.preprocess_data(df)
            result = self.predict(sequences)

            if show_interpretation:
                self.interpret_prediction(result, close_price)

            if save_to_db:
                self.save_prediction(result, close_price, seq_start, seq_end)

            print("\nPrediction completed!")
            return result

        except Exception as e:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def view_history(db_path='data/btc_ohlcv.db', limit=20):
    db_path = ROOT_DIR / db_path if not Path(db_path).is_absolute() else Path(db_path)
    conn = sqlite3.connect(str(db_path))

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    if not cursor.fetchone():
        print("No predictions found")
        conn.close()
        return

    df = pd.read_sql_query(f"""
        SELECT timestamp, prediction, confidence, close_price, model_type
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT {limit}
    """, conn)
    conn.close()

    if len(df) == 0:
        print("No predictions found")
        return

    print(f"\n{'='*60}")
    print(f"PREDICTION HISTORY (Last {len(df)})")
    print(f"{'='*60}\n")

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
    df['close_price'] = df['close_price'].apply(lambda x: f"${x:,.0f}")

    print(df.to_string(index=False))

    df_full = pd.read_sql_query("SELECT * FROM predictions", sqlite3.connect(str(db_path)))
    pred_counts = df_full['prediction'].value_counts()

    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}\n")

    for pred in ['DOWN', 'HOLD', 'UP']:
        count = pred_counts.get(pred, 0)
        pct = count / len(df_full) * 100 if len(df_full) > 0 else 0
        print(f"  {pred}: {count:3d} ({pct:5.1f}%)")

    print(f"\n{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='BTC Multi-Input Prediction (6 Regime Features)')
    parser.add_argument('--model', default='models/multi_input_lstm_optimized_full.pth')
    parser.add_argument('--preprocessed', default='preprocessed_data_multi_lstm_1h')
    parser.add_argument('--db', default='data/btc_ohlcv.db')
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--min-confidence', type=float, default=0.30)

    args = parser.parse_args()

    if args.history:
        view_history(args.db, args.limit)
        return

    required_files = {
        'Model': args.model,
        'Preprocessed': args.preprocessed,
        'Database': args.db
    }

    missing = []
    for name, path in required_files.items():
        full_path = ROOT_DIR / path if not Path(path).is_absolute() else Path(path)
        if not full_path.exists():
            missing.append(f"{name}: {full_path}")

    if missing:
        print(f"\n{'='*60}")
        print("MISSING FILES")
        print(f"{'='*60}")
        for m in missing:
            print(f"  {m}")
        print(f"\nRun: python train_multi_input_lstm.py first")
        print(f"{'='*60}\n")
        return

    config = {
        'thresholds': {
            'strong': 0.60,
            'moderate': 0.50,
            'weak': 0.40
        },
        'min_confidence_to_save': args.min_confidence,
        'price_change_thresholds': {
            'up': 1.0,
            'down': -1.0
        }
    }

    try:
        predictor = BTCMultiInputPredictor(
            model_path=args.model,
            preprocessed_dir=args.preprocessed,
            db_path=args.db,
            config=config
        )

        result = predictor.run(save_to_db=not args.no_save)

        if result:
            print(f"\n{'='*60}")
            print("SUCCESS")
            print(f"{'='*60}")
            print(f"\nCommands:")
            print(f"  View history: python predict_multi_input_lstm.py --history")
            print(f"  No save:      python predict_multi_input_lstm.py --no-save")
            print(f"\n{'='*60}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user\n")
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
