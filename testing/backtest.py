import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Paths
ROOT_DIR = Path.cwd()
MODELS_DIR = ROOT_DIR / 'models'
sys.path.insert(0, str(MODELS_DIR))

from models import LSTMAttentionEnhanced, LSTMAttention, SimpleLSTM


class BTCBacktesterOptimized:
    def __init__(self, model_path, scaler_path, feature_cols_path,
                 sequence_length_path, db_path='data/btc_ohlcv.db',
                 initial_capital=10000, trading_fee=0.001, position_size=0.50,
                 stop_loss=0.06, min_confidence=0.50, max_capital_per_trade=None,
                 max_trades_per_day=10, slippage=0.0003, min_hold_hours=4,
                 max_position_hours=24, use_trailing_stop=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Paths
        self.model_path = ROOT_DIR / model_path if not Path(model_path).is_absolute() else Path(model_path)
        self.scaler_path = ROOT_DIR / scaler_path if not Path(scaler_path).is_absolute() else Path(scaler_path)
        self.feature_cols_path = ROOT_DIR / feature_cols_path if not Path(feature_cols_path).is_absolute() else Path(feature_cols_path)
        self.sequence_length_path = ROOT_DIR / sequence_length_path if not Path(sequence_length_path).is_absolute() else Path(sequence_length_path)
        self.db_path = ROOT_DIR / db_path if not Path(db_path).is_absolute() else Path(db_path)

        # Trading parameters (OPTIMIZED)
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.min_confidence = min_confidence
        self.max_capital_per_trade = max_capital_per_trade or (initial_capital * 2)  # More conservative
        self.max_trades_per_day = max_trades_per_day
        self.slippage = slippage
        self.min_hold_hours = min_hold_hours
        self.max_position_hours = max_position_hours
        self.use_trailing_stop = use_trailing_stop

        # Results storage
        self.trades = []
        self.equity_curve = []
        self.predictions_log = []
        self.daily_trades_count = {}
        self.skipped_trades = {
            'confidence': 0,
            'daily_limit': 0,
            'hold_time': 0,
            'time_gap': 0,
            'capital_insufficient': 0
        }

        # Load model
        self._load_model()

        # Load preprocessing artifacts
        self._load_artifacts()

    def _load_model(self):
        """Load trained model"""
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

    def _load_artifacts(self):
        """Load preprocessing artifacts"""
        scaler_data = joblib.load(str(self.scaler_path))
        self.scaler = scaler_data['robust'] if isinstance(scaler_data, dict) else scaler_data
        self.feature_cols = joblib.load(str(self.feature_cols_path))
        self.sequence_length = joblib.load(str(self.sequence_length_path))

    def fetch_backtest_data(self, start_date=None, end_date=None):
        """Fetch historical data - ALIGNED with preprocessing"""
        print(f"\n{'='*80}")
        print(f"📊 BACKTEST CONFIGURATION")
        print(f"{'='*80}")
        print(f"Capital:      ${self.initial_capital:,.2f}")
        print(f"Stop Loss:    {self.stop_loss*100:.1f}%")
        print(f"Position:     {self.position_size*100:.0f}%")
        print(f"Confidence:   {self.min_confidence*100:.0f}%")
        print(f"Max Trades:   {self.max_trades_per_day}/day")
        print(f"Hold Time:    {self.min_hold_hours}-{self.max_position_hours}h")
        print(f"{'='*80}\n")

        print(f"Loading data...")

        conn = sqlite3.connect(str(self.db_path))

        try:
            date_filter = ""
            if start_date:
                date_filter += f" AND timestamp >= '{start_date}'"
            if end_date:
                date_filter += f" AND timestamp <= '{end_date}'"

            # Load OHLCV
            query_ohlcv = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM btc_1h
                WHERE 1=1 {date_filter}
                ORDER BY timestamp ASC
            """
            df_ohlcv = pd.read_sql_query(query_ohlcv, conn)

            # Load Technical Indicators
            query_indicators = f"""
                SELECT * FROM smc_btc_1h_technical_indicators
                WHERE 1=1 {date_filter}
                ORDER BY timestamp ASC
            """
            df_indicators = pd.read_sql_query(query_indicators, conn)

            # Load Optimized Features
            query_optimized = f"""
                SELECT * FROM smc_btc_1h_optimized_features
                WHERE 1=1 {date_filter}
                ORDER BY timestamp ASC
            """
            df_optimized = pd.read_sql_query(query_optimized, conn)

        finally:
            conn.close()

        # Convert timestamps
        df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp']).dt.tz_localize(None)
        df_indicators['timestamp'] = pd.to_datetime(df_indicators['timestamp']).dt.tz_localize(None)
        df_optimized['timestamp'] = pd.to_datetime(df_optimized['timestamp']).dt.tz_localize(None)

        # Merge all data
        df = pd.merge(df_ohlcv, df_indicators, on='timestamp', how='inner')
        df = pd.merge(df, df_optimized, on='timestamp', how='inner')
        df = df.sort_values('timestamp').reset_index(drop=True)

        # NaN handling (SAME as preprocessing)
        nan_counts_before = df.isna().sum().sum()
        if nan_counts_before > 0:
            df = df.ffill(limit=2)
            df = df.bfill(limit=2)
            rows_before = len(df)
            df = df.dropna()

        # Validate sufficient data
        min_required = self.sequence_length + 100
        if len(df) < min_required:
            raise ValueError(
                f"Insufficient clean data: need {min_required}, got {len(df)}"
            )

        print(f"Loaded {len(df):,} candles: {df['timestamp'].iloc[0].strftime('%Y-%m-%d')} → {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"Price: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}\n")

        return df

    def validate_sequence_continuity(self, seq_data):
        """Validate sequence has no time gaps"""
        if 'timestamp' not in seq_data.columns or len(seq_data) < 2:
            return True

        timestamps = seq_data['timestamp']
        time_diffs = timestamps.diff()[1:]

        expected_diff = pd.Timedelta(hours=1)
        tolerance = pd.Timedelta(minutes=5)

        is_continuous = (
            (time_diffs >= expected_diff - tolerance) &
            (time_diffs <= expected_diff + tolerance)
        ).all()

        return is_continuous

    def create_sequences(self, df):
        """Create sequences with validation"""
        sequences = []
        timestamps = []
        prices = []
        indices = []

        skipped_gaps = 0

        # Need at least 4 hours ahead for future price
        for i in range(len(df) - self.sequence_length - 4):
            seq_data = df.iloc[i:i+self.sequence_length].copy()

            # Validate continuity
            if not self.validate_sequence_continuity(seq_data):
                skipped_gaps += 1
                continue

            # Check features
            missing = [col for col in self.feature_cols if col not in seq_data.columns]
            if missing:
                continue

            # Scale features
            X = seq_data[self.feature_cols].values
            X_scaled = self.scaler.transform(X)
            X_scaled = np.clip(X_scaled, -10, 10)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            sequences.append(X_scaled)
            timestamps.append(df['timestamp'].iloc[i+self.sequence_length])
            prices.append(df['close'].iloc[i+self.sequence_length])
            indices.append(i + self.sequence_length)

        return np.array(sequences), timestamps, prices, indices

    def predict_batch(self, sequences, batch_size=64):
        """Make predictions in batches"""
        num_sequences = sequences.shape[0]
        all_predictions = []
        all_confidences = []
        all_probs = []

        for i in range(0, num_sequences, batch_size):
            batch_end = min(i + batch_size, num_sequences)
            batch_sequences = sequences[i:batch_end]

            X_tensor = torch.FloatTensor(batch_sequences).to(self.device)
            batch_len = batch_sequences.shape[0]
            masks = torch.ones(batch_len, self.sequence_length).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_tensor, mask=masks)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

            batch_predictions = np.argmax(probs, axis=1)
            batch_confidences = np.max(probs, axis=1)

            all_predictions.extend(batch_predictions)
            all_confidences.extend(batch_confidences)
            all_probs.extend(probs)

            del X_tensor, masks, outputs
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            if (i // batch_size) % 10 == 0:
                progress = (batch_end / num_sequences) * 100
                print(f"   Predicting... {progress:.0f}%", end='\r')

        print(f"   Predicting... 100%  ")

        return np.array(all_predictions), np.array(all_confidences), np.array(all_probs)

    def _can_trade_today(self, timestamp):
        """Check daily trade limit"""
        date_key = timestamp.date()
        trades_today = self.daily_trades_count.get(date_key, 0)
        return trades_today < self.max_trades_per_day

    def _increment_daily_trade(self, timestamp):
        """Increment daily trade counter"""
        date_key = timestamp.date()
        self.daily_trades_count[date_key] = self.daily_trades_count.get(date_key, 0) + 1

    def _check_min_hold_time(self, current_time, entry_time):
        """Check minimum hold time"""
        time_diff = (current_time - entry_time).total_seconds() / 3600
        return time_diff >= self.min_hold_hours

    def _check_max_hold_time(self, current_time, entry_time):
        """Check if max hold time exceeded"""
        time_diff = (current_time - entry_time).total_seconds() / 3600
        return time_diff >= self.max_position_hours

    def run_backtest(self, df):
        """Run OPTIMIZED backtesting simulation"""
        print(f"Creating sequences...")
        sequences, timestamps, entry_prices, indices = self.create_sequences(df)
        print(f"Created {len(sequences):,} sequences\n")

        print(f"Making predictions...")
        predictions, confidences, probs = self.predict_batch(sequences)
        print(f"Completed {len(predictions):,} predictions\n")

        # Initialize trading
        capital = self.initial_capital
        position = None
        entry_price = 0
        entry_capital = 0
        entry_time = None
        entry_idx = None
        position_size_btc = 0
        highest_price = 0  # For trailing stop
        lowest_price = float('inf')

        self.trades = []
        self.equity_curve = [{'timestamp': timestamps[0], 'equity': capital, 'position': None}]

        print(f"Simulating trades...")
        class_names = ['DOWN', 'HOLD', 'UP']

        for i in range(len(predictions)):
            pred = predictions[i]
            conf = confidences[i]
            current_price = entry_prices[i]
            timestamp = timestamps[i]
            current_idx = indices[i]

            # Get future price (4 hours ahead - aligned with training)
            future_idx = current_idx + 4
            if future_idx >= len(df):
                break

            future_price = df.iloc[future_idx]['close']
            high_price = df.iloc[future_idx]['high']
            low_price = df.iloc[future_idx]['low']

            # Log prediction
            self.predictions_log.append({
                'timestamp': timestamp,
                'prediction': class_names[pred],
                'confidence': conf,
                'prob_down': probs[i][0],
                'prob_hold': probs[i][1],
                'prob_up': probs[i][2],
                'price': current_price,
                'future_price': future_price
            })

            # POSITION MANAGEMENT
            if position is not None:
                # Check minimum hold time
                if not self._check_min_hold_time(timestamp, entry_time):
                    self.equity_curve.append({
                        'timestamp': timestamp,
                        'equity': capital,
                        'position': position,
                        'entry_confidence': conf
                    })
                    continue

                # Update trailing stop levels
                if position == 'LONG':
                    highest_price = max(highest_price, high_price)
                elif position == 'SHORT':
                    lowest_price = min(lowest_price, low_price)

                # Check exit conditions
                should_exit = False
                exit_reason = None
                exit_price = None

                # 1. Stop Loss
                if position == 'LONG':
                    stop_price = entry_price * (1 - self.stop_loss)
                    if self.use_trailing_stop:
                        trailing_stop = highest_price * (1 - self.stop_loss)
                        stop_price = max(stop_price, trailing_stop)

                    if low_price <= stop_price:
                        exit_price = stop_price * (1 - self.slippage)
                        should_exit = True
                        exit_reason = 'STOP_LOSS'

                elif position == 'SHORT':
                    stop_price = entry_price * (1 + self.stop_loss)
                    if self.use_trailing_stop:
                        trailing_stop = lowest_price * (1 + self.stop_loss)
                        stop_price = min(stop_price, trailing_stop)

                    if high_price >= stop_price:
                        exit_price = stop_price * (1 + self.slippage)
                        should_exit = True
                        exit_reason = 'STOP_LOSS'

                # 2. Max Hold Time
                if not should_exit and self._check_max_hold_time(timestamp, entry_time):
                    if position == 'LONG':
                        exit_price = future_price * (1 - self.slippage)
                    else:
                        exit_price = future_price * (1 + self.slippage)
                    should_exit = True
                    exit_reason = 'MAX_TIME'

                # 3. Time Exit (normal 4h exit)
                if not should_exit:
                    if position == 'LONG':
                        exit_price = future_price * (1 - self.slippage)
                    else:
                        exit_price = future_price * (1 + self.slippage)
                    should_exit = True
                    exit_reason = 'TIME_EXIT'

                # Execute exit
                if should_exit:
                    # Calculate P&L
                    if position == 'LONG':
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price

                    # Apply fees
                    gross_pnl = entry_capital * pnl_pct
                    fees = entry_capital * self.trading_fee * 2  # Entry + Exit
                    net_pnl = gross_pnl - fees

                    # SAFETY: Cap P&L to prevent extreme losses
                    max_loss = -entry_capital * 0.95  # Max 95% loss per trade
                    max_gain = entry_capital * 5  # Max 500% gain per trade
                    net_pnl = np.clip(net_pnl, max_loss, max_gain)

                    # Update capital
                    capital += net_pnl

                    # SAFETY: Prevent negative capital
                    if capital < self.initial_capital * 0.01:  # Keep at least 1%
                        capital = self.initial_capital * 0.01

                    # Log trade
                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_usd': net_pnl,
                        'capital_after': capital,
                        'reason': exit_reason,
                        'confidence': self.equity_curve[-1].get('entry_confidence', 0),
                        'hold_hours': (timestamp - entry_time).total_seconds() / 3600
                    })

                    # Reset position
                    position = None
                    entry_time = None
                    entry_idx = None
                    highest_price = 0
                    lowest_price = float('inf')

            # OPEN NEW POSITION
            if position is None and capital > 0:
                # Check confidence threshold
                if conf < self.min_confidence:
                    self.skipped_trades['confidence'] += 1
                # Check daily trade limit
                elif not self._can_trade_today(timestamp):
                    self.skipped_trades['daily_limit'] += 1
                # Check sufficient capital
                elif capital < self.initial_capital * 0.05:  # Need at least 5% of initial
                    self.skipped_trades['capital_insufficient'] += 1
                # Open position (UP or DOWN only)
                elif pred in [0, 2]:
                    if pred == 2:  # UP - Go LONG
                        position = 'LONG'
                        entry_price = current_price * (1 + self.slippage)
                        highest_price = entry_price
                    elif pred == 0:  # DOWN - Go SHORT
                        position = 'SHORT'
                        entry_price = current_price * (1 - self.slippage)
                        lowest_price = entry_price

                    entry_time = timestamp
                    entry_idx = current_idx

                    # Calculate position size
                    entry_capital = min(
                        capital * self.position_size,
                        self.max_capital_per_trade,
                        capital * 0.95  # Never use more than 95% of capital
                    )

                    entry_fee = entry_capital * self.trading_fee
                    capital -= entry_fee
                    position_size_btc = entry_capital / entry_price

                    # Increment daily counter
                    self._increment_daily_trade(timestamp)

            # Update equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': capital,
                'position': position,
                'entry_confidence': conf if position else None
            })

        # Close any remaining position
        if position is not None:
            final_price = df['close'].iloc[-1]

            if position == 'LONG':
                exit_price = final_price * (1 - self.slippage)
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                exit_price = final_price * (1 + self.slippage)
                pnl_pct = (entry_price - exit_price) / entry_price

            gross_pnl = entry_capital * pnl_pct
            fees = entry_capital * self.trading_fee
            net_pnl = gross_pnl - fees

            # Cap P&L
            net_pnl = np.clip(net_pnl, -entry_capital * 0.95, entry_capital * 5)

            capital += net_pnl
            if capital < self.initial_capital * 0.01:
                capital = self.initial_capital * 0.01

            self.trades.append({
                'entry_time': entry_time,
                'exit_time': df['timestamp'].iloc[-1],
                'position': position,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct * 100,
                'pnl_usd': net_pnl,
                'capital_after': capital,
                'reason': 'FINAL_CLOSE',
                'confidence': self.equity_curve[-1].get('entry_confidence', 0),
                'hold_hours': (df['timestamp'].iloc[-1] - entry_time).total_seconds() / 3600
            })

        print(f"Completed {len(self.trades):,} trades\n")

        return self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if len(self.trades) == 0:
            return None

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        # Basic metrics
        final_capital = equity_df['equity'].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100

        # Time metrics
        total_days = (equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days
        total_years = total_days / 365.25
        annualized_return = ((final_capital / self.initial_capital) ** (1/total_years) - 1) * 100 if total_years > 0 else 0
        avg_trades_per_day = len(trades_df) / total_days if total_days > 0 else 0

        # Win rate
        winning_trades = trades_df[trades_df['pnl_usd'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0

        # Average win/loss
        avg_win = winning_trades['pnl_usd'].mean() if len(winning_trades) > 0 else 0
        losing_trades = trades_df[trades_df['pnl_usd'] <= 0]
        avg_loss = losing_trades['pnl_usd'].mean() if len(losing_trades) > 0 else 0

        # Drawdown
        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
        else:
            sharpe_ratio = 0

        # Profit factor
        total_wins = winning_trades['pnl_usd'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl_usd'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Best/worst trades
        best_trade = trades_df['pnl_usd'].max()
        worst_trade = trades_df['pnl_usd'].min()

        # Average hold time
        avg_hold_hours = trades_df['hold_hours'].mean() if 'hold_hours' in trades_df.columns else 0

        # Long vs Short
        long_trades = trades_df[trades_df['position'] == 'LONG']
        short_trades = trades_df[trades_df['position'] == 'SHORT']

        long_win_rate = len(long_trades[long_trades['pnl_usd'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0
        short_win_rate = len(short_trades[short_trades['pnl_usd'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0

        long_pnl = long_trades['pnl_usd'].sum() if len(long_trades) > 0 else 0
        short_pnl = short_trades['pnl_usd'].sum() if len(short_trades) > 0 else 0

        # Exit reasons breakdown
        exit_reasons = trades_df['reason'].value_counts()

        return {
            'final_capital': final_capital,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'total_trades': len(trades_df),
            'total_days': total_days,
            'avg_trades_per_day': avg_trades_per_day,
            'avg_hold_hours': avg_hold_hours,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'exit_reasons': exit_reasons,
            'skipped_trades': self.skipped_trades,
            'trades_df': trades_df,
            'equity_df': equity_df
        }

    def print_summary(self, metrics):
        """Print detailed performance summary"""
        if metrics is None:
            print("❌ No results to display")
            return

        print(f"\n{'='*80}")
        print(f"📊 OPTIMIZED BACKTEST RESULTS")
        print(f"{'='*80}")

        print(f"\n💰 CAPITAL & RETURNS:")
        print(f"   Initial Capital:    ${self.initial_capital:>12,.2f}")
        print(f"   Final Capital:      ${metrics['final_capital']:>12,.2f}")
        print(f"   Total Return:       {metrics['total_return']:>12.2f}%")
        print(f"   Annualized Return:  {metrics['annualized_return']:>12.2f}%")
        print(f"   Period:             {metrics['total_days']:>12,} days ({metrics['total_days']/365.25:.1f} years)")

        # Performance rating
        if metrics['total_return'] > 200:
            rating = "🔥 EXCEPTIONAL"
        elif metrics['total_return'] > 100:
            rating = "✅ EXCELLENT"
        elif metrics['total_return'] > 50:
            rating = "👍 GOOD"
        elif metrics['total_return'] > 0:
            rating = "⚠️  MODEST"
        else:
            rating = "❌ LOSS"
        print(f"   Performance:        {rating}")

        # Realistic expectation
        expected_live = metrics['annualized_return'] * 0.4  # 40% of backtest
        print(f"   Expected Live (~):  {expected_live:>12.1f}% per year ⚠️")

        print(f"\n📈 RISK METRICS:")
        print(f"   Sharpe Ratio:       {metrics['sharpe_ratio']:>12.2f}")
        print(f"   Max Drawdown:       {metrics['max_drawdown']:>12.2f}%")
        print(f"   Profit Factor:      {metrics['profit_factor']:>12.2f}")

        # Risk assessment
        if abs(metrics['max_drawdown']) < 20:
            risk_level = "✅ LOW RISK"
        elif abs(metrics['max_drawdown']) < 40:
            risk_level = "⚠️  MODERATE RISK"
        else:
            risk_level = "❌ HIGH RISK"
        print(f"   Risk Level:         {risk_level}")

        print(f"\n🎯 TRADING STATISTICS:")
        print(f"   Total Trades:       {metrics['total_trades']:>12,}")
        print(f"   Trades/Day (avg):   {metrics['avg_trades_per_day']:>12.1f}")
        print(f"   Avg Hold Time:      {metrics['avg_hold_hours']:>12.1f} hours")
        print(f"   Win Rate:           {metrics['win_rate']:>12.1f}%")
        print(f"   Average Win:        ${metrics['avg_win']:>11,.2f}")
        print(f"   Average Loss:       ${metrics['avg_loss']:>11,.2f}")
        print(f"   Best Trade:         ${metrics['best_trade']:>11,.2f}")
        print(f"   Worst Trade:        ${metrics['worst_trade']:>11,.2f}")

        print(f"\n🚫 SKIPPED TRADES:")
        print(f"   Low Confidence:     {metrics['skipped_trades']['confidence']:>12,}")
        print(f"   Daily Limit:        {metrics['skipped_trades']['daily_limit']:>12,}")
        print(f"   Insufficient Cap:   {metrics['skipped_trades']['capital_insufficient']:>12,}")

        print(f"\n📊 POSITION BREAKDOWN:")
        print(f"   Long Trades:        {metrics['long_trades']:>12,} ({metrics['long_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Long Win Rate:      {metrics['long_win_rate']:>12.1f}%")
        print(f"   Long P&L:           ${metrics['long_pnl']:>11,.2f}")
        print(f"   Short Trades:       {metrics['short_trades']:>12,} ({metrics['short_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Short Win Rate:     {metrics['short_win_rate']:>12.1f}%")
        print(f"   Short P&L:          ${metrics['short_pnl']:>11,.2f}")

        print(f"\n🚪 EXIT REASONS:")
        for reason, count in metrics['exit_reasons'].items():
            pct = count / metrics['total_trades'] * 100
            print(f"   {reason:15s}     {count:>12,} ({pct:.1f}%)")

        print(f"\n💡 STRATEGY ASSESSMENT:")

        # Win rate assessment
        if metrics['win_rate'] >= 55:
            print(f"   ✅ Excellent win rate ({metrics['win_rate']:.1f}%)")
        elif metrics['win_rate'] >= 45:
            print(f"   👍 Good win rate ({metrics['win_rate']:.1f}%)")
        else:
            print(f"   ⚠️  Low win rate ({metrics['win_rate']:.1f}%) - needs improvement")

        # Profit factor assessment
        if metrics['profit_factor'] >= 2.0:
            print(f"   ✅ Excellent profit factor ({metrics['profit_factor']:.2f})")
        elif metrics['profit_factor'] >= 1.5:
            print(f"   👍 Good profit factor ({metrics['profit_factor']:.2f})")
        elif metrics['profit_factor'] >= 1.0:
            print(f"   ⚠️  Marginal profit factor ({metrics['profit_factor']:.2f})")
        else:
            print(f"   ❌ Poor profit factor ({metrics['profit_factor']:.2f}) - unprofitable")

        # Position bias assessment
        long_bias = metrics['long_trades'] / metrics['total_trades'] * 100
        if 40 <= long_bias <= 60:
            print(f"   ✅ Balanced Long/Short ratio")
        else:
            bias_type = "LONG" if long_bias > 60 else "SHORT"
            print(f"   ⚠️  Biased towards {bias_type} ({long_bias:.1f}% / {100-long_bias:.1f}%)")

        # Performance comparison
        if metrics['long_pnl'] > 0 and metrics['short_pnl'] > 0:
            print(f"   ✅ Both Long and Short profitable")
        elif metrics['long_pnl'] > 0:
            print(f"   ⚠️  Only Long trades profitable")
        elif metrics['short_pnl'] > 0:
            print(f"   ⚠️  Only Short trades profitable")
        else:
            print(f"   ❌ Both Long and Short unprofitable")

        print(f"\n⚙️  OPTIMIZED PARAMETERS:")
        print(f"   Trading Fee:        {self.trading_fee*100:>12.2f}%")
        print(f"   Slippage:           {self.slippage*100:>12.3f}%")
        print(f"   Position Size:      {self.position_size*100:>12.1f}%")
        print(f"   Stop Loss:          {self.stop_loss*100:>12.1f}% ⚡")
        print(f"   Min Confidence:     {self.min_confidence*100:>12.1f}%")
        print(f"   Max Capital/Trade:  ${self.max_capital_per_trade:>11,.2f}")
        print(f"   Max Trades/Day:     {self.max_trades_per_day:>12}")
        print(f"   Min Hold Time:      {self.min_hold_hours:>12.1f}h")
        print(f"   Max Hold Time:      {self.max_position_hours:>12.1f}h")

        print(f"\n{'='*80}\n")

    def plot_results(self, metrics, save_path='backtest_optimized.png'):
        """Create comprehensive visualization"""
        if metrics is None:
            print("❌ No metrics to plot")
            return

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.3)

        trades_df = metrics['trades_df']
        equity_df = metrics['equity_df']

        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity_df['timestamp'], equity_df['equity'], linewidth=2, color='#2E86AB', label='Equity')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.fill_between(equity_df['timestamp'], self.initial_capital, equity_df['equity'],
                         where=(equity_df['equity'] >= self.initial_capital),
                         interpolate=True, alpha=0.3, color='green')
        ax1.fill_between(equity_df['timestamp'], self.initial_capital, equity_df['equity'],
                         where=(equity_df['equity'] < self.initial_capital),
                         interpolate=True, alpha=0.3, color='red')
        ax1.set_title('Equity Curve (Optimized)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax * 100
        ax2.fill_between(equity_df['timestamp'], 0, drawdown, color='red', alpha=0.3)
        ax2.plot(equity_df['timestamp'], drawdown, color='darkred', linewidth=1)
        ax2.set_title(f'Drawdown (Max: {drawdown.min():.2f}%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Win/Loss Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        trade_results = ['Win' if pnl > 0 else 'Loss' for pnl in trades_df['pnl_usd']]
        trade_counts = pd.Series(trade_results).value_counts()
        colors = ['#06D6A0' if x == 'Win' else '#EF476F' for x in trade_counts.index]
        ax3.bar(trade_counts.index, trade_counts.values, color=colors, alpha=0.7)
        ax3.set_title(f'Win/Loss (WR: {metrics["win_rate"]:.1f}%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Trades')
        for i, (idx, val) in enumerate(trade_counts.items()):
            ax3.text(i, val, f'{val}\n({val/len(trades_df)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')

        # 4. Long vs Short
        ax4 = fig.add_subplot(gs[2, 1])
        position_counts = trades_df['position'].value_counts()
        colors_pos = ['#118AB2', '#FF6B35']
        bars = ax4.bar(position_counts.index, position_counts.values, color=colors_pos, alpha=0.7)
        ax4.set_title('Long vs Short', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Trades')
        for i, (idx, val) in enumerate(position_counts.items()):
            wr = metrics['long_win_rate'] if idx == 'LONG' else metrics['short_win_rate']
            ax4.text(i, val, f'{val}\nWR: {wr:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 5. P&L Distribution
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.hist(trades_df['pnl_usd'], bins=40, color='#073B4C', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax5.set_title('P&L Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('P&L ($)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Cumulative Returns
        ax6 = fig.add_subplot(gs[3, 0])
        cumulative_returns = trades_df['pnl_usd'].cumsum()
        ax6.plot(range(len(cumulative_returns)), cumulative_returns, color='#06D6A0', linewidth=2)
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax6.fill_between(range(len(cumulative_returns)), 0, cumulative_returns,
                         where=(cumulative_returns >= 0), interpolate=True, alpha=0.3, color='green')
        ax6.fill_between(range(len(cumulative_returns)), 0, cumulative_returns,
                         where=(cumulative_returns < 0), interpolate=True, alpha=0.3, color='red')
        ax6.set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Trade #')
        ax6.set_ylabel('Cumulative P&L ($)')
        ax6.grid(True, alpha=0.3)

        # 7. Monthly Returns
        ax7 = fig.add_subplot(gs[3, 1])
        trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
        monthly_pnl = trades_df.groupby('month')['pnl_usd'].sum()
        colors_monthly = ['#06D6A0' if x > 0 else '#EF476F' for x in monthly_pnl.values]
        ax7.bar(range(len(monthly_pnl)), monthly_pnl.values, color=colors_monthly, alpha=0.7)
        ax7.set_title('Monthly Returns', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Month')
        ax7.set_ylabel('P&L ($)')
        ax7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax7.set_xticks(range(0, len(monthly_pnl), max(1, len(monthly_pnl)//12)))
        ax7.set_xticklabels([str(m) for m in monthly_pnl.index[::max(1, len(monthly_pnl)//12)]],
                           rotation=45, ha='right')
        ax7.grid(True, alpha=0.3, axis='y')

        # 8. Hold Time Distribution
        ax8 = fig.add_subplot(gs[3, 2])
        ax8.hist(trades_df['hold_hours'], bins=30, color='#118AB2', alpha=0.7, edgecolor='black')
        ax8.axvline(x=self.min_hold_hours, color='red', linestyle='--',
                   linewidth=2, label=f'Min: {self.min_hold_hours}h')
        ax8.axvline(x=self.max_position_hours, color='orange', linestyle='--',
                   linewidth=2, label=f'Max: {self.max_position_hours}h')
        ax8.set_title(f'Hold Time (Avg: {metrics["avg_hold_hours"]:.1f}h)', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Hours')
        ax8.set_ylabel('Frequency')
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')

        # 9. Exit Reasons
        ax9 = fig.add_subplot(gs[4, 0])
        exit_reasons = metrics['exit_reasons']
        colors_exit = ['#FF6B35', '#06D6A0', '#118AB2', '#FFD23F']
        ax9.pie(exit_reasons.values, labels=exit_reasons.index, autopct='%1.1f%%',
               colors=colors_exit[:len(exit_reasons)], startangle=90)
        ax9.set_title('Exit Reasons', fontsize=12, fontweight='bold')

        # 10. Long vs Short P&L
        ax10 = fig.add_subplot(gs[4, 1])
        position_pnl = [metrics['long_pnl'], metrics['short_pnl']]
        position_labels = ['LONG', 'SHORT']
        colors_pnl = ['#118AB2', '#FF6B35']
        bars = ax10.bar(position_labels, position_pnl, color=colors_pnl, alpha=0.7)
        ax10.set_title('P&L by Position Type', fontsize=12, fontweight='bold')
        ax10.set_ylabel('Total P&L ($)')
        ax10.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        for i, (label, pnl) in enumerate(zip(position_labels, position_pnl)):
            ax10.text(i, pnl, f'${pnl:,.0f}', ha='center',
                     va='bottom' if pnl > 0 else 'top', fontweight='bold')

        # 11. Daily Trade Frequency
        ax11 = fig.add_subplot(gs[4, 2])
        trades_df['date'] = pd.to_datetime(trades_df['exit_time']).dt.date
        daily_count = trades_df.groupby('date').size()
        ax11.hist(daily_count, bins=min(30, int(daily_count.max())),
                 color='#FF6B35', alpha=0.7, edgecolor='black')
        ax11.axvline(x=self.max_trades_per_day, color='red', linestyle='--',
                    linewidth=2, label=f'Limit: {self.max_trades_per_day}')
        ax11.set_title(f'Daily Frequency (Avg: {metrics["avg_trades_per_day"]:.1f})',
                      fontsize=12, fontweight='bold')
        ax11.set_xlabel('Trades/Day')
        ax11.set_ylabel('Frequency')
        ax11.legend()
        ax11.grid(True, alpha=0.3, axis='y')

        # 12. Performance Metrics Table
        ax12 = fig.add_subplot(gs[5, :])
        ax12.axis('off')

        metrics_text = f"""
        OPTIMIZED BACKTEST PERFORMANCE
        {'='*70}
        RETURNS                          RISK METRICS
        Total Return:      {metrics['total_return']:>8.2f}%    Sharpe Ratio:       {metrics['sharpe_ratio']:>8.2f}
        Annualized:        {metrics['annualized_return']:>8.2f}%    Max Drawdown:       {metrics['max_drawdown']:>8.2f}%
        Expected Live:     {metrics['annualized_return']*0.4:>8.1f}%    Profit Factor:      {metrics['profit_factor']:>8.2f}

        TRADING STATISTICS               POSITION BREAKDOWN
        Total Trades:      {metrics['total_trades']:>8,}      Long Trades:        {metrics['long_trades']:>8,} ({metrics['long_trades']/metrics['total_trades']*100:.0f}%)
        Trades/Day:        {metrics['avg_trades_per_day']:>8.1f}      Long Win Rate:      {metrics['long_win_rate']:>8.1f}%
        Avg Hold:          {metrics['avg_hold_hours']:>8.1f}h     Long P&L:           ${metrics['long_pnl']:>7,.0f}
        Win Rate:          {metrics['win_rate']:>8.1f}%    Short Trades:       {metrics['short_trades']:>8,} ({metrics['short_trades']/metrics['total_trades']*100:.0f}%)
        Avg Win:           ${metrics['avg_win']:>7,.0f}    Short Win Rate:     {metrics['short_win_rate']:>8.1f}%
        Avg Loss:          ${metrics['avg_loss']:>7,.0f}    Short P&L:          ${metrics['short_pnl']:>7,.0f}
        Best Trade:        ${metrics['best_trade']:>7,.0f}
        Worst Trade:       ${metrics['worst_trade']:>7,.0f}    SKIPPED TRADES
                                         Low Confidence:     {metrics['skipped_trades']['confidence']:>8,}
        PARAMETERS USED                  Daily Limit:        {metrics['skipped_trades']['daily_limit']:>8,}
        Stop Loss:         {self.stop_loss*100:>8.1f}%    Insufficient Cap:   {metrics['skipped_trades']['capital_insufficient']:>8,}
        Position Size:     {self.position_size*100:>8.1f}%
        Min Confidence:    {self.min_confidence*100:>8.1f}%
        Max Trades/Day:    {self.max_trades_per_day:>8}
        Trading Fee:       {self.trading_fee*100:>8.2f}%
        Slippage:          {self.slippage*100:>8.3f}%
        """

        ax12.text(0.05, 0.95, metrics_text, transform=ax12.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('OPTIMIZED Backtest Results - BTC LSTM Prediction Model',
                    fontsize=16, fontweight='bold', y=0.998)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved: {save_path}\n")
        plt.close()

    def export_trades(self, filename='trades_optimized.csv'):
        """Export trades to CSV"""
        if len(self.trades) == 0:
            print("❌ No trades to export")
            return
        pd.DataFrame(self.trades).to_csv(filename, index=False)
        print(f"💾 Trades exported: {filename}")

    def export_predictions(self, filename='predictions_optimized.csv'):
        """Export predictions to CSV"""
        if len(self.predictions_log) == 0:
            print("❌ No predictions to export")
            return
        pd.DataFrame(self.predictions_log).to_csv(filename, index=False)
        print(f"💾 Predictions exported: {filename}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='OPTIMIZED BTC Backtester')

    # Paths
    parser.add_argument('--model', default='models/lstm_full.pth')
    parser.add_argument('--scaler', default='preprocessed_data_lstm_1h/scalers.pkl')
    parser.add_argument('--features', default='preprocessed_data_lstm_1h/feature_cols.pkl')
    parser.add_argument('--sequence', default='preprocessed_data_lstm_1h/sequence_length.pkl')
    parser.add_argument('--db', default='data/btc_ohlcv.db')

    # Trading params (OPTIMIZED DEFAULTS)
    parser.add_argument('--capital', type=float, default=10000)
    parser.add_argument('--fee', type=float, default=0.001, help='0.1%')
    parser.add_argument('--slippage', type=float, default=0.0003, help='0.03%')
    parser.add_argument('--position-size', type=float, default=0.50, help='50%')
    parser.add_argument('--stop-loss', type=float, default=0.06, help='6% - OPTIMIZED')
    parser.add_argument('--min-confidence', type=float, default=0.50, help='50%')
    parser.add_argument('--max-capital-per-trade', type=float, default=None)
    parser.add_argument('--max-trades-per-day', type=int, default=10)
    parser.add_argument('--min-hold-hours', type=float, default=4)
    parser.add_argument('--max-position-hours', type=float, default=24, help='Max hold: 24h')
    parser.add_argument('--trailing-stop', action='store_true', help='Use trailing stop')

    # Date range
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)

    # Output
    parser.add_argument('--output', default='backtest_optimized.png')
    parser.add_argument('--export-trades', action='store_true')
    parser.add_argument('--export-predictions', action='store_true')

    args = parser.parse_args()

    # Validate files
    required = {
        'Model': args.model,
        'Scaler': args.scaler,
        'Features': args.features,
        'Sequence': args.sequence,
        'Database': args.db
    }

    missing = []
    for name, path in required.items():
        full_path = ROOT_DIR / path if not Path(path).is_absolute() else Path(path)
        if not full_path.exists():
            missing.append(f"{name}: {full_path}")

    if missing:
        print(f"\n{'='*80}")
        print("❌ MISSING FILES")
        print(f"{'='*80}")
        for m in missing:
            print(f"   {m}")
        print(f"{'='*80}\n")
        return

    try:
        # Initialize
        backtester = BTCBacktesterOptimized(
            model_path=args.model,
            scaler_path=args.scaler,
            feature_cols_path=args.features,
            sequence_length_path=args.sequence,
            db_path=args.db,
            initial_capital=args.capital,
            trading_fee=args.fee,
            position_size=args.position_size,
            stop_loss=args.stop_loss,
            min_confidence=args.min_confidence,
            max_capital_per_trade=args.max_capital_per_trade,
            max_trades_per_day=args.max_trades_per_day,
            slippage=args.slippage,
            min_hold_hours=args.min_hold_hours,
            max_position_hours=args.max_position_hours,
            use_trailing_stop=args.trailing_stop
        )

        # Fetch data
        df = backtester.fetch_backtest_data(
            start_date=args.start_date,
            end_date=args.end_date
        )

        if len(df) < backtester.sequence_length + 100:
            print(f"\n❌ Insufficient data: need {backtester.sequence_length + 100}, got {len(df)}")
            return

        # Run backtest
        metrics = backtester.run_backtest(df)

        if metrics is None:
            print(f"\n❌ Backtest failed - no trades")
            return

        # Print summary
        backtester.print_summary(metrics)

        # Plot
        backtester.plot_results(metrics, save_path=args.output)

        # Export
        if args.export_trades:
            backtester.export_trades('trades_optimized.csv')

        if args.export_predictions:
            backtester.export_predictions('predictions_optimized.csv')

        print(f"\n{'='*80}")
        print(f"✅ BACKTEST COMPLETED")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR")
        print(f"{'='*80}")
        print(f"{str(e)}\n")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
