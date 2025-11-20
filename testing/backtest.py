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


class BTCBacktester:
    """
    Enhanced Realistic Backtesting System for BTC Prediction Model

    Key Features:
    - Trade frequency limiting (max trades per day)
    - Realistic slippage modeling
    - Proper position sizing with capital cap
    - Multiple timeframe testing support
    """

    def __init__(self, model_path, scaler_path, feature_cols_path,
                 sequence_length_path, db_path='data/btc_ohlcv.db',
                 initial_capital=10000, trading_fee=0.001, position_size=0.95,
                 stop_loss=0.05, min_confidence=0.45, max_capital_per_trade=None,
                 max_trades_per_day=10, slippage=0.0005, min_hold_hours=4):
        """
        Initialize backtester with realistic trading constraints

        Args:
            model_path: Path to trained model
            scaler_path: Path to scaler
            feature_cols_path: Path to feature columns
            sequence_length_path: Path to sequence length
            db_path: Path to database
            initial_capital: Starting capital in USD
            trading_fee: Trading fee percentage (0.001 = 0.1%)
            position_size: Fraction of capital to use per trade (0.95 = 95%)
            stop_loss: Stop loss percentage (0.05 = 5%)
            min_confidence: Minimum confidence to take a trade (0.45 = 45%)
            max_capital_per_trade: Maximum capital per trade (None = unlimited)
            max_trades_per_day: Maximum number of trades per day (default: 10)
            slippage: Slippage percentage per trade (0.0005 = 0.05%)
            min_hold_hours: Minimum hours to hold a position (default: 4)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Paths
        self.model_path = ROOT_DIR / model_path if not Path(model_path).is_absolute() else Path(model_path)
        self.scaler_path = ROOT_DIR / scaler_path if not Path(scaler_path).is_absolute() else Path(scaler_path)
        self.feature_cols_path = ROOT_DIR / feature_cols_path if not Path(feature_cols_path).is_absolute() else Path(feature_cols_path)
        self.sequence_length_path = ROOT_DIR / sequence_length_path if not Path(sequence_length_path).is_absolute() else Path(sequence_length_path)
        self.db_path = ROOT_DIR / db_path if not Path(db_path).is_absolute() else Path(db_path)

        # Trading parameters
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.min_confidence = min_confidence
        self.max_capital_per_trade = max_capital_per_trade or (initial_capital * 10)
        self.max_trades_per_day = max_trades_per_day
        self.slippage = slippage
        self.min_hold_hours = min_hold_hours

        # Results storage
        self.trades = []
        self.equity_curve = []
        self.predictions_log = []
        self.daily_trades_count = {}
        self.skipped_trades = {'confidence': 0, 'daily_limit': 0, 'hold_time': 0}

        print(f"\n{'='*80}")
        print(f"🔧 Initializing Enhanced BTC Backtester")
        print(f"{'='*80}")
        print(f"Device: {self.device}")

        # Load model
        self._load_model()

        # Load preprocessing artifacts
        self._load_artifacts()

        print(f"\n💰 Trading Parameters:")
        print(f"   Initial Capital:     ${self.initial_capital:,.2f}")
        print(f"   Trading Fee:         {self.trading_fee*100:.2f}%")
        print(f"   Slippage:            {self.slippage*100:.3f}%")
        print(f"   Position Size:       {self.position_size*100:.1f}%")
        print(f"   Stop Loss:           {self.stop_loss*100:.1f}%")
        print(f"   Min Confidence:      {self.min_confidence*100:.1f}%")
        print(f"   Max Capital/Trade:   ${self.max_capital_per_trade:,.2f}")
        print(f"   Max Trades/Day:      {self.max_trades_per_day}")
        print(f"   Min Hold Time:       {self.min_hold_hours} hours")
        print(f"{'='*80}")

    def _load_model(self):
        """Load trained model"""
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

    def _load_artifacts(self):
        """Load preprocessing artifacts"""
        print(f"\n📦 Loading preprocessing artifacts...")
        scaler_data = joblib.load(str(self.scaler_path))
        self.scaler = scaler_data['robust'] if isinstance(scaler_data, dict) else scaler_data
        self.feature_cols = joblib.load(str(self.feature_cols_path))
        self.sequence_length = joblib.load(str(self.sequence_length_path))
        print(f"   ✓ Scaler loaded")
        print(f"   ✓ Features: {len(self.feature_cols)}")
        print(f"   ✓ Sequence length: {self.sequence_length}")

    def fetch_backtest_data(self, start_date=None, end_date=None):
        """Fetch historical data for backtesting"""
        print(f"\n{'='*80}")
        print(f"📥 Fetching Backtest Data")
        print(f"{'='*80}")

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

        # Handle NaN
        nan_counts = df.isna().sum().sum()
        if nan_counts > 0:
            df = df.ffill(limit=3)
            df = df.bfill(limit=3)
            df = df.fillna(0)
            print(f"   ✓ Cleaned {nan_counts:,} NaN values")

        print(f"   ✓ Loaded: {len(df)} candles")
        print(f"   ✓ Period: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        print(f"   ✓ Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
        print(f"{'='*80}")

        return df

    def create_sequences(self, df):
        """Create sequences for prediction"""
        sequences = []
        timestamps = []
        prices = []

        for i in range(len(df) - self.sequence_length - 4):
            seq_data = df.iloc[i:i+self.sequence_length]

            missing = [col for col in self.feature_cols if col not in seq_data.columns]
            if missing:
                continue

            X = seq_data[self.feature_cols].values
            X_scaled = self.scaler.transform(X)
            X_scaled = np.clip(X_scaled, -10, 10)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            sequences.append(X_scaled)
            timestamps.append(df['timestamp'].iloc[i+self.sequence_length])
            prices.append(df['close'].iloc[i+self.sequence_length])

        return np.array(sequences), timestamps, prices

    def predict_batch(self, sequences, batch_size=32):
        """Make predictions for batch of sequences"""
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
                print(f"   Progress: {progress:.1f}% ({batch_end:,}/{num_sequences:,})", end='\r')

        print(f"   Progress: 100.0% ({num_sequences:,}/{num_sequences:,})")

        return np.array(all_predictions), np.array(all_confidences), np.array(all_probs)

    def _can_trade_today(self, timestamp):
        """Check if we can open new trade today"""
        date_key = timestamp.date()
        trades_today = self.daily_trades_count.get(date_key, 0)
        return trades_today < self.max_trades_per_day

    def _increment_daily_trade(self, timestamp):
        """Increment daily trade counter"""
        date_key = timestamp.date()
        self.daily_trades_count[date_key] = self.daily_trades_count.get(date_key, 0) + 1

    def _check_min_hold_time(self, current_time, entry_time):
        """Check if minimum hold time has passed"""
        time_diff = (current_time - entry_time).total_seconds() / 3600
        return time_diff >= self.min_hold_hours

    def run_backtest(self, df):
        """Run backtesting simulation with realistic constraints"""
        print(f"\n{'='*80}")
        print(f"🚀 Running Enhanced Backtest")
        print(f"{'='*80}")

        # Create sequences
        print(f"\n📊 Creating sequences...")
        sequences, timestamps, entry_prices = self.create_sequences(df)
        print(f"   ✓ Created {len(sequences):,} sequences")

        # Make predictions
        print(f"\n🔮 Making predictions...")
        predictions, confidences, probs = self.predict_batch(sequences)
        print(f"   ✓ Completed {len(predictions):,} predictions")

        # Initialize trading
        capital = self.initial_capital
        position = None
        entry_price = 0
        entry_capital = 0
        entry_time = None
        position_size_btc = 0

        self.trades = []
        self.equity_curve = [{'timestamp': timestamps[0], 'equity': capital, 'position': None}]

        print(f"\n💰 Simulating trades with realistic constraints...")
        class_names = ['DOWN', 'HOLD', 'UP']

        for i in range(len(predictions)):
            pred = predictions[i]
            conf = confidences[i]
            current_price = entry_prices[i]
            timestamp = timestamps[i]

            # Get actual future price
            future_idx = df[df['timestamp'] == timestamp].index[0] + 4
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

            # Close existing position if criteria met
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

                # Apply slippage to exit price
                slippage_factor = 1 - self.slippage if position == 'LONG' else 1 + self.slippage

                # Check stop loss
                if position == 'LONG':
                    stop_price = entry_price * (1 - self.stop_loss)
                    if low_price <= stop_price:
                        exit_price = stop_price * slippage_factor
                        pnl_pct = (exit_price - entry_price) / entry_price
                        reason = 'STOP_LOSS'
                    else:
                        exit_price = future_price * slippage_factor
                        pnl_pct = (exit_price - entry_price) / entry_price
                        reason = 'TIME_EXIT'

                elif position == 'SHORT':
                    stop_price = entry_price * (1 + self.stop_loss)
                    if high_price >= stop_price:
                        exit_price = stop_price * slippage_factor
                        pnl_pct = (entry_price - exit_price) / entry_price
                        reason = 'STOP_LOSS'
                    else:
                        exit_price = future_price * slippage_factor
                        pnl_pct = (entry_price - exit_price) / entry_price
                        reason = 'TIME_EXIT'

                # Calculate P&L with fees
                gross_pnl = entry_capital * pnl_pct
                fees = entry_capital * self.trading_fee * 2
                net_pnl = gross_pnl - fees

                # Cap P&L
                max_pnl = entry_capital * 10
                min_pnl = -entry_capital
                net_pnl = np.clip(net_pnl, min_pnl, max_pnl)

                capital += net_pnl
                if capital < 0:
                    capital = 0

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
                    'reason': reason,
                    'confidence': self.equity_curve[-1].get('entry_confidence', 0),
                    'hold_hours': (timestamp - entry_time).total_seconds() / 3600
                })

                position = None
                entry_time = None

            # Open new position with constraints
            if capital > 0:
                # Check confidence threshold
                if conf < self.min_confidence:
                    self.skipped_trades['confidence'] += 1
                # Check daily trade limit
                elif not self._can_trade_today(timestamp):
                    self.skipped_trades['daily_limit'] += 1
                # Open position
                elif pred in [0, 2]:  # UP or DOWN
                    if pred == 2:  # UP - Go LONG
                        position = 'LONG'
                        # Apply slippage to entry
                        entry_price = current_price * (1 + self.slippage)
                    elif pred == 0:  # DOWN - Go SHORT
                        position = 'SHORT'
                        entry_price = current_price * (1 - self.slippage)

                    entry_time = timestamp
                    entry_capital = min(capital * self.position_size, self.max_capital_per_trade)
                    entry_fee = entry_capital * self.trading_fee
                    capital -= entry_fee
                    position_size_btc = entry_capital / entry_price

                    # Increment daily trade counter
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
            slippage_factor = 1 - self.slippage if position == 'LONG' else 1 + self.slippage
            exit_price = final_price * slippage_factor

            if position == 'LONG':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            gross_pnl = entry_capital * pnl_pct
            fees = entry_capital * self.trading_fee
            net_pnl = gross_pnl - fees
            net_pnl = np.clip(net_pnl, -entry_capital, entry_capital * 10)

            capital += net_pnl
            if capital < 0:
                capital = 0

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

        print(f"   ✓ Executed {len(self.trades):,} trades")
        print(f"   ✓ Skipped trades:")
        print(f"      - Low confidence:  {self.skipped_trades['confidence']:,}")
        print(f"      - Daily limit:     {self.skipped_trades['daily_limit']:,}")
        print(f"{'='*80}")

        return self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate performance metrics"""
        if len(self.trades) == 0:
            return None

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        # Basic metrics
        final_capital = equity_df['equity'].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100

        # Time-based metrics
        total_days = (equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days
        total_years = total_days / 365.25
        annualized_return = ((final_capital / self.initial_capital) ** (1/total_years) - 1) * 100 if total_years > 0 else 0
        avg_trades_per_day = len(trades_df) / total_days if total_days > 0 else 0

        # Win rate
        winning_trades = trades_df[trades_df['pnl_usd'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100

        # Average win/loss
        avg_win = winning_trades['pnl_usd'].mean() if len(winning_trades) > 0 else 0
        losing_trades = trades_df[trades_df['pnl_usd'] <= 0]
        avg_loss = losing_trades['pnl_usd'].mean() if len(losing_trades) > 0 else 0

        # Max drawdown
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

        # Best and worst trade
        best_trade = trades_df['pnl_usd'].max()
        worst_trade = trades_df['pnl_usd'].min()

        # Average hold time
        avg_hold_hours = trades_df['hold_hours'].mean() if 'hold_hours' in trades_df.columns else 0

        # Long vs Short performance
        long_trades = trades_df[trades_df['position'] == 'LONG']
        short_trades = trades_df[trades_df['position'] == 'SHORT']

        long_win_rate = len(long_trades[long_trades['pnl_usd'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0
        short_win_rate = len(short_trades[short_trades['pnl_usd'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0

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
            'skipped_trades': self.skipped_trades,
            'trades_df': trades_df,
            'equity_df': equity_df
        }

    def print_summary(self, metrics):
        """Print detailed summary with realistic assessment"""
        if metrics is None:
            print("❌ No results to display")
            return

        print(f"\n{'='*80}")
        print(f"📊 ENHANCED BACKTEST RESULTS SUMMARY")
        print(f"{'='*80}")

        print(f"\n💰 CAPITAL & RETURNS:")
        print(f"   Initial Capital:    ${self.initial_capital:>12,.2f}")
        print(f"   Final Capital:      ${metrics['final_capital']:>12,.2f}")
        print(f"   Total Return:       {metrics['total_return']:>12.2f}%")
        print(f"   Annualized Return:  {metrics['annualized_return']:>12.2f}%")
        print(f"   Period:             {metrics['total_days']:>12,} days ({metrics['total_days']/365.25:.1f} years)")

        # Realistic performance assessment
        if metrics['annualized_return'] > 100:
            print(f"   Backtest Rating:    🔥 EXCEPTIONAL (unrealistic for live)")
        elif metrics['annualized_return'] > 50:
            print(f"   Backtest Rating:    ✅ EXCELLENT")
        elif metrics['annualized_return'] > 20:
            print(f"   Backtest Rating:    👍 GOOD")
        elif metrics['annualized_return'] > 0:
            print(f"   Backtest Rating:    ⚠️  MODEST")
        else:
            print(f"   Backtest Rating:    ❌ LOSS")

        # Realistic expectation
        expected_live = metrics['annualized_return'] * 0.3  # Conservative: 30% of backtest
        print(f"   Expected Live (~):  {expected_live:>12.1f}% per year ⚠️")

        print(f"\n📈 RISK METRICS:")
        print(f"   Sharpe Ratio:       {metrics['sharpe_ratio']:>12.2f}")
        print(f"   Max Drawdown:       {metrics['max_drawdown']:>12.2f}%")
        print(f"   Profit Factor:      {metrics['profit_factor']:>12.2f}")

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

        print(f"\n📊 POSITION BREAKDOWN:")
        print(f"   Long Trades:        {metrics['long_trades']:>12,} ({metrics['long_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Long Win Rate:      {metrics['long_win_rate']:>12.1f}%")
        print(f"   Short Trades:       {metrics['short_trades']:>12,} ({metrics['short_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Short Win Rate:     {metrics['short_win_rate']:>12.1f}%")

        # Strategy assessment
        print(f"\n💡 STRATEGY ASSESSMENT:")
        if metrics['win_rate'] > 55:
            print(f"   ✅ Strong predictive accuracy")
        elif metrics['win_rate'] > 45:
            print(f"   👍 Decent predictive accuracy")
        else:
            print(f"   ⚠️  Low predictive accuracy - consider tuning")

        if metrics['profit_factor'] > 1.5:
            print(f"   ✅ Excellent profit factor")
        elif metrics['profit_factor'] > 1.0:
            print(f"   👍 Profitable strategy")
        else:
            print(f"   ❌ Unprofitable - losses exceed wins")

        if abs(metrics['max_drawdown']) < 15:
            print(f"   ✅ Low drawdown - good risk management")
        elif abs(metrics['max_drawdown']) < 25:
            print(f"   ⚠️  Moderate drawdown")
        else:
            print(f"   ❌ High drawdown - risky strategy")

        # Trade frequency assessment
        if metrics['avg_trades_per_day'] > 20:
            print(f"   ⚠️  Very high trade frequency - not realistic for manual trading")
        elif metrics['avg_trades_per_day'] > 10:
            print(f"   ⚠️  High trade frequency - consider reducing")
        elif metrics['avg_trades_per_day'] > 5:
            print(f"   👍 Moderate trade frequency - manageable")
        else:
            print(f"   ✅ Conservative trade frequency")

        print(f"\n⚙️  PARAMETERS USED:")
        print(f"   Trading Fee:        {self.trading_fee*100:>12.2f}%")
        print(f"   Slippage:           {self.slippage*100:>12.3f}%")
        print(f"   Position Size:      {self.position_size*100:>12.1f}%")
        print(f"   Stop Loss:          {self.stop_loss*100:>12.1f}%")
        print(f"   Min Confidence:     {self.min_confidence*100:>12.1f}%")
        print(f"   Max Capital/Trade:  ${self.max_capital_per_trade:>11,.2f}")
        print(f"   Max Trades/Day:     {self.max_trades_per_day:>12}")
        print(f"   Min Hold Time:      {self.min_hold_hours:>12.1f} hours")

        print(f"\n⚠️  REALITY CHECK:")
        print(f"   📌 Backtest results are OPTIMISTIC")
        print(f"   📌 Live trading will have:")
        print(f"      • Higher slippage during volatility")
        print(f"      • Execution delays")
        print(f"      • Psychological factors")
        print(f"      • Market regime changes")
        print(f"   📌 Expect 30-50% of backtest returns in live trading")
        print(f"   📌 Always start with paper trading!")

        print(f"\n{'='*80}\n")

    def plot_results(self, metrics, save_path='backtest_results.png'):
        """Create comprehensive visualization"""
        if metrics is None:
            print("❌ No metrics to plot")
            return

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)

        trades_df = metrics['trades_df']
        equity_df = metrics['equity_df']

        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity_df['timestamp'], equity_df['equity'], linewidth=2, color='#2E86AB')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.fill_between(equity_df['timestamp'], self.initial_capital, equity_df['equity'],
                         where=(equity_df['equity'] >= self.initial_capital),
                         interpolate=True, alpha=0.3, color='green', label='Profit')
        ax1.fill_between(equity_df['timestamp'], self.initial_capital, equity_df['equity'],
                         where=(equity_df['equity'] < self.initial_capital),
                         interpolate=True, alpha=0.3, color='red', label='Loss')
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
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
        ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Trade Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        trade_results = ['Win' if pnl > 0 else 'Loss' for pnl in trades_df['pnl_usd']]
        trade_counts = pd.Series(trade_results).value_counts()
        colors = ['#06D6A0' if x == 'Win' else '#EF476F' for x in trade_counts.index]
        ax3.bar(trade_counts.index, trade_counts.values, color=colors, alpha=0.7)
        ax3.set_title('Win/Loss Distribution', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Trades')
        for i, (idx, val) in enumerate(trade_counts.items()):
            ax3.text(i, val, f'{val}\n({val/len(trades_df)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')

        # 4. Position Type Distribution
        ax4 = fig.add_subplot(gs[2, 1])
        position_counts = trades_df['position'].value_counts()
        colors_pos = ['#118AB2', '#FF6B35']
        ax4.bar(position_counts.index, position_counts.values, color=colors_pos, alpha=0.7)
        ax4.set_title('Long vs Short Trades', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Number of Trades')
        for i, (idx, val) in enumerate(position_counts.items()):
            ax4.text(i, val, f'{val}', ha='center', va='bottom', fontweight='bold')

        # 5. P&L Distribution
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.hist(trades_df['pnl_usd'], bins=30, color='#073B4C', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax5.set_title('P&L Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('P&L ($)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Cumulative Returns
        ax6 = fig.add_subplot(gs[3, 0])
        cumulative_returns = trades_df['pnl_usd'].cumsum()
        ax6.plot(range(len(cumulative_returns)), cumulative_returns,
                color='#06D6A0', linewidth=2)
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax6.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Trade Number')
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
        if 'hold_hours' in trades_df.columns:
            ax8.hist(trades_df['hold_hours'], bins=30, color='#118AB2', alpha=0.7, edgecolor='black')
            ax8.axvline(x=self.min_hold_hours, color='red', linestyle='--',
                       linewidth=2, label=f'Min Hold: {self.min_hold_hours}h')
            ax8.set_title('Hold Time Distribution', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Hours')
            ax8.set_ylabel('Frequency')
            ax8.legend()
            ax8.grid(True, alpha=0.3, axis='y')

        # 9. Daily Trade Frequency
        ax9 = fig.add_subplot(gs[4, 0])
        trades_df['date'] = pd.to_datetime(trades_df['exit_time']).dt.date
        daily_trade_count = trades_df.groupby('date').size()
        ax9.hist(daily_trade_count, bins=min(30, daily_trade_count.max()),
                color='#FF6B35', alpha=0.7, edgecolor='black')
        ax9.axvline(x=self.max_trades_per_day, color='red', linestyle='--',
                   linewidth=2, label=f'Limit: {self.max_trades_per_day}')
        ax9.set_title('Daily Trade Frequency', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Trades per Day')
        ax9.set_ylabel('Frequency')
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')

        # 10. Performance Metrics Table
        ax10 = fig.add_subplot(gs[4, 1:])
        ax10.axis('off')

        metrics_text = f"""
        PERFORMANCE METRICS
        {'='*50}
        Total Return:          {metrics['total_return']:>10.2f}%
        Annualized Return:     {metrics['annualized_return']:>10.2f}%
        Expected Live (~30%):  {metrics['annualized_return']*0.3:>10.1f}%
        Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}
        Max Drawdown:          {metrics['max_drawdown']:>10.2f}%
        Profit Factor:         {metrics['profit_factor']:>10.2f}

        TRADING STATS
        {'='*50}
        Total Trades:          {metrics['total_trades']:>10,}
        Trades/Day (avg):      {metrics['avg_trades_per_day']:>10.1f}
        Avg Hold Time:         {metrics['avg_hold_hours']:>10.1f}h
        Win Rate:              {metrics['win_rate']:>10.1f}%
        Avg Win:               ${metrics['avg_win']:>9,.0f}
        Avg Loss:              ${metrics['avg_loss']:>9,.0f}
        Best Trade:            ${metrics['best_trade']:>9,.0f}
        Worst Trade:           ${metrics['worst_trade']:>9,.0f}

        POSITION BREAKDOWN
        {'='*50}
        Long Trades:           {metrics['long_trades']:>10,} ({metrics['long_trades']/metrics['total_trades']*100:.0f}%)
        Long Win Rate:         {metrics['long_win_rate']:>10.1f}%
        Short Trades:          {metrics['short_trades']:>10,} ({metrics['short_trades']/metrics['total_trades']*100:.0f}%)
        Short Win Rate:        {metrics['short_win_rate']:>10.1f}%

        SKIPPED TRADES
        {'='*50}
        Low Confidence:        {metrics['skipped_trades']['confidence']:>10,}
        Daily Limit:           {metrics['skipped_trades']['daily_limit']:>10,}
        """

        ax10.text(0.05, 0.95, metrics_text, transform=ax10.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('Enhanced Backtesting Results - BTC Prediction Model',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Results saved to: {save_path}")
        plt.close()

    def export_trades(self, filename='backtest_trades.csv'):
        """Export trades to CSV"""
        if len(self.trades) == 0:
            print("❌ No trades to export")
            return

        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(filename, index=False)
        print(f"💾 Trades exported to: {filename}")

    def export_predictions(self, filename='backtest_predictions.csv'):
        """Export predictions log to CSV"""
        if len(self.predictions_log) == 0:
            print("❌ No predictions to export")
            return

        pred_df = pd.DataFrame(self.predictions_log)
        pred_df.to_csv(filename, index=False)
        print(f"💾 Predictions exported to: {filename}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Backtest BTC Prediction Model')

    # Model paths
    parser.add_argument('--model', default='models/lstm_balanced_full.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--scaler', default='preprocessed_data_lstm_1h/scalers.pkl',
                       help='Path to scaler file')
    parser.add_argument('--features', default='preprocessed_data_lstm_1h/feature_cols.pkl',
                       help='Path to feature columns file')
    parser.add_argument('--sequence', default='preprocessed_data_lstm_1h/sequence_length.pkl',
                       help='Path to sequence length file')
    parser.add_argument('--db', default='data/btc_ohlcv.db',
                       help='Path to database')

    # Trading parameters
    parser.add_argument('--capital', type=float, default=10000,
                       help='Initial capital in USD (default: 10000)')
    parser.add_argument('--fee', type=float, default=0.002,
                       help='Trading fee percentage (default: 0.002 = 0.2%%)')
    parser.add_argument('--slippage', type=float, default=0.0005,
                       help='Slippage percentage (default: 0.0005 = 0.05%%)')
    parser.add_argument('--position-size', type=float, default=0.50,
                       help='Position size as fraction of capital (default: 0.50 = 50%%)')
    parser.add_argument('--stop-loss', type=float, default=0.03,
                       help='Stop loss percentage (default: 0.03 = 3%%)')
    parser.add_argument('--min-confidence', type=float, default=0.55,
                       help='Minimum confidence to trade (default: 0.55 = 55%%)')
    parser.add_argument('--max-capital-per-trade', type=float, default=None,
                       help='Maximum capital per trade (default: 10x initial capital)')
    parser.add_argument('--max-trades-per-day', type=int, default=10,
                       help='Maximum trades per day (default: 10)')
    parser.add_argument('--min-hold-hours', type=float, default=4,
                       help='Minimum hours to hold position (default: 4)')

    # Date range
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for backtest (YYYY-MM-DD)')

    # Output options
    parser.add_argument('--output', default='backtest_results_enhanced.png',
                       help='Output filename for chart')
    parser.add_argument('--export-trades', action='store_true',
                       help='Export trades to CSV')
    parser.add_argument('--export-predictions', action='store_true',
                       help='Export predictions to CSV')

    args = parser.parse_args()

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
        print(f"\n{'='*80}")
        print("❌ MISSING REQUIRED FILES")
        print(f"{'='*80}")
        for m in missing:
            print(f"   - {m}")
        print(f"\n💡 Make sure you have:")
        print(f"   1. Trained the model (run train.py)")
        print(f"   2. Preprocessed data exists")
        print(f"   3. Database is accessible")
        print(f"{'='*80}\n")
        return

    try:
        # Initialize backtester
        backtester = BTCBacktester(
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
            min_hold_hours=args.min_hold_hours
        )

        # Fetch data
        df = backtester.fetch_backtest_data(
            start_date=args.start_date,
            end_date=args.end_date
        )

        if len(df) < backtester.sequence_length + 100:
            print(f"\n❌ Insufficient data for backtesting")
            print(f"   Need at least {backtester.sequence_length + 100} candles")
            print(f"   Got: {len(df)} candles")
            return

        # Run backtest
        metrics = backtester.run_backtest(df)

        if metrics is None:
            print(f"\n❌ Backtesting failed - no trades executed")
            return

        # Print summary
        backtester.print_summary(metrics)

        # Plot results
        backtester.plot_results(metrics, save_path=args.output)

        # Export if requested
        if args.export_trades:
            backtester.export_trades('backtest_trades_enhanced.csv')

        if args.export_predictions:
            backtester.export_predictions('backtest_predictions_enhanced.csv')

        print(f"\n{'='*80}")
        print(f"✅ ENHANCED BACKTESTING COMPLETED")
        print(f"{'='*80}")
        print(f"\n💡 RECOMMENDED TESTING SCENARIOS:")
        print(f"\n   1️⃣  CONSERVATIVE (Recommended for beginners):")
        print(f"      python testing/backtest.py --min-confidence 0.65 \\")
        print(f"             --position-size 0.30 --max-trades-per-day 5 \\")
        print(f"             --stop-loss 0.02")
        print(f"\n   2️⃣  MODERATE (Balanced approach):")
        print(f"      python testing/backtest.py --min-confidence 0.55 \\")
        print(f"             --position-size 0.50 --max-trades-per-day 10")
        print(f"\n   3️⃣  AGGRESSIVE (Higher risk/reward):")
        print(f"      python testing/backtest.py --min-confidence 0.45 \\")
        print(f"             --position-size 0.70 --max-trades-per-day 15")
        print(f"\n   4️⃣  TEST DIFFERENT PERIODS:")
        print(f"      # Bear market (2022)")
        print(f"      python testing/backtest.py --start-date 2022-01-01 --end-date 2022-12-31")
        print(f"\n      # Bull market (2024)")
        print(f"      python testing/backtest.py --start-date 2024-01-01")
        print(f"\n   5️⃣  WALK-FORWARD (Most realistic):")
        print(f"      # Test only on recent unseen data")
        print(f"      python testing/backtest.py --start-date 2024-06-01")
        print(f"\n{'='*80}")
        print(f"\n⚠️  IMPORTANT REMINDERS:")
        print(f"   • Backtest returns are ALWAYS optimistic")
        print(f"   • Expect 30-50% of backtest performance in live trading")
        print(f"   • Start with paper trading for 3-6 months")
        print(f"   • Never risk more than you can afford to lose")
        print(f"   • Markets change - retest strategy regularly")
        print(f"\n{'='*80}\n")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR OCCURRED")
        print(f"{'='*80}")
        print(f"{str(e)}\n")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
