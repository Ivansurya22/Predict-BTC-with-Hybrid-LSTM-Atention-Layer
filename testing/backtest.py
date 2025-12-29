import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import sqlite3
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path.cwd()
MODELS_DIR = ROOT_DIR / 'models'
sys.path.insert(0, str(MODELS_DIR))

from models import MultiInputBidirectionalLSTMAttention


class BTCFuturesBacktester:
    """Futures Trading Backtester with Correct P&L Calculation"""

    def __init__(self, model_path, preprocessed_dir, db_path='data/btc_ohlcv.db',
                 initial_capital=10000, trading_fee=0.0005, position_size=0.50,
                 leverage=5, stop_loss=0.04, take_profit=0.08, min_confidence=0.50,
                 max_trades_per_day=10, slippage=0.0003, min_hold_hours=1,
                 max_position_hours=48, use_trailing_stop=False,
                 maintenance_margin=0.005, funding_rate=0.0001):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Paths
        self.model_path = ROOT_DIR / model_path if not Path(model_path).is_absolute() else Path(model_path)
        self.preprocessed_dir = ROOT_DIR / preprocessed_dir if not Path(preprocessed_dir).is_absolute() else Path(preprocessed_dir)
        self.db_path = ROOT_DIR / db_path if not Path(db_path).is_absolute() else Path(db_path)

        # Trading parameters
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.position_size = position_size
        self.leverage = leverage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.min_confidence = min_confidence
        self.max_trades_per_day = max_trades_per_day
        self.slippage = slippage
        self.min_hold_hours = min_hold_hours
        self.max_position_hours = max_position_hours
        self.use_trailing_stop = use_trailing_stop
        self.maintenance_margin = maintenance_margin
        self.funding_rate = funding_rate

        # Tracking
        self.liquidations = 0
        self.funding_paid = 0
        self.funding_received = 0
        self.trades = []
        self.equity_curve = []
        self.predictions_log = []
        self.daily_trades_count = {}
        self.skipped_trades = {'confidence': 0, 'daily_limit': 0, 'capital_insufficient': 0}

        print(f"\n{'='*80}")
        print(f"INITIALIZING FUTURES BACKTESTER")
        print(f"{'='*80}")

        self._load_model()
        self._load_artifacts()

    def _load_model(self):
        """Load trained model"""
        print(f"\nLoading model from: {self.model_path.name}")
        checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
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
        print(f"Model loaded: {total_params:,} parameters")
        print(f"Input branches: {self.input_sizes}")

    def _load_artifacts(self):
        """Load preprocessing artifacts"""
        print(f"\nLoading preprocessing artifacts...")
        self.scalers = joblib.load(str(self.preprocessed_dir / 'scalers.pkl'))
        self.sequence_length = joblib.load(str(self.preprocessed_dir / 'sequence_length.pkl'))
        metadata = joblib.load(str(self.preprocessed_dir / 'metadata.pkl'))
        print(f"Sequence length: {self.sequence_length}")
        print(f"Total features: {metadata['total_features']}")

    def calculate_liquidation_price(self, entry_price, position_type):
        """
        Calculate liquidation price
        Liquidation occurs when losses reach (1/leverage - maintenance_margin) of position
        """
        if position_type == 'LONG':
            # For long: price drops to entry * (1 - liquidation_threshold)
            liquidation_threshold = (1 / self.leverage) - self.maintenance_margin
            return entry_price * (1 - liquidation_threshold)
        else:
            # For short: price rises to entry * (1 + liquidation_threshold)
            liquidation_threshold = (1 / self.leverage) - self.maintenance_margin
            return entry_price * (1 + liquidation_threshold)

    def calculate_funding_fee(self, position_value, hours_held):
        """
        Calculate funding fee (applied every 8 hours)
        Funding fee = position_value * funding_rate * number_of_periods
        """
        funding_periods = int(hours_held / 8)  # Only count complete 8-hour periods
        return position_value * self.funding_rate * funding_periods

    def fetch_backtest_data(self, start_date=None, end_date=None):
        """Fetch historical data"""
        print(f"\n{'='*80}")
        print(f"FUTURES BACKTEST CONFIGURATION")
        print(f"{'='*80}")
        print(f"Capital:        ${self.initial_capital:,.2f}")
        print(f"Leverage:       {self.leverage}x")
        print(f"Stop Loss:      {self.stop_loss*100:.1f}%")
        print(f"Take Profit:    {self.take_profit*100:.1f}%")
        print(f"Position Size:  {self.position_size*100:.0f}%")
        print(f"Min Confidence: {self.min_confidence*100:.0f}%")
        print(f"Trading Fee:    {self.trading_fee*100:.3f}%")
        print(f"Funding Rate:   {self.funding_rate*100:.3f}% per 8h")
        print(f"Max Trades/Day: {self.max_trades_per_day}")
        print(f"{'='*80}\n")

        print(f"Loading data from database...")
        conn = sqlite3.connect(str(self.db_path))

        date_filter = ""
        if start_date:
            date_filter += f" AND timestamp >= '{start_date}'"
        if end_date:
            date_filter += f" AND timestamp <= '{end_date}'"

        # Load data efficiently
        df_ohlcv = pd.read_sql_query(
            f"SELECT * FROM btc_1h WHERE 1=1 {date_filter} ORDER BY timestamp ASC",
            conn
        )
        df_indicators = pd.read_sql_query(
            f"SELECT * FROM smc_btc_1h_technical_indicators WHERE 1=1 {date_filter} ORDER BY timestamp ASC",
            conn
        )
        df_regimes = pd.read_sql_query(
            f"SELECT * FROM smc_btc_1h_market_regimes WHERE 1=1 {date_filter} ORDER BY timestamp ASC",
            conn
        )
        conn.close()

        print(f"Loaded {len(df_ohlcv):,} candles from database")

        # Merge and clean
        for df in [df_ohlcv, df_indicators, df_regimes]:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

        df = pd.merge(df_ohlcv, df_indicators, on='timestamp', how='inner')
        df = pd.merge(df, df_regimes, on='timestamp', how='inner')
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Handle NaN efficiently
        critical_features = ['close', 'volume', 'ema_21', 'rsi_14']
        df = df.dropna(subset=critical_features)
        df = df.ffill(limit=3).bfill(limit=3).dropna()

        # Convert to float32 for memory efficiency
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype('float32')

        print(f"Clean data: {len(df):,} candles")
        print(f"Date range: {df['timestamp'].iloc[0].strftime('%Y-%m-%d')} → {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}\n")

        return df

    def validate_sequence_continuity(self, seq_data):
        """Validate no time gaps in sequence"""
        if len(seq_data) < 2:
            return True

        time_diffs = seq_data['timestamp'].diff()[1:]
        expected = pd.Timedelta(hours=1)
        tolerance = pd.Timedelta(minutes=5)

        return ((time_diffs >= expected - tolerance) & (time_diffs <= expected + tolerance)).all()

    def create_sequences(self, df):
        """Create multi-input sequences efficiently (with warning suppression)"""
        print(f"Creating sequences...")
        sequences = {group: [] for group in self.feature_groups.keys()}
        timestamps, prices, indices = [], [], []

        # Pre-check features
        all_features = [f for feats in self.feature_groups.values() for f in feats]
        missing_features = [f for f in all_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        skipped = 0
        total_possible = len(df) - self.sequence_length - 4

        # Suppress sklearn warnings about feature names
        import warnings
        from sklearn.exceptions import DataConversionWarning

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            warnings.filterwarnings('ignore', category=DataConversionWarning)

            for i in range(total_possible):
                seq_data = df.iloc[i:i+self.sequence_length]

                # Validate continuity
                if not self.validate_sequence_continuity(seq_data):
                    skipped += 1
                    continue

                # Scale by group
                valid_sequence = True
                group_sequences = {}

                for group_name, feature_cols in self.feature_groups.items():
                    # Keep as DataFrame to preserve feature names for sklearn
                    X_group = seq_data[feature_cols].astype('float32')

                    # Check for invalid values
                    if np.any(~np.isfinite(X_group.values)):
                        valid_sequence = False
                        break

                    # Transform with DataFrame (preserves feature names)
                    X_scaled = self.scalers[group_name].transform(X_group)
                    X_scaled = np.clip(X_scaled, -10, 10)
                    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                    group_sequences[group_name] = X_scaled

                if not valid_sequence:
                    skipped += 1
                    continue

                # Add valid sequence
                for group_name, seq in group_sequences.items():
                    sequences[group_name].append(seq)

                timestamps.append(df['timestamp'].iloc[i+self.sequence_length])
                prices.append(df['close'].iloc[i+self.sequence_length])
                indices.append(i + self.sequence_length)

        # Convert to numpy arrays
        for group_name in sequences.keys():
            sequences[group_name] = np.array(sequences[group_name], dtype='float32')

        print(f"Created {len(timestamps):,} sequences (skipped {skipped} due to gaps/invalid data)\n")

        return sequences, timestamps, prices, indices

    def predict_batch(self, sequences, batch_size=128):
        """Make predictions in batches efficiently"""
        print(f"Making predictions...")
        num_sequences = sequences[list(sequences.keys())[0]].shape[0]
        all_predictions = np.zeros(num_sequences, dtype=np.int64)
        all_confidences = np.zeros(num_sequences, dtype=np.float32)
        all_probs = np.zeros((num_sequences, 3), dtype=np.float32)

        for i in range(0, num_sequences, batch_size):
            batch_end = min(i + batch_size, num_sequences)
            batch_inputs = {
                name: torch.FloatTensor(arr[i:batch_end]).to(self.device)
                for name, arr in sequences.items()
            }

            with torch.no_grad():
                outputs = self.model(batch_inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

            all_predictions[i:batch_end] = np.argmax(probs, axis=1)
            all_confidences[i:batch_end] = np.max(probs, axis=1)
            all_probs[i:batch_end] = probs

            # Cleanup
            for tensor in batch_inputs.values():
                del tensor
            del outputs
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            if (i // batch_size + 1) % 100 == 0:
                progress = (batch_end / num_sequences) * 100
                print(f"  Progress: {progress:.1f}%")

        print(f"Completed {len(all_predictions):,} predictions\n")

        return all_predictions, all_confidences, all_probs

    def calculate_pnl(self, entry_price, exit_price, position_type, margin_used):
        """
        CORRECT P&L Calculation for Futures Trading

        Formula:
        - Price change % = (exit - entry) / entry  [for LONG]
        - Price change % = (entry - exit) / entry  [for SHORT]
        - P&L in USD = margin * leverage * price_change_pct

        Example with $1000 margin, 5x leverage:
        - Position value = $5000
        - If price moves 2%: P&L = $1000 * 5 * 0.02 = $100
        """
        if position_type == 'LONG':
            price_change_pct = (exit_price - entry_price) / entry_price
        else:  # SHORT
            price_change_pct = (entry_price - exit_price) / entry_price

        # P&L = margin * leverage * price_change
        pnl = margin_used * self.leverage * price_change_pct

        return pnl, price_change_pct

    def run_backtest(self, df):
        """Run futures backtesting with correct calculations"""
        # Create sequences and predict
        sequences, timestamps, entry_prices, indices = self.create_sequences(df)
        predictions, confidences, probs = self.predict_batch(sequences)

        print(f"{'='*80}")
        print(f"SIMULATING FUTURES TRADES")
        print(f"{'='*80}\n")

        # Initialize state
        capital = float(self.initial_capital)
        position = None
        margin_used = 0.0
        position_value = 0.0
        entry_price = 0.0
        entry_time = None
        entry_idx = 0
        liquidation_price = 0.0
        take_profit_price = 0.0
        stop_loss_price = 0.0
        highest_price = 0.0
        lowest_price = float('inf')

        self.equity_curve = [{'timestamp': timestamps[0], 'equity': capital}]
        class_names = ['DOWN', 'HOLD', 'UP']

        trade_count = 0
        last_progress = 0

        # Main backtest loop
        for i in range(len(predictions)):
            pred = predictions[i]
            conf = confidences[i]
            current_price = entry_prices[i]
            timestamp = timestamps[i]
            current_idx = indices[i]

            # Progress indicator
            progress = (i / len(predictions)) * 100
            if progress - last_progress >= 10:
                print(f"Progress: {progress:.0f}% | Trades: {trade_count} | Capital: ${capital:,.2f}")
                last_progress = progress

            # Check if we have future price data
            future_idx = current_idx + 1  # Look 1 hour ahead (more realistic)
            if future_idx >= len(df):
                break

            future_candle = df.iloc[future_idx]
            future_price = future_candle['close']
            high_price = future_candle['high']
            low_price = future_candle['low']

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

            # MANAGE OPEN POSITION
            if position is not None:
                hours_held = (timestamp - entry_time).total_seconds() / 3600

                # Check minimum hold time
                if hours_held < self.min_hold_hours:
                    # Update equity with unrealized P&L
                    unrealized_pnl, _ = self.calculate_pnl(entry_price, current_price, position, margin_used)
                    current_equity = capital + margin_used + unrealized_pnl
                    self.equity_curve.append({'timestamp': timestamp, 'equity': current_equity})
                    continue

                # Update trailing stop prices
                if position == 'LONG':
                    highest_price = max(highest_price, high_price)
                    if self.use_trailing_stop:
                        stop_loss_price = max(stop_loss_price, highest_price * (1 - self.stop_loss))
                else:  # SHORT
                    lowest_price = min(lowest_price, low_price)
                    if self.use_trailing_stop:
                        stop_loss_price = min(stop_loss_price, lowest_price * (1 + self.stop_loss))

                should_exit = False
                exit_reason = None
                exit_price = None

                # 1. CHECK LIQUIDATION (highest priority)
                if position == 'LONG' and low_price <= liquidation_price:
                    exit_price = liquidation_price
                    should_exit = True
                    exit_reason = 'LIQUIDATION'
                    self.liquidations += 1
                elif position == 'SHORT' and high_price >= liquidation_price:
                    exit_price = liquidation_price
                    should_exit = True
                    exit_reason = 'LIQUIDATION'
                    self.liquidations += 1

                # 2. CHECK TAKE PROFIT
                if not should_exit:
                    if position == 'LONG' and high_price >= take_profit_price:
                        exit_price = take_profit_price
                        should_exit = True
                        exit_reason = 'TAKE_PROFIT'
                    elif position == 'SHORT' and low_price <= take_profit_price:
                        exit_price = take_profit_price
                        should_exit = True
                        exit_reason = 'TAKE_PROFIT'

                # 3. CHECK STOP LOSS
                if not should_exit:
                    if position == 'LONG' and low_price <= stop_loss_price:
                        exit_price = stop_loss_price
                        should_exit = True
                        exit_reason = 'STOP_LOSS'
                    elif position == 'SHORT' and high_price >= stop_loss_price:
                        exit_price = stop_loss_price
                        should_exit = True
                        exit_reason = 'STOP_LOSS'

                # 4. CHECK MAX POSITION TIME
                if not should_exit and hours_held >= self.max_position_hours:
                    exit_price = future_price
                    should_exit = True
                    exit_reason = 'MAX_TIME'

                # 5. SIGNAL-BASED EXIT
                if not should_exit:
                    # Exit if signal changes or HOLD signal
                    if pred == 1:  # HOLD
                        exit_price = future_price
                        should_exit = True
                        exit_reason = 'SIGNAL_EXIT'
                    elif (position == 'LONG' and pred == 0):  # Long but got DOWN signal
                        exit_price = future_price
                        should_exit = True
                        exit_reason = 'SIGNAL_EXIT'
                    elif (position == 'SHORT' and pred == 2):  # Short but got UP signal
                        exit_price = future_price
                        should_exit = True
                        exit_reason = 'SIGNAL_EXIT'

                # EXECUTE EXIT
                if should_exit:
                    # Apply slippage
                    if position == 'LONG':
                        exit_price = exit_price * (1 - self.slippage)
                    else:
                        exit_price = exit_price * (1 + self.slippage)

                    # Calculate P&L correctly
                    gross_pnl, pnl_pct = self.calculate_pnl(entry_price, exit_price, position, margin_used)

                    # Calculate fees on position value
                    total_fees = position_value * self.trading_fee * 2  # Entry + Exit

                    # Calculate funding fee
                    funding_fee = self.calculate_funding_fee(position_value, hours_held)
                    if position == 'LONG':
                        self.funding_paid += funding_fee
                    else:
                        self.funding_received += funding_fee
                        funding_fee = -funding_fee  # Shorts receive funding

                    # Net P&L
                    net_pnl = gross_pnl - total_fees - funding_fee

                    # LIQUIDATION: lose all margin
                    if exit_reason == 'LIQUIDATION':
                        net_pnl = -margin_used

                    # Update capital
                    capital += margin_used  # Return margin
                    capital += net_pnl      # Add P&L

                    # Safety checks
                    if capital < 0:
                        capital = self.initial_capital * 0.01
                    elif capital > self.initial_capital * 1000:  # Prevent unrealistic growth
                        capital = self.initial_capital * 1000

                    # Calculate ROI on margin
                    roi_on_margin = (net_pnl / margin_used) * 100 if margin_used > 0 else 0

                    # Record trade
                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'liquidation_price': liquidation_price,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_usd': net_pnl,
                        'roi_on_margin': roi_on_margin,
                        'margin_used': margin_used,
                        'position_value': position_value,
                        'leverage': self.leverage,
                        'fees': total_fees,
                        'funding_fee': funding_fee,
                        'capital_after': capital,
                        'reason': exit_reason,
                        'confidence': conf,
                        'hold_hours': hours_held
                    })

                    trade_count += 1

                    # Reset position
                    position = None
                    margin_used = 0.0
                    position_value = 0.0
                    highest_price = 0.0
                    lowest_price = float('inf')

            # OPEN NEW POSITION
            if position is None and capital > 0:
                date_key = timestamp.date()
                trades_today = self.daily_trades_count.get(date_key, 0)

                # Check filters
                if conf < self.min_confidence:
                    self.skipped_trades['confidence'] += 1
                elif trades_today >= self.max_trades_per_day:
                    self.skipped_trades['daily_limit'] += 1
                elif capital < self.initial_capital * 0.05:
                    self.skipped_trades['capital_insufficient'] += 1
                elif pred in [0, 2]:  # DOWN or UP signal
                    # Calculate position size (max 40% of capital to avoid overexposure)
                    margin_used = min(
                        capital * self.position_size,
                        capital * 0.40
                    )

                    # Position value with leverage
                    position_value = margin_used * self.leverage

                    # Apply slippage to entry
                    if pred == 2:  # LONG
                        position = 'LONG'
                        entry_price = current_price * (1 + self.slippage)
                        highest_price = entry_price
                        take_profit_price = entry_price * (1 + self.take_profit)
                        stop_loss_price = entry_price * (1 - self.stop_loss)
                    else:  # SHORT (pred == 0)
                        position = 'SHORT'
                        entry_price = current_price * (1 - self.slippage)
                        lowest_price = entry_price
                        take_profit_price = entry_price * (1 - self.take_profit)
                        stop_loss_price = entry_price * (1 + self.stop_loss)

                    liquidation_price = self.calculate_liquidation_price(entry_price, position)
                    entry_time = timestamp
                    entry_idx = i

                    # Lock margin
                    capital -= margin_used

                    # Update daily trade count
                    self.daily_trades_count[date_key] = trades_today + 1

            # Update equity curve
            if position is None:
                current_equity = capital
            else:
                # Calculate unrealized P&L
                unrealized_pnl, _ = self.calculate_pnl(entry_price, current_price, position, margin_used)
                current_equity = capital + margin_used + unrealized_pnl

            self.equity_curve.append({'timestamp': timestamp, 'equity': current_equity})

        # Close any remaining position at end of backtest
        if position is not None:
            final_price = df['close'].iloc[-1]
            exit_price = final_price * (1 - self.slippage if position == 'LONG' else 1 + self.slippage)

            hours_held = (df['timestamp'].iloc[-1] - entry_time).total_seconds() / 3600
            gross_pnl, pnl_pct = self.calculate_pnl(entry_price, exit_price, position, margin_used)

            total_fees = position_value * self.trading_fee * 2
            funding_fee = self.calculate_funding_fee(position_value, hours_held)
            net_pnl = gross_pnl - total_fees - funding_fee

            capital += margin_used + net_pnl
            capital = max(capital, self.initial_capital * 0.01)

            if capital > self.initial_capital * 1000:
                capital = self.initial_capital * 1000

            self.trades.append({
                'entry_time': entry_time,
                'exit_time': df['timestamp'].iloc[-1],
                'position': position,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'liquidation_price': liquidation_price,
                'pnl_pct': pnl_pct * 100,
                'pnl_usd': net_pnl,
                'roi_on_margin': (net_pnl/margin_used)*100,
                'margin_used': margin_used,
                'position_value': position_value,
                'leverage': self.leverage,
                'fees': total_fees,
                'funding_fee': funding_fee,
                'capital_after': capital,
                'reason': 'FINAL_CLOSE',
                'confidence': conf,
                'hold_hours': hours_held
            })

            trade_count += 1

        print(f"\nCompleted {trade_count} trades\n")

        return self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if len(self.trades) == 0:
            return None

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        final_capital = equity_df['equity'].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100

        # Annualized return
        total_days = (equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days
        total_years = max(total_days / 365.25, 0.01)
        annualized_return = ((final_capital / self.initial_capital) ** (1/total_years) - 1) * 100

        # Win rate
        winning_trades = trades_df[trades_df['pnl_usd'] > 0]
        losing_trades = trades_df[trades_df['pnl_usd'] <= 0]
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0

        # Drawdown
        equity_series = equity_df['equity'].values
        cummax = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = np.min(drawdown)

        # Sharpe Ratio (annualized)
        returns = pd.Series(equity_series).pct_change().dropna()
        sharpe_ratio = 0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(24 * 365)

        # Sortino Ratio (downside risk)
        downside_returns = returns[returns < 0]
        sortino_ratio = 0
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(24 * 365)

        # Profit factor
        total_wins = winning_trades['pnl_usd'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl_usd'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Position breakdown
        long_trades = trades_df[trades_df['position'] == 'LONG']
        short_trades = trades_df[trades_df['position'] == 'SHORT']

        # Expectancy
        avg_win = winning_trades['pnl_usd'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_usd'].mean() if len(losing_trades) > 0 else 0
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        return {
            'final_capital': final_capital,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'total_trades': len(trades_df),
            'total_days': total_days,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'avg_roi': trades_df['roi_on_margin'].mean(),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_factor': profit_factor,
            'best_trade': trades_df['pnl_usd'].max(),
            'worst_trade': trades_df['pnl_usd'].min(),
            'avg_hold_hours': trades_df['hold_hours'].mean(),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': len(long_trades[long_trades['pnl_usd'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0,
            'short_win_rate': len(short_trades[short_trades['pnl_usd'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0,
            'long_pnl': long_trades['pnl_usd'].sum() if len(long_trades) > 0 else 0,
            'short_pnl': short_trades['pnl_usd'].sum() if len(short_trades) > 0 else 0,
            'liquidations': self.liquidations,
            'total_fees': trades_df['fees'].sum(),
            'net_funding': self.funding_paid - self.funding_received,
            'skipped_trades': self.skipped_trades,
            'trades_df': trades_df,
            'equity_df': equity_df
        }

    def print_summary(self, metrics):
        """Print comprehensive summary"""
        if not metrics:
            print("\nNo trades executed\n")
            return

        print(f"\n{'='*80}")
        print(f"FUTURES BACKTEST RESULTS")
        print(f"{'='*80}\n")

        print(f"CAPITAL & RETURNS")
        print(f"  Initial:       ${self.initial_capital:>12,.2f}")
        print(f"  Final:         ${metrics['final_capital']:>12,.2f}")
        print(f"  Total Return:  {metrics['total_return']:>12.2f}%")
        print(f"  Annualized:    {metrics['annualized_return']:>12.2f}%")
        print(f"  Period:        {metrics['total_days']:>12,} days ({metrics['total_days']/365:.1f} years)\n")

        print(f"RISK METRICS")
        print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:>12.2f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:>12.2f}")
        print(f"  Max Drawdown:  {metrics['max_drawdown']:>12.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:>12.2f}\n")

        print(f"TRADING PERFORMANCE")
        print(f"  Total Trades:  {metrics['total_trades']:>12,}")
        print(f"  Win Rate:      {metrics['win_rate']:>12.1f}%")
        print(f"  Avg Win:       ${metrics['avg_win']:>11,.2f}")
        print(f"  Avg Loss:      ${metrics['avg_loss']:>11,.2f}")
        print(f"  Expectancy:    ${metrics['expectancy']:>11,.2f}")
        print(f"  Best Trade:    ${metrics['best_trade']:>11,.2f}")
        print(f"  Worst Trade:   ${metrics['worst_trade']:>11,.2f}")
        print(f"  Avg ROI:       {metrics['avg_roi']:>12.1f}%")
        print(f"  Avg Hold:      {metrics['avg_hold_hours']:>12.1f} hours\n")

        print(f"FUTURES COSTS")
        print(f"  Leverage:      {self.leverage:>12.0f}x")
        print(f"  Liquidations:  {metrics['liquidations']:>12,}")
        print(f"  Total Fees:    ${metrics['total_fees']:>11,.2f}")
        print(f"  Net Funding:   ${metrics['net_funding']:>11,.2f}")
        print(f"  Total Costs:   ${metrics['total_fees'] + metrics['net_funding']:>11,.2f}\n")

        print(f"POSITION BREAKDOWN")
        print(f"  Long:  {metrics['long_trades']:>5,} trades ({metrics['long_win_rate']:>5.1f}% WR)  P&L: ${metrics['long_pnl']:>10,.2f}")
        print(f"  Short: {metrics['short_trades']:>5,} trades ({metrics['short_win_rate']:>5.1f}% WR)  P&L: ${metrics['short_pnl']:>10,.2f}\n")

        print(f"TRADE FILTERS")
        print(f"  Low Confidence:     {metrics['skipped_trades']['confidence']:>7,}")
        print(f"  Daily Limit:        {metrics['skipped_trades']['daily_limit']:>7,}")
        print(f"  Insufficient Cap:   {metrics['skipped_trades']['capital_insufficient']:>7,}\n")

        print(f"{'='*80}\n")

    def export_results(self, prefix='backtest'):
        """Export all results"""
        if self.trades:
            trades_file = f"{prefix}_trades.csv"
            pd.DataFrame(self.trades).to_csv(trades_file, index=False)
            print(f"✓ Trades exported to: {trades_file}")

        if self.predictions_log:
            pred_file = f"{prefix}_predictions.csv"
            pd.DataFrame(self.predictions_log).to_csv(pred_file, index=False)
            print(f"✓ Predictions exported to: {pred_file}")

        if self.equity_curve:
            equity_file = f"{prefix}_equity.csv"
            pd.DataFrame(self.equity_curve).to_csv(equity_file, index=False)
            print(f"✓ Equity curve exported to: {equity_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Futures Backtester')
    parser.add_argument('--model', default='models/multi_input_lstm_optimized_full.pth')
    parser.add_argument('--preprocessed', default='preprocessed_data_multi_lstm_1h')
    parser.add_argument('--db', default='data/btc_ohlcv.db')
    parser.add_argument('--capital', type=float, default=10000)
    parser.add_argument('--leverage', type=int, default=5)
    parser.add_argument('--fee', type=float, default=0.0005)
    parser.add_argument('--position-size', type=float, default=0.40)
    parser.add_argument('--stop-loss', type=float, default=0.03)
    parser.add_argument('--take-profit', type=float, default=0.06)
    parser.add_argument('--min-confidence', type=float, default=0.55)
    parser.add_argument('--max-trades-per-day', type=int, default=5)
    parser.add_argument('--trailing-stop', action='store_true')
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--export', action='store_true')

    args = parser.parse_args()

    # Validate files
    required_paths = {
        'Model': args.model,
        'Preprocessed': args.preprocessed,
        'Database': args.db
    }

    missing = []
    for name, path in required_paths.items():
        full_path = ROOT_DIR / path if not Path(path).is_absolute() else Path(path)
        if not full_path.exists():
            missing.append(f"  ✗ {name}: {full_path}")

    if missing:
        print(f"\n{'='*80}")
        print("MISSING FILES")
        print(f"{'='*80}")
        for m in missing:
            print(m)
        print(f"\nℹ Run training first: python train_multi_input_lstm.py")
        print(f"{'='*80}\n")
        return

    try:
        # Initialize backtester
        backtester = BTCFuturesBacktester(
            model_path=args.model,
            preprocessed_dir=args.preprocessed,
            db_path=args.db,
            initial_capital=args.capital,
            leverage=args.leverage,
            trading_fee=args.fee,
            position_size=args.position_size,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            min_confidence=args.min_confidence,
            max_trades_per_day=args.max_trades_per_day,
            use_trailing_stop=args.trailing_stop
        )

        # Fetch data
        df = backtester.fetch_backtest_data(args.start_date, args.end_date)

        if len(df) < backtester.sequence_length + 100:
            print(f"\nInsufficient data: need {backtester.sequence_length + 100}, got {len(df)}\n")
            return

        # Run backtest
        metrics = backtester.run_backtest(df)

        # Display results
        if metrics:
            backtester.print_summary(metrics)

            if args.export:
                backtester.export_results()
                print()
        else:
            print("\nNo trades executed\n")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR OCCURRED")
        print(f"{'='*80}")
        print(f"{str(e)}\n")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
