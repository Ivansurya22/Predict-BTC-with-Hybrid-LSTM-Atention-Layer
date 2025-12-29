import os
import sqlite3

import pandas as pd
import numpy as np
import talib
from smartmoneyconcepts import smc
from hmmlearn.hmm import GaussianHMM

# Configuration
DB_PATH = os.getenv("OHLCV_DB_PATH", "data/btc_ohlcv.db")
OHLCV_TABLES = ["btc_1h"]


def get_smc_tables(timeframe):
    """Get SMC table names for a specific timeframe."""
    return {
        "swing_highs_lows": f"smc_{timeframe}_swing_highs_lows",
        "fvg": f"smc_{timeframe}_fvg",
        "bos_choch": f"smc_{timeframe}_bos_choch",
        "order_block": f"smc_{timeframe}_order_block",
        "liquidity": f"smc_{timeframe}_liquidity",
        "retracements": f"smc_{timeframe}_retracements",  # NEW
        "technical_indicators": f"smc_{timeframe}_technical_indicators",
        "market_regimes": f"smc_{timeframe}_market_regimes",
    }


def table_exists(conn, table_name):
    """Check if a table exists in the database."""
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(query, (table_name,)).fetchone() is not None


def get_available_ohlcv_tables():
    """Get list of available OHLCV tables in the database."""
    available_tables = []
    with sqlite3.connect(DB_PATH) as conn:
        for table in OHLCV_TABLES:
            if table_exists(conn, table):
                available_tables.append(table)
    return available_tables


def load_ohlcv(table_name):
    """Load OHLCV data from specific table."""
    with sqlite3.connect(DB_PATH) as conn:
        if not table_exists(conn, table_name):
            raise ValueError(f"Table '{table_name}' not found.")
        df = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY timestamp", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def save_to_db(df, table_name, mode="replace"):
    """Save dataframe to database with proper dtype handling."""
    df_to_save = df.copy()
    df_to_save.index.name = "timestamp"

    for col in df_to_save.columns:
        if df_to_save[col].dtype not in ['float64', 'float32', 'int64', 'int32']:
            df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce')

    with sqlite3.connect(DB_PATH) as conn:
        df_to_save.to_sql(table_name, conn, if_exists=mode, index=True)


def load_smc_table(table_name):
    """Load SMC table with proper type conversion."""
    with sqlite3.connect(DB_PATH) as conn:
        if not table_exists(conn, table_name):
            return None
        df = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY timestamp", conn)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)

    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def add_technical_indicators(df):
    """Add technical indicators."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    indicators = pd.DataFrame(index=df.index)

    try:
        indicators["ema_9"] = talib.EMA(close, timeperiod=9)
        indicators["ema_21"] = talib.EMA(close, timeperiod=21)
        indicators["ema_50"] = talib.EMA(close, timeperiod=50)
        indicators["ema_100"] = talib.EMA(close, timeperiod=100)
        indicators["ema_200"] = talib.EMA(close, timeperiod=200)

        macd, macd_signal, macd_hist = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        indicators["macd"] = macd
        indicators["macd_signal"] = macd_signal
        indicators["macd_hist"] = macd_hist

        indicators["rsi_14"] = talib.RSI(close, timeperiod=14)
        indicators["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
        indicators["adx_14"] = talib.ADX(high, low, close, timeperiod=14)

        upper, middle, lower = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        indicators["bb_upper"] = upper
        indicators["bb_middle"] = middle
        indicators["bb_lower"] = lower
        indicators["bb_width"] = (upper - lower) / middle * 100

        indicators["obv"] = talib.OBV(close, volume)

    except Exception as e:
        print(f"Technical Indicators error: {e}")

    return indicators.fillna(0)


def add_hmm_regime(df, n_states=3):
    """Hidden Markov Model - 2 features only."""
    features = pd.DataFrame(index=df.index)

    try:
        returns = df['close'].pct_change().fillna(0)
        volatility = df['close'].rolling(14).std().fillna(0)
        X = np.column_stack([returns, volatility])

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        model.fit(X)

        hidden_states = model.predict(X)

        state_volatility = []
        for state in range(n_states):
            mask = hidden_states == state
            avg_vol = volatility[mask].mean()
            state_volatility.append((state, avg_vol))

        state_volatility.sort(key=lambda x: x[1])
        state_mapping = {old: new for new, (old, _) in enumerate(state_volatility)}
        mapped_states = np.array([state_mapping[s] for s in hidden_states])

        features['hmm_regime_high'] = (mapped_states == 2).astype(int)

        regime_changes = (pd.Series(mapped_states, index=df.index) !=
                         pd.Series(mapped_states, index=df.index).shift(1)).cumsum()
        features['hmm_regime_duration'] = regime_changes.groupby(regime_changes).cumcount() + 1

    except Exception as e:
        print(f"HMM error: {e}")
        features['hmm_regime_high'] = 0
        features['hmm_regime_duration'] = 1

    return features


def add_trend_regime(df):
    """Trend regime detection - 2 features only."""
    features = pd.DataFrame(index=df.index)
    close = df['close']

    try:
        if 'adx_14' in df.columns:
            adx = df['adx_14']
        else:
            high = df['high'].values
            low = df['low'].values
            close_arr = close.values
            adx = pd.Series(talib.ADX(high, low, close_arr, timeperiod=14), index=df.index)

        if 'ema_21' in df.columns:
            ema_21 = df['ema_21']
            ema_50 = df['ema_50']
            ema_200 = df['ema_200']
        else:
            ema_21 = close.ewm(span=21).mean()
            ema_50 = close.ewm(span=50).mean()
            ema_200 = close.ewm(span=200).mean()

        above_ema21 = (close > ema_21).astype(int)
        above_ema50 = (close > ema_50).astype(int)
        above_ema200 = (close > ema_200).astype(int)

        alignment = above_ema21 + above_ema50 + above_ema200
        features['ema_alignment'] = alignment - 1.5

        ema_trend = ((ema_21 > ema_50).astype(int) * 2 - 1)
        trend_strong = (adx > 25).astype(int)

        features['trend_strong_bull'] = ((trend_strong == 1) & (ema_trend == 1)).astype(int)

    except Exception as e:
        print(f"Trend regime error: {e}")
        features['ema_alignment'] = 0
        features['trend_strong_bull'] = 0

    return features


def add_volume_regime(df):
    """Volume-based market regime - 2 features only."""
    features = pd.DataFrame(index=df.index)
    volume = df['volume']

    try:
        features['volume_percentile'] = volume.rolling(100, min_periods=1).apply(
            lambda x: (pd.Series(x).rank(pct=True).iloc[-1] * 100) if len(x) > 0 else 50
        )

        vol_ma_5 = volume.rolling(5, min_periods=1).mean()
        vol_ma_20 = volume.rolling(20, min_periods=1).mean()
        features['volume_trend'] = (vol_ma_5 / vol_ma_20) - 1

    except Exception as e:
        print(f"Volume regime error: {e}")
        features['volume_percentile'] = 50
        features['volume_trend'] = 0

    return features


def add_smc_features(df):
    """Add Smart Money Concepts features."""
    df_smc = df.reset_index()
    ohlc = df_smc[["open", "high", "low", "close"]]
    ohlcv = df_smc[["open", "high", "low", "close", "volume"]]

    SWING_LENGTH = 15
    JOIN_CONSECUTIVE_FVG = True
    CLOSE_BREAK = True
    CLOSE_MITIGATION = True
    LIQUIDITY_RANGE_PERCENT = 0.01

    features = {}

    # Swing Highs and Lows
    try:
        swing_highs_lows = smc.swing_highs_lows(ohlc, swing_length=SWING_LENGTH)
        features["swing_highs_lows"] = swing_highs_lows if isinstance(swing_highs_lows, pd.DataFrame) else \
            pd.DataFrame(np.nan, index=df_smc.index, columns=["HighLow", "Level"])
    except Exception as e:
        print(f"Swing detection error: {e}")
        features["swing_highs_lows"] = pd.DataFrame(np.nan, index=df_smc.index, columns=["HighLow", "Level"])

    # Fair Value Gap
    try:
        fvg = smc.fvg(ohlc, join_consecutive=JOIN_CONSECUTIVE_FVG)
        features["fvg"] = fvg.fillna(0) if isinstance(fvg, pd.DataFrame) else \
            pd.DataFrame(0, index=df_smc.index, columns=["FVG", "Top", "Bottom", "MitigatedIndex"])
    except Exception as e:
        print(f"FVG error: {e}")
        features["fvg"] = pd.DataFrame(0, index=df_smc.index, columns=["FVG", "Top", "Bottom", "MitigatedIndex"])

    # BOS & CHOCH
    try:
        bos_choch = smc.bos_choch(ohlc, features["swing_highs_lows"], close_break=CLOSE_BREAK)
        features["bos_choch"] = bos_choch if isinstance(bos_choch, pd.DataFrame) else \
            pd.DataFrame(np.nan, index=df_smc.index, columns=["BOS", "CHOCH", "Level", "BrokenIndex"])
    except Exception as e:
        print(f"BOS/CHOCH error: {e}")
        features["bos_choch"] = pd.DataFrame(np.nan, index=df_smc.index, columns=["BOS", "CHOCH", "Level", "BrokenIndex"])

    # Order Blocks
    try:
        order_block = smc.ob(ohlcv, features["swing_highs_lows"], close_mitigation=CLOSE_MITIGATION)
        features["order_block"] = order_block.fillna(0) if isinstance(order_block, pd.DataFrame) else \
            pd.DataFrame(0, index=df_smc.index, columns=["OB", "Top", "Bottom", "OBVolume", "Percentage"])
    except Exception as e:
        print(f"Order Block error: {e}")
        features["order_block"] = pd.DataFrame(0, index=df_smc.index, columns=["OB", "Top", "Bottom", "OBVolume", "Percentage"])

    # Liquidity
    try:
        liquidity = smc.liquidity(ohlc, features["swing_highs_lows"], range_percent=LIQUIDITY_RANGE_PERCENT)
        features["liquidity"] = liquidity.fillna(0) if isinstance(liquidity, pd.DataFrame) else \
            pd.DataFrame(0, index=df_smc.index, columns=["Liquidity", "Level", "End", "Swept"])
    except Exception as e:
        print(f"Liquidity error: {e}")
        features["liquidity"] = pd.DataFrame(0, index=df_smc.index, columns=["Liquidity", "Level", "End", "Swept"])

    # Retracements (NEW)
    try:
        retracements = smc.retracements(ohlc, features["swing_highs_lows"])
        features["retracements"] = retracements.fillna(0) if isinstance(retracements, pd.DataFrame) else \
            pd.DataFrame(0, index=df_smc.index, columns=["Direction", "CurrentRetracement%", "DeepestRetracement%"])

        print(f"  ✓ Retracements: {len(retracements)} rows")
    except Exception as e:
        print(f"Retracements error: {e}")
        features["retracements"] = pd.DataFrame(0, index=df_smc.index, columns=["Direction", "CurrentRetracement%", "DeepestRetracement%"])

    # Technical Indicators
    features["technical_indicators"] = add_technical_indicators(df)

    # Market Regimes (6 features only)
    hmm_regime = add_hmm_regime(df)
    trend_regime = add_trend_regime(df)
    volume_regime = add_volume_regime(df)

    features["market_regimes"] = pd.concat([hmm_regime, trend_regime, volume_regime], axis=1)

    return features


def get_last_smc_timestamp(table_name):
    """Get the timestamp of the last SMC record."""
    with sqlite3.connect(DB_PATH) as conn:
        if table_exists(conn, table_name):
            ts = conn.execute(f"SELECT MAX(timestamp) FROM {table_name}").fetchone()[0]
            return pd.Timestamp(ts, tz="UTC") if ts else None
        return None


def all_smc_tables_exist(timeframe):
    """Check if all SMC tables exist."""
    smc_tables = get_smc_tables(timeframe)
    with sqlite3.connect(DB_PATH) as conn:
        return all(table_exists(conn, table) for table in smc_tables.values())


def clean_smc_duplicate_rows(timeframe):
    """Remove duplicate rows from SMC tables."""
    smc_tables = get_smc_tables(timeframe)
    with sqlite3.connect(DB_PATH) as conn:
        for table in smc_tables.values():
            if table_exists(conn, table):
                conn.execute(f"""
                    DELETE FROM {table}
                    WHERE rowid NOT IN (
                        SELECT MIN(rowid)
                        FROM {table}
                        GROUP BY timestamp
                    )
                """)
                conn.commit()


def sync_smc_features_for_timeframe(timeframe):
    """Synchronize SMC features with latest OHLCV data."""
    print(f"\n{'─' * 60}")
    print(f"Processing {timeframe.upper()}")
    print(f"{'─' * 60}")

    try:
        df_ohlcv = load_ohlcv(timeframe)
    except ValueError as e:
        print(f"Error: {e}")
        return False

    ohlcv_count = len(df_ohlcv)
    start_date = df_ohlcv.index[0].strftime('%Y-%m-%d')
    end_date = df_ohlcv.index[-1].strftime('%Y-%m-%d')
    print(f"OHLCV: {ohlcv_count:,} rows ({start_date} to {end_date})")

    smc_tables = get_smc_tables(timeframe)
    smc_complete = all_smc_tables_exist(timeframe)

    last_timestamps = []
    for table in smc_tables.values():
        ts = get_last_smc_timestamp(table)
        if ts:
            last_timestamps.append(ts)

    last_ts = min(last_timestamps) if last_timestamps else None
    needs_full_sync = last_ts is None or not smc_complete

    if needs_full_sync:
        print(f"Full sync required")
        print(f"Processing {ohlcv_count:,} candles...")
        smc_features = add_smc_features(df_ohlcv)

        saved_count = 0
        for key, table in smc_tables.items():
            if key in smc_features:
                df_smc = smc_features[key]
                if isinstance(df_smc, pd.DataFrame):
                    if "timestamp" not in df_smc.columns:
                        df_smc["timestamp"] = df_ohlcv.index
                    df_smc.set_index("timestamp", inplace=True)
                    save_to_db(df_smc, table)
                    saved_count += 1
                    print(f"  ✓ Saved: {table} ({len(df_smc)} rows)")

        print(f"Saved {saved_count}/{len(smc_tables)} tables")
        return True

    new_ohlcv_data = df_ohlcv[df_ohlcv.index > last_ts]

    if new_ohlcv_data.empty:
        print(f"Up to date (Last: {last_ts.strftime('%Y-%m-%d %H:%M')})")
        return True

    print(f"Incremental sync: {len(new_ohlcv_data)} new candles")
    smc_features = add_smc_features(new_ohlcv_data)

    saved_count = 0
    for key, table in smc_tables.items():
        if key in smc_features and isinstance(smc_features[key], pd.DataFrame):
            df_smc = smc_features[key]
            if "timestamp" not in df_smc.columns:
                df_smc["timestamp"] = new_ohlcv_data.index
            df_smc.set_index("timestamp", inplace=True)
            save_to_db(df_smc, table, mode="append")
            saved_count += 1

    clean_smc_duplicate_rows(timeframe)
    print(f"Saved {saved_count}/{len(smc_tables)} tables")
    return True


def sync_all_smc_features():
    """Synchronize SMC features for all available timeframes."""
    available_tables = get_available_ohlcv_tables()

    if not available_tables:
        print("No OHLCV tables available!")
        return

    print(f"\n{'═' * 60}")
    print(f"SMC FEATURE ENGINEERING")
    print(f"{'═' * 60}")
    print(f"Database: {DB_PATH}")
    print(f"Timeframes: {', '.join(available_tables)}")
    print(f"Features: SMC (~25) + Technical (16) + Market Regimes (6) = ~47 total")


    success_count = 0
    for table in available_tables:
        if sync_smc_features_for_timeframe(table):
            success_count += 1

    print(f"\n{'═' * 60}")
    print(f"Completed: {success_count}/{len(available_tables)} timeframes")
    print(f"{'═' * 60}\n")


def verify_smc_sync():
    """Verify SMC synchronization status."""
    available_tables = get_available_ohlcv_tables()

    print(f"\n{'═' * 60}")
    print("VERIFICATION REPORT")
    print(f"{'═' * 60}")

    for timeframe in available_tables:
        print(f"\n{timeframe.upper()}")
        print("─" * 40)

        try:
            df_ohlcv = load_ohlcv(timeframe)
            ohlcv_count = len(df_ohlcv)

            smc_tables = get_smc_tables(timeframe)
            all_synced = True

            with sqlite3.connect(DB_PATH) as conn:
                for key, table in smc_tables.items():
                    if table_exists(conn, table):
                        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        diff = ohlcv_count - count

                        if key == "market_regimes":
                            df_test = load_smc_table(table)
                            col_count = len(df_test.columns) if df_test is not None else 0
                            status_msg = f"{count:,} rows (diff: {diff}) | Columns: {col_count}"
                        elif key == "retracements":
                            df_test = load_smc_table(table)
                            col_count = len(df_test.columns) if df_test is not None else 0
                            status_msg = f"{count:,} rows (diff: {diff}) | Columns: {col_count}"
                        else:
                            status_msg = f"{count:,} rows (diff: {diff})"

                        status = "✅" if diff == 0 else "⚠️"
                        if diff != 0:
                            all_synced = False

                        print(f"{status} {key:20s}: {status_msg}")
                    else:
                        print(f"❌ {key:20s}: Table not found")
                        all_synced = False

            print(f"\n{'✅ All tables in sync!' if all_synced else '⚠️ Sync required'}")

        except Exception as e:
            print(f"Error: {e}")

    print(f"\n{'═' * 60}\n")


def main():
    """Main function to run the SMC feature engineering pipeline."""
    try:
        sync_all_smc_features()
        verify_smc_sync()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
