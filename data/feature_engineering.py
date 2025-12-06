import os
import sqlite3

import pandas as pd
import numpy as np
import talib
from smartmoneyconcepts import smc

# Configuration
DB_PATH = os.getenv("OHLCV_DB_PATH", "data/btc_ohlcv.db")

# OHLCV Tables (only 1h)
OHLCV_TABLES = ["btc_1h"]


# Smart Money Concepts tables for 1h timeframe
def get_smc_tables(timeframe):
    """Get SMC table names for a specific timeframe."""
    return {
        "swing_highs_lows": f"smc_{timeframe}_swing_highs_lows",
        "fvg": f"smc_{timeframe}_fvg",
        "bos_choch": f"smc_{timeframe}_bos_choch",
        "order_block": f"smc_{timeframe}_order_block",
        "liquidity": f"smc_{timeframe}_liquidity",
        "retracements": f"smc_{timeframe}_retracements",
        "technical_indicators": f"smc_{timeframe}_technical_indicators",
        "optimized_features": f"smc_{timeframe}_optimized_features",
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
            raise ValueError(f"❌ Table '{table_name}' tidak ditemukan.")
        df = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY timestamp", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def save_to_db(df, table_name, mode="replace"):
    """Save dataframe to database with proper dtype handling."""
    df_to_save = df.copy()
    df_to_save.index.name = "timestamp"

    # Convert all numeric columns to float64 BEFORE saving
    for col in df_to_save.columns:
        if df_to_save[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Keep as numeric
            continue
        else:
            # Try to convert to numeric, keep NaN as NaN
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

    # CRITICAL FIX: Convert all columns to numeric (except timestamp)
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def add_technical_indicators(df):
    """Add Technical Indicators using TA-Lib."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_price = df["open"].values
    volume = df["volume"].values

    indicators = pd.DataFrame(index=df.index)

    try:
        # EMAs
        indicators["ema_9"] = talib.EMA(close, timeperiod=9)
        indicators["ema_21"] = talib.EMA(close, timeperiod=21)
        indicators["ema_50"] = talib.EMA(close, timeperiod=50)
        indicators["ema_100"] = talib.EMA(close, timeperiod=100)
        indicators["ema_200"] = talib.EMA(close, timeperiod=200)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        indicators["macd"] = macd
        indicators["macd_signal"] = macd_signal
        indicators["macd_hist"] = macd_hist

        # RSI
        indicators["rsi_14"] = talib.RSI(close, timeperiod=14)

        # Stochastic
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=14, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )
        indicators["stoch_k"] = slowk
        indicators["stoch_d"] = slowd

        # ATR
        indicators["atr_14"] = talib.ATR(high, low, close, timeperiod=14)

        # ADX (untuk trend strength)
        indicators["adx_14"] = talib.ADX(high, low, close, timeperiod=14)

        # Directional Movement (untuk trend direction)
        indicators["plus_di_14"] = talib.PLUS_DI(high, low, close, timeperiod=14)
        indicators["minus_di_14"] = talib.MINUS_DI(high, low, close, timeperiod=14)

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        indicators["bb_upper_20"] = upper
        indicators["bb_middle_20"] = middle
        indicators["bb_lower_20"] = lower
        indicators["bb_width"] = (upper - lower) / middle * 100

        # OBV
        indicators["obv"] = talib.OBV(close, volume)

        # MFI
        indicators["mfi_14"] = talib.MFI(high, low, close, volume, timeperiod=14)

    except Exception as e:
        print(f"    ⚠️ Technical Indicators error: {e}")

    # Fill NaN with 0
    indicators = indicators.fillna(0)
    return indicators


def add_optimized_features(df):
    features = pd.DataFrame(index=df.index)

    print(f"    🔧 Adding optimized features...")

    # ========== 1. TEMPORAL FEATURES (6 features) ==========
    hour = df.index.hour
    day_of_week = df.index.dayofweek
    day_of_month = df.index.day

    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    features["dom_sin"] = np.sin(2 * np.pi * day_of_month / 31)
    features["dom_cos"] = np.cos(2 * np.pi * day_of_month / 31)

    # ========== 2. MULTI-TIMEFRAME RETURNS (6 features) ==========
    features["returns_1h"] = df["close"].pct_change()
    features["returns_4h"] = df["close"].pct_change(4)
    features["returns_24h"] = df["close"].pct_change(24)
    features["returns_48h"] = df["close"].pct_change(48)
    features["returns_168h"] = df["close"].pct_change(168)  # 1 week

    # Volume momentum
    volume_ma_24h = df["volume"].rolling(24).mean()
    features["volume_ratio"] = df["volume"] / volume_ma_24h

    # ========== 3. VOLATILITY REGIME (3 features) ==========
    volatility_pct = (df["close"].rolling(14).std() / df["close"]) * 100
    quantiles = volatility_pct.quantile([0.33, 0.67])

    regime = pd.Series(1, index=df.index)
    regime[volatility_pct <= quantiles.iloc[0]] = 0
    regime[volatility_pct >= quantiles.iloc[1]] = 2

    features["vol_regime_low"] = (regime == 0).astype(int)
    features["vol_regime_medium"] = (regime == 1).astype(int)
    features["vol_regime_high"] = (regime == 2).astype(int)

    features = features.fillna(0)
    feature_count = len(features.columns)
    print(f"    ✅ Added {feature_count} optimized features")

    return features


def add_smc_features(df):
    """Add Smart Money Concepts features for 1h timeframe with optimized parameters."""
    df_smc = df.reset_index()
    ohlc = df_smc[["open", "high", "low", "close"]]
    ohlcv = df_smc[["open", "high", "low", "close", "volume"]]

    # Parameters - FIXED
    SWING_LENGTH = 15
    JOIN_CONSECUTIVE_FVG = True
    CLOSE_BREAK = True
    CLOSE_MITIGATION = True
    LIQUIDITY_RANGE_PERCENT = 0.01

    features = {}

    # 1. Swing Highs and Lows - KEEP NaN!
    try:
        swing_highs_lows = smc.swing_highs_lows(ohlc, swing_length=SWING_LENGTH)
        if isinstance(swing_highs_lows, pd.DataFrame):
            # DON'T fillna(0) - NaN means "not a swing point"
            features["swing_highs_lows"] = swing_highs_lows
        else:
            features["swing_highs_lows"] = pd.DataFrame(
                np.nan, index=df_smc.index, columns=["HighLow", "Level"]
            )
    except Exception as e:
        print(f"    ⚠️ Swing detection error: {e}")
        features["swing_highs_lows"] = pd.DataFrame(
            np.nan, index=df_smc.index, columns=["HighLow", "Level"]
        )

    # 2. Fair Value Gap
    try:
        fvg = smc.fvg(ohlc, join_consecutive=JOIN_CONSECUTIVE_FVG)
        if isinstance(fvg, pd.DataFrame):
            fvg = fvg.fillna(0)
            features["fvg"] = fvg
        else:
            features["fvg"] = pd.DataFrame(
                0, index=df_smc.index, columns=["FVG", "Top", "Bottom", "MitigatedIndex"]
            )
    except Exception as e:
        print(f"    ⚠️ FVG error: {e}")
        features["fvg"] = pd.DataFrame(
            0, index=df_smc.index, columns=["FVG", "Top", "Bottom", "MitigatedIndex"]
        )

    # 3. BOS & CHOCH - KEEP NaN AS NaN, DON'T FILLNA!
    try:
        bos_choch = smc.bos_choch(
            ohlc, features["swing_highs_lows"], close_break=CLOSE_BREAK
        )
        if isinstance(bos_choch, pd.DataFrame):
            # CRITICAL: DON'T fillna(0) for BOS/CHOCH!
            features["bos_choch"] = bos_choch
        else:
            features["bos_choch"] = pd.DataFrame(
                np.nan, index=df_smc.index, columns=["BOS", "CHOCH", "Level", "BrokenIndex"]
            )
    except Exception as e:
        print(f"    ⚠️ BOS/CHOCH error: {e}")
        features["bos_choch"] = pd.DataFrame(
            np.nan, index=df_smc.index, columns=["BOS", "CHOCH", "Level", "BrokenIndex"]
        )

    # 4. Order Blocks
    try:
        order_block = smc.ob(
            ohlcv, features["swing_highs_lows"], close_mitigation=CLOSE_MITIGATION
        )
        if isinstance(order_block, pd.DataFrame):
            order_block = order_block.fillna(0)
            features["order_block"] = order_block
        else:
            features["order_block"] = pd.DataFrame(
                0, index=df_smc.index,
                columns=["OB", "Top", "Bottom", "OBVolume", "Percentage"]
            )
    except Exception as e:
        print(f"    ⚠️ Order Block error: {e}")
        features["order_block"] = pd.DataFrame(
            0, index=df_smc.index,
            columns=["OB", "Top", "Bottom", "OBVolume", "Percentage"]
        )

    # 5. Liquidity
    try:
        liquidity = smc.liquidity(
            ohlc, features["swing_highs_lows"], range_percent=LIQUIDITY_RANGE_PERCENT
        )
        if isinstance(liquidity, pd.DataFrame):
            liquidity = liquidity.fillna(0)
            features["liquidity"] = liquidity
        else:
            features["liquidity"] = pd.DataFrame(
                0, index=df_smc.index, columns=["Liquidity", "Level", "End", "Swept"]
            )
    except Exception as e:
        print(f"    ⚠️ Liquidity error: {e}")
        features["liquidity"] = pd.DataFrame(
            0, index=df_smc.index, columns=["Liquidity", "Level", "End", "Swept"]
        )

    # 6. Retracements
    try:
        retracements = smc.retracements(ohlc, features["swing_highs_lows"])
        if isinstance(retracements, pd.DataFrame):
            retracements = retracements.fillna(0)
            features["retracements"] = retracements
        else:
            features["retracements"] = pd.DataFrame(
                0, index=df_smc.index,
                columns=["Direction", "CurrentRetracement%", "DeepestRetracement%"]
            )
    except Exception as e:
        print(f"    ⚠️ Retracements error: {e}")
        features["retracements"] = pd.DataFrame(
            0, index=df_smc.index,
            columns=["Direction", "CurrentRetracement%", "DeepestRetracement%"]
        )

    # 7. Technical Indicators
    features["technical_indicators"] = add_technical_indicators(df)

    # 8. Optimized Features
    features["optimized_features"] = add_optimized_features(df)

    return features


def get_last_smc_timestamp(table_name):
    """Get the timestamp of the last SMC record for a specific table."""
    with sqlite3.connect(DB_PATH) as conn:
        if table_exists(conn, table_name):
            ts = conn.execute(f"SELECT MAX(timestamp) FROM {table_name}").fetchone()[0]
            return pd.Timestamp(ts, tz="UTC") if ts else None
        return None


def all_smc_tables_exist(timeframe):
    """Check if all SMC tables exist for a specific timeframe."""
    smc_tables = get_smc_tables(timeframe)
    with sqlite3.connect(DB_PATH) as conn:
        return all(table_exists(conn, table) for table in smc_tables.values())


def clean_smc_duplicate_rows(timeframe):
    """Remove duplicate rows from SMC tables for specific timeframe."""
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


def print_smc_summary(features):
    """Print concise summary of detected SMC features - FIXED VERSION."""
    summary = []

    # Swing Points
    if "swing_highs_lows" in features:
        swings = features["swing_highs_lows"]
        swing_count = int((swings["HighLow"].notna() & (swings["HighLow"] != 0)).sum())
        summary.append(f"Swings: {swing_count}")

    # FVG
    if "fvg" in features:
        fvg = features["fvg"]
        bullish = int((fvg["FVG"] == 1).sum())
        bearish = int((fvg["FVG"] == -1).sum())
        summary.append(f"FVG: {bullish}↑ {bearish}↓")

    # BOS/CHOCH - CORRECT COUNTING METHOD
    if "bos_choch" in features:
        bc = features["bos_choch"]

        # Ensure columns are numeric type
        if "BOS" in bc.columns:
            bc["BOS"] = pd.to_numeric(bc["BOS"], errors='coerce')
            bos_count = int((bc["BOS"].notna() & (bc["BOS"] != 0)).sum())
        else:
            bos_count = 0

        if "CHOCH" in bc.columns:
            bc["CHOCH"] = pd.to_numeric(bc["CHOCH"], errors='coerce')
            choch_count = int((bc["CHOCH"].notna() & (bc["CHOCH"] != 0)).sum())
        else:
            choch_count = 0

        summary.append(f"BOS: {bos_count} | CHOCH: {choch_count}")

    # Order Blocks
    if "order_block" in features:
        ob = features["order_block"]
        bullish = int((ob["OB"] == 1).sum())
        bearish = int((ob["OB"] == -1).sum())
        summary.append(f"OB: {bullish}↑ {bearish}↓")

    # Liquidity
    if "liquidity" in features:
        liq = features["liquidity"]
        bullish = int((liq["Liquidity"] == 1).sum())
        bearish = int((liq["Liquidity"] == -1).sum())
        summary.append(f"Liq: {bullish}↑ {bearish}↓")

    # Optimized Features
    if "optimized_features" in features:
        opt = features["optimized_features"]
        summary.append(f"Opt: {len(opt.columns)} features")

    return " | ".join(summary)


def sync_smc_features_for_timeframe(timeframe):
    """Synchronize SMC features with latest OHLCV data for 1h timeframe."""
    print(f"\n{'─' * 60}")
    print(f"⚙️  Processing {timeframe.upper()}")
    print(f"{'─' * 60}")

    # Load OHLCV data
    try:
        df_ohlcv = load_ohlcv(timeframe)
    except ValueError as e:
        print(f"❌ {e}")
        return False

    ohlcv_count = len(df_ohlcv)
    start_date = df_ohlcv.index[0].strftime('%Y-%m-%d')
    end_date = df_ohlcv.index[-1].strftime('%Y-%m-%d')
    print(f"📊 OHLCV: {ohlcv_count:,} rows ({start_date} to {end_date})")

    # Get SMC tables for this timeframe
    smc_tables = get_smc_tables(timeframe)

    # Check if SMC tables exist
    smc_complete = all_smc_tables_exist(timeframe)

    # Get the earliest last timestamp from all SMC tables
    last_timestamps = []
    for table in smc_tables.values():
        ts = get_last_smc_timestamp(table)
        if ts:
            last_timestamps.append(ts)

    last_ts = min(last_timestamps) if last_timestamps else None

    # Check if we need full sync
    needs_full_sync = last_ts is None or not smc_complete

    if needs_full_sync:
        print(f"🔄 Full sync required (SMC tables empty/incomplete)")
        print(f"⏳ Processing {ohlcv_count:,} candles...")

        # Full sync
        smc_features = add_smc_features(df_ohlcv)

        # Print summary
        print(f"✅ {print_smc_summary(smc_features)}")

        # Save SMC features to database
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

        print(f"💾 Saved {saved_count}/{len(smc_tables)} tables")
        return True

    # Check for new data
    new_ohlcv_data = df_ohlcv[df_ohlcv.index > last_ts]

    if new_ohlcv_data.empty:
        print(f"✅ Up to date (Last: {last_ts.strftime('%Y-%m-%d %H:%M')})")
        return True

    print(f"🔄 Incremental sync: {len(new_ohlcv_data)} new candles")
    print(f"⏳ Processing...")

    # Incremental sync for SMC features
    smc_features = add_smc_features(new_ohlcv_data)

    # Print summary
    print(f"✅ {print_smc_summary(smc_features)}")

    # Save
    saved_count = 0
    for key, table in smc_tables.items():
        if key in smc_features and isinstance(smc_features[key], pd.DataFrame):
            df_smc = smc_features[key]
            if "timestamp" not in df_smc.columns:
                df_smc["timestamp"] = new_ohlcv_data.index
            df_smc.set_index("timestamp", inplace=True)
            save_to_db(df_smc, table, mode="append")
            saved_count += 1

    # Clean up duplicate rows
    clean_smc_duplicate_rows(timeframe)
    print(f"💾 Saved {saved_count}/{len(smc_tables)} tables")
    return True


def sync_all_smc_features():
    """Synchronize SMC features for all available timeframes."""
    available_tables = get_available_ohlcv_tables()

    if not available_tables:
        print("❌ No OHLCV tables available!")
        return

    print(f"\n{'═' * 60}")
    print(f"🚀 SMC FEATURE ENGINEERING (TYPE-SAFE VERSION)")
    print(f"{'═' * 60}")
    print(f"📂 Database: {DB_PATH}")
    print(f"🎯 Timeframes: {', '.join(available_tables)}")
    print(f"✨ Features: SMC + Technical + Optimized (~34 total)")
    print(f"🔧 Parameters: swing_length=24, close_break=True")
    print(f"🛡️  Type Safety: Explicit numeric conversion on save/load")

    success_count = 0
    for table in available_tables:
        if sync_smc_features_for_timeframe(table):
            success_count += 1

    print(f"\n{'═' * 60}")
    print(f"✅ COMPLETED: {success_count}/{len(available_tables)} timeframes")
    print(f"{'═' * 60}\n")


def verify_smc_sync():
    """Verify SMC synchronization status for all timeframes."""
    available_tables = get_available_ohlcv_tables()

    print(f"\n{'═' * 60}")
    print("📋 VERIFICATION REPORT (WITH TYPE CHECK)")
    print(f"{'═' * 60}")

    for timeframe in available_tables:
        print(f"\n📈 {timeframe.upper()}")
        print("─" * 40)

        try:
            df_ohlcv = load_ohlcv(timeframe)
            ohlcv_count = len(df_ohlcv)

            # Check each SMC table
            smc_tables = get_smc_tables(timeframe)
            all_synced = True

            with sqlite3.connect(DB_PATH) as conn:
                for key, table in smc_tables.items():
                    if table_exists(conn, table):
                        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        diff = ohlcv_count - count

                        # Load and check data types for BOS/CHOCH table
                        if key == "bos_choch":
                            df_test = load_smc_table(table)
                            if df_test is not None:
                                bos_dtype = df_test["BOS"].dtype if "BOS" in df_test.columns else "N/A"
                                bos_valid = (df_test["BOS"].notna() & (df_test["BOS"] != 0)).sum() if "BOS" in df_test.columns else 0
                                status_msg = f"{count:,} rows (diff: {diff}) | BOS dtype: {bos_dtype} | Valid: {bos_valid}"
                            else:
                                status_msg = f"{count:,} rows (diff: {diff})"
                        else:
                            status_msg = f"{count:,} rows (diff: {diff})"

                        if diff == 0:
                            status = "✅"
                        else:
                            status = "⚠️"
                            all_synced = False

                        print(f"{status} {key:20s}: {status_msg}")
                    else:
                        print(f"❌ {key:20s}: Table not found")
                        all_synced = False

            if all_synced:
                print(f"\n✅ All tables in sync!")
            else:
                print(f"\n⚠️ Sync required")

        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function to run the SMC feature engineering pipeline."""
    try:
        sync_all_smc_features()
        verify_smc_sync()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
