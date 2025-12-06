import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "data/btc_ohlcv.db"


def get_latest_price():
    """Get harga BTC terkini dari database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = "SELECT close FROM btc_1h ORDER BY timestamp DESC LIMIT 1"
            result = conn.execute(query).fetchone()
            return result[0] if result else None
    except Exception as e:
        print(f"❌ Error getting latest price: {e}")
        return None


def get_latest_prediction():
    """
    Get prediksi terbaru dari MODEL dan SMC

    Returns:
    {
        'model': {
            'timestamp': str,
            'signal': str (UP/DOWN/HOLD),
            'confidence': float,
            'probabilities': {...},
            'close_price': float,
            'sequence_length': int,
            'model_type': str
        },
        'smc': {
            'timestamp': str,
            'smc_signal': str (Buy/Sell/Hold),
            'entry_price': float,
            'stop_loss': float,
            'take_profit_1': float,
            'take_profit_2': float,
            'take_profit_3': float,
            'risk_reward': float,
            'setup_quality': str,
            'confidence': float,
            'setup_type': str,
            'supporting_factors': str,
            'conflicting_factors': str,
            'invalidation_level': float
        },
        'aligned': bool (True jika Model dan SMC sama-sama BUY/SELL)
    }
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Get latest model prediction
            model_query = """
                SELECT
                    timestamp,
                    prediction as signal,
                    confidence,
                    prob_down,
                    prob_hold,
                    prob_up,
                    close_price,
                    sequence_length,
                    model_type
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT 1
            """
            model_result = conn.execute(model_query).fetchone()

            if not model_result:
                return None

            # Get latest SMC signal
            smc_query = """
                SELECT
                    timestamp,
                    smc_signal,
                    entry_price,
                    stop_loss,
                    take_profit_1,
                    take_profit_2,
                    take_profit_3,
                    risk_reward,
                    setup_quality,
                    confidence,
                    setup_type,
                    supporting_factors,
                    conflicting_factors,
                    invalidation_level
                FROM smc_signals
                ORDER BY timestamp DESC
                LIMIT 1
            """
            smc_result = conn.execute(smc_query).fetchone()

            if not smc_result:
                return None

            # Parse model prediction
            model_data = {
                'timestamp': model_result[0],
                'signal': model_result[1],  # UP/DOWN/HOLD
                'confidence': model_result[2],
                'probabilities': {
                    'down': model_result[3],
                    'hold': model_result[4],
                    'up': model_result[5]
                },
                'close_price': model_result[6],
                'sequence_length': model_result[7],
                'model_type': model_result[8]
            }

            # Parse SMC signal
            smc_data = {
                'timestamp': smc_result[0],
                'smc_signal': smc_result[1],  # Buy/Sell/Hold
                'entry_price': smc_result[2],
                'stop_loss': smc_result[3],
                'take_profit_1': smc_result[4],
                'take_profit_2': smc_result[5],
                'take_profit_3': smc_result[6],
                'risk_reward': smc_result[7],
                'setup_quality': smc_result[8],
                'confidence': smc_result[9],
                'setup_type': smc_result[10],
                'supporting_factors': smc_result[11],
                'conflicting_factors': smc_result[12],
                'invalidation_level': smc_result[13]
            }

            # Check alignment (CRITICAL LOGIC)
            aligned = check_signal_alignment(
                model_data['signal'],
                smc_data['smc_signal']
            )

            return {
                'model': model_data,
                'smc': smc_data,
                'aligned': aligned
            }

    except Exception as e:
        print(f"❌ Error getting latest prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_signal_alignment(model_signal, smc_signal):
    """
    Check apakah model dan SMC aligned (sama-sama BUY atau SELL)

    Logic:
    - Model UP + SMC Buy = ALIGNED ✅
    - Model DOWN + SMC Sell = ALIGNED ✅
    - Selain itu = NOT ALIGNED ❌

    Args:
        model_signal: UP/DOWN/HOLD (dari model prediction)
        smc_signal: Buy/Sell/Hold (dari SMC)

    Returns:
        bool: True jika aligned, False jika tidak
    """
    model_signal = model_signal.upper() if model_signal else ""
    smc_signal = smc_signal.upper() if smc_signal else ""

    # Case 1: Both BUY
    if model_signal == "UP" and smc_signal == "BUY":
        return True

    # Case 2: Both SELL
    if model_signal == "DOWN" and smc_signal == "SELL":
        return True

    # All other cases: NOT ALIGNED
    return False


def get_prediction_history(limit=20):
    """Get history prediksi dengan alignment status"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Join predictions dengan smc_signals
            query = f"""
                SELECT
                    p.timestamp,
                    p.prediction as model_signal,
                    p.confidence as model_confidence,
                    s.smc_signal,
                    s.confidence as smc_confidence,
                    s.setup_quality,
                    s.risk_reward,
                    p.close_price
                FROM predictions p
                LEFT JOIN smc_signals s ON
                    datetime(p.timestamp) = datetime(s.timestamp)
                ORDER BY p.timestamp DESC
                LIMIT {limit}
            """

            df = pd.read_sql_query(query, conn)

            if df.empty:
                return None

            # Add alignment column
            df['aligned'] = df.apply(
                lambda row: check_signal_alignment(
                    row['model_signal'],
                    row['smc_signal']
                ),
                axis=1
            )

            return df

    except Exception as e:
        print(f"❌ Error getting prediction history: {e}")
        return None


def get_market_stats():
    """Get statistik market terkini"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Get latest 24h data
            query = """
                SELECT
                    MIN(low) as low_24h,
                    MAX(high) as high_24h,
                    AVG(volume) as avg_volume_24h,
                    close
                FROM btc_1h
                ORDER BY timestamp DESC
                LIMIT 24
            """

            result = conn.execute(query).fetchone()

            if not result:
                return None

            # Get current price
            current_price = result[3]

            # Get price 24h ago
            query_24h = """
                SELECT close
                FROM btc_1h
                ORDER BY timestamp DESC
                LIMIT 1 OFFSET 24
            """
            price_24h = conn.execute(query_24h).fetchone()
            price_24h = price_24h[0] if price_24h else current_price

            price_change_24h = current_price - price_24h
            price_change_pct_24h = (price_change_24h / price_24h) * 100

            return {
                'current_price': current_price,
                'low_24h': result[0],
                'high_24h': result[1],
                'avg_volume_24h': result[2],
                'price_change_24h': price_change_24h,
                'price_change_pct_24h': price_change_pct_24h
            }

    except Exception as e:
        print(f"❌ Error getting market stats: {e}")
        return None


def count_aligned_signals(hours=24):
    """Count berapa kali signal aligned dalam N jam terakhir"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = f"""
                SELECT
                    p.prediction as model_signal,
                    s.smc_signal
                FROM predictions p
                LEFT JOIN smc_signals s ON
                    datetime(p.timestamp) = datetime(s.timestamp)
                WHERE p.timestamp >= datetime('now', '-{hours} hours')
                ORDER BY p.timestamp DESC
            """

            df = pd.read_sql_query(query, conn)

            if df.empty:
                return 0

            # Count aligned signals
            aligned_count = df.apply(
                lambda row: check_signal_alignment(
                    row['model_signal'],
                    row['smc_signal']
                ),
                axis=1
            ).sum()

            return aligned_count

    except Exception as e:
        print(f"❌ Error counting aligned signals: {e}")
        return 0


if __name__ == "__main__":
    # Test functions
    print("=" * 60)
    print("🧪 TESTING DATABASE HANDLER")
    print("=" * 60)

    print("\n1️⃣ Testing get_latest_price()...")
    price = get_latest_price()
    print(f"   Current Price: ${price:,.2f}" if price else "   ❌ No price data")

    print("\n2️⃣ Testing get_latest_prediction()...")
    pred = get_latest_prediction()
    if pred:
        print(f"   Model: {pred['model']['signal']} ({pred['model']['confidence']*100:.1f}%)")
        print(f"   SMC: {pred['smc']['smc_signal']} ({pred['smc']['confidence']:.1f}%)")
        print(f"   Aligned: {'✅ YES' if pred['aligned'] else '❌ NO'}")
    else:
        print("   ❌ No prediction data")

    print("\n3️⃣ Testing get_market_stats()...")
    stats = get_market_stats()
    if stats:
        print(f"   24h High: ${stats['high_24h']:,.2f}")
        print(f"   24h Low: ${stats['low_24h']:,.2f}")
        print(f"   24h Change: {stats['price_change_pct_24h']:+.2f}%")
    else:
        print("   ❌ No market stats")

    print("\n4️⃣ Testing count_aligned_signals()...")
    count = count_aligned_signals(24)
    print(f"   Aligned signals (24h): {count}")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
