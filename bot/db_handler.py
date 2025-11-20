import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), "data", "ohlcv.db")  # Path DB

def get_latest_price():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT close FROM ohlcv ORDER BY timestamp DESC LIMIT 1"
    price = pd.read_sql(query, conn).values[0][0]
    conn.close()
    return price

def get_last_24h_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT timestamp, open, high, low, close
    FROM ohlcv 
    WHERE timestamp >= (SELECT MAX(timestamp) - 86400 FROM ohlcv)
    """
    df = pd.read_sql(query, conn, parse_dates=["timestamp"])
    conn.close()
    return df

def get_latest_prediction():
    """Mendapatkan hasil prediksi terakhir dari database"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        query = """
        SELECT timestamp, signal, confidence, prob_buy, prob_sell, prob_hold
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT 1
        """
        pred_df = pd.read_sql(query, conn)
        
        # Coba dapatkan evaluasi SMC terkait jika ada
        if not pred_df.empty:
            timestamp = pred_df['timestamp'].iloc[0]
            query_smc = """
            SELECT timestamp, model_signal, smc_confidence, setup_type, supporting_factors, conflicting_factors
            FROM smc_evaluations
            WHERE timestamp = ?
            """
            smc_df = pd.read_sql(query_smc, conn, params=(timestamp,))
            conn.close()
            
            if not smc_df.empty:
                return {
                    'model': {
                        'timestamp': pred_df['timestamp'].iloc[0],
                        'signal': pred_df['signal'].iloc[0],
                        'confidence': pred_df['confidence'].iloc[0],
                        'probabilities': {
                            'buy': pred_df['prob_buy'].iloc[0],
                            'sell': pred_df['prob_sell'].iloc[0],
                            'hold': pred_df['prob_hold'].iloc[0]
                        }
                    },
                    'smc': {
                        'confidence': smc_df['smc_confidence'].iloc[0],
                        'setup_type': smc_df['setup_type'].iloc[0],
                        'supporting_factors': smc_df['supporting_factors'].iloc[0].split('|') if smc_df['supporting_factors'].iloc[0] else [],
                        'conflicting_factors': smc_df['conflicting_factors'].iloc[0].split('|') if smc_df['conflicting_factors'].iloc[0] else []
                    }
                }
            else:
                # Hanya return hasil prediksi model jika tidak ada evaluasi SMC
                return {
                    'model': {
                        'timestamp': pred_df['timestamp'].iloc[0],
                        'signal': pred_df['signal'].iloc[0],
                        'confidence': pred_df['confidence'].iloc[0],
                        'probabilities': {
                            'buy': pred_df['prob_buy'].iloc[0],
                            'sell': pred_df['prob_sell'].iloc[0],
                            'hold': pred_df['prob_hold'].iloc[0]
                        }
                    },
                    'smc': None
                }
        else:
            conn.close()
            return None
    except Exception as e:
        print(f"Error memuat prediksi: {e}")
        return None