import sqlite3
import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = "data/ohlcv.db"
SMC_TABLES = {
    "swing_highs_lows": "smc_swing_highs_lows",
    "fvg": "smc_fvg",
    "bos_choch": "smc_bos_choch",
    "order_block": "smc_order_block",
    "liquidity": "smc_liquidity",
    "retracements": "smc_retracements"
}


class SimpleSMCEvaluator:
    def __init__(self, db_path: str = DB_PATH):
        """Initialize the SMC Evaluator for BTC data."""
        self.db_path = db_path
        self.recent_candles = 5  # Number of recent candles to consider
        self.smc_data = {}  # Dictionary to store data from each SMC table
        self.current_price = None
    
    def load_smc_data(self, limit: int = 30) -> bool:
        """Load SMC data from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current price
                df = pd.read_sql("SELECT close FROM ohlcv ORDER BY timestamp DESC LIMIT 1", conn)
                if not df.empty:
                    self.current_price = df['close'].iloc[0]
                
                # Load data from each SMC table
                for key, table in SMC_TABLES.items():
                    df = pd.read_sql(f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT ?", 
                                    conn, params=(limit,), parse_dates=['timestamp'])
                    self.smc_data[key] = df
                
                return all(not df.empty for df in self.smc_data.values())
                
        except Exception as e:
            logger.error(f"Error loading SMC data: {e}")
            return False
    
    def evaluate_signal(self, prediction: Dict) -> Dict:
        """Evaluate a trading signal using SMC principles."""
        if not self.smc_data or not all(not df.empty for df in self.smc_data.values() if df is not None):
            return self._create_empty_result(prediction['signal'])
        
        signal_type = prediction['signal']  # 'Buy', 'Sell', or 'Hold'
        
        # Initialize evaluation result
        confidence_score = 0.0
        supporting_factors = []
        conflicting_factors = []
        
        # Evaluate each SMC component
        confidence_score = self._evaluate_all_components(signal_type, supporting_factors, 
                                                        conflicting_factors, confidence_score)
        
        # Identify setup type
        setup_type = self._identify_setup_type(signal_type)
        
        # Cap confidence score between 0-100%
        confidence_score = max(0, min(confidence_score, 100))
        
        return {
            'original_signal': signal_type,
            'smc_confidence': round(confidence_score, 2),
            'setup_type': setup_type,
            'supporting_factors': supporting_factors,
            'conflicting_factors': conflicting_factors,
            'timestamp': prediction.get('timestamp', datetime.now().isoformat())
        }
    
    def _create_empty_result(self, signal_type: str) -> Dict:
        """Create an empty result when no data is available."""
        return {
            'original_signal': signal_type,
            'smc_confidence': 0.0,
            'setup_type': 'Unknown',
            'supporting_factors': [],
            'conflicting_factors': ["No SMC data available for evaluation"],
            'timestamp': datetime.now().isoformat()
        }
    
    def _evaluate_all_components(self, signal_type: str, supporting_factors: List[str], 
                                conflicting_factors: List[str], confidence_score: float) -> float:
        """Evaluate all SMC components in a single method."""
        signal_lower = signal_type.lower()
        is_buy = signal_lower == 'buy'
        is_sell = signal_lower == 'sell'
        
        # 1. Evaluate FVG (Fair Value Gap)
        if 'fvg' in self.smc_data and not self.smc_data['fvg'].empty:
            recent_fvg = self.smc_data['fvg'].head(self.recent_candles)
            
            # For Buy signals
            if is_buy:
                if (recent_fvg['FVG'] == 1).any():
                    confidence_score += 10
                    supporting_factors.append("Bullish Fair Value Gap detected")
                    
                    if (recent_fvg.head(3)['FVG'] == 1).any():
                        confidence_score += 5
                        supporting_factors.append("Recently formed Bullish FVG")
                
                # Check conflicting bearish FVG
                bearish_fvg = recent_fvg[(recent_fvg['FVG'] == -1) & (recent_fvg['MitigatedIndex'] == 0)]
                if not bearish_fvg.empty:
                    confidence_score -= 10
                    conflicting_factors.append("Unmitigated Bearish FVG present")
            
            # For Sell signals
            elif is_sell:
                if (recent_fvg['FVG'] == -1).any():
                    confidence_score += 10
                    supporting_factors.append("Bearish Fair Value Gap detected")
                    
                    if (recent_fvg.head(3)['FVG'] == -1).any():
                        confidence_score += 5
                        supporting_factors.append("Recently formed Bearish FVG")
                
                # Check conflicting bullish FVG
                bullish_fvg = recent_fvg[(recent_fvg['FVG'] == 1) & (recent_fvg['MitigatedIndex'] == 0)]
                if not bullish_fvg.empty:
                    confidence_score -= 10
                    conflicting_factors.append("Unmitigated Bullish FVG present")
        
        # 2. Evaluate Swing Points
        if 'swing_highs_lows' in self.smc_data and not self.smc_data['swing_highs_lows'].empty and self.current_price:
            recent_swings = self.smc_data['swing_highs_lows'].head(10)
            
            # For Buy signals
            if is_buy:
                # Check if price is near a swing low
                swing_lows = recent_swings[recent_swings['HighLow'] == -1]
                if not swing_lows.empty:
                    nearest_low = swing_lows.iloc[0]['Level']
                    if abs(self.current_price - nearest_low) / nearest_low < 0.01:  # Within 1%
                        confidence_score += 15
                        supporting_factors.append("Price near swing low")
                
                # Check if price broke above a swing high
                swing_highs = recent_swings[recent_swings['HighLow'] == 1]
                if not swing_highs.empty and self.current_price > swing_highs.iloc[0]['Level']:
                    confidence_score += 10
                    supporting_factors.append("Price above recent swing high")
            
            # For Sell signals  
            elif is_sell:
                # Check if price is near a swing high
                swing_highs = recent_swings[recent_swings['HighLow'] == 1]
                if not swing_highs.empty:
                    nearest_high = swing_highs.iloc[0]['Level']
                    if abs(self.current_price - nearest_high) / self.current_price < 0.01:  # Within 1%
                        confidence_score += 15
                        supporting_factors.append("Price near swing high")
                
                # Check if price broke below a swing low
                swing_lows = recent_swings[recent_swings['HighLow'] == -1]
                if not swing_lows.empty and self.current_price < swing_lows.iloc[0]['Level']:
                    confidence_score += 10
                    supporting_factors.append("Price below recent swing low")
        
        # 3. Evaluate BOS/CHOCH (Break of Structure/Change of Character)
        if 'bos_choch' in self.smc_data and not self.smc_data['bos_choch'].empty:
            recent_bos = self.smc_data['bos_choch'].head(self.recent_candles)
            
            # For Buy signals
            if is_buy:
                # Check for bullish BOS
                if (recent_bos['BOS'] == 1).any():
                    confidence_score += 20
                    supporting_factors.append("Bullish Break of Structure")
                
                # Check for bullish CHOCH
                if (recent_bos['CHOCH'] == 1).any():
                    confidence_score += 25
                    supporting_factors.append("Bullish Change of Character")
                
                # Check for bearish structures as conflicting factors
                if (recent_bos['BOS'] == -1).any() or (recent_bos['CHOCH'] == -1).any():
                    confidence_score -= 15
                    conflicting_factors.append("Recent bearish structure break")
            
            # For Sell signals
            elif is_sell:
                # Check for bearish BOS
                if (recent_bos['BOS'] == -1).any():
                    confidence_score += 20
                    supporting_factors.append("Bearish Break of Structure")
                
                # Check for bearish CHOCH
                if (recent_bos['CHOCH'] == -1).any():
                    confidence_score += 25
                    supporting_factors.append("Bearish Change of Character")
                
                # Check for bullish structures as conflicting factors
                if (recent_bos['BOS'] == 1).any() or (recent_bos['CHOCH'] == 1).any():
                    confidence_score -= 15
                    conflicting_factors.append("Recent bullish structure break")
        
        # 4. Evaluate Order Blocks
        if 'order_block' in self.smc_data and not self.smc_data['order_block'].empty and self.current_price:
            recent_obs = self.smc_data['order_block'].head(10)
            
            # For Buy signals
            if is_buy:
                bullish_obs = recent_obs[recent_obs['OB'] == 1]
                if not bullish_obs.empty:
                    # High volume OBs
                    if (bullish_obs['OBVolume'] > bullish_obs['OBVolume'].median()).any():
                        confidence_score += 15
                        supporting_factors.append("High volume bullish Order Block")
                    
                    # Strong OBs
                    if (bullish_obs['Percentage'] > 70).any():
                        confidence_score += 5
                        supporting_factors.append("Strong bullish Order Block (>70%)")
                    
                    # Price near OB
                    nearest_ob = bullish_obs.iloc[0]
                    if 'Bottom' in nearest_ob and 'Top' in nearest_ob:
                        if nearest_ob['Bottom'] <= self.current_price <= nearest_ob['Top']:
                            confidence_score += 10
                            supporting_factors.append("Price near bullish Order Block")
            
            # For Sell signals
            elif is_sell:
                bearish_obs = recent_obs[recent_obs['OB'] == -1]
                if not bearish_obs.empty:
                    # High volume OBs
                    if (bearish_obs['OBVolume'] > bearish_obs['OBVolume'].median()).any():
                        confidence_score += 15
                        supporting_factors.append("High volume bearish Order Block")
                    
                    # Strong OBs
                    if (bearish_obs['Percentage'] > 70).any():
                        confidence_score += 5
                        supporting_factors.append("Strong bearish Order Block (>70%)")
                    
                    # Price near OB
                    nearest_ob = bearish_obs.iloc[0]
                    if 'Bottom' in nearest_ob and 'Top' in nearest_ob:
                        if nearest_ob['Bottom'] <= self.current_price <= nearest_ob['Top']:
                            confidence_score += 10
                            supporting_factors.append("Price near bearish Order Block")
        
        # 5. Evaluate Liquidity
        if 'liquidity' in self.smc_data and not self.smc_data['liquidity'].empty:
            recent_liq = self.smc_data['liquidity'].head(self.recent_candles)
            
            # For Buy signals
            if is_buy:
                bullish_swept = recent_liq[(recent_liq['Liquidity'] == 1) & (recent_liq['Swept'] == 1)]
                if not bullish_swept.empty:
                    confidence_score += 15
                    supporting_factors.append("Recently swept bullish liquidity")
                else:
                    confidence_score -= 5
                    conflicting_factors.append("No swept bullish liquidity")
            
            # For Sell signals
            elif is_sell:
                bearish_swept = recent_liq[(recent_liq['Liquidity'] == -1) & (recent_liq['Swept'] == 1)]
                if not bearish_swept.empty:
                    confidence_score += 15
                    supporting_factors.append("Recently swept bearish liquidity")
                else:
                    confidence_score -= 5
                    conflicting_factors.append("No swept bearish liquidity")
        
        # 6. Evaluate Retracements
        if 'retracements' in self.smc_data and not self.smc_data['retracements'].empty:
            recent_retr = self.smc_data['retracements'].head(self.recent_candles)
            
            # For Buy signals
            if is_buy:
                up_retracements = recent_retr[recent_retr['Direction'] == 1]
                if not up_retracements.empty:
                    current_retr = up_retracements.iloc[0]['CurrentRetracement%']
                    if 38 <= current_retr <= 62:
                        confidence_score += 10
                        supporting_factors.append(f"Bullish retracement in ideal range ({current_retr:.1f}%)")
                    elif current_retr > 70:
                        confidence_score -= 5
                        conflicting_factors.append(f"Deeper than ideal bullish retracement ({current_retr:.1f}%)")
            
            # For Sell signals
            elif is_sell:
                down_retracements = recent_retr[recent_retr['Direction'] == -1]
                if not down_retracements.empty:
                    current_retr = down_retracements.iloc[0]['CurrentRetracement%']
                    if 38 <= current_retr <= 62:
                        confidence_score += 10
                        supporting_factors.append(f"Bearish retracement in ideal range ({current_retr:.1f}%)")
                    elif current_retr > 70:
                        confidence_score -= 5
                        conflicting_factors.append(f"Deeper than ideal bearish retracement ({current_retr:.1f}%)")
        
        return confidence_score
    
    def _identify_setup_type(self, signal_type: str) -> str:
        """Identify the trading setup type based on SMC indicators."""
        signal_lower = signal_type.lower()
        
        # Get indicator statuses
        bull_indicators = self._has_bullish_indicators()
        bear_indicators = self._has_bearish_indicators()
        
        # Check for setup types
        if signal_lower == 'buy':
            if (bull_indicators['swing_low'] and 
                bull_indicators['bullish_ob'] and 
                (bull_indicators['bullish_bos'] or bull_indicators['bullish_choch'])):
                return "Bull Reversal"
            
            if (bull_indicators['bullish_fvg'] and 
                self._has_bullish_retracement() and 
                self._has_bullish_liquidity()):
                return "Bull Continuation"
        
        elif signal_lower == 'sell':
            if (bear_indicators['swing_high'] and 
                bear_indicators['bearish_ob'] and 
                (bear_indicators['bearish_bos'] or bear_indicators['bearish_choch'])):
                return "Bear Reversal"
            
            if (bear_indicators['bearish_fvg'] and 
                self._has_bearish_retracement() and 
                self._has_bearish_liquidity()):
                return "Bear Continuation"
        
        # Check for range-bound conditions
        if (self._has_no_bos() and 
            self._has_no_choch() and 
            bull_indicators['swing_low'] and 
            bear_indicators['swing_high']):
            return "Range Bound"
        
        # Check for consolidation
        if self._has_shallow_retracement() and self._has_no_significant_fvg():
            return "Consolidation"
        
        # Default
        return "Undefined"
    
    # Refactored helper methods to avoid ambiguous DataFrame boolean operations
    def _has_bullish_indicators(self):
        """Check for presence of bullish indicators, returning dict of booleans."""
        return {
            'swing_low': self._check_data('swing_highs_lows', 'HighLow', -1),
            'bullish_ob': self._check_data('order_block', 'OB', 1),
            'bullish_bos': self._check_data('bos_choch', 'BOS', 1),
            'bullish_choch': self._check_data('bos_choch', 'CHOCH', 1),
            'bullish_fvg': self._check_data('fvg', 'FVG', 1)
        }
    
    def _has_bearish_indicators(self):
        """Check for presence of bearish indicators, returning dict of booleans."""
        return {
            'swing_high': self._check_data('swing_highs_lows', 'HighLow', 1),
            'bearish_ob': self._check_data('order_block', 'OB', -1),
            'bearish_bos': self._check_data('bos_choch', 'BOS', -1),
            'bearish_choch': self._check_data('bos_choch', 'CHOCH', -1),
            'bearish_fvg': self._check_data('fvg', 'FVG', -1)
        }
    
    # Helper methods
    def _check_data(self, table_key: str, column: str, value: int) -> bool:
        """Check if a condition exists in the data table."""
        if table_key not in self.smc_data or self.smc_data[table_key].empty or column not in self.smc_data[table_key].columns:
            return False
        return (self.smc_data[table_key][column] == value).any()
    
    # Specialized helper methods
    def _has_bullish_retracement(self) -> bool:
        if 'retracements' not in self.smc_data or self.smc_data['retracements'].empty:
            return False
        return (self.smc_data['retracements']['Direction'] == 1).any()
    
    def _has_bearish_retracement(self) -> bool:
        if 'retracements' not in self.smc_data or self.smc_data['retracements'].empty:
            return False
        return (self.smc_data['retracements']['Direction'] == -1).any()
    
    def _has_bullish_liquidity(self) -> bool:
        if 'liquidity' not in self.smc_data or self.smc_data['liquidity'].empty:
            return False
        liquidity = self.smc_data['liquidity']
        return ((liquidity['Liquidity'] == 1) & (liquidity['Swept'] == 1)).any()
    
    def _has_bearish_liquidity(self) -> bool:
        if 'liquidity' not in self.smc_data or self.smc_data['liquidity'].empty:
            return False
        liquidity = self.smc_data['liquidity']
        return ((liquidity['Liquidity'] == -1) & (liquidity['Swept'] == 1)).any()
    
    def _has_no_bos(self) -> bool:
        if 'bos_choch' not in self.smc_data or self.smc_data['bos_choch'].empty:
            return True
        recent_data = self.smc_data['bos_choch'].head(self.recent_candles)
        if 'BOS' not in recent_data.columns:
            return True
        return ((recent_data['BOS'] == 0) | recent_data['BOS'].isna()).all()
    
    def _has_no_choch(self) -> bool:
        if 'bos_choch' not in self.smc_data or self.smc_data['bos_choch'].empty:
            return True
        recent_data = self.smc_data['bos_choch'].head(self.recent_candles)
        if 'CHOCH' not in recent_data.columns:
            return True
        return ((recent_data['CHOCH'] == 0) | recent_data['CHOCH'].isna()).all()
    
    def _has_shallow_retracement(self) -> bool:
        if 'retracements' not in self.smc_data or self.smc_data['retracements'].empty:
            return False
        recent_data = self.smc_data['retracements'].head(self.recent_candles)
        if 'CurrentRetracement%' not in recent_data.columns:
            return False
        return (recent_data['CurrentRetracement%'] < 30).any()
    
    def _has_no_significant_fvg(self) -> bool:
        if 'fvg' not in self.smc_data or self.smc_data['fvg'].empty:
            return True
        recent_data = self.smc_data['fvg'].head(self.recent_candles)
        if 'FVG' not in recent_data.columns:
            return True
        return ((recent_data['FVG'] == 0) | recent_data['FVG'].isna()).all()


def validate_signal(prediction: Dict) -> Dict:
    """Validate a trading signal against SMC principles."""
    try:
        evaluator = SimpleSMCEvaluator()
        
        # Load SMC data
        data_loaded = evaluator.load_smc_data()
        if not data_loaded:
            logger.warning("Failed to load SMC data")
        
        # Evaluate the signal
        result = evaluator.evaluate_signal(prediction)
        
        logger.info(f"SMC Evaluation: {prediction['signal']} signal, {result['smc_confidence']}% confidence")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during SMC validation: {e}")
        return {
            'original_signal': prediction['signal'],
            'smc_confidence': 0.0,
            'setup_type': 'Error',
            'supporting_factors': [],
            'conflicting_factors': [f"Error: {str(e)}"],
            'timestamp': prediction.get('timestamp', datetime.now().isoformat())
        }


def save_evaluation(evaluation: Dict, db_path: str = DB_PATH) -> bool:
    """Save the SMC evaluation result to database."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create evaluations table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS smc_evaluations (
                timestamp TEXT PRIMARY KEY,
                model_signal TEXT,
                smc_confidence REAL,
                setup_type TEXT,
                supporting_factors TEXT,
                conflicting_factors TEXT
            )
            ''')
            
            # Convert lists to string representation
            supporting_str = "|".join(evaluation['supporting_factors'])
            conflicting_str = "|".join(evaluation['conflicting_factors'])
            
            # Save the evaluation
            cursor.execute('''
            INSERT OR REPLACE INTO smc_evaluations 
            (timestamp, model_signal, smc_confidence, setup_type, supporting_factors, conflicting_factors)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                evaluation['timestamp'],
                evaluation['original_signal'],
                evaluation['smc_confidence'],
                evaluation['setup_type'],
                supporting_str,
                conflicting_str
            ))
            
            conn.commit()
            logger.info(f"✅ Evaluation saved to database")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error saving evaluation: {e}")
        return False


def get_latest_prediction(db_path: str = DB_PATH) -> Optional[Dict]:
    """Get the latest prediction from database."""
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT timestamp, signal, confidence 
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            df = pd.read_sql(query, conn)
            if not df.empty:
                return {
                    'timestamp': df['timestamp'].iloc[0],
                    'signal': df['signal'].iloc[0],
                    'confidence': df['confidence'].iloc[0]
                }
            return None
                
    except Exception as e:
        logger.error(f"Error retrieving latest prediction: {e}")
        return None


def evaluate_latest_prediction() -> Optional[Dict]:
    """Evaluate the latest model prediction with SMC principles."""
    try:
        # Get the latest prediction
        prediction = get_latest_prediction()
        if not prediction:
            logger.warning("No prediction available")
            return None
        
        # Validate and save the evaluation
        evaluation = validate_signal(prediction)
        save_evaluation(evaluation)
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Error in evaluation process: {e}")
        return None


def get_historical_evaluations(limit: int = 5, db_path: str = DB_PATH) -> List[Dict]:
    """Get historical SMC evaluations from database."""
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT * FROM smc_evaluations
            ORDER BY timestamp DESC
            LIMIT ?
            """
            
            df = pd.read_sql(query, conn, params=(limit,))
            
            # Convert string-formatted lists back to actual lists
            results = []
            for _, row in df.iterrows():
                result = row.to_dict()
                result['supporting_factors'] = row['supporting_factors'].split('|') if row['supporting_factors'] else []
                result['conflicting_factors'] = row['conflicting_factors'].split('|') if row['conflicting_factors'] else []
                results.append(result)
            
            return results
                
    except Exception as e:
        logger.error(f"Error retrieving evaluations: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    try:
        # Evaluate the latest prediction
        latest_evaluation = evaluate_latest_prediction()
        
        if latest_evaluation:
            # Print results
            print(f"Signal: {latest_evaluation['original_signal']}")
            print(f"Confidence: {latest_evaluation['smc_confidence']}%")
            print(f"Setup type: {latest_evaluation['setup_type']}")
            print("Supporting factors:")
            for factor in latest_evaluation['supporting_factors']:
                print(f"- {factor}")
            print("Conflicting factors:")
            for factor in latest_evaluation['conflicting_factors']:
                print(f"- {factor}")
            
            # Get historical evaluations
            historical = get_historical_evaluations()
            if historical:
                print(f"\nRecent evaluations:")
                for eval in historical:
                    print(f"{eval['timestamp']}: {eval['model_signal']} - {eval['smc_confidence']}% - {eval['setup_type']}")
        else:
            print("No evaluation available")
            
    except Exception as e:
        print(f"Error: {str(e)}")