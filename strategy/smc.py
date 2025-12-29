import sqlite3
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DB_PATH = "data/btc_ohlcv.db"

# Nama tabel
OHLCV_TABLE = "btc_1h"
TECHNICAL_TABLE = "smc_btc_1h_technical_indicators"

SMC_TABLES = {
    "bos_choch": "smc_btc_1h_bos_choch",
    "order_block": "smc_btc_1h_order_block",
    "fvg": "smc_btc_1h_fvg",
    "liquidity": "smc_btc_1h_liquidity",
    "swing_highs_lows": "smc_btc_1h_swing_highs_lows",
    "retracements": "smc_btc_1h_retracements"
}


class MarketBias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SetupQuality(Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    NO_TRADE = "No Trade"


@dataclass
class SMCContext:
    """Market context dari analisis SMC"""
    bias: MarketBias
    trend_strength: float
    key_levels: Dict[str, float]
    active_pois: List[Dict]


@dataclass
class TradeSetup:
    """Setup trading yang teridentifikasi"""
    signal: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward: float
    quality: SetupQuality
    confidence: float
    setup_type: str
    supporting_factors: List[str]
    conflicting_factors: List[str]
    invalidation_level: float


class EnhancedSMCStrategy:
    """
    Enhanced SMC Strategy - BALANCED OPTIMIZATION
    
    Key Fixes:
    1. Symmetric evaluation untuk BUY dan SELL
    2. Confidence calculation yang lebih realistis
    3. Threshold yang adjusted untuk actual score range
    4. No default bias - evaluate each signal independently
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.lookback_candles = 50
        self.recent_candles = 10
        self.smc_data = {}
        self.ohlcv_data = None
        self.technical_data = None
        self.current_price = None

        # Risk parameters
        self.max_risk_percent = 1.0
        self.min_rr_ratio = 2.0
        self.atr_multiplier_sl = 1.5
        self.atr_multiplier_tp = 3.0

        # Base weights - will be adjusted dynamically
        self.base_weights = {
            'market_structure': 25,
            'order_block': 20,
            'fvg': 12,
            'liquidity': 13,
            'premium_discount': 10,
            'retracement': 5,
            'technical_confluence': 15
        }

        self.weights = self.base_weights.copy()

    def load_data(self) -> bool:
        """Load semua data yang diperlukan"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load OHLCV data
                self.ohlcv_data = pd.read_sql(
                    f"SELECT * FROM {OHLCV_TABLE} ORDER BY timestamp DESC LIMIT {self.lookback_candles}",
                    conn, parse_dates=['timestamp']
                )

                if self.ohlcv_data.empty:
                    logger.error("No OHLCV data found")
                    return False

                self.current_price = self.ohlcv_data['close'].iloc[0]
                self.current_high = self.ohlcv_data['high'].iloc[0]
                self.current_low = self.ohlcv_data['low'].iloc[0]

                # Load Technical Indicators
                self.technical_data = pd.read_sql(
                    f"SELECT * FROM {TECHNICAL_TABLE} WHERE atr_14 > 0 ORDER BY timestamp DESC LIMIT {self.lookback_candles}",
                    conn, parse_dates=['timestamp']
                )

                # ATR calculation
                atr_val = None
                if not self.technical_data.empty and 'atr_14' in self.technical_data.columns:
                    atr_val = self.technical_data['atr_14'].iloc[0]
                    if pd.notna(atr_val) and atr_val > 0:
                        self.atr = float(atr_val)
                        logger.info(f"ATR loaded: {self.atr:.2f}")
                    else:
                        atr_val = None

                if atr_val is None or atr_val <= 0:
                    self._calculate_atr()
                    logger.info(f"ATR calculated: {self.atr:.2f}")

                # Load SMC tables
                for key, table in SMC_TABLES.items():
                    try:
                        df = pd.read_sql(
                            f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT {self.lookback_candles}",
                            conn, parse_dates=['timestamp']
                        )
                        self.smc_data[key] = df
                    except Exception as table_err:
                        logger.warning(f"Could not load table {table}: {table_err}")
                        self.smc_data[key] = pd.DataFrame()

                return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def _calculate_atr(self, period: int = 14):
        """Calculate ATR untuk sizing dan levels"""
        if self.ohlcv_data is None or len(self.ohlcv_data) < period:
            self.atr = 100.0
            logger.warning(f"Not enough data for ATR, using default: {self.atr}")
            return

        df = self.ohlcv_data.copy()
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        df['atr'] = df['tr'].rolling(window=period, min_periods=period).mean()

        valid_atr = df['atr'].dropna()
        if not valid_atr.empty:
            self.atr = float(valid_atr.iloc[-1])
        else:
            self.atr = float(df['tr'].mean()) if df['tr'].notna().any() else 100.0

        if pd.isna(self.atr) or self.atr <= 0:
            self.atr = 100.0

    def analyze_market_context(self) -> SMCContext:
        """
        FIXED: Symmetric market analysis tanpa bias
        """
        # 1. Market Structure Analysis
        structure_bias, structure_strength = self._analyze_market_structure()

        # 2. Premium/Discount Zone
        premium_discount = self._identify_premium_discount_zone()

        # 3. Active POIs
        active_pois = self._identify_active_pois()

        # 4. Swing Structure
        swing_bias = self._analyze_swing_structure()

        # 5. Technical Trend
        tech_bias, tech_strength = self._analyze_technical_trend()

        # FIXED: Balanced bias calculation
        bias_score = 0.0
        
        # Market structure (40% weight)
        if structure_bias == MarketBias.BULLISH:
            bias_score += structure_strength * 0.4
        elif structure_bias == MarketBias.BEARISH:
            bias_score -= structure_strength * 0.4

        # Swing structure (30% weight)
        if swing_bias == MarketBias.BULLISH:
            bias_score += 0.3
        elif swing_bias == MarketBias.BEARISH:
            bias_score -= 0.3

        # Technical trend (30% weight)
        if tech_bias == MarketBias.BULLISH:
            bias_score += tech_strength * 0.3
        elif tech_bias == MarketBias.BEARISH:
            bias_score -= tech_strength * 0.3

        # FIXED: Stricter threshold - require strong consensus
        if bias_score > 0.25:
            bias = MarketBias.BULLISH
        elif bias_score < -0.25:
            bias = MarketBias.BEARISH
        else:
            bias = MarketBias.NEUTRAL

        trend_strength = min(abs(bias_score), 1.0)

        # Adjust weights
        self._adjust_weights_based_on_context(bias, trend_strength, active_pois)

        return SMCContext(
            bias=bias,
            trend_strength=trend_strength,
            key_levels=premium_discount,
            active_pois=active_pois
        )

    def _analyze_market_structure(self) -> Tuple[MarketBias, float]:
        """FIXED: Symmetric BOS/CHOCH analysis"""
        if 'bos_choch' not in self.smc_data or self.smc_data['bos_choch'].empty:
            return MarketBias.NEUTRAL, 0.0

        df = self.smc_data['bos_choch'].head(self.recent_candles)

        # FIXED: Exponential decay dengan weight yang sama untuk bull dan bear
        decay_factor = 0.8
        bull_score = 0.0
        bear_score = 0.0

        for i, row in df.iterrows():
            weight = decay_factor ** i
            
            if row['BOS'] == 1:
                bull_score += weight * 1.0
            elif row['BOS'] == -1:
                bear_score += weight * 1.0
                
            if row['CHOCH'] == 1:
                bull_score += weight * 2.0  # CHOCH weighted higher
            elif row['CHOCH'] == -1:
                bear_score += weight * 2.0

        # FIXED: Require minimum threshold to avoid weak signals
        total = bull_score + bear_score
        if total < 0.5:  # Minimum activity threshold
            return MarketBias.NEUTRAL, 0.0

        # Calculate normalized strength
        if bull_score > bear_score * 1.3:  # Require 30% dominance
            strength = bull_score / (bull_score + bear_score)
            return MarketBias.BULLISH, strength
        elif bear_score > bull_score * 1.3:
            strength = bear_score / (bull_score + bear_score)
            return MarketBias.BEARISH, strength
        
        return MarketBias.NEUTRAL, 0.0

    def _analyze_technical_trend(self) -> Tuple[MarketBias, float]:
        """FIXED: Symmetric technical analysis"""
        if self.technical_data is None or self.technical_data.empty:
            return MarketBias.NEUTRAL, 0.0

        tech = self.technical_data.iloc[0]
        bull_score = 0.0
        bear_score = 0.0
        max_score = 0.0

        # EMA alignment (weight: 35%)
        ema_9 = tech.get('ema_9', 0)
        ema_21 = tech.get('ema_21', 0)
        ema_50 = tech.get('ema_50', 0)

        if ema_9 > 0 and ema_21 > 0 and ema_50 > 0:
            max_score += 0.35
            if ema_9 > ema_21 > ema_50:
                bull_score += 0.35
            elif ema_9 < ema_21 < ema_50:
                bear_score += 0.35

        # ADX (weight: 25%)
        adx = tech.get('adx_14', 0)
        plus_di = tech.get('plus_di_14', 0)
        minus_di = tech.get('minus_di_14', 0)

        if adx > 20:  # Only if trend exists
            max_score += 0.25
            strength = min(adx / 50, 1.0)
            if plus_di > minus_di:
                bull_score += 0.25 * strength
            else:
                bear_score += 0.25 * strength

        # MACD (weight: 20%)
        macd = tech.get('macd', 0)
        macd_signal = tech.get('macd_signal', 0)

        if macd != 0 and macd_signal != 0:
            max_score += 0.20
            if macd > macd_signal:
                bull_score += 0.20
            else:
                bear_score += 0.20

        # RSI (weight: 20%)
        rsi = tech.get('rsi_14', 50)
        if rsi > 0:
            max_score += 0.20
            if 50 < rsi < 70:
                bull_score += 0.20
            elif 30 < rsi < 50:
                bear_score += 0.20

        if max_score == 0:
            return MarketBias.NEUTRAL, 0.0

        # FIXED: Require clear dominance
        total_score = bull_score + bear_score
        if total_score < max_score * 0.4:  # Need at least 40% of max
            return MarketBias.NEUTRAL, 0.0

        if bull_score > bear_score * 1.2:
            return MarketBias.BULLISH, bull_score / max_score
        elif bear_score > bull_score * 1.2:
            return MarketBias.BEARISH, bear_score / max_score

        return MarketBias.NEUTRAL, 0.0

    def _analyze_swing_structure(self) -> MarketBias:
        """FIXED: Symmetric swing analysis"""
        if 'swing_highs_lows' not in self.smc_data:
            return MarketBias.NEUTRAL

        df = self.smc_data['swing_highs_lows'].head(10)

        highs = df[df['HighLow'] == 1]['Level'].dropna().tolist()
        lows = df[df['HighLow'] == -1]['Level'].dropna().tolist()

        if len(highs) < 2 or len(lows) < 2:
            return MarketBias.NEUTRAL

        # Higher highs AND higher lows = bullish
        hh = highs[0] > highs[1]
        hl = lows[0] > lows[1]

        # Lower highs AND lower lows = bearish
        lh = highs[0] < highs[1]
        ll = lows[0] < lows[1]

        if hh and hl:
            return MarketBias.BULLISH
        elif lh and ll:
            return MarketBias.BEARISH

        return MarketBias.NEUTRAL

    def _identify_premium_discount_zone(self) -> Dict[str, float]:
        """Identifikasi Premium/Discount Zone"""
        if 'swing_highs_lows' not in self.smc_data:
            return {}

        df = self.smc_data['swing_highs_lows'].head(20)

        swing_highs = df[df['HighLow'] == 1]['Level'].dropna()
        swing_lows = df[df['HighLow'] == -1]['Level'].dropna()

        if swing_highs.empty or swing_lows.empty:
            return {}

        range_high = swing_highs.max()
        range_low = swing_lows.min()
        range_size = range_high - range_low

        equilibrium = (range_high + range_low) / 2
        premium_zone = equilibrium + (range_size * 0.25)
        discount_zone = equilibrium - (range_size * 0.25)

        return {
            'range_high': range_high,
            'range_low': range_low,
            'equilibrium': equilibrium,
            'premium_zone': premium_zone,
            'discount_zone': discount_zone,
            'current_zone': 'premium' if self.current_price > equilibrium else 'discount'
        }

    def _identify_active_pois(self) -> List[Dict]:
        """Identifikasi Point of Interest"""
        pois = []

        # Active Order Blocks
        if 'order_block' in self.smc_data:
            obs = self.smc_data['order_block']
            active_obs = obs[(obs['OB'] != 0) & (obs['MitigatedIndex'] == 0)]

            for _, ob in active_obs.head(5).iterrows():
                pois.append({
                    'type': 'order_block',
                    'direction': 'bullish' if ob['OB'] == 1 else 'bearish',
                    'top': ob['Top'],
                    'bottom': ob['Bottom'],
                    'strength': ob['Percentage'],
                    'volume': ob['OBVolume']
                })

        # Active FVG
        if 'fvg' in self.smc_data:
            fvgs = self.smc_data['fvg']
            active_fvgs = fvgs[(fvgs['FVG'] != 0) & (fvgs['MitigatedIndex'] == 0)]

            for _, fvg in active_fvgs.head(5).iterrows():
                pois.append({
                    'type': 'fvg',
                    'direction': 'bullish' if fvg['FVG'] == 1 else 'bearish',
                    'top': fvg['Top'],
                    'bottom': fvg['Bottom']
                })

        # Liquidity Levels
        if 'liquidity' in self.smc_data:
            liqs = self.smc_data['liquidity']
            active_liqs = liqs[(liqs['Liquidity'] != 0) & (liqs['Swept'] == 0)]

            for _, liq in active_liqs.head(5).iterrows():
                pois.append({
                    'type': 'liquidity',
                    'direction': 'bullish' if liq['Liquidity'] == 1 else 'bearish',
                    'level': liq['Level']
                })

        return pois

    def _adjust_weights_based_on_context(self, bias: MarketBias,
                                        trend_strength: float,
                                        active_pois: List[Dict]):
        """Dynamic weight adjustment"""
        self.weights = self.base_weights.copy()

        if trend_strength > 0.6:
            self.weights['market_structure'] = 30
            self.weights['technical_confluence'] = 20
            self.weights['order_block'] = 18
            self.weights['premium_discount'] = 7

        if len(active_pois) > 3:
            self.weights['order_block'] = 25
            self.weights['fvg'] = 15
            self.weights['liquidity'] = 15
            self.weights['market_structure'] = 20

        if bias == MarketBias.NEUTRAL:
            self.weights['premium_discount'] = 15
            self.weights['liquidity'] = 18
            self.weights['market_structure'] = 20
            self.weights['technical_confluence'] = 12

        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: (v / total) * 100 for k, v in self.weights.items()}

    def generate_trade_setup(self, prediction: Dict) -> TradeSetup:
        """Generate complete trade setup"""
        if not self.load_data():
            return self._create_no_trade_setup("Failed to load data")

        signal = prediction.get('signal', 'Hold').lower()
        context = self.analyze_market_context()

        # FIXED: Evaluate both BUY and SELL independently
        buy_score, buy_supporting, buy_conflicting = self._evaluate_signal_quality('buy', context)
        sell_score, sell_supporting, sell_conflicting = self._evaluate_signal_quality('sell', context)

        # FIXED: Choose best signal or NO TRADE
        if buy_score > sell_score and buy_score >= 45:  # Minimum threshold
            final_signal = 'buy'
            score = buy_score
            supporting = buy_supporting
            conflicting = buy_conflicting
        elif sell_score > buy_score and sell_score >= 45:
            final_signal = 'sell'
            score = sell_score
            supporting = sell_supporting
            conflicting = sell_conflicting
        else:
            # NO TRADE if both scores are low or too close
            return self._create_no_trade_setup(f"Insufficient signal strength (Buy: {buy_score:.1f}, Sell: {sell_score:.1f})")

        # Determine quality
        quality = self._determine_setup_quality(score, final_signal, context)

        # Calculate levels
        entry, sl, tp1, tp2, tp3, invalidation = self._calculate_trade_levels(
            final_signal, context, quality
        )

        # Calculate RR
        if entry and sl and tp1:
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)
            rr = reward / risk if risk > 0 else 0
        else:
            rr = 0

        # FIXED: Realistic confidence calculation
        confidence = self._calculate_final_confidence(
            score, quality, rr, context, supporting, conflicting
        )

        # Setup type
        setup_type = self._identify_setup_type(final_signal, context, supporting)

        return TradeSetup(
            signal=final_signal.capitalize(),
            entry_price=entry or self.current_price,
            stop_loss=sl or 0,
            take_profit_1=tp1 or 0,
            take_profit_2=tp2 or 0,
            take_profit_3=tp3 or 0,
            risk_reward=round(rr, 2),
            quality=quality,
            confidence=round(confidence, 2),
            setup_type=setup_type,
            supporting_factors=supporting,
            conflicting_factors=conflicting,
            invalidation_level=invalidation or 0
        )

    def _evaluate_signal_quality(self, signal: str, context: SMCContext) -> Tuple[float, List[str], List[str]]:
        """
        FIXED: Symmetric evaluation untuk BUY dan SELL
        Score range: 0-100
        """
        score = 0.0
        supporting = []
        conflicting = []

        # 1. Market Structure (weight: 25%)
        ms_score = self._eval_market_structure(signal, context, supporting, conflicting)
        score += ms_score * (self.weights['market_structure'] / 100) * 100

        # 2. Order Block (weight: 20%)
        ob_score = self._eval_order_block(signal, context, supporting, conflicting)
        score += ob_score * (self.weights['order_block'] / 100) * 100

        # 3. FVG (weight: 12%)
        fvg_score = self._eval_fvg(signal, supporting, conflicting)
        score += fvg_score * (self.weights['fvg'] / 100) * 100

        # 4. Liquidity (weight: 13%)
        liq_score = self._eval_liquidity(signal, supporting, conflicting)
        score += liq_score * (self.weights['liquidity'] / 100) * 100

        # 5. Premium/Discount (weight: 10%)
        pd_score = self._eval_premium_discount(signal, context, supporting, conflicting)
        score += pd_score * (self.weights['premium_discount'] / 100) * 100

        # 6. Retracement (weight: 5%)
        ret_score = self._eval_retracement(signal, supporting, conflicting)
        score += ret_score * (self.weights['retracement'] / 100) * 100

        # 7. Technical (weight: 15%)
        tech_score = self._eval_technical(signal, supporting, conflicting)
        score += tech_score * (self.weights['technical_confluence'] / 100) * 100

        return score, supporting, conflicting

    def _eval_market_structure(self, signal: str, context: SMCContext,
                               supporting: List, conflicting: List) -> float:
        """
        FIXED: Score 0-1 based on alignment
        0 = completely against, 0.5 = neutral, 1 = perfect alignment
        """
        if 'bos_choch' not in self.smc_data:
            return 0.5

        is_buy = signal == 'buy'
        score = 0.5  # Start neutral

        # Bias alignment (50% of component)
        if is_buy:
            if context.bias == MarketBias.BULLISH:
                score += 0.25 * context.trend_strength
                supporting.append(f"Bullish market bias (strength: {context.trend_strength:.0%})")
            elif context.bias == MarketBias.BEARISH:
                score -= 0.3 * context.trend_strength
                conflicting.append("Buy signal against bearish market bias")
        else:
            if context.bias == MarketBias.BEARISH:
                score += 0.25 * context.trend_strength
                supporting.append(f"Bearish market bias (strength: {context.trend_strength:.0%})")
            elif context.bias == MarketBias.BULLISH:
                score -= 0.3 * context.trend_strength
                conflicting.append("Sell signal against bullish market bias")

        # Recent structure (50% of component)
        df = self.smc_data['bos_choch'].head(3)  # Only most recent
        
        for i, row in df.iterrows():
            weight = 0.17 if i == 0 else 0.08  # Recent = higher weight
            
            if is_buy:
                if row['CHOCH'] == 1:
                    score += weight
                    if i == 0:
                        supporting.append("Recent bullish CHoCH")
                elif row['CHOCH'] == -1:
                    score -= weight
                    if i == 0:
                        conflicting.append("Recent bearish CHoCH")
            else:
                if row['CHOCH'] == -1:
                    score += weight
                    if i == 0:
                        supporting.append("Recent bearish CHoCH")
                elif row['CHOCH'] == 1:
                    score -= weight
                    if i == 0:
                        conflicting.append("Recent bullish CHoCH")

        return max(0, min(1, score))

    def _eval_order_block(self, signal: str, context: SMCContext,
                         supporting: List, conflicting: List) -> float:
        """FIXED: Symmetric OB evaluation"""
        if 'order_block' not in self.smc_data:
            return 0.5

        df = self.smc_data['order_block']
        is_buy = signal == 'buy'
        
        # Find aligned OB
        aligned_obs = df[(df['OB'] == (1 if is_buy else -1)) & (df['MitigatedIndex'] == 0)]
        
        # Find opposing OB
        opposing_obs = df[(df['OB'] == (-1 if is_buy else 1)) & (df['MitigatedIndex'] == 0)]

        score = 0.5  # Start neutral

        # Check aligned OB
        if not aligned_obs.empty:
            ob = aligned_obs.iloc[0]
            top, bottom = ob['Top'], ob['Bottom']
            
            # Price within OB
            if bottom <= self.current_price <= top:
                score = 0.8
                strength_bonus = min(ob['Percentage'] / 100, 0.2)
                score += strength_bonus
                supporting.append(f"Price at {'bullish' if is_buy else 'bearish'} OB (strength: {ob['Percentage']:.0f}%)")
            # Price near OB
            elif abs(self.current_price - (top + bottom) / 2) < (top - bottom) * 2:
                score = 0.6
                supporting.append(f"Price near {'bullish' if is_buy else 'bearish'} OB")
        else:
            score = 0.4  # Penalty for no aligned OB

        # Check opposing OB
        if not opposing_obs.empty:
            ob = opposing_obs.iloc[0]
            if ob['Bottom'] <= self.current_price <= ob['Top']:
                score -= 0.3
                conflicting.append(f"Price at opposing {'bearish' if is_buy else 'bullish'} OB")

        return max(0, min(1, score))

    def _eval_fvg(self, signal: str, supporting: List, conflicting: List) -> float:
        """FIXED: Symmetric FVG evaluation"""
        if 'fvg' not in self.smc_data:
            return 0.5

        df = self.smc_data['fvg'].head(self.recent_candles)
        is_buy = signal == 'buy'
        
        target_fvg = 1 if is_buy else -1
        opposing_fvg = -1 if is_buy else 1

        score = 0.5  # Start neutral

        # Check aligned FVG
        aligned = df[(df['FVG'] == target_fvg) & (df['MitigatedIndex'] == 0)]
        if not aligned.empty:
            fvg = aligned.iloc[0]
            if fvg['Bottom'] <= self.current_price <= fvg['Top']:
                score = 0.9
                supporting.append(f"Price at {'bullish' if is_buy else 'bearish'} FVG")
            else:
                score = 0.6
                supporting.append(f"Unmitigated {'bullish' if is_buy else 'bearish'} FVG present")
        else:
            score = 0.4  # Penalty for no aligned FVG

        # Check opposing FVG
        opposing = df[(df['FVG'] == opposing_fvg) & (df['MitigatedIndex'] == 0)]
        if not opposing.empty:
            score -= 0.2
            conflicting.append(f"Unmitigated {'bearish' if is_buy else 'bullish'} FVG present")

        return max(0, min(1, score))

    def _eval_liquidity(self, signal: str, supporting: List, conflicting: List) -> float:
        """FIXED: Symmetric liquidity evaluation"""
        if 'liquidity' not in self.smc_data:
            return 0.5

        df = self.smc_data['liquidity'].head(self.recent_candles)
        is_buy = signal == 'buy'

        score = 0.5  # Start neutral

        if is_buy:
            # Check for swept sell-side liquidity (bullish)
            swept_sell = df[(df['Liquidity'] == -1) & (df['Swept'] != 0)]
            if not swept_sell.empty and swept_sell.index[0] < 3:
                score = 0.8
                supporting.append("Recent sell-side liquidity swept (bullish)")
            
            # Check for unswept buy-side target
            unswept_buy = df[(df['Liquidity'] == 1) & (df['Swept'] == 0)]
            if not unswept_buy.empty:
                score += 0.2
                supporting.append("Buy-side liquidity target above")
        else:
            # Check for swept buy-side liquidity (bearish)
            swept_buy = df[(df['Liquidity'] == 1) & (df['Swept'] != 0)]
            if not swept_buy.empty and swept_buy.index[0] < 3:
                score = 0.8
                supporting.append("Recent buy-side liquidity swept (bearish)")
            
            # Check for unswept sell-side target
            unswept_sell = df[(df['Liquidity'] == -1) & (df['Swept'] == 0)]
            if not unswept_sell.empty:
                score += 0.2
                supporting.append("Sell-side liquidity target below")

        return max(0, min(1, score))

    def _eval_premium_discount(self, signal: str, context: SMCContext,
                               supporting: List, conflicting: List) -> float:
        """FIXED: Symmetric zone evaluation"""
        if 'equilibrium' not in context.key_levels:
            return 0.5

        eq = context.key_levels['equilibrium']
        range_high = context.key_levels.get('range_high', eq)
        range_low = context.key_levels.get('range_low', eq)
        range_size = range_high - range_low
        
        if range_size == 0:
            return 0.5

        # Calculate position (0 = bottom, 1 = top)
        position = (self.current_price - range_low) / range_size
        is_buy = signal == 'buy'

        if is_buy:
            # Ideal: deep discount (position < 0.3)
            if position < 0.3:
                score = 1.0
                supporting.append("Price in deep discount zone (optimal for buy)")
            elif position < 0.5:
                score = 0.7
                supporting.append("Price in discount zone (good for buy)")
            elif position > 0.7:
                score = 0.2
                conflicting.append("Buy signal in premium zone (risky)")
            else:
                score = 0.5
        else:
            # Ideal: deep premium (position > 0.7)
            if position > 0.7:
                score = 1.0
                supporting.append("Price in deep premium zone (optimal for sell)")
            elif position > 0.5:
                score = 0.7
                supporting.append("Price in premium zone (good for sell)")
            elif position < 0.3:
                score = 0.2
                conflicting.append("Sell signal in discount zone (risky)")
            else:
                score = 0.5

        return score

    def _eval_retracement(self, signal: str, supporting: List, conflicting: List) -> float:
        """FIXED: Symmetric retracement evaluation"""
        if 'retracements' not in self.smc_data:
            return 0.5

        df = self.smc_data['retracements'].head(self.recent_candles)
        is_buy = signal == 'buy'
        
        target_dir = 1 if is_buy else -1
        retr = df[df['Direction'] == target_dir]

        if retr.empty:
            return 0.4  # Slight penalty

        current_retr = retr.iloc[0]['CurrentRetracement%']

        # Optimal: 50-70%
        if 50 <= current_retr <= 70:
            score = 1.0
            supporting.append(f"Optimal retracement ({current_retr:.1f}%)")
        elif 38 <= current_retr < 50:
            score = 0.7
        elif 70 < current_retr <= 79:
            score = 0.6
        elif current_retr > 79:
            score = 0.3
            conflicting.append(f"Very deep retracement ({current_retr:.1f}%)")
        else:
            score = 0.4

        return score

    def _eval_technical(self, signal: str, supporting: List, conflicting: List) -> float:
        """FIXED: Symmetric technical evaluation"""
        if self.technical_data is None or self.technical_data.empty:
            return 0.5

        tech = self.technical_data.iloc[0]
        is_buy = signal == 'buy'
        score = 0.5  # Start neutral

        # EMA (35%)
        ema_9 = tech.get('ema_9', 0)
        ema_21 = tech.get('ema_21', 0)
        ema_50 = tech.get('ema_50', 0)

        if ema_9 > 0 and ema_21 > 0 and ema_50 > 0:
            if is_buy:
                if ema_9 > ema_21 > ema_50:
                    score += 0.25
                    supporting.append("EMA alignment bullish")
                elif ema_9 < ema_21 < ema_50:
                    score -= 0.25
                    conflicting.append("EMA alignment bearish")
                
                if self.current_price > ema_21:
                    score += 0.1
                    supporting.append("Price above EMA 21")
            else:
                if ema_9 < ema_21 < ema_50:
                    score += 0.25
                    supporting.append("EMA alignment bearish")
                elif ema_9 > ema_21 > ema_50:
                    score -= 0.25
                    conflicting.append("EMA alignment bullish")
                
                if self.current_price < ema_21:
                    score += 0.1
                    supporting.append("Price below EMA 21")

        # RSI (20%)
        rsi = tech.get('rsi_14', 50)
        if rsi > 0:
            if is_buy:
                if 30 <= rsi <= 50:
                    score += 0.2
                    supporting.append(f"RSI in buy zone ({rsi:.1f})")
                elif rsi > 70:
                    score -= 0.2
                    conflicting.append(f"RSI overbought ({rsi:.1f})")
            else:
                if 50 <= rsi <= 70:
                    score += 0.2
                    supporting.append(f"RSI in sell zone ({rsi:.1f})")
                elif rsi < 30:
                    score -= 0.2
                    conflicting.append(f"RSI oversold ({rsi:.1f})")

        # MACD (20%)
        macd = tech.get('macd', 0)
        macd_signal = tech.get('macd_signal', 0)

        if macd != 0 and macd_signal != 0:
            if is_buy:
                if macd > macd_signal:
                    score += 0.2
                    supporting.append("MACD bullish crossover")
                else:
                    score -= 0.1
                    conflicting.append("MACD bearish")
            else:
                if macd < macd_signal:
                    score += 0.2
                    supporting.append("MACD bearish crossover")
                else:
                    score -= 0.1
                    conflicting.append("MACD bullish")

        # ADX (25%)
        adx = tech.get('adx_14', 0)
        plus_di = tech.get('plus_di_14', 0)
        minus_di = tech.get('minus_di_14', 0)

        if adx > 20:
            if is_buy and plus_di > minus_di:
                score += 0.15
                supporting.append(f"Strong bullish trend (ADX: {adx:.1f})")
            elif not is_buy and minus_di > plus_di:
                score += 0.15
                supporting.append(f"Strong bearish trend (ADX: {adx:.1f})")

        return max(0, min(1, score))

    def _determine_setup_quality(self, score: float, signal: str, context: SMCContext) -> SetupQuality:
        """
        FIXED: Realistic quality grading
        Score range: 0-100
        """
        is_buy = signal == 'buy'
        is_sell = signal == 'sell'

        bias_aligned = (
            (is_buy and context.bias == MarketBias.BULLISH) or
            (is_sell and context.bias == MarketBias.BEARISH)
        )
        strong_trend = context.trend_strength > 0.6

        # FIXED: Adjusted thresholds for actual score range
        if score >= 70:
            if bias_aligned and strong_trend:
                return SetupQuality.A_PLUS
            elif bias_aligned:
                return SetupQuality.A
            else:
                return SetupQuality.B
        elif score >= 55:
            if bias_aligned:
                return SetupQuality.A
            else:
                return SetupQuality.B
        elif score >= 45:
            return SetupQuality.B
        elif score >= 35:
            return SetupQuality.C
        else:
            return SetupQuality.NO_TRADE

    def _calculate_trade_levels(self, signal: str, context: SMCContext,
                                quality: SetupQuality) -> Tuple[float, float, float, float, float, float]:
        """Calculate entry, SL, TP levels"""
        is_buy = signal == 'buy'
        entry = self.current_price

        if pd.isna(self.atr) or self.atr <= 0:
            self.atr = entry * 0.02

        swing_df = self.smc_data.get('swing_highs_lows', pd.DataFrame())
        ob_df = self.smc_data.get('order_block', pd.DataFrame())

        quality_multipliers = {
            SetupQuality.A_PLUS: 0.8,
            SetupQuality.A: 0.9,
            SetupQuality.B: 1.0,
            SetupQuality.C: 1.2,
            SetupQuality.NO_TRADE: 1.5
        }
        risk_multiplier = quality_multipliers.get(quality, 1.0)

        if is_buy:
            # Find support
            if not swing_df.empty and 'HighLow' in swing_df.columns:
                swing_lows = swing_df[swing_df['HighLow'] == -1]['Level'].dropna()
                recent_low = float(swing_lows.iloc[0]) if not swing_lows.empty else self.current_low
            else:
                recent_low = self.current_low

            if not ob_df.empty and 'OB' in ob_df.columns:
                bullish_obs = ob_df[(ob_df['OB'] == 1) & (ob_df['Bottom'] < entry) & (ob_df['MitigatedIndex'] == 0)]
                ob_bottom = float(bullish_obs['Bottom'].iloc[0]) if not bullish_obs.empty else recent_low
            else:
                ob_bottom = recent_low

            structural_sl = max(recent_low, ob_bottom)
            atr_buffer = self.atr * 0.3 * risk_multiplier
            sl = structural_sl - atr_buffer
            invalidation = structural_sl - (self.atr * 0.8)

            risk = entry - sl
            if risk <= 0:
                risk = self.atr * risk_multiplier
                sl = entry - risk

            if quality == SetupQuality.A_PLUS:
                tp1 = entry + (risk * 2.0)
                tp2 = entry + (risk * 3.5)
                tp3 = entry + (risk * 5.0)
            elif quality == SetupQuality.A:
                tp1 = entry + (risk * 1.8)
                tp2 = entry + (risk * 3.0)
                tp3 = entry + (risk * 4.5)
            else:
                tp1 = entry + (risk * 1.5)
                tp2 = entry + (risk * 2.5)
                tp3 = entry + (risk * 4.0)

        else:  # Sell
            # Find resistance
            if not swing_df.empty and 'HighLow' in swing_df.columns:
                swing_highs = swing_df[swing_df['HighLow'] == 1]['Level'].dropna()
                recent_high = float(swing_highs.iloc[0]) if not swing_highs.empty else self.current_high
            else:
                recent_high = self.current_high

            if not ob_df.empty and 'OB' in ob_df.columns:
                bearish_obs = ob_df[(ob_df['OB'] == -1) & (ob_df['Top'] > entry) & (ob_df['MitigatedIndex'] == 0)]
                ob_top = float(bearish_obs['Top'].iloc[0]) if not bearish_obs.empty else recent_high
            else:
                ob_top = recent_high

            structural_sl = min(recent_high, ob_top)
            atr_buffer = self.atr * 0.3 * risk_multiplier
            sl = structural_sl + atr_buffer
            invalidation = structural_sl + (self.atr * 0.8)

            risk = sl - entry
            if risk <= 0:
                risk = self.atr * risk_multiplier
                sl = entry + risk

            if quality == SetupQuality.A_PLUS:
                tp1 = entry - (risk * 2.0)
                tp2 = entry - (risk * 3.5)
                tp3 = entry - (risk * 5.0)
            elif quality == SetupQuality.A:
                tp1 = entry - (risk * 1.8)
                tp2 = entry - (risk * 3.0)
                tp3 = entry - (risk * 4.5)
            else:
                tp1 = entry - (risk * 1.5)
                tp2 = entry - (risk * 2.5)
                tp3 = entry - (risk * 4.0)

        return entry, sl, tp1, tp2, tp3, invalidation

    def _calculate_final_confidence(self, score: float, quality: SetupQuality,
                                   rr: float, context: SMCContext,
                                   supporting: List[str], conflicting: List[str]) -> float:
        """
        FIXED: Realistic confidence calculation
        Base score already 0-100, apply reasonable adjustments
        """
        confidence = score

        # Quality adjustment (±10)
        quality_adj = {
            SetupQuality.A_PLUS: 10,
            SetupQuality.A: 5,
            SetupQuality.B: 0,
            SetupQuality.C: -5,
            SetupQuality.NO_TRADE: -15
        }
        confidence += quality_adj.get(quality, 0)

        # Risk-reward bonus (max: +10)
        if rr >= 4:
            confidence += 10
        elif rr >= 3:
            confidence += 7
        elif rr >= 2.5:
            confidence += 5
        elif rr >= 2:
            confidence += 3
        elif rr < 1.5:
            confidence -= 10

        # Trend strength bonus (max: +8)
        confidence += context.trend_strength * 8

        # Factor ratio (±7)
        total_factors = len(supporting) + len(conflicting)
        if total_factors > 0:
            support_ratio = len(supporting) / total_factors
            if support_ratio > 0.75:
                confidence += 7
            elif support_ratio > 0.6:
                confidence += 4
            elif support_ratio < 0.4:
                confidence -= 7

        # POI bonus (max: +5)
        poi_count = len(context.active_pois)
        if poi_count >= 3:
            confidence += 5
        elif poi_count >= 2:
            confidence += 3

        return max(0, min(100, confidence))

    def _identify_setup_type(self, signal: str, context: SMCContext,
                            supporting: List[str]) -> str:
        """Identify setup type"""
        has_choch = any('CHoCH' in s for s in supporting)
        has_bos = any('BOS' in s for s in supporting)
        has_ob = any('OB' in s or 'Order Block' in s for s in supporting)
        has_fvg = any('FVG' in s for s in supporting)
        has_liq = any('swept' in s.lower() for s in supporting)
        
        direction = 'Bullish' if signal == 'buy' else 'Bearish'

        if has_choch and has_ob and has_liq:
            return f"{direction} Reversal Setup"
        if has_bos and has_ob and has_fvg:
            return f"{direction} Continuation Setup"
        if has_ob:
            return f"{direction} OB Retest"
        if has_fvg:
            return f"{direction} FVG Fill"
        if has_liq:
            return f"{direction} Liquidity Grab"
        if context.bias == MarketBias.NEUTRAL:
            return "Range Trade"
        
        return f"{direction} Setup"

    def _create_no_trade_setup(self, reason: str) -> TradeSetup:
        """Create no-trade setup"""
        return TradeSetup(
            signal="Hold",
            entry_price=0,
            stop_loss=0,
            take_profit_1=0,
            take_profit_2=0,
            take_profit_3=0,
            risk_reward=0,
            quality=SetupQuality.NO_TRADE,
            confidence=0,
            setup_type="No Setup",
            supporting_factors=[],
            conflicting_factors=[reason],
            invalidation_level=0
        )


def generate_smc_signal(prediction: Dict, db_path: str = DB_PATH) -> Dict:
    """Main function untuk generate SMC-validated signal"""
    try:
        strategy = EnhancedSMCStrategy(db_path)
        setup = strategy.generate_trade_setup(prediction)

        return {
            'timestamp': datetime.now().isoformat(),
            'original_signal': prediction.get('signal', 'Hold'),
            'smc_signal': setup.signal,
            'entry_price': setup.entry_price,
            'stop_loss': setup.stop_loss,
            'take_profit_1': setup.take_profit_1,
            'take_profit_2': setup.take_profit_2,
            'take_profit_3': setup.take_profit_3,
            'risk_reward': setup.risk_reward,
            'setup_quality': setup.quality.value,
            'confidence': setup.confidence,
            'setup_type': setup.setup_type,
            'supporting_factors': setup.supporting_factors,
            'conflicting_factors': setup.conflicting_factors,
            'invalidation_level': setup.invalidation_level
        }

    except Exception as e:
        logger.error(f"Error generating SMC signal: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'original_signal': prediction.get('signal', 'Hold'),
            'smc_signal': 'Hold',
            'error': str(e)
        }


def save_smc_signal(signal_data: Dict, db_path: str = DB_PATH) -> bool:
    """Save SMC signal ke database"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS smc_signals (
                timestamp TEXT PRIMARY KEY,
                original_signal TEXT,
                smc_signal TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit_1 REAL,
                take_profit_2 REAL,
                take_profit_3 REAL,
                risk_reward REAL,
                setup_quality TEXT,
                confidence REAL,
                setup_type TEXT,
                supporting_factors TEXT,
                conflicting_factors TEXT,
                invalidation_level REAL
            )
            ''')

            cursor.execute('''
            INSERT OR REPLACE INTO smc_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data['timestamp'],
                signal_data['original_signal'],
                signal_data['smc_signal'],
                signal_data.get('entry_price', 0),
                signal_data.get('stop_loss', 0),
                signal_data.get('take_profit_1', 0),
                signal_data.get('take_profit_2', 0),
                signal_data.get('take_profit_3', 0),
                signal_data.get('risk_reward', 0),
                signal_data.get('setup_quality', 'No Trade'),
                signal_data.get('confidence', 0),
                signal_data.get('setup_type', 'Unknown'),
                '|'.join(signal_data.get('supporting_factors', [])),
                '|'.join(signal_data.get('conflicting_factors', [])),
                signal_data.get('invalidation_level', 0)
            ))

            conn.commit()
            logger.info("SMC signal saved successfully")
            return True

    except Exception as e:
        logger.error(f"Error saving SMC signal: {e}")
        return False


def get_market_analysis(db_path: str = DB_PATH) -> Dict:
    """Get comprehensive market analysis"""
    try:
        strategy = EnhancedSMCStrategy(db_path)
        if not strategy.load_data():
            return {'error': 'Failed to load data'}

        context = strategy.analyze_market_context()
        atr_value = strategy.atr if pd.notna(strategy.atr) else 0.0

        return {
            'timestamp': datetime.now().isoformat(),
            'current_price': float(strategy.current_price),
            'atr': round(float(atr_value), 2),
            'market_bias': context.bias.value,
            'trend_strength': round(context.trend_strength * 100, 1),
            'key_levels': {k: round(float(v), 2) if isinstance(v, (int, float)) and pd.notna(v) else v
                         for k, v in context.key_levels.items()},
            'active_pois': context.active_pois[:5],
            'active_weights': strategy.weights,
            'recommendation': _get_recommendation(context)
        }

    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        return {'error': str(e)}


def _get_recommendation(context: SMCContext) -> str:
    """Generate trading recommendation"""
    if context.bias == MarketBias.BULLISH and context.trend_strength > 0.5:
        zone = context.key_levels.get('current_zone', '')
        if zone == 'discount':
            return "Look for bullish setups at discount zone"
        return "Bullish bias - wait for pullback to discount"
    elif context.bias == MarketBias.BEARISH and context.trend_strength > 0.5:
        zone = context.key_levels.get('current_zone', '')
        if zone == 'premium':
            return "Look for bearish setups at premium zone"
        return "Bearish bias - wait for pullback to premium"
    return "Neutral market - wait for clear structure break"


if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced SMC Strategy - BALANCED OPTIMIZATION")
    print("=" * 60)

    analysis = get_market_analysis()
    if 'error' not in analysis:
        print(f"\n📊 Market Analysis:")
        print(f"  Current Price: ${analysis['current_price']:,.2f}")
        print(f"  ATR (14): ${analysis['atr']:,.2f}")
        print(f"  Market Bias: {analysis['market_bias'].upper()}")
        print(f"  Trend Strength: {analysis['trend_strength']}%")
        print(f"  Recommendation: {analysis['recommendation']}")

        if analysis['key_levels']:
            print(f"\n📍 Key Levels:")
            for k, v in analysis['key_levels'].items():
                if isinstance(v, float):
                    print(f"  {k}: ${v:,.2f}")
                else:
                    print(f"  {k}: {v}")

        if analysis.get('active_pois'):
            print(f"\n🎯 Active POIs ({len(analysis['active_pois'])}):")
            for poi in analysis['active_pois'][:3]:
                if poi['type'] == 'order_block':
                    print(f"  • {poi['direction'].upper()} OB: ${poi['bottom']:,.2f} - ${poi['top']:,.2f}")
                elif poi['type'] == 'fvg':
                    print(f"  • {poi['direction'].upper()} FVG: ${poi['bottom']:,.2f} - ${poi['top']:,.2f}")
                elif poi['type'] == 'liquidity':
                    print(f"  • {poi['direction'].upper()} Liquidity: ${poi['level']:,.2f}")

    print(f"\n{'='*60}")
    print("🎯 Generating Trade Signal:")
    print(f"{'='*60}")

    test_prediction = {'signal': 'Buy', 'confidence': 75}
    signal = generate_smc_signal(test_prediction)

    if 'error' not in signal:
        print(f"\n🎯 Trade Setup:")
        print(f"  SMC Signal: {signal['smc_signal']}")
        print(f"  Quality: {signal['setup_quality']}")
        print(f"  Confidence: {signal['confidence']}%")
        print(f"  Setup Type: {signal['setup_type']}")

        print(f"\n💰 Trade Levels:")
        print(f"  Entry: ${signal['entry_price']:,.2f}")
        print(f"  Stop Loss: ${signal['stop_loss']:,.2f}")
        print(f"  TP1: ${signal['take_profit_1']:,.2f}")
        print(f"  TP2: ${signal['take_profit_2']:,.2f}")
        print(f"  TP3: ${signal['take_profit_3']:,.2f}")
        print(f"  Risk/Reward: {signal['risk_reward']}")

        if signal['supporting_factors']:
            print(f"\n✅ Supporting ({len(signal['supporting_factors'])}):")
            for f in signal['supporting_factors']:
                print(f"  • {f}")

        if signal['conflicting_factors']:
            print(f"\n⚠️ Conflicting ({len(signal['conflicting_factors'])}):")
            for f in signal['conflicting_factors']:
                print(f"  • {f}")

        if signal['confidence'] >= 55:
            print(f"\n✅ RECOMMENDATION: Consider this trade")
        elif signal['confidence'] >= 40:
            print(f"\n⚠️ RECOMMENDATION: Marginal setup")
        else:
            print(f"\n❌ RECOMMENDATION: Skip this trade")

        if save_smc_signal(signal):
            print(f"\n✅ Signal saved to database")

    print(f"\n{'='*60}")