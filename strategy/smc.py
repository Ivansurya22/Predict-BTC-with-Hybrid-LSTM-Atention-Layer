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
    Enhanced SMC Strategy untuk Intraday Trading BTC 1H - OPTIMIZED VERSION

    Key Improvements:
    1. Dynamic weight adjustment berdasarkan market conditions
    2. Better confidence calculation dengan normalization
    3. Multi-factor scoring dengan decay untuk historical data
    4. Improved setup quality grading system
    5. Better risk management dengan adaptive levels
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.lookback_candles = 50
        self.recent_candles = 10
        self.smc_data = {}
        self.ohlcv_data = None
        self.technical_data = None
        self.current_price = None

        # Risk parameters untuk intraday
        self.max_risk_percent = 1.0
        self.min_rr_ratio = 2.0
        self.atr_multiplier_sl = 1.5
        self.atr_multiplier_tp = 3.0

        # OPTIMIZED: Base weights yang akan disesuaikan dinamis
        self.base_weights = {
            'market_structure': 25,
            'order_block': 20,
            'fvg': 12,
            'liquidity': 13,
            'premium_discount': 10,
            'retracement': 5,
            'technical_confluence': 15
        }

        # Weights akan diupdate berdasarkan kondisi market
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

                # Use ATR from technical indicators if available and valid
                atr_val = None
                if not self.technical_data.empty and 'atr_14' in self.technical_data.columns:
                    atr_val = self.technical_data['atr_14'].iloc[0]
                    if pd.notna(atr_val) and atr_val > 0:
                        self.atr = float(atr_val)
                        logger.info(f"ATR loaded from technical indicators: {self.atr:.2f}")
                    else:
                        atr_val = None

                # Fallback: Calculate ATR manually if not available
                if atr_val is None or atr_val <= 0:
                    self._calculate_atr()
                    logger.info(f"ATR calculated manually: {self.atr:.2f}")

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
            logger.warning(f"Not enough data for ATR calculation, using default: {self.atr}")
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
            logger.warning(f"ATR calculation resulted in invalid value, using default: {self.atr}")

    def analyze_market_context(self) -> SMCContext:
        """
        OPTIMIZED: Analisis konteks pasar dengan weighted scoring yang lebih baik
        """
        bias = MarketBias.NEUTRAL
        trend_strength = 0.0
        key_levels = {}
        active_pois = []

        # 1. Analisis Market Structure
        structure_bias, structure_strength = self._analyze_market_structure()

        # 2. Premium/Discount Zone
        premium_discount = self._identify_premium_discount_zone()
        key_levels.update(premium_discount)

        # 3. Active POIs
        active_pois = self._identify_active_pois()

        # 4. Swing Structure
        swing_bias = self._analyze_swing_structure()

        # 5. OPTIMIZED: Technical trend confirmation
        tech_bias, tech_strength = self._analyze_technical_trend()

        # OPTIMIZED: Improved bias calculation dengan weighted confidence
        bias_score = 0.0
        confidence_sum = 0.0

        # Market structure (40% weight)
        if structure_bias == MarketBias.BULLISH:
            bias_score += structure_strength * 0.4
            confidence_sum += 0.4
        elif structure_bias == MarketBias.BEARISH:
            bias_score -= structure_strength * 0.4
            confidence_sum += 0.4

        # Swing structure (30% weight)
        if swing_bias == MarketBias.BULLISH:
            bias_score += 0.3
            confidence_sum += 0.3
        elif swing_bias == MarketBias.BEARISH:
            bias_score -= 0.3
            confidence_sum += 0.3

        # Technical trend (30% weight)
        if tech_bias == MarketBias.BULLISH:
            bias_score += tech_strength * 0.3
            confidence_sum += 0.3
        elif tech_bias == MarketBias.BEARISH:
            bias_score -= tech_strength * 0.3
            confidence_sum += 0.3

        # Normalize bias score
        if confidence_sum > 0:
            normalized_bias_score = bias_score / confidence_sum
        else:
            normalized_bias_score = 0

        # Determine final bias dengan threshold yang lebih strict
        if normalized_bias_score > 0.35:
            bias = MarketBias.BULLISH
        elif normalized_bias_score < -0.35:
            bias = MarketBias.BEARISH
        else:
            bias = MarketBias.NEUTRAL

        trend_strength = min(abs(normalized_bias_score), 1.0)

        # OPTIMIZED: Adjust weights based on market condition
        self._adjust_weights_based_on_context(bias, trend_strength, active_pois)

        return SMCContext(
            bias=bias,
            trend_strength=trend_strength,
            key_levels=key_levels,
            active_pois=active_pois
        )

    def _analyze_technical_trend(self) -> Tuple[MarketBias, float]:
        """
        OPTIMIZED: Analisis technical trend dengan multi-timeframe perspective
        """
        if self.technical_data is None or self.technical_data.empty:
            return MarketBias.NEUTRAL, 0.0

        tech = self.technical_data.iloc[0]
        score = 0.0
        max_score = 0.0

        # EMA alignment (weight: 0.35)
        ema_9 = tech.get('ema_9', 0)
        ema_21 = tech.get('ema_21', 0)
        ema_50 = tech.get('ema_50', 0)

        if ema_9 > 0 and ema_21 > 0 and ema_50 > 0:
            if ema_9 > ema_21 > ema_50:
                score += 0.35
            elif ema_9 < ema_21 < ema_50:
                score -= 0.35
            max_score += 0.35

        # ADX trend strength (weight: 0.25)
        adx = tech.get('adx_14', 0)
        plus_di = tech.get('plus_di_14', 0)
        minus_di = tech.get('minus_di_14', 0)

        if adx > 0 and plus_di > 0 and minus_di > 0:
            if adx > 25:  # Strong trend
                if plus_di > minus_di:
                    score += 0.25 * (adx / 50)  # Scale by ADX strength
                else:
                    score -= 0.25 * (adx / 50)
            max_score += 0.25

        # MACD (weight: 0.20)
        macd = tech.get('macd', 0)
        macd_signal = tech.get('macd_signal', 0)
        macd_hist = tech.get('macd_hist', 0)

        if macd != 0 and macd_signal != 0:
            if macd > macd_signal and macd_hist > 0:
                score += 0.20
            elif macd < macd_signal and macd_hist < 0:
                score -= 0.20
            max_score += 0.20

        # RSI momentum (weight: 0.20)
        rsi = tech.get('rsi_14', 50)
        if rsi > 0:
            if 50 < rsi < 70:
                score += 0.20
            elif 30 < rsi < 50:
                score -= 0.20
            max_score += 0.20

        # Determine bias and strength
        if max_score > 0:
            normalized_score = score / max_score
            if normalized_score > 0.3:
                return MarketBias.BULLISH, abs(normalized_score)
            elif normalized_score < -0.3:
                return MarketBias.BEARISH, abs(normalized_score)

        return MarketBias.NEUTRAL, 0.0

    def _adjust_weights_based_on_context(self, bias: MarketBias,
                                        trend_strength: float,
                                        active_pois: List[Dict]):
        """
        OPTIMIZED: Dynamic weight adjustment berdasarkan kondisi market
        """
        self.weights = self.base_weights.copy()

        # Jika trend kuat, prioritaskan market structure dan momentum
        if trend_strength > 0.6:
            self.weights['market_structure'] = 30  # +5
            self.weights['technical_confluence'] = 20  # +5
            self.weights['order_block'] = 18  # -2
            self.weights['premium_discount'] = 7  # -3

        # Jika banyak POIs aktif, prioritaskan POI analysis
        if len(active_pois) > 3:
            self.weights['order_block'] = 25  # +5
            self.weights['fvg'] = 15  # +3
            self.weights['liquidity'] = 15  # +2
            self.weights['market_structure'] = 20  # -5

        # Jika market neutral/ranging, prioritaskan levels
        if bias == MarketBias.NEUTRAL:
            self.weights['premium_discount'] = 15  # +5
            self.weights['liquidity'] = 18  # +5
            self.weights['market_structure'] = 20  # -5
            self.weights['technical_confluence'] = 12  # -3

        # Normalize weights to 100
        total = sum(self.weights.values())
        self.weights = {k: (v / total) * 100 for k, v in self.weights.items()}

    def _analyze_market_structure(self) -> Tuple[MarketBias, float]:
        """Analisis BOS dan CHOCH untuk menentukan market structure"""
        if 'bos_choch' not in self.smc_data or self.smc_data['bos_choch'].empty:
            return MarketBias.NEUTRAL, 0.0

        df = self.smc_data['bos_choch'].head(self.recent_candles)

        bullish_bos = (df['BOS'] == 1).sum()
        bearish_bos = (df['BOS'] == -1).sum()
        bullish_choch = (df['CHOCH'] == 1).sum()
        bearish_choch = (df['CHOCH'] == -1).sum()

        # OPTIMIZED: Weight recent signals more heavily with exponential decay
        decay_factor = 0.8
        bull_score = 0
        bear_score = 0

        for i, row in df.iterrows():
            weight = decay_factor ** i
            if row['BOS'] == 1:
                bull_score += weight * 1
            elif row['BOS'] == -1:
                bear_score += weight * 1
            if row['CHOCH'] == 1:
                bull_score += weight * 2  # CHOCH more important
            elif row['CHOCH'] == -1:
                bear_score += weight * 2

        total = bull_score + bear_score
        if total == 0:
            return MarketBias.NEUTRAL, 0.0

        if bull_score > bear_score:
            return MarketBias.BULLISH, bull_score / (total + 1)
        elif bear_score > bull_score:
            return MarketBias.BEARISH, bear_score / (total + 1)

        return MarketBias.NEUTRAL, 0.0

    def _identify_premium_discount_zone(self) -> Dict[str, float]:
        """Identifikasi Premium/Discount Zone berdasarkan swing range"""
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
        """Identifikasi Point of Interest yang masih aktif"""
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

    def _analyze_swing_structure(self) -> MarketBias:
        """Analisis swing highs dan lows untuk trend direction"""
        if 'swing_highs_lows' not in self.smc_data:
            return MarketBias.NEUTRAL

        df = self.smc_data['swing_highs_lows'].head(10)

        highs = df[df['HighLow'] == 1]['Level'].dropna().tolist()
        lows = df[df['HighLow'] == -1]['Level'].dropna().tolist()

        if len(highs) < 2 or len(lows) < 2:
            return MarketBias.NEUTRAL

        # Higher highs dan higher lows = bullish
        hh = highs[0] > highs[1] if len(highs) >= 2 else False
        hl = lows[0] > lows[1] if len(lows) >= 2 else False

        # Lower highs dan lower lows = bearish
        lh = highs[0] < highs[1] if len(highs) >= 2 else False
        ll = lows[0] < lows[1] if len(lows) >= 2 else False

        if hh and hl:
            return MarketBias.BULLISH
        elif lh and ll:
            return MarketBias.BEARISH

        return MarketBias.NEUTRAL

    def generate_trade_setup(self, prediction: Dict) -> TradeSetup:
        """Generate complete trade setup dengan entry, SL, TP"""
        if not self.load_data():
            return self._create_no_trade_setup("Failed to load data")

        signal = prediction.get('signal', 'Hold').lower()
        context = self.analyze_market_context()

        # Evaluate signal quality
        score, supporting, conflicting = self._evaluate_signal_quality(signal, context)

        # Determine setup quality
        quality = self._determine_setup_quality(score, signal, context)

        # Calculate entry, SL, TP
        entry, sl, tp1, tp2, tp3, invalidation = self._calculate_trade_levels(
            signal, context, quality
        )

        # Calculate risk-reward
        if entry and sl and tp1:
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)
            rr = reward / risk if risk > 0 else 0
        else:
            rr = 0

        # OPTIMIZED: Final confidence calculation
        confidence = self._calculate_final_confidence(
            score, quality, rr, context, supporting, conflicting
        )

        # Identify setup type
        setup_type = self._identify_setup_type(signal, context, supporting)

        return TradeSetup(
            signal=signal.capitalize(),
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
        OPTIMIZED: Evaluate signal dengan normalization dan weighted scoring
        """
        score = 0.0
        supporting = []
        conflicting = []

        # 1. Market Structure Alignment
        ms_score = self._evaluate_market_structure_alignment(
            signal, context, supporting, conflicting
        )
        score += ms_score * (self.weights['market_structure'] / 100)

        # 2. Order Block Proximity
        ob_score = self._evaluate_order_block(
            signal, context, supporting, conflicting
        )
        score += ob_score * (self.weights['order_block'] / 100)

        # 3. FVG Analysis
        fvg_score = self._evaluate_fvg(
            signal, supporting, conflicting
        )
        score += fvg_score * (self.weights['fvg'] / 100)

        # 4. Liquidity Analysis
        liq_score = self._evaluate_liquidity(
            signal, supporting, conflicting
        )
        score += liq_score * (self.weights['liquidity'] / 100)

        # 5. Premium/Discount Zone
        pd_score = self._evaluate_premium_discount(
            signal, context, supporting, conflicting
        )
        score += pd_score * (self.weights['premium_discount'] / 100)

        # 6. Retracement Quality
        ret_score = self._evaluate_retracement(
            signal, supporting, conflicting
        )
        score += ret_score * (self.weights['retracement'] / 100)

        # 7. Technical Indicator Confluence
        tech_score = self._evaluate_technical_confluence(
            signal, supporting, conflicting
        )
        score += tech_score * (self.weights['technical_confluence'] / 100)

        # OPTIMIZED: Convert to 0-100 scale
        final_score = score * 100

        return final_score, supporting, conflicting

    def _evaluate_technical_confluence(self, signal: str,
                                       supporting: List, conflicting: List) -> float:
        """
        OPTIMIZED: Evaluate technical confluence dengan better normalization
        """
        if self.technical_data is None or self.technical_data.empty:
            return 0.5  # Neutral score instead of 0

        score = 0.0
        max_possible = 0.0
        tech = self.technical_data.iloc[0]
        is_buy = signal == 'buy'

        # 1. EMA Alignment (max: 0.35)
        ema_9 = tech.get('ema_9', 0)
        ema_21 = tech.get('ema_21', 0)
        ema_50 = tech.get('ema_50', 0)

        if ema_9 > 0 and ema_21 > 0 and ema_50 > 0:
            max_possible += 0.35
            if is_buy:
                if ema_9 > ema_21 > ema_50:
                    score += 0.25
                    supporting.append("EMA alignment bullish (9>21>50)")
                elif ema_9 < ema_21 < ema_50:
                    conflicting.append("EMA alignment bearish")
                if self.current_price > ema_21:
                    score += 0.1
                    supporting.append("Price above EMA 21")
            else:
                if ema_9 < ema_21 < ema_50:
                    score += 0.25
                    supporting.append("EMA alignment bearish (9<21<50)")
                elif ema_9 > ema_21 > ema_50:
                    conflicting.append("EMA alignment bullish")
                if self.current_price < ema_21:
                    score += 0.1
                    supporting.append("Price below EMA 21")

        # 2. RSI Confluence (max: 0.20)
        rsi = tech.get('rsi_14', 50)
        if rsi > 0:
            max_possible += 0.20
            if is_buy:
                if 30 <= rsi <= 50:
                    score += 0.20
                    supporting.append(f"RSI in buy zone ({rsi:.1f})")
                elif rsi < 30:
                    score += 0.15
                    supporting.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    conflicting.append(f"RSI overbought ({rsi:.1f})")
            else:
                if 50 <= rsi <= 70:
                    score += 0.20
                    supporting.append(f"RSI in sell zone ({rsi:.1f})")
                elif rsi > 70:
                    score += 0.15
                    supporting.append(f"RSI overbought ({rsi:.1f})")
                elif rsi < 30:
                    conflicting.append(f"RSI oversold ({rsi:.1f})")

        # 3. MACD Confluence (max: 0.20)
        macd = tech.get('macd', 0)
        macd_signal = tech.get('macd_signal', 0)
        macd_hist = tech.get('macd_hist', 0)

        if macd != 0 and macd_signal != 0:
            max_possible += 0.20
            if is_buy:
                if macd > macd_signal and macd_hist > 0:
                    score += 0.20
                    supporting.append("MACD bullish crossover")
                elif macd < macd_signal:
                    conflicting.append("MACD bearish")
            else:
                if macd < macd_signal and macd_hist < 0:
                    score += 0.20
                    supporting.append("MACD bearish crossover")
                elif macd > macd_signal:
                    conflicting.append("MACD bullish")

        # 4. ADX Trend Strength (max: 0.15)
        adx = tech.get('adx_14', 0)
        plus_di = tech.get('plus_di_14', 0)
        minus_di = tech.get('minus_di_14', 0)

        if adx > 0:
            max_possible += 0.15
            if adx > 25:
                if is_buy and plus_di > minus_di:
                    score += 0.15
                    supporting.append(f"Strong bullish trend (ADX: {adx:.1f})")
                elif not is_buy and minus_di > plus_di:
                    score += 0.15
                    supporting.append(f"Strong bearish trend (ADX: {adx:.1f})")
                elif is_buy and minus_di > plus_di:
                    conflicting.append("ADX shows bearish dominance")
                elif not is_buy and plus_di > minus_di:
                    conflicting.append("ADX shows bullish dominance")
            elif adx < 20:
                conflicting.append(f"Weak trend (ADX: {adx:.1f})")

        # 5. Bollinger Bands (max: 0.10)
        bb_upper = tech.get('bb_upper_20', 0)
        bb_lower = tech.get('bb_lower_20', 0)

        if bb_upper > 0 and bb_lower > 0:
            max_possible += 0.10
            if is_buy:
                if self.current_price <= bb_lower:
                    score += 0.10
                    supporting.append("Price at lower Bollinger Band")
                elif self.current_price >= bb_upper:
                    conflicting.append("Price at upper Bollinger Band")
            else:
                if self.current_price >= bb_upper:
                    score += 0.10
                    supporting.append("Price at upper Bollinger Band")
                elif self.current_price <= bb_lower:
                    conflicting.append("Price at lower Bollinger Band")

        # Normalize to 0-1 range
        return (score / max_possible) if max_possible > 0 else 0.5

    def _evaluate_market_structure_alignment(self, signal: str, context: SMCContext,
                                            supporting: List, conflicting: List) -> float:
        """
        OPTIMIZED: Better scoring dengan recency weight
        """
        score = 0.0
        max_possible = 1.0

        if 'bos_choch' not in self.smc_data:
            return 0.5

        df = self.smc_data['bos_choch'].head(self.recent_candles)
        is_buy = signal == 'buy'

        # Bias alignment (40% of total)
        if is_buy:
            if context.bias == MarketBias.BULLISH:
                score += 0.4 * context.trend_strength
                supporting.append(f"Signal aligns with bullish bias (strength: {context.trend_strength:.0%})")
            elif context.bias == MarketBias.BEARISH:
                conflicting.append("Buy signal against bearish market bias")
        else:
            if context.bias == MarketBias.BEARISH:
                score += 0.4 * context.trend_strength
                supporting.append(f"Signal aligns with bearish bias (strength: {context.trend_strength:.0%})")
            elif context.bias == MarketBias.BULLISH:
                conflicting.append("Sell signal against bullish market bias")

        # Recent structure confirmations (60% of total)
        decay_factor = 0.85
        for i, row in df.iterrows():
            weight = decay_factor ** i

            if is_buy:
                if row['CHOCH'] == 1:
                    score += 0.3 * weight
                    if i == 0:
                        supporting.append("Recent bullish CHoCH confirms trend change")
                elif row['BOS'] == 1:
                    score += 0.2 * weight
                    if i == 0:
                        supporting.append("Recent bullish BOS confirms continuation")
                elif row['CHOCH'] == -1:
                    if i == 0:
                        conflicting.append("Recent bearish CHoCH present")
            else:
                if row['CHOCH'] == -1:
                    score += 0.3 * weight
                    if i == 0:
                        supporting.append("Recent bearish CHoCH confirms trend change")
                elif row['BOS'] == -1:
                    score += 0.2 * weight
                    if i == 0:
                        supporting.append("Recent bearish BOS confirms continuation")
                elif row['CHOCH'] == 1:
                    if i == 0:
                        conflicting.append("Recent bullish CHoCH present")

        return min(score / max_possible, 1.0)

    def _evaluate_order_block(self, signal: str, context: SMCContext,
                             supporting: List, conflicting: List) -> float:
        """
        OPTIMIZED: Better OB scoring dengan distance and quality factors
        """
        if 'order_block' not in self.smc_data:
            return 0.5

        score = 0.0
        max_possible = 1.0
        df = self.smc_data['order_block']
        is_buy = signal == 'buy'

        # Find relevant OBs
        obs = df[df['OB'] == (1 if is_buy else -1)]

        if obs.empty:
            return 0.4  # Slight penalty for no OB

        # Check proximity and quality
        for _, ob in obs.head(3).iterrows():
            if ob['MitigatedIndex'] != 0:
                continue

            top, bottom = ob['Top'], ob['Bottom']
            ob_mid = (top + bottom) / 2
            distance = abs(self.current_price - ob_mid)
            ob_size = top - bottom

            # Price within OB (max: 0.5)
            if bottom <= self.current_price <= top:
                base_score = 0.5

                # Quality bonus (max: +0.2)
                if ob['Percentage'] > 70:
                    base_score += 0.2
                    supporting.append(f"Strong OB (strength: {ob['Percentage']:.0f}%)")
                elif ob['Percentage'] > 50:
                    base_score += 0.1

                # Volume bonus (max: +0.15)
                if ob['OBVolume'] > df['OBVolume'].median():
                    base_score += 0.15
                    supporting.append("High volume Order Block")

                score += base_score
                supporting.append(f"Price at {'bullish' if is_buy else 'bearish'} OB zone")
                break

            # Price approaching OB (max: 0.3)
            elif distance < ob_size * 2:
                proximity_score = 0.3 * (1 - distance / (ob_size * 2))
                score += proximity_score
                supporting.append(f"Price near {'bullish' if is_buy else 'bearish'} OB")
                break

        return min(score / max_possible, 1.0)

    def _evaluate_fvg(self, signal: str, supporting: List, conflicting: List) -> float:
        """
        OPTIMIZED: Better FVG evaluation dengan proximity
        """
        if 'fvg' not in self.smc_data:
            return 0.5

        score = 0.0
        max_possible = 1.0
        df = self.smc_data['fvg'].head(self.recent_candles)
        is_buy = signal == 'buy'

        target_fvg = 1 if is_buy else -1
        opposite_fvg = -1 if is_buy else 1

        # Check for aligned unmitigated FVG
        aligned_fvgs = df[(df['FVG'] == target_fvg) & (df['MitigatedIndex'] == 0)]

        if not aligned_fvgs.empty:
            fvg = aligned_fvgs.iloc[0]

            # FVG exists (base: 0.4)
            score += 0.4
            supporting.append(f"Unmitigated {'bullish' if is_buy else 'bearish'} FVG present")

            # Price within FVG (bonus: +0.4)
            if fvg['Bottom'] <= self.current_price <= fvg['Top']:
                score += 0.4
                supporting.append(f"Price at {'bullish' if is_buy else 'bearish'} FVG zone")
            # Price approaching FVG (bonus: +0.2)
            else:
                fvg_mid = (fvg['Top'] + fvg['Bottom']) / 2
                distance = abs(self.current_price - fvg_mid)
                fvg_size = fvg['Top'] - fvg['Bottom']
                if distance < fvg_size * 1.5:
                    score += 0.2
        else:
            # Mitigated FVG (small bonus: 0.2)
            mitigated = df[(df['FVG'] == target_fvg) & (df['MitigatedIndex'] != 0)]
            if not mitigated.empty:
                score += 0.2
                supporting.append(f"{'Bullish' if is_buy else 'Bearish'} FVG detected (mitigated)")

        # Check for conflicting FVG
        opposite_unmit = df[(df['FVG'] == opposite_fvg) & (df['MitigatedIndex'] == 0)]
        if not opposite_unmit.empty:
            conflicting.append(f"Unmitigated {'bearish' if is_buy else 'bullish'} FVG present")

        return min(score / max_possible, 1.0)

    def _evaluate_liquidity(self, signal: str, supporting: List, conflicting: List) -> float:
        """
        OPTIMIZED: Better liquidity evaluation
        """
        if 'liquidity' not in self.smc_data:
            return 0.5

        score = 0.0
        max_possible = 1.0
        df = self.smc_data['liquidity'].head(self.recent_candles)
        is_buy = signal == 'buy'

        # Recent liquidity sweep (high value)
        if is_buy:
            swept = df[(df['Liquidity'] == -1) & (df['Swept'] != 0)]
            if not swept.empty and swept.index[0] < 3:  # Very recent
                score += 0.7
                supporting.append("Recent sell-side liquidity swept (bullish)")

            unswept = df[(df['Liquidity'] == 1) & (df['Swept'] == 0)]
            if not unswept.empty:
                score += 0.3
                supporting.append("Buy-side liquidity target above")
        else:
            swept = df[(df['Liquidity'] == 1) & (df['Swept'] != 0)]
            if not swept.empty and swept.index[0] < 3:
                score += 0.7
                supporting.append("Recent buy-side liquidity swept (bearish)")

            unswept = df[(df['Liquidity'] == -1) & (df['Swept'] == 0)]
            if not unswept.empty:
                score += 0.3
                supporting.append("Sell-side liquidity target below")

        return min(score / max_possible, 1.0)

    def _evaluate_premium_discount(self, signal: str, context: SMCContext,
                                   supporting: List, conflicting: List) -> float:
        """
        OPTIMIZED: Better zone evaluation dengan gradient scoring
        """
        score = 0.5  # Start neutral

        if 'equilibrium' not in context.key_levels:
            return score

        eq = context.key_levels['equilibrium']
        premium_zone = context.key_levels.get('premium_zone', eq)
        discount_zone = context.key_levels.get('discount_zone', eq)
        is_buy = signal == 'buy'

        # Calculate position in range (0 = bottom, 1 = top)
        range_size = context.key_levels.get('range_high', eq) - context.key_levels.get('range_low', eq)
        if range_size > 0:
            position = (self.current_price - context.key_levels.get('range_low', eq)) / range_size
        else:
            return score

        if is_buy:
            # Ideal: deep discount (position < 0.3)
            if position < 0.3:
                score = 1.0
                supporting.append(f"Price in deep discount zone (optimal for buy)")
            # Good: discount (position < 0.5)
            elif position < 0.5:
                score = 0.7
                supporting.append("Price in discount zone (good for buy)")
            # Neutral: equilibrium (position 0.4-0.6)
            elif position < 0.6:
                score = 0.5
            # Bad: premium (position > 0.7)
            elif position > 0.7:
                score = 0.2
                conflicting.append("Buy signal in premium zone (risky)")
            else:
                score = 0.4
        else:  # Sell
            # Ideal: deep premium (position > 0.7)
            if position > 0.7:
                score = 1.0
                supporting.append(f"Price in deep premium zone (optimal for sell)")
            # Good: premium (position > 0.5)
            elif position > 0.5:
                score = 0.7
                supporting.append("Price in premium zone (good for sell)")
            # Neutral: equilibrium
            elif position > 0.4:
                score = 0.5
            # Bad: discount (position < 0.3)
            elif position < 0.3:
                score = 0.2
                conflicting.append("Sell signal in discount zone (risky)")
            else:
                score = 0.4

        return score

    def _evaluate_retracement(self, signal: str, supporting: List, conflicting: List) -> float:
        """
        OPTIMIZED: Better retracement evaluation
        """
        if 'retracements' not in self.smc_data:
            return 0.5

        score = 0.5  # Neutral default
        df = self.smc_data['retracements'].head(self.recent_candles)
        is_buy = signal == 'buy'

        target_dir = 1 if is_buy else -1
        retr = df[df['Direction'] == target_dir]

        if not retr.empty:
            current_retr = retr.iloc[0]['CurrentRetracement%']

            # Optimal: 50-70% retracement
            if 50 <= current_retr <= 70:
                score = 1.0
                supporting.append(f"Optimal retracement ({current_retr:.1f}%)")
            # Good: 38-50% or 70-79%
            elif 38 <= current_retr < 50:
                score = 0.7
                supporting.append(f"Shallow retracement ({current_retr:.1f}%)")
            elif 70 < current_retr <= 79:
                score = 0.6
                supporting.append(f"Deep but acceptable retracement ({current_retr:.1f}%)")
            # Risky: >79%
            elif current_retr > 79:
                score = 0.3
                conflicting.append(f"Very deep retracement risk ({current_retr:.1f}%)")
            # Too shallow: <38%
            elif current_retr < 38:
                score = 0.4
                conflicting.append(f"Insufficient retracement ({current_retr:.1f}%)")

        return score

    def _determine_setup_quality(self, score: float, signal: str, context: SMCContext) -> SetupQuality:
        """
        OPTIMIZED: Better quality grading dengan multiple factors
        """
        is_buy = signal == 'buy'
        is_sell = signal == 'sell'

        # Bias alignment bonus
        bias_aligned = (
            (is_buy and context.bias == MarketBias.BULLISH) or
            (is_sell and context.bias == MarketBias.BEARISH)
        )

        # Strong trend bonus
        strong_trend = context.trend_strength > 0.6

        # Adjusted thresholds
        if score >= 75:
            if bias_aligned and strong_trend:
                return SetupQuality.A_PLUS
            elif bias_aligned or strong_trend:
                return SetupQuality.A
            else:
                return SetupQuality.B

        elif score >= 60:
            if bias_aligned and strong_trend:
                return SetupQuality.A
            elif bias_aligned:
                return SetupQuality.B
            else:
                return SetupQuality.C

        elif score >= 45:
            if bias_aligned:
                return SetupQuality.B
            else:
                return SetupQuality.C

        elif score >= 35:
            return SetupQuality.C

        else:
            return SetupQuality.NO_TRADE

    def _calculate_trade_levels(self, signal: str, context: SMCContext,
                                quality: SetupQuality) -> Tuple[float, float, float, float, float, float]:
        """
        OPTIMIZED: Better level calculation dengan adaptive sizing
        """
        is_buy = signal == 'buy'
        entry = self.current_price

        # Ensure valid ATR
        if pd.isna(self.atr) or self.atr <= 0:
            self.atr = entry * 0.02

        swing_df = self.smc_data.get('swing_highs_lows', pd.DataFrame())
        ob_df = self.smc_data.get('order_block', pd.DataFrame())

        # Quality-based risk adjustment
        quality_multipliers = {
            SetupQuality.A_PLUS: 0.8,  # Tighter SL for A+ setups
            SetupQuality.A: 0.9,
            SetupQuality.B: 1.0,
            SetupQuality.C: 1.2,
            SetupQuality.NO_TRADE: 1.5
        }
        risk_multiplier = quality_multipliers.get(quality, 1.0)

        if is_buy:
            # Find structural support
            if not swing_df.empty and 'HighLow' in swing_df.columns:
                swing_lows = swing_df[swing_df['HighLow'] == -1]['Level'].dropna()
                recent_low = float(swing_lows.iloc[0]) if not swing_lows.empty else self.current_low
            else:
                recent_low = self.current_low

            # Find OB support
            if not ob_df.empty and 'OB' in ob_df.columns:
                bullish_obs = ob_df[(ob_df['OB'] == 1) & (ob_df['Bottom'] < entry) & (ob_df['MitigatedIndex'] == 0)]
                ob_bottom = float(bullish_obs['Bottom'].iloc[0]) if not bullish_obs.empty else recent_low
            else:
                ob_bottom = recent_low

            # Use the higher support
            structural_sl = max(recent_low, ob_bottom)
            atr_buffer = self.atr * 0.3 * risk_multiplier
            sl = structural_sl - atr_buffer
            invalidation = structural_sl - (self.atr * 0.8)

            # Find resistance for targets
            if not swing_df.empty and 'HighLow' in swing_df.columns:
                swing_highs = swing_df[swing_df['HighLow'] == 1]['Level'].dropna()
                if not swing_highs.empty:
                    highs_above = swing_highs[swing_highs > entry]
                    nearest_high = float(highs_above.min()) if not highs_above.empty else float(swing_highs.max())
                else:
                    nearest_high = entry + (self.atr * 3)
            else:
                nearest_high = entry + (self.atr * 3)

            risk = entry - sl
            if risk <= 0:
                risk = self.atr * risk_multiplier
                sl = entry - risk

            # Dynamic R:R based on quality
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

            # Adjust TP3 to structural target
            tp3 = max(tp3, nearest_high)

        else:  # Sell
            # Find structural resistance
            if not swing_df.empty and 'HighLow' in swing_df.columns:
                swing_highs = swing_df[swing_df['HighLow'] == 1]['Level'].dropna()
                recent_high = float(swing_highs.iloc[0]) if not swing_highs.empty else self.current_high
            else:
                recent_high = self.current_high

            # Find OB resistance
            if not ob_df.empty and 'OB' in ob_df.columns:
                bearish_obs = ob_df[(ob_df['OB'] == -1) & (ob_df['Top'] > entry) & (ob_df['MitigatedIndex'] == 0)]
                ob_top = float(bearish_obs['Top'].iloc[0]) if not bearish_obs.empty else recent_high
            else:
                ob_top = recent_high

            # Use the lower resistance
            structural_sl = min(recent_high, ob_top)
            atr_buffer = self.atr * 0.3 * risk_multiplier
            sl = structural_sl + atr_buffer
            invalidation = structural_sl + (self.atr * 0.8)

            # Find support for targets
            if not swing_df.empty and 'HighLow' in swing_df.columns:
                swing_lows = swing_df[swing_df['HighLow'] == -1]['Level'].dropna()
                if not swing_lows.empty:
                    lows_below = swing_lows[swing_lows < entry]
                    nearest_low = float(lows_below.max()) if not lows_below.empty else float(swing_lows.min())
                else:
                    nearest_low = entry - (self.atr * 3)
            else:
                nearest_low = entry - (self.atr * 3)

            risk = sl - entry
            if risk <= 0:
                risk = self.atr * risk_multiplier
                sl = entry + risk

            # Dynamic R:R based on quality
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

            # Adjust TP3 to structural target
            tp3 = min(tp3, nearest_low)

        return entry, sl, tp1, tp2, tp3, invalidation

    def _calculate_final_confidence(self, score: float, quality: SetupQuality,
                                   rr: float, context: SMCContext,
                                   supporting: List[str], conflicting: List[str]) -> float:
        """
        OPTIMIZED: Better confidence calculation dengan multiple weighted factors
        """
        # Start with base score (0-100)
        confidence = score

        # Quality bonus/penalty (±10)
        quality_adjustments = {
            SetupQuality.A_PLUS: 10,
            SetupQuality.A: 5,
            SetupQuality.B: 0,
            SetupQuality.C: -5,
            SetupQuality.NO_TRADE: -15
        }
        confidence += quality_adjustments.get(quality, 0)

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
        elif rr < 2:
            confidence -= 5

        # Trend strength bonus (max: +8)
        confidence += context.trend_strength * 8

        # Supporting vs conflicting factors ratio (±7)
        total_factors = len(supporting) + len(conflicting)
        if total_factors > 0:
            support_ratio = len(supporting) / total_factors
            if support_ratio > 0.75:
                confidence += 7
            elif support_ratio > 0.6:
                confidence += 4
            elif support_ratio < 0.4:
                confidence -= 7
            elif support_ratio < 0.5:
                confidence -= 4

        # Active POI bonus (max: +5)
        poi_count = len(context.active_pois)
        if poi_count >= 3:
            confidence += 5
        elif poi_count >= 2:
            confidence += 3
        elif poi_count >= 1:
            confidence += 1

        # Ensure confidence is in valid range
        return max(0, min(100, confidence))

    def _identify_setup_type(self, signal: str, context: SMCContext,
                            supporting: List[str]) -> str:
        """Identify specific SMC setup type"""
        has_choch = any('CHoCH' in s for s in supporting)
        has_bos = any('BOS' in s for s in supporting)
        has_ob = any('Order Block' in s or 'OB' in s for s in supporting)
        has_fvg = any('FVG' in s for s in supporting)
        has_liq_sweep = any('swept' in s.lower() for s in supporting)
        has_optimal_retr = any('Optimal retracement' in s for s in supporting)
        has_strong_trend = any('Strong' in s and 'trend' in s.lower() for s in supporting)

        is_buy = signal == 'buy'
        direction = 'Bullish' if is_buy else 'Bearish'

        # Priority-based identification
        if has_choch and has_ob and has_liq_sweep:
            return f"{direction} Reversal (A+ Setup)"

        if has_bos and has_ob and has_fvg:
            return f"{direction} Continuation (OB+FVG)"

        if has_ob and has_optimal_retr and has_strong_trend:
            return f"{direction} OB Retest (High Probability)"

        if has_ob and has_optimal_retr:
            return f"{direction} OB Retest"

        if has_fvg and has_optimal_retr:
            return f"{direction} FVG Fill"

        if has_liq_sweep and (has_ob or has_fvg):
            return f"{direction} Liquidity Grab"

        if has_bos and has_strong_trend:
            return f"{direction} Trend Continuation"

        if has_bos:
            return f"{direction} Breakout"

        if has_choch:
            return f"{direction} Trend Change"

        if context.bias == MarketBias.NEUTRAL:
            return "Range Trade"

        return f"{direction} Setup"

    def _create_no_trade_setup(self, reason: str) -> TradeSetup:
        """Create a no-trade setup"""
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
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def _get_recommendation(context: SMCContext) -> str:
    """Generate trading recommendation based on context"""
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
    print("Enhanced SMC Strategy - OPTIMIZED VERSION")
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
            print(f"\n🎯 Active Points of Interest ({len(analysis['active_pois'])}):")
            for poi in analysis['active_pois'][:3]:
                if poi['type'] == 'order_block':
                    print(f"  • {poi['direction'].upper()} OB: ${poi['bottom']:,.2f} - ${poi['top']:,.2f} (Strength: {poi['strength']:.0f}%)")
                elif poi['type'] == 'fvg':
                    print(f"  • {poi['direction'].upper()} FVG: ${poi['bottom']:,.2f} - ${poi['top']:,.2f}")
                elif poi['type'] == 'liquidity':
                    print(f"  • {poi['direction'].upper()} Liquidity: ${poi['level']:,.2f}")

        if analysis.get('active_weights'):
            print(f"\n⚖️ Dynamic Weights (adjusted for current conditions):")
            for k, v in sorted(analysis['active_weights'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {k.replace('_', ' ').title()}: {v:.1f}%")
    else:
        print(f"\n❌ Market Analysis Error: {analysis['error']}")

    # Test with main prediction signal
    print(f"\n{'='*60}")
    print("🎯 Generating Primary Trade Signal:")
    print(f"{'='*60}")

    test_prediction = {'signal': 'Buy', 'confidence': 75}
    signal = generate_smc_signal(test_prediction)

    if 'error' not in signal:
        print(f"\n🎯 Trade Setup for {signal['original_signal'].upper()} Signal:")
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
        print(f"  Invalidation: ${signal['invalidation_level']:,.2f}")

        if signal['supporting_factors']:
            print(f"\n✅ Supporting Factors ({len(signal['supporting_factors'])}):")
            for f in signal['supporting_factors']:
                print(f"  • {f}")

        if signal['conflicting_factors']:
            print(f"\n⚠️ Conflicting Factors ({len(signal['conflicting_factors'])}):")
            for f in signal['conflicting_factors']:
                print(f"  • {f}")

        # Determine if trade should be taken
        if signal['confidence'] >= 60 and signal['setup_quality'] in ['A+', 'A', 'B']:
            print(f"\n✅ RECOMMENDATION: Consider taking this trade")
        elif signal['confidence'] >= 45:
            print(f"\n⚠️ RECOMMENDATION: Marginal setup, proceed with caution")
        else:
            print(f"\n❌ RECOMMENDATION: Skip this trade")

        # Save to database
        print(f"\n💾 Saving signal to database...")
        if save_smc_signal(signal):
            print(f"✅ Signal saved successfully to database")
        else:
            print(f"❌ Failed to save signal to database")

    else:
        print(f"\n❌ Signal Generation Error: {signal.get('error', 'Unknown error')}")

    print(f"\n{'='*60}")
    print("Analysis completed.")
    print(f"{'='*60}")
