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

# FIX 1: SMC_TABLES harus berupa dictionary dengan syntax yang benar
SMC_TABLES = {
    "bos_choch": "smc_btc_1h_bos_choch",
    "order_block": "smc_btc_1h_order_block",
    "fvg": "smc_btc_1h_fvg",
    "liquidity": "smc_btc_1h_liquidity",
    "swing_highs_lows": "smc_btc_1h_swing_highs_lows",
    "retracements": "smc_btc_1h_retracements"
}

# FIX 2: MarketBias Enum harus didefinisikan SEBELUM digunakan
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
    active_pois: List[Dict]  # Points of Interest


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
    Enhanced SMC Strategy untuk Intraday Trading BTC 1H

    Improvements:
    1. Multi-timeframe context (menggunakan data yang ada)
    2. Premium/Discount zone analysis
    3. POI (Point of Interest) identification
    4. Entry model dengan konfirmasi
    5. Dynamic SL/TP berdasarkan struktur
    6. Risk management terintegrasi
    7. Session-aware trading
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

        # Weight scoring untuk setiap komponen SMC
        self.weights = {
            'market_structure': 25,
            'order_block': 20,
            'fvg': 12,
            'liquidity': 13,
            'premium_discount': 10,
            'retracement': 5,
            'technical_confluence': 15
        }

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

                # Load Technical Indicators - ambil data terbaru yang valid (bukan 0)
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
            self.atr = 100.0  # Default fallback untuk BTC
            logger.warning(f"Not enough data for ATR calculation, using default: {self.atr}")
            return

        df = self.ohlcv_data.copy()

        # Pastikan data diurutkan dari lama ke baru untuk kalkulasi rolling
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate ATR dengan rolling mean
        df['atr'] = df['tr'].rolling(window=period, min_periods=period).mean()

        # Ambil ATR terbaru yang valid
        valid_atr = df['atr'].dropna()
        if not valid_atr.empty:
            self.atr = float(valid_atr.iloc[-1])  # Ambil nilai terakhir (terbaru)
        else:
            # Fallback: gunakan average True Range dari semua data yang ada
            self.atr = float(df['tr'].mean()) if df['tr'].notna().any() else 100.0

        # Validasi final
        if pd.isna(self.atr) or self.atr <= 0:
            self.atr = 100.0  # Default fallback untuk BTC
            logger.warning(f"ATR calculation resulted in invalid value, using default: {self.atr}")

    def analyze_market_context(self) -> SMCContext:
        """
        Analisis konteks pasar secara komprehensif
        Menentukan bias, trend strength, dan key levels
        """
        bias = MarketBias.NEUTRAL
        trend_strength = 0.0
        key_levels = {}
        active_pois = []

        # 1. Analisis Market Structure dari BOS/CHOCH
        structure_bias, structure_strength = self._analyze_market_structure()

        # 2. Identifikasi Premium/Discount Zone
        premium_discount = self._identify_premium_discount_zone()
        key_levels.update(premium_discount)

        # 3. Identifikasi Active POIs (Order Blocks, FVG yang belum mitigated)
        active_pois = self._identify_active_pois()

        # 4. Analisis Swing Structure
        swing_bias = self._analyze_swing_structure()

        # Combine untuk final bias
        bias_score = 0
        if structure_bias == MarketBias.BULLISH:
            bias_score += structure_strength * 0.5
        elif structure_bias == MarketBias.BEARISH:
            bias_score -= structure_strength * 0.5

        if swing_bias == MarketBias.BULLISH:
            bias_score += 0.3
        elif swing_bias == MarketBias.BEARISH:
            bias_score -= 0.3

        # Determine final bias
        if bias_score > 0.3:
            bias = MarketBias.BULLISH
        elif bias_score < -0.3:
            bias = MarketBias.BEARISH
        else:
            bias = MarketBias.NEUTRAL

        trend_strength = abs(bias_score)

        return SMCContext(
            bias=bias,
            trend_strength=min(trend_strength, 1.0),
            key_levels=key_levels,
            active_pois=active_pois
        )

    def _analyze_market_structure(self) -> Tuple[MarketBias, float]:
        """Analisis BOS dan CHOCH untuk menentukan market structure"""
        if 'bos_choch' not in self.smc_data or self.smc_data['bos_choch'].empty:
            return MarketBias.NEUTRAL, 0.0

        df = self.smc_data['bos_choch'].head(self.recent_candles)

        bullish_bos = (df['BOS'] == 1).sum()
        bearish_bos = (df['BOS'] == -1).sum()
        bullish_choch = (df['CHOCH'] == 1).sum()
        bearish_choch = (df['CHOCH'] == -1).sum()

        # CHOCH lebih penting karena menandakan perubahan karakter
        bull_score = bullish_bos * 1 + bullish_choch * 2
        bear_score = bearish_bos * 1 + bearish_choch * 2

        total = bull_score + bear_score
        if total == 0:
            return MarketBias.NEUTRAL, 0.0

        if bull_score > bear_score:
            return MarketBias.BULLISH, bull_score / (total + 1)
        elif bear_score > bull_score:
            return MarketBias.BEARISH, bear_score / (total + 1)

        return MarketBias.NEUTRAL, 0.0

    def _identify_premium_discount_zone(self) -> Dict[str, float]:
        """
        Identifikasi Premium/Discount Zone berdasarkan swing range
        Premium: Upper 50% dari range (ideal untuk sell)
        Discount: Lower 50% dari range (ideal untuk buy)
        """
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
        """
        Identifikasi Point of Interest yang masih aktif:
        - Unmitigated Order Blocks
        - Unmitigated FVG
        - Key Liquidity Levels
        """
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

        # Liquidity Levels yang belum di-sweep
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
        """
        Generate complete trade setup dengan entry, SL, TP
        """
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

        # Final confidence adjustment
        confidence = self._calculate_final_confidence(score, quality, rr, context)

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
        """Evaluate signal berdasarkan semua komponen SMC"""
        score = 0.0
        supporting = []
        conflicting = []

        # FIX 3: Hapus variabel yang tidak digunakan (is_buy, is_sell tidak dipakai di sini)
        # is_buy = signal == 'buy'
        # is_sell = signal == 'sell'

        # 1. Market Structure Alignment (30%)
        ms_score = self._evaluate_market_structure_alignment(
            signal, context, supporting, conflicting
        )
        score += ms_score * self.weights['market_structure']

        # 2. Order Block Proximity (25%)
        ob_score = self._evaluate_order_block(
            signal, context, supporting, conflicting
        )
        score += ob_score * self.weights['order_block']

        # 3. FVG Analysis (15%)
        fvg_score = self._evaluate_fvg(
            signal, supporting, conflicting
        )
        score += fvg_score * self.weights['fvg']

        # 4. Liquidity Analysis (15%)
        liq_score = self._evaluate_liquidity(
            signal, supporting, conflicting
        )
        score += liq_score * self.weights['liquidity']

        # 5. Premium/Discount Zone (10%)
        pd_score = self._evaluate_premium_discount(
            signal, context, supporting, conflicting
        )
        score += pd_score * self.weights['premium_discount']

        # 6. Retracement Quality (5%)
        ret_score = self._evaluate_retracement(
            signal, supporting, conflicting
        )
        score += ret_score * self.weights['retracement']

        # 7. Technical Indicator Confluence (15%)
        tech_score = self._evaluate_technical_confluence(
            signal, supporting, conflicting
        )
        score += tech_score * self.weights['technical_confluence']

        return score, supporting, conflicting

    def _evaluate_technical_confluence(self, signal: str,
                                       supporting: List, conflicting: List) -> float:
        """Evaluate technical indicator confluence untuk konfirmasi tambahan"""
        if self.technical_data is None or self.technical_data.empty:
            return 0.0

        score = 0.0
        tech = self.technical_data.iloc[0]
        is_buy = signal == 'buy'

        # 1. EMA Alignment
        ema_9 = tech.get('ema_9', 0)
        ema_21 = tech.get('ema_21', 0)
        ema_50 = tech.get('ema_50', 0)

        if ema_9 > 0 and ema_21 > 0 and ema_50 > 0:
            if is_buy:
                if ema_9 > ema_21 > ema_50:
                    score += 0.25
                    supporting.append("EMA alignment bullish (9>21>50)")
                elif ema_9 < ema_21 < ema_50:
                    score -= 0.15
                    conflicting.append("EMA alignment bearish")
                # Price above EMA
                if self.current_price > ema_21:
                    score += 0.1
                    supporting.append("Price above EMA 21")
            else:
                if ema_9 < ema_21 < ema_50:
                    score += 0.25
                    supporting.append("EMA alignment bearish (9<21<50)")
                elif ema_9 > ema_21 > ema_50:
                    score -= 0.15
                    conflicting.append("EMA alignment bullish")
                if self.current_price < ema_21:
                    score += 0.1
                    supporting.append("Price below EMA 21")

        # 2. RSI Confluence
        rsi = tech.get('rsi_14', 50)
        if rsi > 0:
            if is_buy:
                if 30 <= rsi <= 50:
                    score += 0.2
                    supporting.append(f"RSI in buy zone ({rsi:.1f})")
                elif rsi < 30:
                    score += 0.15
                    supporting.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    score -= 0.2
                    conflicting.append(f"RSI overbought ({rsi:.1f})")
            else:
                if 50 <= rsi <= 70:
                    score += 0.2
                    supporting.append(f"RSI in sell zone ({rsi:.1f})")
                elif rsi > 70:
                    score += 0.15
                    supporting.append(f"RSI overbought ({rsi:.1f})")
                elif rsi < 30:
                    score -= 0.2
                    conflicting.append(f"RSI oversold ({rsi:.1f})")

        # 3. MACD Confluence
        macd = tech.get('macd', 0)
        macd_signal = tech.get('macd_signal', 0)
        macd_hist = tech.get('macd_hist', 0)

        if macd != 0 and macd_signal != 0:
            if is_buy:
                if macd > macd_signal and macd_hist > 0:
                    score += 0.2
                    supporting.append("MACD bullish crossover")
                elif macd < macd_signal:
                    score -= 0.1
                    conflicting.append("MACD bearish")
            else:
                if macd < macd_signal and macd_hist < 0:
                    score += 0.2
                    supporting.append("MACD bearish crossover")
                elif macd > macd_signal:
                    score -= 0.1
                    conflicting.append("MACD bullish")

        # 4. ADX Trend Strength
        adx = tech.get('adx_14', 0)
        plus_di = tech.get('plus_di_14', 0)
        minus_di = tech.get('minus_di_14', 0)

        if adx > 0:
            if adx > 25:  # Strong trend
                if is_buy and plus_di > minus_di:
                    score += 0.15
                    supporting.append(f"Strong bullish trend (ADX: {adx:.1f})")
                elif not is_buy and minus_di > plus_di:
                    score += 0.15
                    supporting.append(f"Strong bearish trend (ADX: {adx:.1f})")
                elif is_buy and minus_di > plus_di:
                    score -= 0.1
                    conflicting.append("ADX shows bearish dominance")
                elif not is_buy and plus_di > minus_di:
                    score -= 0.1
                    conflicting.append("ADX shows bullish dominance")
            elif adx < 20:
                conflicting.append(f"Weak trend (ADX: {adx:.1f})")

        # 5. Bollinger Bands
        bb_upper = tech.get('bb_upper_20', 0)
        bb_lower = tech.get('bb_lower_20', 0)
        # FIX 4: Hapus variabel yang tidak digunakan
        # bb_middle = tech.get('bb_middle_20', 0)

        if bb_upper > 0 and bb_lower > 0:
            if is_buy:
                if self.current_price <= bb_lower:
                    score += 0.1
                    supporting.append("Price at lower Bollinger Band")
                elif self.current_price >= bb_upper:
                    score -= 0.1
                    conflicting.append("Price at upper Bollinger Band")
            else:
                if self.current_price >= bb_upper:
                    score += 0.1
                    supporting.append("Price at upper Bollinger Band")
                elif self.current_price <= bb_lower:
                    score -= 0.1
                    conflicting.append("Price at lower Bollinger Band")

        return max(0, min(1, score))

    def _evaluate_market_structure_alignment(self, signal: str, context: SMCContext,
                                            supporting: List, conflicting: List) -> float:
        """Check jika signal align dengan market structure"""
        score = 0.0

        if 'bos_choch' not in self.smc_data:
            return 0.0

        df = self.smc_data['bos_choch'].head(self.recent_candles)
        is_buy = signal == 'buy'

        if is_buy:
            if context.bias == MarketBias.BULLISH:
                score += 0.5
                supporting.append("Signal aligns with bullish market bias")
            elif context.bias == MarketBias.BEARISH:
                score -= 0.3
                conflicting.append("Buy signal against bearish market bias")

            if (df['CHOCH'] == 1).any():
                score += 0.3
                supporting.append("Bullish CHoCH confirms trend change")
            if (df['BOS'] == 1).any():
                score += 0.2
                supporting.append("Bullish BOS confirms continuation")

            # Conflicting bearish structure
            if (df['CHOCH'] == -1).any():
                score -= 0.2
                conflicting.append("Recent bearish CHoCH present")
        else:  # sell
            if context.bias == MarketBias.BEARISH:
                score += 0.5
                supporting.append("Signal aligns with bearish market bias")
            elif context.bias == MarketBias.BULLISH:
                score -= 0.3
                conflicting.append("Sell signal against bullish market bias")

            if (df['CHOCH'] == -1).any():
                score += 0.3
                supporting.append("Bearish CHoCH confirms trend change")
            if (df['BOS'] == -1).any():
                score += 0.2
                supporting.append("Bearish BOS confirms continuation")

            if (df['CHOCH'] == 1).any():
                score -= 0.2
                conflicting.append("Recent bullish CHoCH present")

        return max(0, min(1, score))

    def _evaluate_order_block(self, signal: str, context: SMCContext,
                             supporting: List, conflicting: List) -> float:
        """Evaluate order block proximity dan quality"""
        if 'order_block' not in self.smc_data:
            return 0.0

        score = 0.0
        df = self.smc_data['order_block']
        is_buy = signal == 'buy'

        # Find relevant OBs
        if is_buy:
            obs = df[df['OB'] == 1]
        else:
            obs = df[df['OB'] == -1]

        if obs.empty:
            return 0.0

        # Check if price is near an OB
        for _, ob in obs.head(3).iterrows():
            if ob['MitigatedIndex'] != 0:  # Skip mitigated OBs
                continue

            top, bottom = ob['Top'], ob['Bottom']

            # Price within OB zone
            if bottom <= self.current_price <= top:
                score += 0.6
                supporting.append(f"Price at {'bullish' if is_buy else 'bearish'} OB zone")

                # High quality OB
                if ob['Percentage'] > 70:
                    score += 0.2
                    supporting.append(f"Strong OB (strength: {ob['Percentage']:.0f}%)")

                # High volume OB
                if ob['OBVolume'] > df['OBVolume'].median():
                    score += 0.2
                    supporting.append("High volume Order Block")
                break

            # Price approaching OB
            distance = abs(self.current_price - (top + bottom) / 2)
            ob_size = top - bottom
            if distance < ob_size * 2:
                score += 0.3
                supporting.append(f"Price approaching {'bullish' if is_buy else 'bearish'} OB")

        return max(0, min(1, score))

    def _evaluate_fvg(self, signal: str, supporting: List, conflicting: List) -> float:
        """Evaluate FVG alignment"""
        if 'fvg' not in self.smc_data:
            return 0.0

        score = 0.0
        df = self.smc_data['fvg'].head(self.recent_candles)
        is_buy = signal == 'buy'

        if is_buy:
            bullish_fvgs = df[df['FVG'] == 1]
            if not bullish_fvgs.empty:
                # Check unmitigated FVG
                unmitigated = bullish_fvgs[bullish_fvgs['MitigatedIndex'] == 0]
                if not unmitigated.empty:
                    score += 0.5
                    supporting.append("Unmitigated bullish FVG present")

                    # Price near FVG
                    fvg = unmitigated.iloc[0]
                    if fvg['Bottom'] <= self.current_price <= fvg['Top']:
                        score += 0.3
                        supporting.append("Price at bullish FVG zone")
                else:
                    score += 0.2
                    supporting.append("Bullish FVG detected (mitigated)")

            # Conflicting bearish FVG
            bearish_unmit = df[(df['FVG'] == -1) & (df['MitigatedIndex'] == 0)]
            if not bearish_unmit.empty:
                score -= 0.2
                conflicting.append("Unmitigated bearish FVG above")
        else:
            bearish_fvgs = df[df['FVG'] == -1]
            if not bearish_fvgs.empty:
                unmitigated = bearish_fvgs[bearish_fvgs['MitigatedIndex'] == 0]
                if not unmitigated.empty:
                    score += 0.5
                    supporting.append("Unmitigated bearish FVG present")

                    fvg = unmitigated.iloc[0]
                    if fvg['Bottom'] <= self.current_price <= fvg['Top']:
                        score += 0.3
                        supporting.append("Price at bearish FVG zone")
                else:
                    score += 0.2
                    supporting.append("Bearish FVG detected (mitigated)")

            bullish_unmit = df[(df['FVG'] == 1) & (df['MitigatedIndex'] == 0)]
            if not bullish_unmit.empty:
                score -= 0.2
                conflicting.append("Unmitigated bullish FVG below")

        return max(0, min(1, score))

    def _evaluate_liquidity(self, signal: str, supporting: List, conflicting: List) -> float:
        """Evaluate liquidity sweep"""
        if 'liquidity' not in self.smc_data:
            return 0.0

        score = 0.0
        df = self.smc_data['liquidity'].head(self.recent_candles)
        is_buy = signal == 'buy'

        if is_buy:
            swept = df[(df['Liquidity'] == -1) & (df['Swept'] != 0)]
            if not swept.empty:
                score += 0.7
                supporting.append("Sell-side liquidity swept (bullish)")

            unswept = df[(df['Liquidity'] == 1) & (df['Swept'] == 0)]
            if not unswept.empty:
                score += 0.3
                supporting.append("Buy-side liquidity target above")
        else:
            swept = df[(df['Liquidity'] == 1) & (df['Swept'] != 0)]
            if not swept.empty:
                score += 0.7
                supporting.append("Buy-side liquidity swept (bearish)")

            unswept = df[(df['Liquidity'] == -1) & (df['Swept'] == 0)]
            if not unswept.empty:
                score += 0.3
                supporting.append("Sell-side liquidity target below")

        return max(0, min(1, score))

    def _evaluate_premium_discount(self, signal: str, context: SMCContext,
                                   supporting: List, conflicting: List) -> float:
        """Evaluate if price is in appropriate zone"""
        score = 0.0

        if 'equilibrium' not in context.key_levels:
            return 0.0

        eq = context.key_levels['equilibrium']
        is_buy = signal == 'buy'

        if is_buy:
            if self.current_price < eq:
                score += 0.7
                supporting.append("Price in discount zone (ideal for buy)")
            elif self.current_price > context.key_levels.get('premium_zone', eq):
                score -= 0.3
                conflicting.append("Buy signal in premium zone")
        else:
            if self.current_price > eq:
                score += 0.7
                supporting.append("Price in premium zone (ideal for sell)")
            elif self.current_price < context.key_levels.get('discount_zone', eq):
                score -= 0.3
                conflicting.append("Sell signal in discount zone")

        return max(0, min(1, score))

    def _evaluate_retracement(self, signal: str, supporting: List, conflicting: List) -> float:
        """Evaluate retracement quality"""
        if 'retracements' not in self.smc_data:
            return 0.0

        score = 0.0
        df = self.smc_data['retracements'].head(self.recent_candles)
        is_buy = signal == 'buy'

        if is_buy:
            retr = df[df['Direction'] == 1]
            if not retr.empty:
                current_retr = retr.iloc[0]['CurrentRetracement%']
                if 50 <= current_retr <= 79:
                    score += 0.8
                    supporting.append(f"Optimal retracement ({current_retr:.1f}%)")
                elif 38 <= current_retr < 50:
                    score += 0.4
                    supporting.append(f"Shallow retracement ({current_retr:.1f}%)")
                elif current_retr > 79:
                    score -= 0.2
                    conflicting.append(f"Deep retracement risk ({current_retr:.1f}%)")
        else:
            retr = df[df['Direction'] == -1]
            if not retr.empty:
                current_retr = retr.iloc[0]['CurrentRetracement%']
                if 50 <= current_retr <= 79:
                    score += 0.8
                    supporting.append(f"Optimal retracement ({current_retr:.1f}%)")
                elif 38 <= current_retr < 50:
                    score += 0.4
                    supporting.append(f"Shallow retracement ({current_retr:.1f}%)")
                elif current_retr > 79:
                    score -= 0.2
                    conflicting.append(f"Deep retracement risk ({current_retr:.1f}%)")

        return max(0, min(1, score))

    def _determine_setup_quality(self, score: float, signal: str, context: SMCContext) -> SetupQuality:
        """Determine overall setup quality"""
        is_buy = signal == 'buy'
        is_sell = signal == 'sell'

        bias_aligned = (
            (is_buy and context.bias == MarketBias.BULLISH) or
            (is_sell and context.bias == MarketBias.BEARISH)
        )

        if score >= 75:
            return SetupQuality.A_PLUS if bias_aligned else SetupQuality.A
        elif score >= 60:
            return SetupQuality.A if bias_aligned else SetupQuality.B
        elif score >= 45:
            return SetupQuality.B if bias_aligned else SetupQuality.C
        elif score >= 30:
            return SetupQuality.C
        else:
            return SetupQuality.NO_TRADE

    def _calculate_trade_levels(self, signal: str, context: SMCContext,
                                quality: SetupQuality) -> Tuple[float, float, float, float, float, float]:
        """Calculate entry, SL, TP levels berdasarkan struktur SMC"""
        # Tetap hitung levels meskipun quality rendah untuk informasi
        # if quality == SetupQuality.NO_TRADE:
        #     return None, None, None, None, None, None

        is_buy = signal == 'buy'
        entry = self.current_price

        # Pastikan ATR valid
        if pd.isna(self.atr) or self.atr <= 0:
            self.atr = entry * 0.02  # Fallback: 2% dari harga sebagai proxy ATR

        swing_df = self.smc_data.get('swing_highs_lows', pd.DataFrame())
        ob_df = self.smc_data.get('order_block', pd.DataFrame())

        if is_buy:
            # Cari swing low terdekat
            if not swing_df.empty and 'HighLow' in swing_df.columns:
                swing_lows = swing_df[swing_df['HighLow'] == -1]['Level'].dropna()
                recent_low = float(swing_lows.iloc[0]) if not swing_lows.empty else self.current_low
            else:
                recent_low = self.current_low

            # Cari bullish OB bottom
            if not ob_df.empty and 'OB' in ob_df.columns:
                bullish_obs = ob_df[(ob_df['OB'] == 1) & (ob_df['Bottom'] < entry)]
                ob_bottom = float(bullish_obs['Bottom'].iloc[0]) if not bullish_obs.empty else recent_low
            else:
                ob_bottom = recent_low

            structural_sl = max(recent_low, ob_bottom)
            sl = structural_sl - (self.atr * 0.3)
            invalidation = structural_sl - (self.atr * 0.8)

            # Cari swing high sebagai target
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
                risk = self.atr  # Fallback
                sl = entry - risk

            tp1 = entry + (risk * 1.5)
            tp2 = entry + (risk * 2.5)
            tp3 = max(nearest_high, entry + (risk * 4))

        else:  # Sell
            # Cari swing high terdekat
            if not swing_df.empty and 'HighLow' in swing_df.columns:
                swing_highs = swing_df[swing_df['HighLow'] == 1]['Level'].dropna()
                recent_high = float(swing_highs.iloc[0]) if not swing_highs.empty else self.current_high
            else:
                recent_high = self.current_high

            # Cari bearish OB top
            if not ob_df.empty and 'OB' in ob_df.columns:
                bearish_obs = ob_df[(ob_df['OB'] == -1) & (ob_df['Top'] > entry)]
                ob_top = float(bearish_obs['Top'].iloc[0]) if not bearish_obs.empty else recent_high
            else:
                ob_top = recent_high

            structural_sl = min(recent_high, ob_top)
            sl = structural_sl + (self.atr * 0.3)
            invalidation = structural_sl + (self.atr * 0.8)

            # Cari swing low sebagai target
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
                risk = self.atr  # Fallback
                sl = entry + risk

            tp1 = entry - (risk * 1.5)
            tp2 = entry - (risk * 2.5)
            tp3 = min(nearest_low, entry - (risk * 4))

        return entry, sl, tp1, tp2, tp3, invalidation

    def _calculate_final_confidence(self, score: float, quality: SetupQuality,
                                   rr: float, context: SMCContext) -> float:
        """Calculate final confidence dengan semua faktor"""
        confidence = score

        quality_bonus = {
            SetupQuality.A_PLUS: 10,
            SetupQuality.A: 5,
            SetupQuality.B: 0,
            SetupQuality.C: -5,
            SetupQuality.NO_TRADE: -20
        }
        confidence += quality_bonus.get(quality, 0)

        if rr >= 3:
            confidence += 5
        elif rr >= 2:
            confidence += 2
        elif rr < 1.5:
            confidence -= 10

        confidence += context.trend_strength * 5

        return max(0, min(100, confidence))

    def _identify_setup_type(self, signal: str, context: SMCContext,
                            supporting: List[str]) -> str:
        """Identify specific SMC setup type"""
        has_choch = any('CHoCH' in s for s in supporting)
        has_bos = any('BOS' in s for s in supporting)
        has_ob = any('Order Block' in s for s in supporting)
        has_fvg = any('FVG' in s for s in supporting)
        has_liq_sweep = any('swept' in s.lower() for s in supporting)
        has_optimal_retr = any('Optimal retracement' in s for s in supporting)

        is_buy = signal == 'buy'

        if has_choch and has_ob and has_liq_sweep:
            return f"{'Bullish' if is_buy else 'Bearish'} Reversal (A+ Setup)"

        if has_bos and has_ob and has_fvg:
            return f"{'Bullish' if is_buy else 'Bearish'} Continuation (OB+FVG)"

        if has_ob and has_optimal_retr:
            return f"{'Bullish' if is_buy else 'Bearish'} OB Retest"

        if has_fvg and has_optimal_retr:
            return f"{'Bullish' if is_buy else 'Bearish'} FVG Fill"

        if has_liq_sweep and (has_ob or has_fvg):
            return f"{'Bullish' if is_buy else 'Bearish'} Liquidity Grab"

        if has_bos:
            return f"{'Bullish' if is_buy else 'Bearish'} Breakout"

        if context.bias == MarketBias.NEUTRAL:
            return "Range Trade"

        return f"{'Bullish' if is_buy else 'Bearish'} Setup"

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

        # Pastikan ATR valid untuk output
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
    print("Enhanced SMC Strategy for Intraday Trading")
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
            print(f"\n🎯 Active Points of Interest:")
            for poi in analysis['active_pois'][:3]:
                if poi['type'] == 'order_block':
                    print(f"  • {poi['direction'].upper()} OB: ${poi['bottom']:,.2f} - ${poi['top']:,.2f}")
                elif poi['type'] == 'fvg':
                    print(f"  • {poi['direction'].upper()} FVG: ${poi['bottom']:,.2f} - ${poi['top']:,.2f}")
                elif poi['type'] == 'liquidity':
                    print(f"  • {poi['direction'].upper()} Liquidity: ${poi['level']:,.2f}")
    else:
        print(f"\n❌ Market Analysis Error: {analysis['error']}")

    test_prediction = {'signal': 'Buy', 'confidence': 75}
    signal = generate_smc_signal(test_prediction)

    if 'error' not in signal:
        print(f"\n{'='*60}")
        print(f"🎯 Trade Setup for {signal['original_signal'].upper()} Signal:")
        print(f"{'='*60}")
        print(f"  SMC Signal: {signal['smc_signal']}")
        print(f"  Quality: {signal['setup_quality']}")
        print(f"  Confidence: {signal['confidence']}%")
        print(f"  Setup Type: {signal['setup_type']}")
        print(f"\n💰 Trade Levels:")
        print(f"  Entry: ${signal['entry_price']:,.2f}")
        print(f"  Stop Loss: ${signal['stop_loss']:,.2f}")
        print(f"  TP1 (1.5R): ${signal['take_profit_1']:,.2f}")
        print(f"  TP2 (2.5R): ${signal['take_profit_2']:,.2f}")
        print(f"  TP3 (4R): ${signal['take_profit_3']:,.2f}")
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

        save_smc_signal(signal)
        print(f"\n💾 Signal saved to database")
    else:
        print(f"\n❌ Signal Generation Error: {signal.get('error', 'Unknown error')}")
