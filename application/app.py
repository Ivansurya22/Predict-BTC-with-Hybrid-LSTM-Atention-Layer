import os
import sqlite3
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDateEdit,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tab_backtesting import BacktestingTab
from tab_model import ModelManagementTab

os.environ.pop("QT_STYLE_OVERRIDE", None)

# ============================================================================
# CONSTANTS & STYLING
# ============================================================================

DARK_THEME_QSS = """
QMainWindow {
    background-color: #131722;
}

QTabWidget::pane {
    border: 1px solid #2A2E39;
    background-color: #131722;
}

QTabBar::tab {
    background-color: #1E222D;
    color: #787B86;
    padding: 10px 20px;
    border: 1px solid #2A2E39;
    border-bottom: none;
    font-size: 13px;
    font-weight: bold;
}

QTabBar::tab:selected {
    background-color: #2962FF;
    color: #FFFFFF;
}

QTabBar::tab:hover {
    background-color: #2A2E39;
    color: #D1D4DC;
}

#sidebar {
    background-color: #1E222D;
    border-right: 1px solid #2A2E39;
}

QGroupBox {
    color: #D1D4DC;
    border: 1px solid #2A2E39;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
    font-size: 14px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

QLabel {
    color: #D1D4DC;
    font-size: 14px;
}

#headerLabel {
    font-size: 22px;
    font-weight: bold;
    color: #FFFFFF;
    padding: 10px;
}

#metricLabel {
    font-size: 15px;
    color: #787B86;
}

#metricValue {
    font-size: 26px;
    font-weight: bold;
    color: #2962FF;
    font-family: 'Consolas', 'Courier New', monospace;
}

#signalLabel {
    font-size: 18px;
    font-weight: bold;
    padding: 10px;
    border-radius: 4px;
    margin: 4px 0px;
}

#upSignal {
    background-color: rgba(38, 166, 154, 0.2);
    color: #26A69A;
    border: 1px solid #26A69A;
}

#downSignal {
    background-color: rgba(239, 83, 80, 0.2);
    color: #EF5350;
    border: 1px solid #EF5350;
}

#holdSignal {
    background-color: rgba(41, 98, 255, 0.2);
    color: #2962FF;
    border: 1px solid #2962FF;
}

QPushButton {
    background-color: #2962FF;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 10px 18px;
    font-weight: bold;
    font-size: 14px;
}

QPushButton:hover {
    background-color: #1E53E5;
}

QPushButton:pressed {
    background-color: #1948CC;
}

QPushButton:disabled {
    background-color: #434651;
    color: #787B86;
}

QCheckBox {
    color: #D1D4DC;
    spacing: 8px;
    font-size: 14px;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 3px;
    border: 1px solid #434651;
    background-color: #2A2E39;
}

QCheckBox::indicator:checked {
    background-color: #2962FF;
    border: 1px solid #2962FF;
}

QCheckBox::indicator:hover {
    border: 1px solid #2962FF;
}

QDateEdit {
    background-color: #2A2E39;
    color: #D1D4DC;
    border: 1px solid #434651;
    border-radius: 4px;
    padding: 6px;
    font-size: 14px;
}

QDateEdit:focus {
    border: 1px solid #2962FF;
}

QDateEdit::drop-down {
    border: none;
    width: 20px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #2A2E39;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #434651;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #4A4E59;
}

#infoCard {
    background-color: #1E222D;
    border: 1px solid #2A2E39;
    border-radius: 8px;
    padding: 12px;
}

#metricCard {
    background-color: rgba(41, 98, 255, 0.1);
    border: 1px solid rgba(41, 98, 255, 0.3);
    border-radius: 6px;
    padding: 10px;
}
"""

COLORS = {
    "bg": "#131722",
    "grid": "#2A2E39",
    "text": "#D1D4DC",
    "candle_up": "#26A69A",
    "candle_down": "#EF5350",
    "swing_high": "#26A69A",
    "swing_low": "#EF5350",
    "fvg": "#FFEB3B",
    "bos_bull": "#26A69A",
    "bos_bear": "#EF5350",
    "choch_bull": "#26A69A",
    "choch_bear": "#EF5350",
    "ob_bull": "#4CAF50",
    "ob_bear": "#EF5350",
    "liquidity": "#7E57C2",
    "rsi": "#29B6F6",
    "macd": "#FFA726",
    "signal": "#7E57C2",
    "ema_9": "#FF6B6B",
    "ema_21": "#4ECDC4",
    "ema_50": "#FFE66D",
    "ema_100": "#A8DADC",
    "ema_200": "#F1FAEE",
    "bb_upper": "#A8DADC",
    "bb_middle": "#D1D4DC",
    "bb_lower": "#F1FAEE",
}

DB_PATH = "data/btc_ohlcv.db"

SMC_TABLE_NAMES = {
    "swing": "smc_btc_1h_swing_highs_lows",
    "fvg": "smc_btc_1h_fvg",
    "bos_choch": "smc_btc_1h_bos_choch",
    "ob": "smc_btc_1h_order_block",
    "liquidity": "smc_btc_1h_liquidity",
    "technical": "smc_btc_1h_technical_indicators",
}

# ============================================================================
# CUSTOM CANDLESTICK ITEM
# ============================================================================

class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick chart item"""

    def __init__(self, data, timestamps=None):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.timestamps = timestamps
        self.picture = None
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        painter = pg.QtGui.QPainter(self.picture)
        w = 0.3

        for i, (t, open_, high, low, close) in enumerate(self.data):
            color = COLORS["candle_up"] if close > open_ else COLORS["candle_down"]
            painter.setPen(pg.mkPen(color))
            painter.setBrush(pg.mkBrush(color))
            painter.drawLine(pg.QtCore.QPointF(i, low), pg.QtCore.QPointF(i, high))
            painter.drawRect(pg.QtCore.QRectF(i - w, open_, 2 * w, close - open_))

        painter.end()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())

    def get_candle_at_pos(self, x):
        """Get candle data at x position"""
        idx = int(round(x))
        if 0 <= idx < len(self.data):
            t, open_, high, low, close = self.data[idx]
            timestamp = self.timestamps[idx] if self.timestamps is not None else None
            return {
                'index': idx,
                'timestamp': timestamp,
                'open': open_,
                'high': high,
                'low': low,
                'close': close
            }
        return None

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Handle all database operations"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.last_ohlcv_timestamp = None
        self.last_prediction_timestamp = None

    def get_connection(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def get_date_range(self) -> Tuple[QDate, QDate]:
        """Get min and max dates from database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM btc_1h")
            min_date_str, max_date_str = cursor.fetchone()
            conn.close()

            min_date = QDate.fromString(min_date_str[:10], "yyyy-MM-dd")
            max_date = QDate.fromString(max_date_str[:10], "yyyy-MM-dd")
            return min_date, max_date
        except Exception as e:
            print(f"Error getting date range: {e}")
            today = QDate.currentDate()
            return today.addDays(-365), today

    def load_ohlcv(self, start_date: QDate, end_date: QDate, limit: int = 504) -> pd.DataFrame:
        """Load OHLCV data"""
        try:
            conn = self.get_connection()
            query = f"""
            SELECT * FROM btc_1h
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
            start_str = start_date.toString("yyyy-MM-dd")
            end_str = end_date.addDays(1).toString("yyyy-MM-dd")

            df = pd.read_sql(query, conn, params=(start_str, end_str))
            conn.close()

            df = df.iloc[::-1].reset_index(drop=True)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

            if not df.empty:
                self.last_ohlcv_timestamp = df.iloc[-1]["timestamp"]

            return df
        except Exception as e:
            print(f"Error loading OHLCV: {e}")
            return pd.DataFrame()

    def load_smc_table(self, table_name: str, start_date: QDate, end_date: QDate) -> pd.DataFrame:
        """Load SMC feature table"""
        try:
            conn = self.get_connection()
            query = f"""
            SELECT * FROM {table_name}
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
            """
            start_str = start_date.toString("yyyy-MM-dd")
            end_str = end_date.addDays(1).toString("yyyy-MM-dd")

            df = pd.read_sql(query, conn, params=(start_str, end_str))
            conn.close()

            if df.empty:
                return df

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

            for col in df.columns:
                if col != 'timestamp':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df
        except Exception as e:
            print(f"⚠️ Error loading {table_name}: {e}")
            return pd.DataFrame()

    def get_latest_prediction(self) -> Optional[Dict]:
        """Get latest prediction from database"""
        try:
            conn = self.get_connection()
            query = """
            SELECT timestamp, prediction, confidence, prob_down, prob_hold,
                   prob_up, close_price, model_type
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 1
            """
            pred_df = pd.read_sql(query, conn)
            conn.close()

            if not pred_df.empty:
                row = pred_df.iloc[0]
                self.last_prediction_timestamp = row["timestamp"]
                return {
                    "timestamp": row["timestamp"],
                    "signal": row["prediction"],
                    "confidence": float(row["confidence"]),
                    "prob_down": float(row["prob_down"]),
                    "prob_hold": float(row["prob_hold"]),
                    "prob_up": float(row["prob_up"]),
                    "close_price": float(row["close_price"]),
                    "model_type": row["model_type"]
                }
            return None
        except Exception as e:
            print(f"⚠️ Error loading prediction: {e}")
            return None

    def get_latest_smc_signal(self) -> Optional[Dict]:
        """Get latest SMC rule-based signal from database"""
        try:
            conn = self.get_connection()
            query = """
            SELECT timestamp, original_signal, smc_signal, entry_price,
                   stop_loss, take_profit_1, take_profit_2, take_profit_3,
                   risk_reward, setup_quality, confidence, setup_type,
                   supporting_factors, conflicting_factors, invalidation_level
            FROM smc_signals
            ORDER BY timestamp DESC
            LIMIT 1
            """
            smc_df = pd.read_sql(query, conn)
            conn.close()

            if not smc_df.empty:
                row = smc_df.iloc[0]

                # Debug: Print raw data
                print(f"🧱 SMC Signal Raw Data:")
                print(f"   Timestamp: {row['timestamp']}")
                print(f"   Signal: {row['smc_signal']}")
                print(f"   Entry: {row['entry_price']}")
                print(f"   Setup Type: {row['setup_type']}")
                print(f"   Setup Quality: {row['setup_quality']}")

                # Safe conversion with default values
                def safe_float(value, default=None):
                    try:
                        val = float(value)
                        return val if val > 0 else default
                    except (ValueError, TypeError):
                        return default

                result = {
                    "timestamp": row["timestamp"],
                    "original_signal": str(row["original_signal"]) if pd.notna(row["original_signal"]) else "N/A",
                    "smc_signal": str(row["smc_signal"]) if pd.notna(row["smc_signal"]) else "Hold",
                    "entry_price": safe_float(row["entry_price"]),
                    "stop_loss": safe_float(row["stop_loss"]),
                    "take_profit_1": safe_float(row["take_profit_1"]),
                    "take_profit_2": safe_float(row["take_profit_2"]),
                    "take_profit_3": safe_float(row["take_profit_3"]),
                    "risk_reward": safe_float(row["risk_reward"]),
                    "setup_quality": str(row["setup_quality"]) if pd.notna(row["setup_quality"]) else "No Trade",
                    "confidence": safe_float(row["confidence"]),
                    "setup_type": str(row["setup_type"]) if pd.notna(row["setup_type"]) else "No Setup",
                    "supporting_factors": str(row["supporting_factors"]) if pd.notna(row["supporting_factors"]) else "",
                    "conflicting_factors": str(row["conflicting_factors"]) if pd.notna(row["conflicting_factors"]) else "",
                    "invalidation_level": safe_float(row["invalidation_level"])
                }

                print(f"✅ SMC Signal parsed successfully")
                return result
            else:
                print("⚠️ No SMC signals found in database")
                return None
        except Exception as e:
            print(f"⚠️ Error loading SMC signal: {e}")
            import traceback
            traceback.print_exc()
            return None

    def check_for_new_data(self) -> bool:
        """Check if there's new OHLCV data"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(timestamp) FROM btc_1h")
            latest_timestamp = cursor.fetchone()[0]
            conn.close()

            if latest_timestamp is None or self.last_ohlcv_timestamp is None:
                return False

            latest_dt = pd.to_datetime(latest_timestamp).tz_localize(None)
            last_dt = pd.to_datetime(self.last_ohlcv_timestamp).tz_localize(None)
            return latest_dt > last_dt
        except Exception as e:
            print(f"Error checking for new data: {e}")
            return False

    def check_for_new_prediction(self) -> bool:
        """Check if there's new prediction"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(timestamp) FROM predictions")
            latest_timestamp = cursor.fetchone()[0]
            conn.close()

            if latest_timestamp is None or self.last_prediction_timestamp is None:
                return False

            latest_dt = pd.to_datetime(latest_timestamp).tz_localize(None)
            last_dt = pd.to_datetime(self.last_prediction_timestamp).tz_localize(None)
            return latest_dt > last_dt
        except Exception as e:
            print(f"Error checking for new prediction: {e}")
            return False

# ============================================================================
# METRIC CARD WIDGET
# ============================================================================

class MetricCard(QFrame):
    """Modern metric display card"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setMinimumHeight(85)
        self.setMinimumWidth(140)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricLabel")
        self.title_label.setWordWrap(False)

        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")
        self.value_label.setWordWrap(False)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str):
        """Update metric value"""
        self.value_label.setText(value)

# ============================================================================
# DASHBOARD TAB
# ============================================================================

class DashboardTab(QWidget):
    """Main dashboard tab with charts"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.candle_item = None
        self.hover_label = None
        self.crosshair_v = None
        self.crosshair_h = None
        self.setup_ui()

    def setup_ui(self):
        """Setup dashboard UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Top metrics bar
        info_bar = QWidget()
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(0, 0, 0, 10)

        self.metric_price = MetricCard("Current Price")
        self.metric_change = MetricCard("24h Change")
        self.metric_volume = MetricCard("Volume")
        self.metric_signal = MetricCard("AI Signal")

        info_layout.addWidget(self.metric_price)
        info_layout.addWidget(self.metric_change)
        info_layout.addWidget(self.metric_volume)
        info_layout.addWidget(self.metric_signal)
        info_layout.addStretch()

        layout.addWidget(info_bar)

        # Main candlestick chart
        self.main_plot = pg.PlotWidget()
        self.main_plot.setBackground(COLORS["bg"])
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)
        self.main_plot.setLabel("left", "Price", color=COLORS["text"])
        self.main_plot.setMinimumHeight(400)
        self.main_plot.enableAutoRange(axis='y', enable=True)
        self.main_plot.enableAutoRange(axis='x', enable=True)
        self.main_plot.setMouseEnabled(x=True, y=True)

        # Setup hover functionality
        self.main_plot.scene().sigMouseMoved.connect(self.on_mouse_moved)

        # Create crosshair
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False,
                                          pen=pg.mkPen(COLORS["text"], width=1, style=Qt.DashLine))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False,
                                          pen=pg.mkPen(COLORS["text"], width=1, style=Qt.DashLine))
        self.main_plot.addItem(self.crosshair_v, ignoreBounds=True)
        self.main_plot.addItem(self.crosshair_h, ignoreBounds=True)

        self.hover_label = pg.TextItem(anchor=(0, 1), color=COLORS["text"])
        self.hover_label.setZValue(1000)
        self.main_plot.addItem(self.hover_label, ignoreBounds=True)

        for axis_name in ["bottom", "left"]:
            axis = self.main_plot.getAxis(axis_name)
            axis.setPen(COLORS["text"])
            axis.setTextPen(COLORS["text"])

        layout.addWidget(self.main_plot, stretch=3)

        # RSI chart
        self.rsi_plot = pg.PlotWidget()
        self.rsi_plot.setBackground(COLORS["bg"])
        self.rsi_plot.showGrid(x=True, y=True, alpha=0.3)
        self.rsi_plot.setLabel("left", "RSI", color=COLORS["text"])
        self.rsi_plot.setMinimumHeight(150)
        self.rsi_plot.setMaximumHeight(200)
        self.rsi_plot.setYRange(0, 100)
        self.rsi_plot.addLine(y=70, pen=pg.mkPen(COLORS["candle_down"], width=1, style=Qt.DashLine))
        self.rsi_plot.addLine(y=30, pen=pg.mkPen(COLORS["candle_up"], width=1, style=Qt.DashLine))

        for axis_name in ["bottom", "left"]:
            axis = self.rsi_plot.getAxis(axis_name)
            axis.setPen(COLORS["text"])
            axis.setTextPen(COLORS["text"])

        layout.addWidget(self.rsi_plot, stretch=1)

        # MACD chart
        self.macd_plot = pg.PlotWidget()
        self.macd_plot.setBackground(COLORS["bg"])
        self.macd_plot.showGrid(x=True, y=True, alpha=0.3)
        self.macd_plot.setLabel("left", "MACD", color=COLORS["text"])
        self.macd_plot.setMinimumHeight(150)
        self.macd_plot.setMaximumHeight(200)

        for axis_name in ["bottom", "left"]:
            axis = self.macd_plot.getAxis(axis_name)
            axis.setPen(COLORS["text"])
            axis.setTextPen(COLORS["text"])

        layout.addWidget(self.macd_plot, stretch=1)

    def on_mouse_moved(self, pos):
        """Handle mouse move event"""
        if self.candle_item is None or self.parent_window.df.empty:
            return

        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()

        self.crosshair_v.setPos(x)
        self.crosshair_h.setPos(y)

        candle_data = self.candle_item.get_candle_at_pos(x)

        if candle_data:
            timestamp = candle_data['timestamp']
            open_price = candle_data['open']
            high_price = candle_data['high']
            low_price = candle_data['low']
            close_price = candle_data['close']

            time_str = timestamp.strftime("%Y-%m-%d %H:%M") if timestamp else f"Index: {candle_data['index']}"
            change = close_price - open_price
            change_pct = (change / open_price) * 100 if open_price != 0 else 0

            hover_text = f"""
<div style='background-color: rgba(30, 34, 45, 0.9); padding: 8px; border: 1px solid #2A2E39; border-radius: 4px;'>
<b>{time_str}</b><br>
O: ${open_price:,.2f}<br>
H: ${high_price:,.2f}<br>
L: ${low_price:,.2f}<br>
C: ${close_price:,.2f}<br>
<span style='color: {"#26A69A" if change >= 0 else "#EF5350"}'>{'+' if change >= 0 else ''}{change:.2f} ({'+' if change_pct >= 0 else ''}{change_pct:.2f}%)</span>
</div>
"""
            self.hover_label.setHtml(hover_text)
            self.hover_label.setPos(x + 2, high_price)
        else:
            self.hover_label.setHtml("")

# ============================================================================
# MAIN WINDOW
# ============================================================================

class TradingDashboard(QMainWindow):
    """Main trading dashboard window"""

    def __init__(self):
        super().__init__()
        self.db = DatabaseManager(DB_PATH)
        self.smc_features = {}
        self.df = pd.DataFrame()
        self.smc_plot_items = {
            "swing_high": [],
            "swing_low": [],
            "fvg": [],
            "bos": [],
            "choch": [],
            "ob": [],
            "liquidity": [],
            "indicators": [],
        }

        self.setWindowTitle("📊 BTC Trading Dashboard - Smart Money Concepts")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet(DARK_THEME_QSS)

        self.setup_ui()

        # Timers
        self.check_timer = QTimer()
        self.check_timer.timeout.connect(self.check_for_updates)
        self.check_timer.start(5000)

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.load_data)
        self.refresh_timer.start(300000)

        self.load_data()

    def setup_ui(self):
        """Setup main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)

        sidebar = self.create_sidebar()
        splitter.addWidget(sidebar)

        self.tab_widget = QTabWidget()
        self.dashboard_tab = DashboardTab(self)
        self.tab_widget.addTab(self.dashboard_tab, "📈 Dashboard")

        self.backtest_tab = BacktestingTab(self)
        self.tab_widget.addTab(self.backtest_tab, "📊 Backtesting")

        self.model_tab = ModelManagementTab(self)
        self.tab_widget.addTab(self.model_tab, "🤖 Model")

        splitter.addWidget(self.tab_widget)
        splitter.setSizes([300, 1300])
        splitter.setHandleWidth(1)

        main_layout.addWidget(splitter)

    def create_sidebar(self) -> QWidget:
        """Create left sidebar"""
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(350)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setObjectName("sidebar")

        content = QWidget()
        content.setObjectName("sidebar")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        header = QLabel("⚙️ Configuration")
        header.setObjectName("headerLabel")
        layout.addWidget(header)

        # Date Range Group
        date_group = QGroupBox("📅 Date Range")
        date_layout = QVBoxLayout()

        min_date, max_date = self.db.get_date_range()
        default_start = max_date.addDays(-30)

        date_layout.addWidget(QLabel("Start Date:"))
        self.start_date = QDateEdit()
        self.start_date.setDate(default_start)
        self.start_date.setMinimumDate(min_date)
        self.start_date.setMaximumDate(max_date)
        self.start_date.setCalendarPopup(True)
        date_layout.addWidget(self.start_date)

        date_layout.addWidget(QLabel("End Date:"))
        self.end_date = QDateEdit()
        self.end_date.setDate(max_date)
        self.end_date.setMinimumDate(min_date)
        self.end_date.setMaximumDate(max_date)
        self.end_date.setCalendarPopup(True)
        date_layout.addWidget(self.end_date)

        # Load button for date range
        load_btn = QPushButton("📥 Load Data")
        load_btn.clicked.connect(self.load_data)
        date_layout.addWidget(load_btn)

        date_group.setLayout(date_layout)
        layout.addWidget(date_group)

        # Technical Indicators Group
        tech_group = QGroupBox("📈 Technical Indicators")
        tech_layout = QVBoxLayout()

        self.rsi_check = QCheckBox("RSI (14)")
        self.rsi_check.setChecked(True)
        self.rsi_check.stateChanged.connect(self.update_indicators)
        tech_layout.addWidget(self.rsi_check)

        self.macd_check = QCheckBox("MACD (12,26,9)")
        self.macd_check.setChecked(True)
        self.macd_check.stateChanged.connect(self.update_indicators)
        tech_layout.addWidget(self.macd_check)

        self.ema_check = QCheckBox("EMA (9, 21, 50, 100, 200)")
        self.ema_check.setChecked(False)
        self.ema_check.stateChanged.connect(self.update_overlays)
        tech_layout.addWidget(self.ema_check)

        self.bb_check = QCheckBox("Bollinger Bands")
        self.bb_check.setChecked(False)
        self.bb_check.stateChanged.connect(self.update_overlays)
        tech_layout.addWidget(self.bb_check)

        tech_group.setLayout(tech_layout)
        layout.addWidget(tech_group)

        # SMC Features Group
        smc_group = QGroupBox("🔍 SMC Features")
        smc_layout = QVBoxLayout()

        self.swing_check = QCheckBox("Swing High/Low")
        self.swing_check.setChecked(True)
        self.swing_check.stateChanged.connect(self.update_smc_features)
        smc_layout.addWidget(self.swing_check)

        self.fvg_check = QCheckBox("Fair Value Gap")
        self.fvg_check.setChecked(True)
        self.fvg_check.stateChanged.connect(self.update_smc_features)
        smc_layout.addWidget(self.fvg_check)

        self.bos_check = QCheckBox("BOS/CHOCH")
        self.bos_check.setChecked(True)
        self.bos_check.stateChanged.connect(self.update_smc_features)
        smc_layout.addWidget(self.bos_check)

        self.ob_check = QCheckBox("Order Blocks")
        self.ob_check.setChecked(True)
        self.ob_check.stateChanged.connect(self.update_smc_features)
        smc_layout.addWidget(self.ob_check)

        self.liquidity_check = QCheckBox("Liquidity")
        self.liquidity_check.setChecked(True)
        self.liquidity_check.stateChanged.connect(self.update_smc_features)
        smc_layout.addWidget(self.liquidity_check)

        smc_group.setLayout(smc_layout)
        layout.addWidget(smc_group)

        # Real-time Info Card
        info_card = QFrame()
        info_card.setObjectName("infoCard")
        info_layout = QVBoxLayout(info_card)

        info_header = QLabel("🤖 AI Prediction")
        info_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        info_layout.addWidget(info_header)

        self.price_label = QLabel("Price: --")
        self.signal_label = QLabel("Signal: --")
        self.signal_label.setObjectName("signalLabel")
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setWordWrap(True)
        self.probabilities_label = QLabel("Probabilities: --")
        self.probabilities_label.setWordWrap(True)
        self.model_label = QLabel("Model: --")
        self.update_time_label = QLabel("Updated: --")

        info_layout.addWidget(self.price_label)
        info_layout.addWidget(self.signal_label)
        info_layout.addWidget(self.confidence_label)
        info_layout.addWidget(self.probabilities_label)
        info_layout.addWidget(self.model_label)
        info_layout.addWidget(self.update_time_label)

        layout.addWidget(info_card)

        # SMC Rule-Based Signal Card
        smc_card = QFrame()
        smc_card.setObjectName("infoCard")
        smc_layout = QVBoxLayout(smc_card)

        smc_header = QLabel("🧱 SMC Rule-Based")
        smc_header.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 4px;")
        smc_layout.addWidget(smc_header)

        self.smc_signal_label = QLabel("Signal: --")
        self.smc_signal_label.setObjectName("signalLabel")

        self.smc_setup_label = QLabel("Setup: --")
        self.smc_setup_label.setStyleSheet("font-size: 13px; margin: 2px 0px;")

        self.smc_quality_label = QLabel("Quality: --")
        self.smc_quality_label.setStyleSheet("font-size: 13px; margin: 2px 0px;")

        self.smc_rr_label = QLabel("Risk/Reward: --")
        self.smc_rr_label.setStyleSheet("font-size: 13px; margin: 2px 0px;")

        self.smc_entry_label = QLabel("Entry: --")
        self.smc_entry_label.setStyleSheet("font-size: 12px; margin: 2px 0px;")

        self.smc_sl_label = QLabel("Stop Loss: --")
        self.smc_sl_label.setStyleSheet("font-size: 12px; margin: 2px 0px;")

        self.smc_tp_label = QLabel("Take Profit: --")
        self.smc_tp_label.setStyleSheet("font-size: 12px; margin: 2px 0px;")

        self.smc_update_time_label = QLabel("Updated: --")
        self.smc_update_time_label.setStyleSheet("font-size: 12px; color: #787B86; margin: 2px 0px;")

        smc_layout.addWidget(self.smc_signal_label)
        smc_layout.addWidget(self.smc_setup_label)
        smc_layout.addWidget(self.smc_quality_label)
        smc_layout.addWidget(self.smc_rr_label)
        smc_layout.addWidget(self.smc_entry_label)
        smc_layout.addWidget(self.smc_sl_label)
        smc_layout.addWidget(self.smc_tp_label)
        smc_layout.addWidget(self.smc_update_time_label)

        layout.addWidget(smc_card)

        # Status label
        self.status_label = QLabel("🔄 Ready")
        self.status_label.setStyleSheet("font-size: 12px; color: #787B86; padding: 5px;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        scroll.setWidget(content)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.addWidget(scroll)

        return sidebar

    def check_for_updates(self):
        """Check for new data"""
        try:
            if self.db.check_for_new_data():
                print("📊 New data detected, reloading...")
                self.load_data()

            if self.db.check_for_new_prediction():
                print("🤖 New prediction detected, updating...")
                self.update_prediction()
        except Exception as e:
            print(f"Error checking for updates: {e}")

    def load_data(self):
        """Load all data from database"""
        print("=" * 80)
        print("Loading data...")
        self.status_label.setText("🔄 Loading...")

        # Load OHLCV
        self.df = self.db.load_ohlcv(self.start_date.date(), self.end_date.date())

        if self.df.empty:
            print("No data available")
            self.status_label.setText("⚠️ No data")
            return

        print(f"📊 Loaded {len(self.df)} OHLCV candles")

        # Load technical indicators and merge
        tech_df = self.db.load_smc_table(
            SMC_TABLE_NAMES["technical"], self.start_date.date(), self.end_date.date()
        )

        if not tech_df.empty:
            self.df = pd.merge(self.df, tech_df, on='timestamp', how='left', suffixes=('', '_tech'))
            print(f"✅ Merged technical indicators: {len(tech_df)} rows")

        # Load SMC features
        self.smc_features = {}
        if self.swing_check.isChecked():
            self.smc_features["swing"] = self.db.load_smc_table(
                SMC_TABLE_NAMES["swing"], self.start_date.date(), self.end_date.date()
            )
        if self.fvg_check.isChecked():
            self.smc_features["fvg"] = self.db.load_smc_table(
                SMC_TABLE_NAMES["fvg"], self.start_date.date(), self.end_date.date()
            )
        if self.bos_check.isChecked():
            self.smc_features["bos_choch"] = self.db.load_smc_table(
                SMC_TABLE_NAMES["bos_choch"], self.start_date.date(), self.end_date.date()
            )
        if self.ob_check.isChecked():
            self.smc_features["ob"] = self.db.load_smc_table(
                SMC_TABLE_NAMES["ob"], self.start_date.date(), self.end_date.date()
            )
        if self.liquidity_check.isChecked():
            self.smc_features["liquidity"] = self.db.load_smc_table(
                SMC_TABLE_NAMES["liquidity"], self.start_date.date(), self.end_date.date()
            )

        print("=" * 80)
        self.update_main_chart()
        self.update_indicators()
        self.update_metrics()
        self.update_prediction()
        self.update_smc_signal()
        self.status_label.setText("✅ Ready")

    def update_main_chart(self):
        """Update main candlestick chart"""
        plot = self.dashboard_tab.main_plot
        plot.clear()

        if self.df.empty:
            return

        # Prepare candle data
        candle_data = []
        timestamps = []
        for idx, row in self.df.iterrows():
            candle_data.append((idx, row["open"], row["high"], row["low"], row["close"]))
            timestamps.append(row["timestamp"])

        # Create candle item
        self.dashboard_tab.candle_item = CandlestickItem(candle_data, timestamps)
        plot.addItem(self.dashboard_tab.candle_item)

        # Re-add crosshair and hover label
        plot.addItem(self.dashboard_tab.crosshair_v, ignoreBounds=True)
        plot.addItem(self.dashboard_tab.crosshair_h, ignoreBounds=True)
        plot.addItem(self.dashboard_tab.hover_label, ignoreBounds=True)

        # Set Y range
        y_min = np.min(self.df["low"].values)
        y_max = np.max(self.df["high"].values)
        y_range = y_max - y_min
        padding = y_range * 0.05
        plot.setYRange(y_min - padding, y_max + padding, padding=0)
        plot.setXRange(-2, len(candle_data) + 2, padding=0)

        # Update overlays and SMC features
        self.update_overlays()
        self.update_smc_features()
        self.setup_time_axis(plot)

    def update_overlays(self):
        """Update EMA and Bollinger Bands overlays on main chart"""
        plot = self.dashboard_tab.main_plot

        # Clear existing overlay items
        for item in self.smc_plot_items["indicators"]:
            try:
                plot.removeItem(item)
            except:
                pass
        self.smc_plot_items["indicators"] = []

        if self.df.empty:
            return

        # Get current Y range before adding indicators
        view_range = plot.viewRange()
        y_min, y_max = view_range[1]

        # Add EMAs
        if self.ema_check.isChecked():
            ema_configs = [
                ("ema_9", COLORS["ema_9"], "EMA 9"),
                ("ema_21", COLORS["ema_21"], "EMA 21"),
                ("ema_50", COLORS["ema_50"], "EMA 50"),
                ("ema_100", COLORS["ema_100"], "EMA 100"),
                ("ema_200", COLORS["ema_200"], "EMA 200"),
            ]

            for ema_col, color, label in ema_configs:
                if ema_col in self.df.columns:
                    ema_data = self.df[ema_col].values
                    valid_mask = (~pd.isna(ema_data)) & (ema_data != 0.0)

                    if valid_mask.any():
                        x_data = np.arange(len(self.df))[valid_mask]
                        y_data = ema_data[valid_mask]

                        line = plot.plot(
                            x_data, y_data,
                            pen=pg.mkPen(color, width=2),
                            name=label
                        )
                        self.smc_plot_items["indicators"].append(line)
                        print(f"✅ {label} plotted: {len(y_data)} points")

        # Add Bollinger Bands
        if self.bb_check.isChecked():
            if all(col in self.df.columns for col in ["bb_upper_20", "bb_middle_20", "bb_lower_20"]):
                bb_upper = self.df["bb_upper_20"].values
                bb_middle = self.df["bb_middle_20"].values
                bb_lower = self.df["bb_lower_20"].values

                valid_mask = (~pd.isna(bb_upper)) & (~pd.isna(bb_lower)) & \
                             (bb_upper != 0.0) & (bb_lower != 0.0)

                if valid_mask.any():
                    x_data = np.arange(len(self.df))[valid_mask]

                    upper_line = plot.plot(
                        x_data, bb_upper[valid_mask],
                        pen=pg.mkPen(COLORS["bb_upper"], width=1, style=Qt.DashLine),
                        name="BB Upper"
                    )
                    middle_line = plot.plot(
                        x_data, bb_middle[valid_mask],
                        pen=pg.mkPen(COLORS["bb_middle"], width=1, style=Qt.DotLine),
                        name="BB Middle"
                    )
                    lower_line = plot.plot(
                        x_data, bb_lower[valid_mask],
                        pen=pg.mkPen(COLORS["bb_lower"], width=1, style=Qt.DashLine),
                        name="BB Lower"
                    )

                    self.smc_plot_items["indicators"].extend([upper_line, middle_line, lower_line])
                    print(f"✅ Bollinger Bands plotted: {valid_mask.sum()} points")

        # Restore Y range to keep it stable
        plot.setYRange(y_min, y_max, padding=0)

    def update_smc_features(self):
        """Update SMC features on chart"""
        plot = self.dashboard_tab.main_plot

        # Clear existing SMC items
        for key in ["swing_high", "swing_low", "fvg", "bos", "choch", "ob", "liquidity"]:
            for item in self.smc_plot_items[key]:
                try:
                    plot.removeItem(item)
                except:
                    pass
            self.smc_plot_items[key] = []

        if self.df.empty:
            return

        timestamp_to_idx = {ts: idx for idx, ts in enumerate(self.df["timestamp"])}

        # Add Swing High/Low
        if self.swing_check.isChecked() and "swing" in self.smc_features:
            swing_df = self.smc_features["swing"]
            if not swing_df.empty and "HighLow" in swing_df.columns:
                valid_swings = swing_df[swing_df["HighLow"].notna() & (swing_df["HighLow"] != 0)]

                for _, row in valid_swings.iterrows():
                    ts = row["timestamp"]
                    if ts in timestamp_to_idx:
                        idx = timestamp_to_idx[ts]

                        if row["HighLow"] == 1:
                            scatter = pg.ScatterPlotItem(
                                [idx], [row["Level"]],
                                symbol="t", size=12,
                                brush=pg.mkBrush(COLORS["swing_high"]),
                                pen=pg.mkPen(COLORS["swing_high"], width=2)
                            )
                            plot.addItem(scatter)
                            self.smc_plot_items["swing_high"].append(scatter)
                        elif row["HighLow"] == -1:
                            scatter = pg.ScatterPlotItem(
                                [idx], [row["Level"]],
                                symbol="t1", size=12,
                                brush=pg.mkBrush(COLORS["swing_low"]),
                                pen=pg.mkPen(COLORS["swing_low"], width=2)
                            )
                            plot.addItem(scatter)
                            self.smc_plot_items["swing_low"].append(scatter)

        # Add Fair Value Gap
        if self.fvg_check.isChecked() and "fvg" in self.smc_features:
            fvg_df = self.smc_features["fvg"]
            if not fvg_df.empty and "FVG" in fvg_df.columns:
                valid_fvg = fvg_df[fvg_df["FVG"].notna() & (fvg_df["FVG"] != 0)]

                for _, row in valid_fvg.iterrows():
                    ts = row["timestamp"]
                    if ts in timestamp_to_idx:
                        idx = timestamp_to_idx[ts]
                        rect = pg.QtWidgets.QGraphicsRectItem(
                            idx, row["Bottom"], 15, row["Top"] - row["Bottom"]
                        )
                        rect.setPen(pg.mkPen(COLORS["fvg"], width=1))
                        rect.setBrush(pg.mkBrush(COLORS["fvg"] + "40"))
                        plot.addItem(rect)
                        self.smc_plot_items["fvg"].append(rect)

        # Add BOS/CHOCH
        if self.bos_check.isChecked() and "bos_choch" in self.smc_features:
            bos_df = self.smc_features["bos_choch"]
            if not bos_df.empty:
                if "BOS" in bos_df.columns:
                    valid_bos = bos_df[bos_df["BOS"].notna() & (bos_df["BOS"] != 0)]
                    for _, row in valid_bos.iterrows():
                        ts = row["timestamp"]
                        if ts in timestamp_to_idx:
                            idx = timestamp_to_idx[ts]
                            color = COLORS["bos_bull"] if row["BOS"] == 1 else COLORS["bos_bear"]
                            end_idx = min(idx + 50, len(self.df) - 1)
                            line = plot.plot(
                                [idx, end_idx], [row["Level"], row["Level"]],
                                pen=pg.mkPen(color, width=3, style=Qt.DashDotLine)
                            )
                            self.smc_plot_items["bos"].append(line)

                if "CHOCH" in bos_df.columns:
                    valid_choch = bos_df[bos_df["CHOCH"].notna() & (bos_df["CHOCH"] != 0)]
                    for _, row in valid_choch.iterrows():
                        ts = row["timestamp"]
                        if ts in timestamp_to_idx:
                            idx = timestamp_to_idx[ts]
                            color = COLORS["choch_bull"] if row["CHOCH"] == 1 else COLORS["choch_bear"]
                            end_idx = min(idx + 50, len(self.df) - 1)
                            line = plot.plot(
                                [idx, end_idx], [row["Level"], row["Level"]],
                                pen=pg.mkPen(color, width=2, style=Qt.DashLine)
                            )
                            self.smc_plot_items["choch"].append(line)

        # Add Order Blocks
        if self.ob_check.isChecked() and "ob" in self.smc_features:
            ob_df = self.smc_features["ob"]
            if not ob_df.empty and "OB" in ob_df.columns:
                valid_ob = ob_df[ob_df["OB"].notna() & (ob_df["OB"] != 0)]
                for _, row in valid_ob.iterrows():
                    ts = row["timestamp"]
                    if ts in timestamp_to_idx:
                        idx = timestamp_to_idx[ts]
                        color = COLORS["ob_bull"] if row["OB"] == 1 else COLORS["ob_bear"]
                        width = min(50, len(self.df) - idx)
                        rect = pg.QtWidgets.QGraphicsRectItem(
                            idx, row["Bottom"], width, row["Top"] - row["Bottom"]
                        )
                        rect.setPen(pg.mkPen(color, width=1))
                        rect.setBrush(pg.mkBrush(color + "30"))
                        plot.addItem(rect)
                        self.smc_plot_items["ob"].append(rect)

        # Add Liquidity
        if self.liquidity_check.isChecked() and "liquidity" in self.smc_features:
            liq_df = self.smc_features["liquidity"]
            if not liq_df.empty and "Liquidity" in liq_df.columns:
                valid_liq = liq_df[liq_df["Liquidity"].notna() & (liq_df["Liquidity"] != 0)]
                for _, row in valid_liq.iterrows():
                    ts = row["timestamp"]
                    if ts in timestamp_to_idx:
                        idx = timestamp_to_idx[ts]
                        try:
                            end_idx = int(row["End"])
                            if end_idx < 0 or end_idx >= len(self.df):
                                end_idx = min(idx + 24, len(self.df) - 1)
                        except:
                            end_idx = min(idx + 24, len(self.df) - 1)
                        line = plot.plot(
                            [idx, end_idx], [row["Level"], row["Level"]],
                            pen=pg.mkPen(COLORS["liquidity"], width=2)
                        )
                        self.smc_plot_items["liquidity"].append(line)

    def update_indicators(self):
        """Update technical indicators (RSI, MACD)"""
        if self.df.empty:
            return

        # RSI Plot
        rsi_plot = self.dashboard_tab.rsi_plot
        rsi_plot.clear()

        if self.rsi_check.isChecked() and "rsi_14" in self.df.columns:
            rsi_data = self.df["rsi_14"].values
            valid_mask = (~pd.isna(rsi_data)) & (rsi_data != 0.0)

            if valid_mask.any():
                rsi_plot.show()
                x_data = np.arange(len(rsi_data))[valid_mask]
                y_data = rsi_data[valid_mask]
                rsi_plot.plot(x_data, y_data, pen=pg.mkPen(COLORS["rsi"], width=2))
                rsi_plot.addLine(y=70, pen=pg.mkPen(COLORS["candle_down"], width=1, style=Qt.DashLine))
                rsi_plot.addLine(y=30, pen=pg.mkPen(COLORS["candle_up"], width=1, style=Qt.DashLine))
                print(f"✅ RSI plotted: {len(y_data)} points")
            else:
                rsi_plot.hide()
        else:
            rsi_plot.hide()

        # MACD Plot
        macd_plot = self.dashboard_tab.macd_plot
        macd_plot.clear()

        if self.macd_check.isChecked() and all(col in self.df.columns for col in ["macd", "macd_signal", "macd_hist"]):
            macd = self.df["macd"].values
            signal = self.df["macd_signal"].values
            hist = self.df["macd_hist"].values

            valid_mask = (~pd.isna(macd)) & (~pd.isna(signal)) & \
                         (macd != 0.0) & (signal != 0.0)

            if valid_mask.any():
                macd_plot.show()
                x_data = np.arange(len(macd))[valid_mask]

                macd_plot.plot(x_data, macd[valid_mask], pen=pg.mkPen(COLORS["macd"], width=2))
                macd_plot.plot(x_data, signal[valid_mask], pen=pg.mkPen(COLORS["signal"], width=2))

                hist_valid = hist[valid_mask]
                pos_hist = np.where(hist_valid >= 0, hist_valid, 0)
                neg_hist = np.where(hist_valid < 0, hist_valid, 0)

                bg1 = pg.BarGraphItem(x=x_data, height=pos_hist, width=0.6, brush=COLORS["candle_up"])
                bg2 = pg.BarGraphItem(x=x_data, height=neg_hist, width=0.6, brush=COLORS["candle_down"])

                macd_plot.addItem(bg1)
                macd_plot.addItem(bg2)
                print(f"✅ MACD plotted: {len(x_data)} points")
            else:
                macd_plot.hide()
        else:
            macd_plot.hide()

    def update_metrics(self):
        """Update metric cards"""
        if self.df.empty:
            return

        latest_price = float(self.df.iloc[-1]["close"])
        latest_volume = float(self.df.iloc[-1]["volume"])

        self.dashboard_tab.metric_price.set_value(f"${latest_price:,.2f}")
        self.price_label.setText(f"Price: ${latest_price:,.2f}")

        # 24h change
        if len(self.df) >= 24:
            price_24h_ago = float(self.df.iloc[-24]["close"])
            change = ((latest_price - price_24h_ago) / price_24h_ago) * 100
            change_str = f"{'+' if change >= 0 else ''}{change:.2f}%"
            color = "#26A69A" if change >= 0 else "#EF5350"
            self.dashboard_tab.metric_change.value_label.setText(
                f'<span style="color: {color}">{change_str}</span>'
            )
        else:
            self.dashboard_tab.metric_change.set_value("N/A")

        # Volume
        if latest_volume >= 1_000_000:
            volume_str = f"{latest_volume / 1_000_000:.2f}M"
        elif latest_volume >= 1_000:
            volume_str = f"{latest_volume / 1_000:.2f}K"
        else:
            volume_str = f"{latest_volume:.0f}"
        self.dashboard_tab.metric_volume.set_value(volume_str)

    def update_prediction(self):
        """Update prediction display"""
        prediction = self.db.get_latest_prediction()

        if prediction:
            signal = prediction["signal"].upper()
            confidence = prediction["confidence"]
            prob_down = prediction["prob_down"]
            prob_hold = prediction["prob_hold"]
            prob_up = prediction["prob_up"]
            model_type = prediction["model_type"]

            confidence_pct = f"{confidence * 100:.1f}%"
            self.dashboard_tab.metric_signal.set_value(f"{signal} ({confidence_pct})")

            signal_emoji = {"UP": "🔼", "DOWN": "🔽", "HOLD": "↔️"}.get(signal, "⚠️")
            self.signal_label.setText(f"{signal_emoji} {signal}")

            if signal == "UP":
                self.signal_label.setObjectName("upSignal")
            elif signal == "DOWN":
                self.signal_label.setObjectName("downSignal")
            else:
                self.signal_label.setObjectName("holdSignal")
            self.signal_label.setStyleSheet(DARK_THEME_QSS)

            self.confidence_label.setText(f"Confidence: {confidence_pct}")

            # Color-coded probabilities with proper emoji
            prob_text = (
                f'<span style="color: #26A69A">🔼{prob_up*100:.0f}%</span> '
                f'<span style="color: #2962FF">↔️{prob_hold*100:.0f}%</span> '
                f'<span style="color: #EF5350">🔽{prob_down*100:.0f}%</span>'
            )
            self.probabilities_label.setText(prob_text)

            self.model_label.setText(f"Model: {model_type}")

            try:
                pred_dt = pd.to_datetime(prediction["timestamp"]).tz_localize(None)
                time_ago = datetime.now() - pred_dt
                seconds = time_ago.total_seconds()

                if seconds < 60:
                    time_str = "just now"
                elif seconds < 3600:
                    time_str = f"{int(seconds / 60)}m ago"
                elif seconds < 86400:
                    time_str = f"{int(seconds / 3600)}h ago"
                else:
                    time_str = f"{int(seconds / 86400)}d ago"
                self.update_time_label.setText(f"Updated: {time_str}")
            except:
                self.update_time_label.setText("Updated: --")
        else:
            self.dashboard_tab.metric_signal.set_value("N/A")
            self.signal_label.setText("⚠️ No Data")
            self.confidence_label.setText("Confidence: --")
            self.probabilities_label.setText("Probabilities: --")
            self.model_label.setText("Model: --")
            self.update_time_label.setText("Updated: --")

    def update_smc_signal(self):
        """Update SMC rule-based signal display"""
        print("🔄 Updating SMC signal display...")
        smc_signal = self.db.get_latest_smc_signal()

        if smc_signal:
            print(f"📊 SMC Signal data received: {smc_signal['smc_signal']}")

            signal = smc_signal["smc_signal"].upper()
            setup_type = smc_signal["setup_type"]
            setup_quality = smc_signal["setup_quality"]
            risk_reward = smc_signal["risk_reward"]
            entry_price = smc_signal["entry_price"]
            stop_loss = smc_signal["stop_loss"]
            tp1 = smc_signal["take_profit_1"]
            tp2 = smc_signal["take_profit_2"]
            tp3 = smc_signal["take_profit_3"]

            # Signal emoji
            signal_emoji = {"BUY": "🟧", "SELL": "🟥", "HOLD": "🟦"}.get(signal, "⚠️")
            self.smc_signal_label.setText(f"{signal_emoji} {signal}")

            if signal == "BUY":
                self.smc_signal_label.setObjectName("upSignal")
            elif signal == "SELL":
                self.smc_signal_label.setObjectName("downSignal")
            else:
                self.smc_signal_label.setObjectName("holdSignal")
            self.smc_signal_label.setStyleSheet(DARK_THEME_QSS)

            # Setup type
            self.smc_setup_label.setText(f"Setup: {setup_type}")

            # Setup quality with color
            quality_colors = {
                "High": "#26A69A",
                "Medium": "#FFA726",
                "Low": "#EF5350",
                "No Trade": "#787B86"
            }
            quality_color = quality_colors.get(setup_quality, "#787B86")
            self.smc_quality_label.setText(
                f'Quality: <span style="color: {quality_color}">{setup_quality}</span>'
            )

            # Risk/Reward with color
            if risk_reward and risk_reward > 0:
                rr_color = "#26A69A" if risk_reward >= 2.0 else "#FFA726"
                self.smc_rr_label.setText(
                    f'R/R: <span style="color: {rr_color}">1:{risk_reward:.1f}</span>'
                )
            else:
                self.smc_rr_label.setText("R/R: --")

            # Entry price with color
            if entry_price:
                self.smc_entry_label.setText(
                    f'<span style="color: #D1D4DC">Entry: ${entry_price:,.2f}</span>'
                )
            else:
                self.smc_entry_label.setText("Entry: --")

            # Stop loss with red color
            if stop_loss:
                self.smc_sl_label.setText(
                    f'<span style="color: #EF5350">SL: ${stop_loss:,.2f}</span>'
                )
            else:
                self.smc_sl_label.setText("Stop Loss: --")

            # Take profit with green color
            tp_parts = []
            if tp1:
                tp_parts.append(f"TP1: ${tp1:,.0f}")
            if tp2:
                tp_parts.append(f"TP2: ${tp2:,.0f}")
            if tp3:
                tp_parts.append(f"TP3: ${tp3:,.0f}")

            if tp_parts:
                tp_text = " | ".join(tp_parts)
                self.smc_tp_label.setText(
                    f'<span style="color: #26A69A">{tp_text}</span>'
                )
            else:
                self.smc_tp_label.setText("Take Profit: --")

            # Update timestamp
            try:
                smc_dt = pd.to_datetime(smc_signal["timestamp"])
                if hasattr(smc_dt, 'tz') and smc_dt.tz is not None:
                    smc_dt = smc_dt.tz_localize(None)

                time_ago = datetime.now() - smc_dt
                seconds = time_ago.total_seconds()

                if seconds < 60:
                    time_str = "just now"
                elif seconds < 3600:
                    time_str = f"{int(seconds / 60)}m ago"
                elif seconds < 86400:
                    time_str = f"{int(seconds / 3600)}h ago"
                else:
                    time_str = f"{int(seconds / 86400)}d ago"
                self.smc_update_time_label.setText(f"Updated: {time_str}")
            except Exception as e:
                print(f"⚠️ Error parsing timestamp: {e}")
                self.smc_update_time_label.setText("Updated: --")

            print("✅ SMC signal display updated successfully")
        else:
            print("⚠️ No SMC signal data available")
            self.smc_signal_label.setText("⚠️ No Data")
            self.smc_signal_label.setObjectName("holdSignal")
            self.smc_signal_label.setStyleSheet(DARK_THEME_QSS)
            self.smc_setup_label.setText("Setup: --")
            self.smc_quality_label.setText("Quality: --")
            self.smc_rr_label.setText("R/R: --")
            self.smc_entry_label.setText("Entry: --")
            self.smc_sl_label.setText("Stop Loss: --")
            self.smc_tp_label.setText("Take Profit: --")
            self.smc_update_time_label.setText("Updated: --")

    def setup_time_axis(self, plot):
        """Setup time axis labels"""
        if self.df.empty:
            return

        axis = plot.getAxis("bottom")
        n_labels = min(10, len(self.df))
        indices = np.linspace(0, len(self.df) - 1, n_labels, dtype=int)

        ticks = []
        for idx in indices:
            timestamp = self.df.iloc[idx]["timestamp"]
            label = timestamp.strftime("%m/%d %H:%M")
            ticks.append((idx, label))

        axis.setTicks([ticks])

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TradingDashboard()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
