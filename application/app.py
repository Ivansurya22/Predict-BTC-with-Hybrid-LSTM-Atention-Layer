import os
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDateEdit,
    QFrame,
    QGridLayout,
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

# Import tab modules
from tab_backtesting import BacktestingTab
from tab_model import ModelManagementTab

# Remove Kvantum style override
os.environ.pop("QT_STYLE_OVERRIDE", None)

# ============================================================================
# CONSTANTS & STYLING
# ============================================================================

DARK_THEME_QSS = """
/* Main Window */
QMainWindow {
    background-color: #131722;
}

/* Tab Widget */
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

/* Sidebar */
#sidebar {
    background-color: #1E222D;
    border-right: 1px solid #2A2E39;
}

/* Group Box */
QGroupBox {
    color: #D1D4DC;
    border: 1px solid #2A2E39;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
    font-size: 13px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

/* Labels */
QLabel {
    color: #D1D4DC;
    font-size: 13px;
}

#headerLabel {
    font-size: 20px;
    font-weight: bold;
    color: #FFFFFF;
    padding: 10px;
}

#metricLabel {
    font-size: 14px;
    color: #787B86;
}

#metricValue {
    font-size: 24px;
    font-weight: bold;
    color: #2962FF;
    font-family: 'Consolas', 'Courier New', monospace;
}

#signalLabel {
    font-size: 16px;
    font-weight: bold;
    padding: 8px;
    border-radius: 4px;
}

#buySignal {
    background-color: rgba(38, 166, 154, 0.2);
    color: #26A69A;
    border: 1px solid #26A69A;
}

#sellSignal {
    background-color: rgba(239, 83, 80, 0.2);
    color: #EF5350;
    border: 1px solid #EF5350;
}

#holdSignal {
    background-color: rgba(41, 98, 255, 0.2);
    color: #2962FF;
    border: 1px solid #2962FF;
}

/* Buttons */
QPushButton {
    background-color: #2962FF;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    font-size: 13px;
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

/* CheckBox */
QCheckBox {
    color: #D1D4DC;
    spacing: 8px;
    font-size: 13px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
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

/* DateEdit */
QDateEdit {
    background-color: #2A2E39;
    color: #D1D4DC;
    border: 1px solid #434651;
    border-radius: 4px;
    padding: 6px;
    font-size: 13px;
}

QDateEdit:focus {
    border: 1px solid #2962FF;
}

QDateEdit::drop-down {
    border: none;
    width: 20px;
}

/* SpinBox and DoubleSpinBox */
QSpinBox, QDoubleSpinBox {
    background-color: #2A2E39;
    color: #D1D4DC;
    border: 1px solid #434651;
    border-radius: 4px;
    padding: 6px;
    font-size: 13px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #2962FF;
}

/* ComboBox */
QComboBox {
    background-color: #2A2E39;
    color: #D1D4DC;
    border: 1px solid #434651;
    border-radius: 4px;
    padding: 6px;
    font-size: 13px;
}

QComboBox:focus {
    border: 1px solid #2962FF;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #2A2E39;
    color: #D1D4DC;
    selection-background-color: #2962FF;
}

/* ProgressBar */
QProgressBar {
    background-color: #2A2E39;
    border: 1px solid #434651;
    border-radius: 4px;
    text-align: center;
    color: #D1D4DC;
}

QProgressBar::chunk {
    background-color: #2962FF;
    border-radius: 3px;
}

/* Table Widget */
QTableWidget {
    background-color: #1E222D;
    color: #D1D4DC;
    border: 1px solid #2A2E39;
    gridline-color: #2A2E39;
}

QTableWidget::item {
    padding: 5px;
}

QTableWidget::item:selected {
    background-color: #2962FF;
}

QHeaderView::section {
    background-color: #2A2E39;
    color: #D1D4DC;
    padding: 8px;
    border: none;
    font-weight: bold;
}

/* ScrollArea */
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

QScrollBar:horizontal {
    background-color: #2A2E39;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #434651;
    border-radius: 6px;
    min-width: 20px;
}

/* Info Card */
#infoCard {
    background-color: #1E222D;
    border: 1px solid #2A2E39;
    border-radius: 8px;
    padding: 12px;
}

/* Metric Card */
#metricCard {
    background-color: rgba(41, 98, 255, 0.1);
    border: 1px solid rgba(41, 98, 255, 0.3);
    border-radius: 6px;
    padding: 10px;
}

/* Text Edit */
QTextEdit {
    background-color: #1E222D;
    color: #D1D4DC;
    border: 1px solid #2A2E39;
    border-radius: 4px;
    font-size: 13px;
}
"""

# Chart colors
COLORS = {
    "bg": "#131722",
    "grid": "#2A2E39",
    "text": "#D1D4DC",
    "candle_up": "#26A69A",
    "candle_down": "#EF5350",
    "swing_high": "#26A69A",
    "swing_low": "#EF5350",
    "fvg": "#FFEB3B",
    "bos_bull": "#4CAF50",
    "bos_bear": "#EF5350",
    "choch_bull": "#00BCD4",
    "choch_bear": "#FF9800",
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
    "bb_lower": "#F1FAEE",
}

DB_PATH = "data/btc_ohlcv.db"

# FIXED: Updated table names to match new structure
SMC_TABLE_NAMES = {
    "swing": "smc_btc_1h_swing_highs_lows",
    "fvg": "smc_btc_1h_fvg",
    "bos_choch": "smc_btc_1h_bos_choch",
    "ob": "smc_btc_1h_order_block",
    "liquidity": "smc_btc_1h_liquidity",
    "retracements": "smc_btc_1h_retracements",
    "technical": "smc_btc_1h_technical_indicators",
    "optimized": "smc_btc_1h_optimized_features",
}

# ============================================================================
# CUSTOM CANDLESTICK ITEM
# ============================================================================


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick chart item for PyQtGraph with hover support"""

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
            if close > open_:
                painter.setPen(pg.mkPen(COLORS["candle_up"]))
                painter.setBrush(pg.mkBrush(COLORS["candle_up"]))
            else:
                painter.setPen(pg.mkPen(COLORS["candle_down"]))
                painter.setBrush(pg.mkBrush(COLORS["candle_down"]))

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
    """Handle all database operations with caching and type conversion"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.cache = {}
        self.cache_time = {}
        self.cache_ttl = 300

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

    def load_ohlcv(
        self, start_date: QDate, end_date: QDate, limit: int = 504
    ) -> pd.DataFrame:
        """Load OHLCV data"""
        cache_key = f"ohlcv_{start_date.toString()}_{end_date.toString()}"

        if cache_key in self.cache:
            if datetime.now().timestamp() - self.cache_time[cache_key] < self.cache_ttl:
                return self.cache[cache_key]

        try:
            conn = self.get_connection()
            query = f"""
            SELECT * FROM btc_1h
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
            LIMIT {limit}
            """

            start_str = start_date.toString("yyyy-MM-dd")
            end_str = end_date.addDays(1).toString("yyyy-MM-dd")

            df = pd.read_sql(query, conn, params=(start_str, end_str))
            conn.close()

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

            df = df.reset_index(drop=True)

            self.cache[cache_key] = df
            self.cache_time[cache_key] = datetime.now().timestamp()

            return df
        except Exception as e:
            print(f"Error loading OHLCV: {e}")
            return pd.DataFrame()

    def load_smc_table(
        self, table_name: str, start_date: QDate, end_date: QDate
    ) -> pd.DataFrame:
        """Load SMC feature table with proper type conversion"""
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

            # CRITICAL: Convert all numeric columns (except timestamp)
            for col in df.columns:
                if col != 'timestamp':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"✅ Loaded {table_name}: {len(df)} rows")
            return df
        except Exception as e:
            print(f"⚠️ Error loading {table_name}: {e}")
            return pd.DataFrame()

    def get_latest_price(self) -> Optional[float]:
        """Get latest close price"""
        try:
            conn = self.get_connection()
            result = pd.read_sql(
                "SELECT close FROM btc_1h ORDER BY timestamp DESC LIMIT 1", conn
            )
            conn.close()
            return result["close"].iloc[0] if not result.empty else None
        except Exception:
            return None

    def get_latest_prediction(self) -> Optional[Dict]:
        """Get latest prediction from database"""
        try:
            conn = self.get_connection()
            query = """
            SELECT timestamp, signal, confidence, prob_buy, prob_sell, prob_hold
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 1
            """
            pred_df = pd.read_sql(query, conn)
            conn.close()

            if not pred_df.empty:
                return {
                    "timestamp": pred_df["timestamp"].iloc[0],
                    "signal": pred_df["signal"].iloc[0],
                    "confidence": pred_df["confidence"].iloc[0],
                    "prob_buy": pred_df["prob_buy"].iloc[0],
                    "prob_sell": pred_df["prob_sell"].iloc[0],
                    "prob_hold": pred_df["prob_hold"].iloc[0],
                }
            return None
        except Exception as e:
            print(f"⚠️ Error loading prediction: {e}")
            return None


# ============================================================================
# METRIC CARD WIDGET
# ============================================================================


class MetricCard(QFrame):
    """Modern metric display card"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setFixedHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricLabel")

        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str):
        """Update metric value"""
        self.value_label.setText(value)


# ============================================================================
# DASHBOARD TAB (Main Chart)
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

        # Top info bar with metrics
        info_bar = QWidget()
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(0, 0, 0, 10)

        self.metric_price = MetricCard("Current Price")
        self.metric_change = MetricCard("24h Change")
        self.metric_volume = MetricCard("Volume")
        self.metric_signal = MetricCard("Signal")

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

        # Setup hover functionality
        self.main_plot.scene().sigMouseMoved.connect(self.on_mouse_moved)

        # Create crosshair lines
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False,
                                          pen=pg.mkPen(COLORS["text"], width=1, style=Qt.DashLine))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False,
                                          pen=pg.mkPen(COLORS["text"], width=1, style=Qt.DashLine))
        self.main_plot.addItem(self.crosshair_v, ignoreBounds=True)
        self.main_plot.addItem(self.crosshair_h, ignoreBounds=True)

        # Create hover label
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

        self.rsi_plot.addLine(
            y=70, pen=pg.mkPen(COLORS["candle_down"], width=1, style=Qt.DashLine)
        )
        self.rsi_plot.addLine(
            y=30, pen=pg.mkPen(COLORS["candle_up"], width=1, style=Qt.DashLine)
        )

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

        # Stochastic chart
        self.stoch_plot = pg.PlotWidget()
        self.stoch_plot.setBackground(COLORS["bg"])
        self.stoch_plot.showGrid(x=True, y=True, alpha=0.3)
        self.stoch_plot.setLabel("left", "Stochastic", color=COLORS["text"])
        self.stoch_plot.setMinimumHeight(150)
        self.stoch_plot.setMaximumHeight(200)
        self.stoch_plot.setYRange(0, 100)
        self.stoch_plot.addLine(
            y=80, pen=pg.mkPen(COLORS["candle_down"], width=1, style=Qt.DashLine)
        )
        self.stoch_plot.addLine(
            y=20, pen=pg.mkPen(COLORS["candle_up"], width=1, style=Qt.DashLine)
        )
        self.stoch_plot.hide()

        for axis_name in ["bottom", "left"]:
            axis = self.stoch_plot.getAxis(axis_name)
            axis.setPen(COLORS["text"])
            axis.setTextPen(COLORS["text"])

        layout.addWidget(self.stoch_plot, stretch=1)

        # Volume indicators chart (OBV, MFI, ATR)
        self.volume_plot = pg.PlotWidget()
        self.volume_plot.setBackground(COLORS["bg"])
        self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self.volume_plot.setLabel("left", "Volume Indicators", color=COLORS["text"])
        self.volume_plot.setMinimumHeight(150)
        self.volume_plot.setMaximumHeight(200)
        self.volume_plot.hide()

        for axis_name in ["bottom", "left"]:
            axis = self.volume_plot.getAxis(axis_name)
            axis.setPen(COLORS["text"])
            axis.setTextPen(COLORS["text"])

        layout.addWidget(self.volume_plot, stretch=1)

    def on_mouse_moved(self, pos):
        """Handle mouse move event to show candle info"""
        if self.candle_item is None or self.parent_window.df.empty:
            return

        # Convert mouse position to plot coordinates
        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()

        # Update crosshair
        self.crosshair_v.setPos(x)
        self.crosshair_h.setPos(y)

        # Get candle data at mouse position
        candle_data = self.candle_item.get_candle_at_pos(x)

        if candle_data:
            timestamp = candle_data['timestamp']
            open_price = candle_data['open']
            high_price = candle_data['high']
            low_price = candle_data['low']
            close_price = candle_data['close']

            # Format timestamp
            if timestamp:
                time_str = timestamp.strftime("%Y-%m-%d %H:%M")
            else:
                time_str = f"Index: {candle_data['index']}"

            # Calculate change
            change = close_price - open_price
            change_pct = (change / open_price) * 100 if open_price != 0 else 0

            # Create hover text
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
        self.technical_indicators = pd.DataFrame()

        self.smc_plot_items = {
            "swing_high": [],
            "swing_low": [],
            "fvg": [],
            "bos": [],
            "choch": [],
            "ob": [],
            "liquidity": [],
            "ema": [],
        }

        self.setWindowTitle("📊 BTC Trading Dashboard - Smart Money Concepts")
        self.setGeometry(100, 100, 1600, 900)

        self.setStyleSheet(DARK_THEME_QSS)

        self.setup_ui()

        # Timer untuk full refresh (chart + data)
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(60000)  # Refresh every 1 minute (60000 ms)

        # Initial load
        self.load_data()

        # Timer untuk update metrics saja (lebih sering)
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics_only)
        self.metrics_timer.start(5000)  # Update metrics every 5 seconds

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
        """Create left sidebar with controls"""
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(300)

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
        self.start_date.dateChanged.connect(self.on_date_changed)
        date_layout.addWidget(self.start_date)

        date_layout.addWidget(QLabel("End Date:"))
        self.end_date = QDateEdit()
        self.end_date.setDate(max_date)
        self.end_date.setMinimumDate(min_date)
        self.end_date.setMaximumDate(max_date)
        self.end_date.setCalendarPopup(True)
        self.end_date.dateChanged.connect(self.on_date_changed)
        date_layout.addWidget(self.end_date)

        date_group.setLayout(date_layout)
        layout.addWidget(date_group)

        # Technical Indicators Group
        tech_group = QGroupBox("📈 Technical Indicators")
        tech_layout = QVBoxLayout()

        self.rsi_check = QCheckBox("RSI (14)")
        self.rsi_check.setChecked(True)
        self.rsi_check.stateChanged.connect(self.on_indicator_changed)
        tech_layout.addWidget(self.rsi_check)

        self.macd_check = QCheckBox("MACD (12,26,9)")
        self.macd_check.setChecked(True)
        self.macd_check.stateChanged.connect(self.on_indicator_changed)
        tech_layout.addWidget(self.macd_check)

        self.ema_check = QCheckBox("EMA (9, 21, 50, 100, 200)")
        self.ema_check.setChecked(False)
        self.ema_check.stateChanged.connect(self.on_indicator_changed)
        tech_layout.addWidget(self.ema_check)

        self.bb_check = QCheckBox("Bollinger Bands")
        self.bb_check.setChecked(False)
        self.bb_check.stateChanged.connect(self.on_indicator_changed)
        tech_layout.addWidget(self.bb_check)

        self.stoch_check = QCheckBox("Stochastic (14,3,3)")
        self.stoch_check.setChecked(False)
        self.stoch_check.stateChanged.connect(self.on_indicator_changed)
        tech_layout.addWidget(self.stoch_check)

        self.atr_check = QCheckBox("ATR (14)")
        self.atr_check.setChecked(False)
        self.atr_check.stateChanged.connect(self.on_indicator_changed)
        tech_layout.addWidget(self.atr_check)

        self.obv_check = QCheckBox("OBV")
        self.obv_check.setChecked(False)
        self.obv_check.stateChanged.connect(self.on_indicator_changed)
        tech_layout.addWidget(self.obv_check)

        self.mfi_check = QCheckBox("MFI (14)")
        self.mfi_check.setChecked(False)
        self.mfi_check.stateChanged.connect(self.on_indicator_changed)
        tech_layout.addWidget(self.mfi_check)

        tech_group.setLayout(tech_layout)
        layout.addWidget(tech_group)

        # SMC Features Group
        smc_group = QGroupBox("🔍 SMC Features")
        smc_layout = QVBoxLayout()

        self.swing_check = QCheckBox("Swing High/Low")
        self.swing_check.setChecked(True)
        self.swing_check.stateChanged.connect(self.on_smc_changed)
        smc_layout.addWidget(self.swing_check)

        self.fvg_check = QCheckBox("Fair Value Gap")
        self.fvg_check.setChecked(True)
        self.fvg_check.stateChanged.connect(self.on_smc_changed)
        smc_layout.addWidget(self.fvg_check)

        self.bos_check = QCheckBox("BOS/CHOCH")
        self.bos_check.setChecked(True)
        self.bos_check.stateChanged.connect(self.on_smc_changed)
        smc_layout.addWidget(self.bos_check)

        self.ob_check = QCheckBox("Order Blocks")
        self.ob_check.setChecked(True)
        self.ob_check.stateChanged.connect(self.on_smc_changed)
        smc_layout.addWidget(self.ob_check)

        self.liquidity_check = QCheckBox("Liquidity")
        self.liquidity_check.setChecked(True)
        self.liquidity_check.stateChanged.connect(self.on_smc_changed)
        smc_layout.addWidget(self.liquidity_check)

        smc_group.setLayout(smc_layout)
        layout.addWidget(smc_group)

        # Real-time Info Card
        info_card = QFrame()
        info_card.setObjectName("infoCard")
        info_layout = QVBoxLayout(info_card)

        info_header = QLabel("💹 Real-time")
        info_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        info_layout.addWidget(info_header)

        self.price_label = QLabel("Price: --")
        self.signal_label = QLabel("Signal: --")
        self.signal_label.setObjectName("signalLabel")

        info_layout.addWidget(self.price_label)
        info_layout.addWidget(self.signal_label)

        layout.addWidget(info_card)
        layout.addStretch()

        scroll.setWidget(content)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.addWidget(scroll)

        return sidebar

    def on_date_changed(self):
        """Handle date range change"""
        self.load_data()

    def on_indicator_changed(self):
        """Handle technical indicator checkbox change"""
        self.update_indicators()
        self.update_smc_features()

    def on_smc_changed(self):
        """Handle SMC feature checkbox change"""
        print("SMC checkbox changed, refreshing chart...")
        self.update_smc_features()

    def load_data(self):
        """Load all data from database"""
        print("Loading data...")

        self.df = self.db.load_ohlcv(self.start_date.date(), self.end_date.date())

        if self.df.empty:
            print("No data available for selected date range")
            return

        # Load technical indicators dan merge dengan OHLCV
        tech_df = self.db.load_smc_table(
            SMC_TABLE_NAMES["technical"], self.start_date.date(), self.end_date.date()
        )

        if not tech_df.empty:
            # Merge berdasarkan timestamp
            self.df = pd.merge(
                self.df,
                tech_df,
                on='timestamp',
                how='left',
                suffixes=('', '_tech')
            )
            print(f"✅ Merged {len(tech_df)} technical indicators rows")

        self.technical_indicators = tech_df

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
                SMC_TABLE_NAMES["bos_choch"],
                self.start_date.date(),
                self.end_date.date(),
            )

        if self.ob_check.isChecked():
            self.smc_features["ob"] = self.db.load_smc_table(
                SMC_TABLE_NAMES["ob"], self.start_date.date(), self.end_date.date()
            )

        if self.liquidity_check.isChecked():
            self.smc_features["liquidity"] = self.db.load_smc_table(
                SMC_TABLE_NAMES["liquidity"],
                self.start_date.date(),
                self.end_date.date(),
            )

        print(f"📊 Loaded {len(self.df)} candles with indicators")
        self.update_main_chart()
        self.update_indicators()
        self.update_metrics()

    def update_main_chart(self):
        """Update main candlestick chart"""
        plot = self.dashboard_tab.main_plot
        plot.clear()

        if self.df.empty:
            return

        # Prepare candle data with timestamps
        candle_data = []
        timestamps = []
        for idx, row in self.df.iterrows():
            candle_data.append(
                (idx, row["open"], row["high"], row["low"], row["close"])
            )
            timestamps.append(row["timestamp"])

        # Create and store candle item
        self.dashboard_tab.candle_item = CandlestickItem(candle_data, timestamps)
        plot.addItem(self.dashboard_tab.candle_item)

        # Re-add crosshair and hover label (they were cleared)
        plot.addItem(self.dashboard_tab.crosshair_v, ignoreBounds=True)
        plot.addItem(self.dashboard_tab.crosshair_h, ignoreBounds=True)
        plot.addItem(self.dashboard_tab.hover_label, ignoreBounds=True)

        self.update_smc_features()
        self.setup_time_axis(plot)

    def update_smc_features(self):
        """Update SMC features on chart - FIXED VERSION"""
        plot = self.dashboard_tab.main_plot

        # Clear existing items
        for key in self.smc_plot_items:
            for item in self.smc_plot_items[key]:
                try:
                    plot.removeItem(item)
                except:
                    pass
            self.smc_plot_items[key] = []

        if self.df.empty:
            return

        # Add EMAs - langsung dari df yang sudah di-merge
        if self.ema_check.isChecked():
            for ema_col, color in [
                ("ema_9", COLORS["ema_9"]),
                ("ema_21", COLORS["ema_21"]),
                ("ema_50", COLORS["ema_50"]),
                ("ema_100", COLORS["ema_100"]),
                ("ema_200", COLORS["ema_200"]),
            ]:
                if ema_col in self.df.columns:
                    ema_data = self.df[ema_col].values
                    valid_mask = ~pd.isna(ema_data)

                    if valid_mask.any():
                        x_data = np.arange(len(self.df))[valid_mask]
                        y_data = ema_data[valid_mask]

                        line = plot.plot(
                            x_data,
                            y_data,
                            pen=pg.mkPen(color, width=2),
                            name=ema_col.upper(),
                        )
                        self.smc_plot_items["ema"].append(line)
                        print(f"✅ {ema_col.upper()} plotted: {len(y_data)} points")

        # Add Bollinger Bands
        if self.bb_check.isChecked():
            if "bb_upper_20" in self.df.columns and "bb_lower_20" in self.df.columns:
                bb_upper = self.df["bb_upper_20"].values
                bb_lower = self.df["bb_lower_20"].values
                bb_middle = self.df["bb_middle_20"].values

                valid_mask = ~(pd.isna(bb_upper) | pd.isna(bb_lower))

                if valid_mask.any():
                    x_data = np.arange(len(self.df))[valid_mask]

                    upper_line = plot.plot(
                        x_data,
                        bb_upper[valid_mask],
                        pen=pg.mkPen(COLORS["bb_upper"], width=1, style=Qt.DashLine),
                        name="BB Upper"
                    )

                    middle_line = plot.plot(
                        x_data,
                        bb_middle[valid_mask],
                        pen=pg.mkPen(COLORS["text"], width=1, style=Qt.DotLine),
                        name="BB Middle"
                    )

                    lower_line = plot.plot(
                        x_data,
                        bb_lower[valid_mask],
                        pen=pg.mkPen(COLORS["bb_lower"], width=1, style=Qt.DashLine),
                        name="BB Lower"
                    )

                    self.smc_plot_items["ema"].append(upper_line)
                    self.smc_plot_items["ema"].append(middle_line)
                    self.smc_plot_items["ema"].append(lower_line)
                    print(f"✅ Bollinger Bands plotted: {valid_mask.sum()} points")

        timestamp_to_idx = {ts: idx for idx, ts in enumerate(self.df["timestamp"])}

        # Add Swing High/Low - FIXED: Proper filtering
        if self.swing_check.isChecked() and "swing" in self.smc_features:
            swing_df = self.smc_features["swing"]

            if not swing_df.empty and "HighLow" in swing_df.columns:
                # Filter valid swings (notna AND != 0)
                valid_swings = swing_df[
                    swing_df["HighLow"].notna() & (swing_df["HighLow"] != 0)
                ]

                for _, row in valid_swings.iterrows():
                    ts = row["timestamp"]
                    if ts in timestamp_to_idx:
                        idx = timestamp_to_idx[ts]

                        if row["HighLow"] == 1:
                            scatter = pg.ScatterPlotItem(
                                [idx],
                                [row["Level"]],
                                symbol="t",
                                size=12,
                                brush=pg.mkBrush(COLORS["swing_high"]),
                                pen=pg.mkPen(COLORS["swing_high"], width=2),
                            )
                            plot.addItem(scatter)
                            self.smc_plot_items["swing_high"].append(scatter)

                        elif row["HighLow"] == -1:
                            scatter = pg.ScatterPlotItem(
                                [idx],
                                [row["Level"]],
                                symbol="t1",
                                size=12,
                                brush=pg.mkBrush(COLORS["swing_low"]),
                                pen=pg.mkPen(COLORS["swing_low"], width=2),
                            )
                            plot.addItem(scatter)
                            self.smc_plot_items["swing_low"].append(scatter)

                print(f"✅ Plotted {len(valid_swings)} swing points")

        # Add Fair Value Gap - FIXED: Proper filtering
        if self.fvg_check.isChecked() and "fvg" in self.smc_features:
            fvg_df = self.smc_features["fvg"]

            if not fvg_df.empty and "FVG" in fvg_df.columns:
                # Filter valid FVGs (notna AND != 0)
                valid_fvg = fvg_df[
                    fvg_df["FVG"].notna() & (fvg_df["FVG"] != 0)
                ]

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

                print(f"✅ Plotted {len(valid_fvg)} FVG zones")

        # Add BOS/CHOCH - FIXED: Style like Streamlit (lines with limited length)
        if self.bos_check.isChecked() and "bos_choch" in self.smc_features:
            bos_df = self.smc_features["bos_choch"]

            if not bos_df.empty:
                # Plot BOS signals (DashDotLine style)
                if "BOS" in bos_df.columns:
                    valid_bos = bos_df[
                        bos_df["BOS"].notna() & (bos_df["BOS"] != 0)
                    ]

                    for _, row in valid_bos.iterrows():
                        ts = row["timestamp"]
                        if ts in timestamp_to_idx:
                            idx = timestamp_to_idx[ts]
                            color = (
                                COLORS["bos_bull"]
                                if row["BOS"] == 1
                                else COLORS["bos_bear"]
                            )

                            # Draw line from current index to +25 candles ahead
                            end_idx = min(idx + 25, len(self.df) - 1)

                            line = plot.plot(
                                [idx, end_idx],
                                [row["Level"], row["Level"]],
                                pen=pg.mkPen(color, width=3, style=Qt.DashDotLine),
                                name="BOS"
                            )
                            self.smc_plot_items["bos"].append(line)

                    print(f"✅ Plotted {len(valid_bos)} BOS signals")

                # Plot CHOCH signals (DashLine style)
                if "CHOCH" in bos_df.columns:
                    valid_choch = bos_df[
                        bos_df["CHOCH"].notna() & (bos_df["CHOCH"] != 0)
                    ]

                    for _, row in valid_choch.iterrows():
                        ts = row["timestamp"]
                        if ts in timestamp_to_idx:
                            idx = timestamp_to_idx[ts]
                            color = (
                                COLORS["choch_bull"]
                                if row["CHOCH"] == 1
                                else COLORS["choch_bear"]
                            )

                            # Draw line from current index to +25 candles ahead
                            end_idx = min(idx + 25, len(self.df) - 1)

                            line = plot.plot(
                                [idx, end_idx],
                                [row["Level"], row["Level"]],
                                pen=pg.mkPen(color, width=3, style=Qt.DashLine),
                                name="CHOCH"
                            )
                            self.smc_plot_items["choch"].append(line)

                    print(f"✅ Plotted {len(valid_choch)} CHOCH signals")

        # Add Order Blocks - FIXED: Proper filtering
        if self.ob_check.isChecked() and "ob" in self.smc_features:
            ob_df = self.smc_features["ob"]

            if not ob_df.empty and "OB" in ob_df.columns:
                valid_ob = ob_df[
                    ob_df["OB"].notna() & (ob_df["OB"] != 0)
                ]

                for _, row in valid_ob.iterrows():
                    ts = row["timestamp"]
                    if ts in timestamp_to_idx:
                        idx = timestamp_to_idx[ts]
                        color = (
                            COLORS["ob_bull"] if row["OB"] == 1 else COLORS["ob_bear"]
                        )

                        width = min(50, len(self.df) - idx)

                        rect = pg.QtWidgets.QGraphicsRectItem(
                            idx, row["Bottom"], width, row["Top"] - row["Bottom"]
                        )
                        rect.setPen(pg.mkPen(color, width=1))
                        rect.setBrush(pg.mkBrush(color + "30"))
                        plot.addItem(rect)
                        self.smc_plot_items["ob"].append(rect)

                print(f"✅ Plotted {len(valid_ob)} Order Blocks")

        # Add Liquidity - FIXED: Proper filtering
        if self.liquidity_check.isChecked() and "liquidity" in self.smc_features:
            liq_df = self.smc_features["liquidity"]

            if not liq_df.empty and "Liquidity" in liq_df.columns:
                valid_liq = liq_df[
                    liq_df["Liquidity"].notna() & (liq_df["Liquidity"] != 0)
                ]

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
                            [idx, end_idx],
                            [row["Level"], row["Level"]],
                            pen=pg.mkPen(COLORS["liquidity"], width=2),
                        )
                        self.smc_plot_items["liquidity"].append(line)

                print(f"✅ Plotted {len(valid_liq)} Liquidity zones")

    def update_indicators(self):
        """Update technical indicators (RSI, MACD, Stochastic, Volume)"""
        if self.df.empty:
            return

        # RSI Plot
        rsi_plot = self.dashboard_tab.rsi_plot
        rsi_plot.clear()

        if self.rsi_check.isChecked() and "rsi_14" in self.df.columns:
            rsi_plot.show()
            rsi_data = self.df["rsi_14"].values
            valid_mask = ~pd.isna(rsi_data)

            if valid_mask.any():
                x_data = np.arange(len(rsi_data))[valid_mask]
                y_data = rsi_data[valid_mask]

                rsi_plot.plot(
                    x_data, y_data, pen=pg.mkPen(COLORS["rsi"], width=2)
                )

                rsi_plot.addLine(
                    y=70,
                    pen=pg.mkPen(COLORS["candle_down"], width=1, style=Qt.DashLine),
                )
                rsi_plot.addLine(
                    y=30, pen=pg.mkPen(COLORS["candle_up"], width=1, style=Qt.DashLine)
                )

                print(f"✅ RSI plotted: {len(y_data)} points")
        else:
            rsi_plot.hide()

        # MACD Plot
        macd_plot = self.dashboard_tab.macd_plot
        macd_plot.clear()

        if self.macd_check.isChecked() and "macd" in self.df.columns:
            macd_plot.show()
            macd = self.df["macd"].values
            signal = self.df["macd_signal"].values
            hist = self.df["macd_hist"].values

            valid_mask = ~(pd.isna(macd) | pd.isna(signal) | pd.isna(hist))

            if valid_mask.any():
                x_data = np.arange(len(macd))[valid_mask]

                macd_plot.plot(
                    x_data, macd[valid_mask], pen=pg.mkPen(COLORS["macd"], width=2), name="MACD"
                )
                macd_plot.plot(
                    x_data, signal[valid_mask], pen=pg.mkPen(COLORS["signal"], width=2), name="Signal"
                )

                hist_valid = hist[valid_mask]
                pos_hist = np.where(hist_valid >= 0, hist_valid, 0)
                neg_hist = np.where(hist_valid < 0, hist_valid, 0)

                bg1 = pg.BarGraphItem(
                    x=x_data,
                    height=pos_hist,
                    width=0.6,
                    brush=COLORS["candle_up"],
                )
                bg2 = pg.BarGraphItem(
                    x=x_data,
                    height=neg_hist,
                    width=0.6,
                    brush=COLORS["candle_down"],
                )

                macd_plot.addItem(bg1)
                macd_plot.addItem(bg2)

                print(f"✅ MACD plotted: {len(x_data)} points")
        else:
            macd_plot.hide()

        # Stochastic Plot
        stoch_plot = self.dashboard_tab.stoch_plot
        stoch_plot.clear()

        if self.stoch_check.isChecked() and "stoch_k" in self.df.columns:
            stoch_plot.show()
            stoch_k = self.df["stoch_k"].values
            stoch_d = self.df["stoch_d"].values

            valid_mask = ~(pd.isna(stoch_k) | pd.isna(stoch_d))

            if valid_mask.any():
                x_data = np.arange(len(stoch_k))[valid_mask]

                stoch_plot.plot(
                    x_data, stoch_k[valid_mask],
                    pen=pg.mkPen("#FF6B6B", width=2),
                    name="%K"
                )
                stoch_plot.plot(
                    x_data, stoch_d[valid_mask],
                    pen=pg.mkPen("#4ECDC4", width=2),
                    name="%D"
                )

                stoch_plot.addLine(y=80, pen=pg.mkPen(COLORS["candle_down"], width=1, style=Qt.DashLine))
                stoch_plot.addLine(y=20, pen=pg.mkPen(COLORS["candle_up"], width=1, style=Qt.DashLine))

                print(f"✅ Stochastic plotted: {len(x_data)} points")
        else:
            stoch_plot.hide()

        # Volume Indicators Plot (OBV, MFI, ATR)
        volume_plot = self.dashboard_tab.volume_plot
        volume_plot.clear()

        show_volume_plot = False

        if self.obv_check.isChecked() and "obv" in self.df.columns:
            show_volume_plot = True
            obv_data = self.df["obv"].values
            valid_mask = ~pd.isna(obv_data)

            if valid_mask.any():
                x_data = np.arange(len(obv_data))[valid_mask]
                y_data = obv_data[valid_mask]

                # Normalize OBV untuk plotting
                if y_data.max() != y_data.min():
                    y_data_norm = (y_data - y_data.min()) / (y_data.max() - y_data.min()) * 100
                else:
                    y_data_norm = np.zeros_like(y_data)

                volume_plot.plot(
                    x_data, y_data_norm,
                    pen=pg.mkPen("#9C27B0", width=2),
                    name="OBV (normalized)"
                )
                print(f"✅ OBV plotted: {len(x_data)} points")

        if self.mfi_check.isChecked() and "mfi_14" in self.df.columns:
            show_volume_plot = True
            mfi_data = self.df["mfi_14"].values
            valid_mask = ~pd.isna(mfi_data)

            if valid_mask.any():
                x_data = np.arange(len(mfi_data))[valid_mask]

                volume_plot.plot(
                    x_data, mfi_data[valid_mask],
                    pen=pg.mkPen("#FF9800", width=2),
                    name="MFI"
                )

                volume_plot.addLine(y=80, pen=pg.mkPen(COLORS["candle_down"], width=1, style=Qt.DashLine))
                volume_plot.addLine(y=20, pen=pg.mkPen(COLORS["candle_up"], width=1, style=Qt.DashLine))

                print(f"✅ MFI plotted: {len(x_data)} points")

        if self.atr_check.isChecked() and "atr_14" in self.df.columns:
            show_volume_plot = True
            atr_data = self.df["atr_14"].values
            valid_mask = ~pd.isna(atr_data)

            if valid_mask.any():
                x_data = np.arange(len(atr_data))[valid_mask]
                y_data = atr_data[valid_mask]

                # Normalize ATR
                if y_data.max() != y_data.min():
                    y_data_norm = (y_data - y_data.min()) / (y_data.max() - y_data.min()) * 100
                else:
                    y_data_norm = np.zeros_like(y_data)

                volume_plot.plot(
                    x_data, y_data_norm,
                    pen=pg.mkPen("#00BCD4", width=2),
                    name="ATR (normalized)"
                )
                print(f"✅ ATR plotted: {len(x_data)} points")

        if show_volume_plot:
            volume_plot.show()
        else:
            volume_plot.hide()

    def update_metrics(self):
        """Update metric cards with latest data"""
        if self.df.empty:
            return

        # Get latest data (most recent row)
        latest_price = float(self.df.iloc[-1]["close"])
        latest_volume = float(self.df.iloc[-1]["volume"])

        # Update price
        self.dashboard_tab.metric_price.set_value(f"${latest_price:,.2f}")
        self.price_label.setText(f"Price: ${latest_price:,.2f}")

        # Calculate 24h change (24 candles ago for 1h timeframe)
        if len(self.df) >= 24:
            price_24h_ago = float(self.df.iloc[-24]["close"])
            change = ((latest_price - price_24h_ago) / price_24h_ago) * 100
            change_str = f"{'+' if change >= 0 else ''}{change:.2f}%"

            # Update with color
            if change >= 0:
                change_display = f'<span style="color: #26A69A">{change_str}</span>'
            else:
                change_display = f'<span style="color: #EF5350">{change_str}</span>'

            self.dashboard_tab.metric_change.value_label.setText(change_display)
        else:
            self.dashboard_tab.metric_change.set_value("N/A")

        # Update volume (format large numbers)
        if latest_volume >= 1_000_000:
            volume_str = f"{latest_volume / 1_000_000:.2f}M"
        elif latest_volume >= 1_000:
            volume_str = f"{latest_volume / 1_000:.2f}K"
        else:
            volume_str = f"{latest_volume:.0f}"

        self.dashboard_tab.metric_volume.set_value(volume_str)

        # Update prediction signal
        prediction = self.db.get_latest_prediction()
        if prediction:
            signal = prediction["signal"].upper()
            confidence = f"{prediction['confidence'] * 100:.1f}%"
            self.dashboard_tab.metric_signal.set_value(f"{signal} ({confidence})")

            signal_emoji = {"BUY": "🔼", "SELL": "🔽", "HOLD": "↔️"}.get(signal, "⚠️")
            self.signal_label.setText(f"Signal: {signal_emoji} {signal} ({confidence})")

            if signal == "BUY":
                self.signal_label.setObjectName("buySignal")
            elif signal == "SELL":
                self.signal_label.setObjectName("sellSignal")
            else:
                self.signal_label.setObjectName("holdSignal")

            self.signal_label.setStyleSheet(DARK_THEME_QSS)
        else:
            self.dashboard_tab.metric_signal.set_value("N/A")
            self.signal_label.setText("Signal: ⚠️ No Data")

    def update_metrics_only(self):
        """Update only metrics without reloading entire chart - for frequent updates"""
        # Reload only the latest few rows from database
        try:
            end_date = self.end_date.date()
            start_date = end_date.addDays(-2)  # Get last 2 days to ensure we have latest data

            latest_df = self.db.load_ohlcv(start_date, end_date, limit=50)

            if not latest_df.empty:
                # Update only if we have new data
                latest_timestamp = latest_df.iloc[-1]["timestamp"]
                current_timestamp = self.df.iloc[-1]["timestamp"] if not self.df.empty else None

                if current_timestamp is None or latest_timestamp > current_timestamp:
                    # We have new data, append it
                    self.df = pd.concat([self.df, latest_df]).drop_duplicates(subset=['timestamp']).reset_index(drop=True)
                    print(f"📊 Updated with new data. Latest timestamp: {latest_timestamp}")

                # Always update metrics with latest data
                self.update_metrics()
        except Exception as e:
            print(f"⚠️ Error updating metrics: {e}")

    def setup_time_axis(self, plot):
        """Setup time axis with proper labels"""
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

    def refresh_data(self):
        """Refresh data from database (called by timer)"""
        print("Auto-refreshing data...")
        self.load_data()


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
