import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal, QDate
from PySide6.QtWidgets import (
    QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget,
    QSpinBox, QDateEdit, QMessageBox, QFileDialog, QTextEdit, QCheckBox
)
from datetime import datetime
import sys
from pathlib import Path

# Setup paths
if '__file__' in globals():
    ROOT_DIR = Path(__file__).resolve().parent.parent
else:
    ROOT_DIR = Path.cwd()

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'testing'))

# Import backtester
BACKTEST_AVAILABLE = False
BTCFuturesBacktester = None

try:
    from backtest_futures import BTCFuturesBacktester
    BACKTEST_AVAILABLE = True
except ImportError:
    try:
        from testing.backtest_futures import BTCFuturesBacktester
        BACKTEST_AVAILABLE = True
    except ImportError:
        try:
            backtest_file = ROOT_DIR / 'testing' / 'backtest.py'
            if backtest_file.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("backtest_futures", str(backtest_file))
                module = importlib.util.module_from_spec(spec)
                sys.modules['backtest_futures'] = module
                spec.loader.exec_module(module)
                BTCFuturesBacktester = module.BTCFuturesBacktester
                BACKTEST_AVAILABLE = True
        except Exception:
            pass


class FuturesBacktestWorker(QThread):
    """Background worker for futures backtesting"""
    progress = Signal(int)
    log_message = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self.log_message.emit("Initializing futures backtester...")
            self.progress.emit(10)

            backtester = BTCFuturesBacktester(
                model_path=self.config['model_path'],
                preprocessed_dir=self.config['preprocessed_dir'],
                db_path=self.config['db_path'],
                initial_capital=self.config['initial_capital'],
                trading_fee=self.config['trading_fee'],
                position_size=self.config['position_size'],
                leverage=self.config['leverage'],
                stop_loss=self.config['stop_loss'],
                take_profit=self.config['take_profit'],
                min_confidence=self.config['min_confidence'],
                max_trades_per_day=self.config['max_trades_per_day'],
                slippage=self.config['slippage'],
                min_hold_hours=self.config['min_hold_hours'],
                max_position_hours=self.config['max_position_hours'],
                use_trailing_stop=self.config['use_trailing_stop'],
                maintenance_margin=self.config['maintenance_margin'],
                funding_rate=self.config['funding_rate']
            )

            self.log_message.emit("Backtester initialized")
            self.progress.emit(30)

            self.log_message.emit("Fetching historical data...")
            df = backtester.fetch_backtest_data(
                start_date=self.config.get('start_date'),
                end_date=self.config.get('end_date')
            )

            self.log_message.emit(f"Loaded {len(df):,} candles")
            self.progress.emit(50)

            self.log_message.emit("Running futures backtest...")
            metrics = backtester.run_backtest(df)

            self.progress.emit(90)
            self.log_message.emit("Backtest completed!")

            results = {
                'metrics': metrics,
                'backtester': backtester,
                'trades_df': metrics['trades_df'],
                'equity_df': metrics['equity_df']
            }

            self.progress.emit(100)
            self.finished.emit(results)

        except Exception as e:
            import traceback
            error_details = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.error.emit(error_details)


class FuturesBacktestingTab(QWidget):
    """Futures Backtesting Tab - Clean UI with Graphics Only"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.backtest_results = None
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("Futures Trading Backtesting")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2962FF;")
        layout.addWidget(title)

        if not BACKTEST_AVAILABLE:
            warning = QLabel(
                "Warning: BTCFuturesBacktester not found!\n"
                "Make sure backtest.py is in the testing/ folder."
            )
            warning.setStyleSheet(
                "background-color: #3d2900; color: #FFB74D; "
                "padding: 15px; border-radius: 5px; font-weight: bold;"
            )
            layout.addWidget(warning)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)

        # Sections
        scroll_layout.addWidget(self.create_date_section())
        scroll_layout.addWidget(self.create_params_section())
        scroll_layout.addLayout(self.create_actions_section())
        scroll_layout.addWidget(self.create_progress_section())
        scroll_layout.addWidget(self.create_results_section(), stretch=1)

        layout.addWidget(scroll_widget)

    def create_date_section(self):
        """Date range selection"""
        group = QGroupBox("Backtest Period")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #D1D4DC;
                font-size: 14px;
            }
        """)

        layout = QHBoxLayout()

        start_label = QLabel("Start Date:")
        start_label.setStyleSheet("color: #D1D4DC; font-size: 14px;")
        layout.addWidget(start_label)

        self.start_date = QDateEdit()
        self.start_date.setDate(QDate(2024, 1, 1))
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        self.start_date.setStyleSheet("""
            QDateEdit {
                background-color: #1E222D;
                color: #D1D4DC;
                border: 2px solid #2A2E39;
                border-radius: 4px;
                padding: 6px;
                font-size: 14px;
            }
            QDateEdit:focus {
                border: 2px solid #2962FF;
            }
            QDateEdit:hover {
                border: 2px solid #3D4758;
            }
        """)
        layout.addWidget(self.start_date)

        end_label = QLabel("End Date:")
        end_label.setStyleSheet("color: #D1D4DC; font-size: 14px;")
        layout.addWidget(end_label)

        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        self.end_date.setStyleSheet("""
            QDateEdit {
                background-color: #1E222D;
                color: #D1D4DC;
                border: 2px solid #2A2E39;
                border-radius: 4px;
                padding: 6px;
                font-size: 14px;
            }
            QDateEdit:focus {
                border: 2px solid #2962FF;
            }
            QDateEdit:hover {
                border: 2px solid #3D4758;
            }
        """)
        layout.addWidget(self.end_date)

        separator = QLabel("|")
        separator.setStyleSheet("color: #D1D4DC; font-size: 14px;")
        layout.addWidget(separator)

        for label, days in [("3M", 90), ("6M", 180), ("1Y", 365), ("All", None)]:
            btn = QPushButton(label)
            btn.setMinimumWidth(80)
            btn.setMinimumHeight(32)
            btn.clicked.connect(lambda checked, d=days: self.set_date_range(d))
            layout.addWidget(btn)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_params_section(self):
        """Futures trading parameters"""
        group = QGroupBox("Futures Parameters")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #D1D4DC;
                font-size: 14px;
            }
        """)

        grid = QGridLayout()
        grid.setSpacing(10)

        # Label styling - SAMA DENGAN TAB MODEL
        label_style = """
            QLabel {
                color: #D1D4DC;
                font-size: 14px;
            }
        """

        # Input widgets styling - SAMA DENGAN TAB MODEL
        input_style = """
            QSpinBox, QDoubleSpinBox {
                background-color: #1E222D;
                color: #D1D4DC;
                border: 2px solid #2A2E39;
                border-radius: 4px;
                padding: 6px;
                font-size: 14px;
                min-width: 80px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #2962FF;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border: 2px solid #3D4758;
            }
        """

        # Row 0: Capital & Leverage
        capital_label = QLabel("Initial Capital ($):")
        capital_label.setStyleSheet(label_style)
        grid.addWidget(capital_label, 0, 0)
        self.initial_capital = QDoubleSpinBox()
        self.initial_capital.setRange(100, 1000000)
        self.initial_capital.setValue(10000)
        self.initial_capital.setDecimals(0)
        self.initial_capital.setStyleSheet(input_style)
        grid.addWidget(self.initial_capital, 0, 1)

        leverage_label = QLabel("Leverage:")
        leverage_label.setStyleSheet(label_style)
        grid.addWidget(leverage_label, 0, 2)
        self.leverage = QSpinBox()
        self.leverage.setRange(1, 125)
        self.leverage.setValue(5)
        self.leverage.setStyleSheet(input_style)
        grid.addWidget(self.leverage, 0, 3)

        # Row 1: Position Size & Trading Fee
        position_label = QLabel("Position Size (%):")
        position_label.setStyleSheet(label_style)
        grid.addWidget(position_label, 1, 0)
        self.position_size = QDoubleSpinBox()
        self.position_size.setRange(1, 100)
        self.position_size.setValue(50)
        self.position_size.setDecimals(0)
        self.position_size.setStyleSheet(input_style)
        grid.addWidget(self.position_size, 1, 1)

        fee_label = QLabel("Trading Fee (%):")
        fee_label.setStyleSheet(label_style)
        grid.addWidget(fee_label, 1, 2)
        self.trading_fee = QDoubleSpinBox()
        self.trading_fee.setRange(0, 1)
        self.trading_fee.setValue(0.05)
        self.trading_fee.setDecimals(3)
        self.trading_fee.setStyleSheet(input_style)
        grid.addWidget(self.trading_fee, 1, 3)

        # Row 2: Stop Loss & Take Profit
        sl_label = QLabel("Stop Loss (%):")
        sl_label.setStyleSheet(label_style)
        grid.addWidget(sl_label, 2, 0)
        self.stop_loss = QDoubleSpinBox()
        self.stop_loss.setRange(1, 20)
        self.stop_loss.setValue(4)
        self.stop_loss.setDecimals(1)
        self.stop_loss.setStyleSheet(input_style)
        grid.addWidget(self.stop_loss, 2, 1)

        tp_label = QLabel("Take Profit (%):")
        tp_label.setStyleSheet(label_style)
        grid.addWidget(tp_label, 2, 2)
        self.take_profit = QDoubleSpinBox()
        self.take_profit.setRange(1, 50)
        self.take_profit.setValue(8)
        self.take_profit.setDecimals(1)
        self.take_profit.setStyleSheet(input_style)
        grid.addWidget(self.take_profit, 2, 3)

        # Row 3: Min Confidence & Max Trades/Day
        conf_label = QLabel("Min Confidence (%):")
        conf_label.setStyleSheet(label_style)
        grid.addWidget(conf_label, 3, 0)
        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setRange(0, 100)
        self.min_confidence.setValue(50)
        self.min_confidence.setDecimals(0)
        self.min_confidence.setStyleSheet(input_style)
        grid.addWidget(self.min_confidence, 3, 1)

        trades_label = QLabel("Max Trades/Day:")
        trades_label.setStyleSheet(label_style)
        grid.addWidget(trades_label, 3, 2)
        self.max_trades_per_day = QSpinBox()
        self.max_trades_per_day.setRange(1, 100)
        self.max_trades_per_day.setValue(10)
        self.max_trades_per_day.setStyleSheet(input_style)
        grid.addWidget(self.max_trades_per_day, 3, 3)

        # Row 4: Hold Time
        min_hold_label = QLabel("Min Hold Hours:")
        min_hold_label.setStyleSheet(label_style)
        grid.addWidget(min_hold_label, 4, 0)
        self.min_hold_hours = QDoubleSpinBox()
        self.min_hold_hours.setRange(0, 48)
        self.min_hold_hours.setValue(1)
        self.min_hold_hours.setDecimals(1)
        self.min_hold_hours.setStyleSheet(input_style)
        grid.addWidget(self.min_hold_hours, 4, 1)

        max_pos_label = QLabel("Max Position Hours:")
        max_pos_label.setStyleSheet(label_style)
        grid.addWidget(max_pos_label, 4, 2)
        self.max_position_hours = QDoubleSpinBox()
        self.max_position_hours.setRange(1, 168)
        self.max_position_hours.setValue(48)
        self.max_position_hours.setDecimals(1)
        self.max_position_hours.setStyleSheet(input_style)
        grid.addWidget(self.max_position_hours, 4, 3)

        # Row 5: Slippage & Funding Rate
        slip_label = QLabel("Slippage (%):")
        slip_label.setStyleSheet(label_style)
        grid.addWidget(slip_label, 5, 0)
        self.slippage = QDoubleSpinBox()
        self.slippage.setRange(0, 1)
        self.slippage.setValue(0.03)
        self.slippage.setDecimals(3)
        self.slippage.setStyleSheet(input_style)
        grid.addWidget(self.slippage, 5, 1)

        funding_label = QLabel("Funding Rate (%/8h):")
        funding_label.setStyleSheet(label_style)
        grid.addWidget(funding_label, 5, 2)
        self.funding_rate = QDoubleSpinBox()
        self.funding_rate.setRange(0, 1)
        self.funding_rate.setValue(0.01)
        self.funding_rate.setDecimals(3)
        self.funding_rate.setStyleSheet(input_style)
        grid.addWidget(self.funding_rate, 5, 3)

        # Row 6: Maintenance Margin & Trailing Stop
        margin_label = QLabel("Maintenance Margin (%):")
        margin_label.setStyleSheet(label_style)
        grid.addWidget(margin_label, 6, 0)
        self.maintenance_margin = QDoubleSpinBox()
        self.maintenance_margin.setRange(0, 10)
        self.maintenance_margin.setValue(0.5)
        self.maintenance_margin.setDecimals(2)
        self.maintenance_margin.setStyleSheet(input_style)
        grid.addWidget(self.maintenance_margin, 6, 1)

        trailing_label = QLabel("Trailing Stop:")
        trailing_label.setStyleSheet(label_style)
        grid.addWidget(trailing_label, 6, 2)
        self.use_trailing_stop = QCheckBox("Enable")
        self.use_trailing_stop.setChecked(False)
        self.use_trailing_stop.setStyleSheet("""
            QCheckBox {
                color: #D1D4DC;
                font-size: 14px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 3px;
                border: 2px solid #2A2E39;
                background-color: #1E222D;
            }
            QCheckBox::indicator:checked {
                background-color: #2962FF;
                border: 2px solid #2962FF;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #3D4758;
            }
        """)
        grid.addWidget(self.use_trailing_stop, 6, 3)

        group.setLayout(grid)
        return group

    def create_actions_section(self):
        """Action buttons"""
        layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Futures Backtest")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2962FF;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1E53E5;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888888;
            }
        """)
        self.run_btn.clicked.connect(self.run_backtest)
        layout.addWidget(self.run_btn)

        self.export_btn = QPushButton("Export Results")
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #AB47BC;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #8E24AA;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888888;
            }
        """)
        self.export_btn.clicked.connect(self.export_results)
        layout.addWidget(self.export_btn)

        layout.addStretch()
        return layout

    def create_progress_section(self):
        """Progress tracking - Compact version"""
        group = QGroupBox("Progress")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #D1D4DC;
                font-size: 14px;
            }
        """)

        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2A2E39;
                border-radius: 5px;
                text-align: center;
                background-color: #1E222D;
                height: 25px;
                color: white;
                font-weight: bold;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #2962FF, stop:1 #1E53E5);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        group.setLayout(layout)
        return group

    def create_results_section(self):
        """Results visualization"""
        splitter = QSplitter(Qt.Horizontal)
        splitter.setMinimumHeight(400)

        # Left: Metrics
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel("Performance Metrics")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2962FF;")
        metrics_layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(8)

        self.metric_labels = {}
        metrics_list = [
            ("Final Capital", "final_capital", "#26A69A"),
            ("Total Return", "total_return", "#FFB74D"),
            ("Annualized Return", "annualized_return", "#FFB74D"),
            ("Sharpe Ratio", "sharpe", "#AB47BC"),
            ("Sortino Ratio", "sortino", "#AB47BC"),
            ("Max Drawdown", "drawdown", "#EF5350"),
            ("Profit Factor", "profit_factor", "#26A69A"),
            ("Win Rate", "win_rate", "#2962FF"),
            ("Total Trades", "trades", "#D1D4DC"),
            ("Liquidations", "liquidations", "#EF5350"),
            ("Avg ROI", "avg_roi", "#26A69A"),
            ("Avg Win", "avg_win", "#26A69A"),
            ("Avg Loss", "avg_loss", "#EF5350"),
            ("Best Trade", "best_trade", "#26A69A"),
            ("Worst Trade", "worst_trade", "#EF5350"),
            ("Avg Hold", "avg_hold", "#D1D4DC"),
            ("Total Fees", "fees", "#FFB74D"),
            ("Net Funding", "funding", "#FFB74D"),
        ]

        for i, (label, key, color) in enumerate(metrics_list):
            label_widget = QLabel(f"{label}:")
            label_widget.setStyleSheet("font-size: 11px; color: #D1D4DC;")
            value_widget = QLabel("--")
            value_widget.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 12px;")
            self.metric_labels[key] = value_widget

            grid.addWidget(label_widget, i, 0, Qt.AlignLeft)
            grid.addWidget(value_widget, i, 1, Qt.AlignRight)

        metrics_layout.addLayout(grid)
        metrics_layout.addStretch()

        # Right: Chart
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        chart_title = QLabel("Equity Curve")
        chart_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2962FF;")
        chart_layout.addWidget(chart_title)

        self.equity_plot = pg.PlotWidget()
        self.equity_plot.setMinimumHeight(350)
        self.equity_plot.setBackground("#131722")
        self.equity_plot.showGrid(x=True, y=True, alpha=0.3)
        self.equity_plot.setLabel("left", "Capital ($)", color="#D1D4DC", size="11pt")
        self.equity_plot.setLabel("bottom", "Time", color="#D1D4DC", size="11pt")
        self.equity_plot.addLegend(offset=(10, 10))

        for axis in ['left', 'bottom']:
            self.equity_plot.getAxis(axis).setPen(pg.mkPen(color='#D1D4DC'))
            self.equity_plot.getAxis(axis).setTextPen(pg.mkPen(color='#D1D4DC'))

        chart_layout.addWidget(self.equity_plot)

        splitter.addWidget(metrics_widget)
        splitter.addWidget(chart_widget)
        splitter.setSizes([350, 850])

        return splitter

    def set_date_range(self, days):
        """Quick date range setter"""
        if days is None:
            self.start_date.setDate(QDate(2020, 1, 1))
        else:
            end_date = QDate.currentDate()
            start_date = end_date.addDays(-days)
            self.start_date.setDate(start_date)
        self.end_date.setDate(QDate.currentDate())

    def run_backtest(self):
        """Execute futures backtest"""
        if not BACKTEST_AVAILABLE:
            QMessageBox.critical(
                self, "Backtester Not Available",
                "BTCFuturesBacktester not found. Check testing/backtest.py"
            )
            return

        model_path = ROOT_DIR / 'models' / 'multi_input_lstm_optimized_full.pth'
        if not model_path.exists():
            QMessageBox.warning(
                self, "Model Not Found",
                f"Model file not found:\n{model_path}\n\nTrain the model first."
            )
            return

        config = {
            'model_path': str(model_path),
            'preprocessed_dir': str(ROOT_DIR / 'preprocessed_data_multi_lstm_1h'),
            'db_path': str(ROOT_DIR / 'data' / 'btc_ohlcv.db'),
            'initial_capital': self.initial_capital.value(),
            'trading_fee': self.trading_fee.value() / 100,
            'position_size': self.position_size.value() / 100,
            'leverage': self.leverage.value(),
            'stop_loss': self.stop_loss.value() / 100,
            'take_profit': self.take_profit.value() / 100,
            'min_confidence': self.min_confidence.value() / 100,
            'max_trades_per_day': self.max_trades_per_day.value(),
            'slippage': self.slippage.value() / 100,
            'min_hold_hours': self.min_hold_hours.value(),
            'max_position_hours': self.max_position_hours.value(),
            'use_trailing_stop': self.use_trailing_stop.isChecked(),
            'maintenance_margin': self.maintenance_margin.value() / 100,
            'funding_rate': self.funding_rate.value() / 100,
            'start_date': self.start_date.date().toString('yyyy-MM-dd'),
            'end_date': self.end_date.date().toString('yyyy-MM-dd'),
        }

        reply = QMessageBox.question(
            self, "Run Futures Backtest",
            f"<b>Run futures backtest?</b><br><br>"
            f"<b>Period:</b> {config['start_date']} to {config['end_date']}<br>"
            f"<b>Capital:</b> ${config['initial_capital']:,.0f}<br>"
            f"<b>Leverage:</b> {config['leverage']}x<br>"
            f"<b>Position Size:</b> {self.position_size.value():.0f}%<br>"
            f"<b>Stop Loss:</b> {self.stop_loss.value():.1f}%<br>"
            f"<b>Take Profit:</b> {self.take_profit.value():.1f}%<br><br>"
            f"<i>This may take several minutes...</i>",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Reset UI
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.equity_plot.clear()

        for key in self.metric_labels:
            self.metric_labels[key].setText("--")

        # Start worker
        self.worker = FuturesBacktestWorker(config)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.handle_error)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def handle_error(self, error_msg):
        self.run_btn.setEnabled(True)
        QMessageBox.critical(
            self, "Backtest Error",
            f"An error occurred:\n\n{error_msg[:500]}"
        )

    def display_results(self, results):
        """Display backtest results - Graphics only"""
        self.backtest_results = results
        m = results['metrics']

        # Update metrics
        self.metric_labels['final_capital'].setText(f"${m['final_capital']:,.2f}")
        self.metric_labels['total_return'].setText(f"{m['total_return']:.2f}%")
        self.metric_labels['annualized_return'].setText(f"{m['annualized_return']:.2f}%")
        self.metric_labels['sharpe'].setText(f"{m['sharpe_ratio']:.2f}")
        self.metric_labels['sortino'].setText(f"{m['sortino_ratio']:.2f}")
        self.metric_labels['drawdown'].setText(f"{m['max_drawdown']:.2f}%")
        self.metric_labels['profit_factor'].setText(f"{m['profit_factor']:.2f}")
        self.metric_labels['win_rate'].setText(f"{m['win_rate']:.1f}%")
        self.metric_labels['trades'].setText(f"{m['total_trades']:,}")
        self.metric_labels['liquidations'].setText(f"{m['liquidations']:,}")
        self.metric_labels['avg_roi'].setText(f"{m['avg_roi']:.1f}%")
        self.metric_labels['avg_win'].setText(f"${m['avg_win']:,.2f}")
        self.metric_labels['avg_loss'].setText(f"${m['avg_loss']:,.2f}")
        self.metric_labels['best_trade'].setText(f"${m['best_trade']:,.2f}")
        self.metric_labels['worst_trade'].setText(f"${m['worst_trade']:,.2f}")
        self.metric_labels['avg_hold'].setText(f"{m['avg_hold_hours']:.1f}h")
        self.metric_labels['fees'].setText(f"${m['total_fees']:,.2f}")
        self.metric_labels['funding'].setText(f"${m['net_funding']:,.2f}")

        # Color code returns
        if m['total_return'] > 0:
            self.metric_labels['total_return'].setStyleSheet("font-weight: bold; color: #26A69A; font-size: 12px;")
            self.metric_labels['annualized_return'].setStyleSheet("font-weight: bold; color: #26A69A; font-size: 12px;")
        else:
            self.metric_labels['total_return'].setStyleSheet("font-weight: bold; color: #EF5350; font-size: 12px;")
            self.metric_labels['annualized_return'].setStyleSheet("font-weight: bold; color: #EF5350; font-size: 12px;")

        # Plot equity curve
        equity_df = results['equity_df']
        if equity_df.empty:
            print("Warning: Empty equity dataframe")
            self.export_btn.setEnabled(True)
            self.run_btn.setEnabled(True)
            return

        self.equity_plot.clear()

        # Convert timestamps and create numeric x-axis
        times = pd.to_datetime(equity_df['timestamp'])
        x_data = np.arange(len(equity_df))
        y_data = equity_df['equity'].values

        # Plot equity line
        self.equity_plot.plot(
            x_data,
            y_data,
            pen=pg.mkPen("#2962FF", width=2.5),
            name="Futures Portfolio"
        )

        # Add initial capital reference line
        initial_cap = self.initial_capital.value()
        self.equity_plot.addLine(
            y=initial_cap,
            pen=pg.mkPen("#D1D4DC", width=1.5, style=Qt.DashLine)
        )

        # Set axis ranges with padding
        y_min = min(y_data.min(), initial_cap)
        y_max = max(y_data.max(), initial_cap)
        y_range = y_max - y_min
        padding = y_range * 0.1 if y_range > 0 else initial_cap * 0.1

        self.equity_plot.setYRange(y_min - padding, y_max + padding, padding=0)
        self.equity_plot.setXRange(-1, len(equity_df) + 1, padding=0)

        # Setup time axis labels
        n_labels = min(10, len(equity_df))
        if n_labels > 0:
            indices = np.linspace(0, len(equity_df) - 1, n_labels, dtype=int)
            ticks = []
            for idx in indices:
                timestamp = times.iloc[idx]
                label = timestamp.strftime("%m/%d %H:%M")
                ticks.append((idx, label))

            axis = self.equity_plot.getAxis("bottom")
            axis.setTicks([ticks])

        self.export_btn.setEnabled(True)
        self.run_btn.setEnabled(True)

        # Show summary dialog
        QMessageBox.information(
            self, "Futures Backtest Complete",
            f"<b>Backtest completed successfully!</b><br><br>"
            f"<b>Performance:</b><br>"
            f"Total Return: <b>{m['total_return']:.2f}%</b><br>"
            f"Annualized: <b>{m['annualized_return']:.2f}%</b><br>"
            f"Win Rate: <b>{m['win_rate']:.1f}%</b><br>"
            f"Profit Factor: <b>{m['profit_factor']:.2f}</b><br><br>"
            f"<b>Risk:</b><br>"
            f"Leverage: <b>{self.leverage.value()}x</b><br>"
            f"Max Drawdown: <b>{m['max_drawdown']:.2f}%</b><br>"
            f"Liquidations: <b>{m['liquidations']:,}</b><br>"
            f"Sharpe: <b>{m['sharpe_ratio']:.2f}</b><br><br>"
            f"<b>Trading:</b><br>"
            f"Total Trades: <b>{m['total_trades']:,}</b><br>"
            f"Avg ROI: <b>{m['avg_roi']:.1f}%</b><br>"
            f"Avg Hold: <b>{m['avg_hold_hours']:.1f}h</b><br>"
        )

    def export_results(self):
        """Export backtest results to CSV"""
        if self.backtest_results is None:
            QMessageBox.warning(
                self, "No Results",
                "No backtest results available. Run a backtest first."
            )
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_path = str(ROOT_DIR / f"futures_backtest_{timestamp}")

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            default_path,
            "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            # Export trades
            trades_path = file_path.replace('.csv', '_trades.csv')
            self.backtest_results['trades_df'].to_csv(trades_path, index=False)

            # Export equity curve
            equity_path = file_path.replace('.csv', '_equity.csv')
            self.backtest_results['equity_df'].to_csv(equity_path, index=False)

            # Export summary
            summary_path = file_path.replace('.csv', '_summary.csv')
            m = self.backtest_results['metrics']
            summary_data = {
                'Metric': [
                    'Initial Capital', 'Final Capital', 'Total Return (%)',
                    'Annualized Return (%)', 'Sharpe Ratio', 'Sortino Ratio',
                    'Max Drawdown (%)', 'Profit Factor', 'Win Rate (%)',
                    'Total Trades', 'Liquidations', 'Avg ROI (%)',
                    'Avg Win ($)', 'Avg Loss ($)', 'Best Trade ($)',
                    'Worst Trade ($)', 'Avg Hold (hours)', 'Total Fees ($)',
                    'Net Funding ($)', 'Long Trades', 'Short Trades',
                    'Long Win Rate (%)', 'Short Win Rate (%)',
                    'Long P&L ($)', 'Short P&L ($)'
                ],
                'Value': [
                    self.initial_capital.value(), m['final_capital'], m['total_return'],
                    m['annualized_return'], m['sharpe_ratio'], m['sortino_ratio'],
                    m['max_drawdown'], m['profit_factor'], m['win_rate'],
                    m['total_trades'], m['liquidations'], m['avg_roi'],
                    m['avg_win'], m['avg_loss'], m['best_trade'],
                    m['worst_trade'], m['avg_hold_hours'], m['total_fees'],
                    m['net_funding'], m['long_trades'], m['short_trades'],
                    m['long_win_rate'], m['short_win_rate'],
                    m['long_pnl'], m['short_pnl']
                ]
            }
            pd.DataFrame(summary_data).to_csv(summary_path, index=False)

            QMessageBox.information(
                self, "Export Complete",
                f"Results exported successfully!\n\n"
                f"Trades: {trades_path}\n"
                f"Equity: {equity_path}\n"
                f"Summary: {summary_path}\n"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export results:\n\n{str(e)}"
            )
