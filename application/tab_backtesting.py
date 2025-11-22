import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QSpinBox,
    QDateEdit,
    QMessageBox,
    QFileDialog,
    QTextEdit,
)
from PySide6.QtCore import QDate
from datetime import datetime
import sys
from pathlib import Path

# Import backtesting class
if '__file__' in globals():
    ROOT_DIR = Path(__file__).resolve().parent.parent
else:
    ROOT_DIR = Path.cwd()

sys.path.insert(0, str(ROOT_DIR))

try:
    from testing.backtest import BTCBacktester
    BACKTEST_AVAILABLE = True
    print(f"✓ BTCBacktester imported successfully")
except ImportError as e:
    print(f"✗ Warning: Could not import BTCBacktester: {e}")
    BACKTEST_AVAILABLE = False
    BTCBacktester = None


# ============================================================================
# BACKGROUND WORKER THREAD
# ============================================================================
class BacktestWorker(QThread):
    """Worker thread for running backtests"""
    progress = Signal(int)
    log_message = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self.log_message.emit("🔧 Initializing backtester...")
            self.progress.emit(10)

            # Initialize backtester
            backtester = BTCBacktester(
                model_path=self.config['model_path'],
                scaler_path=self.config['scaler_path'],
                feature_cols_path=self.config['feature_cols_path'],
                sequence_length_path=self.config['sequence_length_path'],
                db_path=self.config['db_path'],
                initial_capital=self.config['initial_capital'],
                trading_fee=self.config['trading_fee'],
                position_size=self.config['position_size'],
                stop_loss=self.config['stop_loss'],
                min_confidence=self.config['min_confidence'],
                max_capital_per_trade=self.config.get('max_capital_per_trade'),
                max_trades_per_day=self.config['max_trades_per_day'],
                slippage=self.config['slippage'],
                min_hold_hours=self.config['min_hold_hours']
            )

            self.log_message.emit("✓ Backtester initialized")
            self.progress.emit(30)

            # Fetch data
            self.log_message.emit("📥 Fetching historical data...")
            df = backtester.fetch_backtest_data(
                start_date=self.config.get('start_date'),
                end_date=self.config.get('end_date')
            )

            self.log_message.emit(f"✓ Loaded {len(df):,} candles")
            self.progress.emit(50)

            # Run backtest
            self.log_message.emit("🚀 Running backtest simulation...")
            metrics = backtester.run_backtest(df)

            self.progress.emit(90)

            # Prepare results
            self.log_message.emit("✓ Backtest completed!")
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


# ============================================================================
# BACKTESTING TAB
# ============================================================================
class BacktestingTab(QWidget):
    """Enhanced Backtesting tab aligned with backtest.py"""

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
        title = QLabel("📊 Backtesting & Performance Analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2962FF;")
        layout.addWidget(title)

        # Check if backtester is available
        if not BACKTEST_AVAILABLE:
            warning = QLabel(
                "⚠️ Warning: BTCBacktester not found!\n"
                "Make sure backtest.py is in the testing/ folder."
            )
            warning.setStyleSheet(
                "background-color: #3d2900; color: #FFB74D; "
                "padding: 15px; border-radius: 5px; font-weight: bold;"
            )
            layout.addWidget(warning)

        # Create scroll area
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)

        # Date Range Section
        date_group = self.create_date_section()
        scroll_layout.addWidget(date_group)

        # Trading Parameters Section
        params_group = self.create_params_section()
        scroll_layout.addWidget(params_group)

        # Action Buttons
        actions_layout = self.create_actions_section()
        scroll_layout.addLayout(actions_layout)

        # Progress Section
        progress_group = self.create_progress_section()
        scroll_layout.addWidget(progress_group)

        # Results Section
        results_splitter = self.create_results_section()
        scroll_layout.addWidget(results_splitter, stretch=1)

        layout.addWidget(scroll_widget)

    def create_date_section(self):
        """Create date range section"""
        group = QGroupBox("📅 Backtest Period")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QHBoxLayout()

        layout.addWidget(QLabel("Start Date:"))
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate(2024, 1, 1))
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        layout.addWidget(self.start_date)

        layout.addWidget(QLabel("End Date:"))
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        layout.addWidget(self.end_date)

        # Quick select buttons
        layout.addWidget(QLabel("|"))

        quick_btns = [
            ("Last 3M", 90),
            ("Last 6M", 180),
            ("Last 1Y", 365),
            ("All", None)
        ]

        for label, days in quick_btns:
            btn = QPushButton(label)
            btn.setMaximumWidth(70)
            btn.clicked.connect(lambda checked, d=days: self.set_date_range(d))
            layout.addWidget(btn)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_params_section(self):
        """Create trading parameters section"""
        group = QGroupBox("⚙️ Trading Parameters")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        grid = QGridLayout()
        grid.setSpacing(10)

        # Row 0: Capital and Position Size
        grid.addWidget(QLabel("Initial Capital ($):"), 0, 0)
        self.initial_capital = QDoubleSpinBox()
        self.initial_capital.setRange(1000, 1000000)
        self.initial_capital.setValue(10000)
        self.initial_capital.setDecimals(0)
        self.initial_capital.setSingleStep(1000)
        grid.addWidget(self.initial_capital, 0, 1)

        grid.addWidget(QLabel("Position Size (%):"), 0, 2)
        self.position_size = QDoubleSpinBox()
        self.position_size.setRange(1, 100)
        self.position_size.setValue(95)
        self.position_size.setDecimals(0)
        self.position_size.setSingleStep(5)
        grid.addWidget(self.position_size, 0, 3)

        # Row 1: Trading Fee and Slippage
        grid.addWidget(QLabel("Trading Fee (%):"), 1, 0)
        self.trading_fee = QDoubleSpinBox()
        self.trading_fee.setRange(0, 1)
        self.trading_fee.setValue(0.1)
        self.trading_fee.setDecimals(3)
        self.trading_fee.setSingleStep(0.01)
        grid.addWidget(self.trading_fee, 1, 1)

        grid.addWidget(QLabel("Slippage (%):"), 1, 2)
        self.slippage = QDoubleSpinBox()
        self.slippage.setRange(0, 1)
        self.slippage.setValue(0.05)
        self.slippage.setDecimals(3)
        self.slippage.setSingleStep(0.01)
        grid.addWidget(self.slippage, 1, 3)

        # Row 2: Stop Loss and Min Confidence
        grid.addWidget(QLabel("Stop Loss (%):"), 2, 0)
        self.stop_loss = QDoubleSpinBox()
        self.stop_loss.setRange(1, 20)
        self.stop_loss.setValue(5)
        self.stop_loss.setDecimals(1)
        self.stop_loss.setSingleStep(0.5)
        grid.addWidget(self.stop_loss, 2, 1)

        grid.addWidget(QLabel("Min Confidence (%):"), 2, 2)
        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setRange(0, 100)
        self.min_confidence.setValue(45)
        self.min_confidence.setDecimals(0)
        self.min_confidence.setSingleStep(5)
        grid.addWidget(self.min_confidence, 2, 3)

        # Row 3: Max Trades/Day and Min Hold Hours
        grid.addWidget(QLabel("Max Trades/Day:"), 3, 0)
        self.max_trades_per_day = QSpinBox()
        self.max_trades_per_day.setRange(1, 100)
        self.max_trades_per_day.setValue(10)
        grid.addWidget(self.max_trades_per_day, 3, 1)

        grid.addWidget(QLabel("Min Hold Hours:"), 3, 2)
        self.min_hold_hours = QDoubleSpinBox()
        self.min_hold_hours.setRange(0, 24)
        self.min_hold_hours.setValue(4)
        self.min_hold_hours.setDecimals(1)
        self.min_hold_hours.setSingleStep(0.5)
        grid.addWidget(self.min_hold_hours, 3, 3)

        group.setLayout(grid)
        return group

    def create_actions_section(self):
        """Create action buttons section"""
        layout = QHBoxLayout()

        self.run_backtest_btn = QPushButton("▶️ Run Backtest")
        self.run_backtest_btn.setStyleSheet("""
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
        self.run_backtest_btn.clicked.connect(self.run_backtest)
        layout.addWidget(self.run_backtest_btn)

        self.export_trades_btn = QPushButton("💾 Export Trades")
        self.export_trades_btn.setEnabled(False)
        self.export_trades_btn.setStyleSheet("""
            QPushButton {
                background-color: #26A69A;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00897B;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888888;
            }
        """)
        self.export_trades_btn.clicked.connect(self.export_trades)
        layout.addWidget(self.export_trades_btn)

        self.save_chart_btn = QPushButton("📊 Save Chart")
        self.save_chart_btn.setEnabled(False)
        self.save_chart_btn.setStyleSheet("""
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
        self.save_chart_btn.clicked.connect(self.save_chart)
        layout.addWidget(self.save_chart_btn)

        layout.addStretch()
        return layout

    def create_progress_section(self):
        """Create progress section"""
        group = QGroupBox("📊 Backtest Progress")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QVBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2A2E39;
                border-radius: 5px;
                text-align: center;
                background-color: #1E222D;
                height: 30px;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #2962FF, stop:1 #1E53E5);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E222D;
                color: #D1D4DC;
                border: 2px solid #2A2E39;
                border-radius: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                padding: 10px;
            }
        """)
        self.log_text.setPlainText(
            "Ready to backtest.\n\n"
            "Configure parameters above and click 'Run Backtest'.\n"
            "The system will:\n"
            "1. Load trained model\n"
            "2. Fetch historical data\n"
            "3. Simulate trading with realistic constraints\n"
            "4. Generate performance report"
        )
        layout.addWidget(self.log_text)

        group.setLayout(layout)
        return group

    def create_results_section(self):
        """Create results visualization section"""
        splitter = QSplitter(Qt.Horizontal)
        splitter.setMinimumHeight(400)

        # Left: Metrics
        metrics_container = QWidget()
        metrics_layout = QVBoxLayout(metrics_container)
        metrics_layout.setContentsMargins(5, 5, 5, 5)

        metrics_title = QLabel("📈 Performance Metrics")
        metrics_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2962FF;")
        metrics_layout.addWidget(metrics_title)

        # Metrics grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(8)

        self.metric_labels = {}
        metrics_list = [
            ("Final Capital", "final_capital", "#26A69A"),
            ("Total Return", "total_return", "#2962FF"),
            ("Annualized Return", "annualized_return", "#2962FF"),
            ("Expected Live (~30%)", "expected_live", "#FFB74D"),
            ("Sharpe Ratio", "sharpe", "#AB47BC"),
            ("Max Drawdown", "drawdown", "#EF5350"),
            ("Profit Factor", "profit_factor", "#26A69A"),
            ("Win Rate", "win_rate", "#2962FF"),
            ("Total Trades", "trades", "#D1D4DC"),
            ("Avg Trades/Day", "avg_trades_day", "#D1D4DC"),
            ("Avg Win", "avg_win", "#26A69A"),
            ("Avg Loss", "avg_loss", "#EF5350"),
            ("Best Trade", "best_trade", "#26A69A"),
            ("Worst Trade", "worst_trade", "#EF5350"),
            ("Avg Hold Time", "avg_hold", "#D1D4DC"),
        ]

        for i, (label, key, color) in enumerate(metrics_list):
            label_widget = QLabel(f"{label}:")
            label_widget.setStyleSheet("font-size: 11px; color: #D1D4DC;")
            value_widget = QLabel("--")
            value_widget.setStyleSheet(f"font-weight: bold; color: {color}; font-size: 12px;")
            self.metric_labels[key] = value_widget

            metrics_grid.addWidget(label_widget, i, 0, Qt.AlignLeft)
            metrics_grid.addWidget(value_widget, i, 1, Qt.AlignRight)

        metrics_layout.addLayout(metrics_grid)
        metrics_layout.addStretch()

        # Right: Charts
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        chart_title = QLabel("📉 Equity Curve")
        chart_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2962FF;")
        chart_layout.addWidget(chart_title)

        # Equity plot
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

        splitter.addWidget(metrics_container)
        splitter.addWidget(chart_container)
        splitter.setSizes([350, 850])

        return splitter

    def set_date_range(self, days):
        """Quick set date range"""
        if days is None:
            # Set to all available data
            self.start_date.setDate(QDate(2020, 1, 1))
        else:
            end_date = QDate.currentDate()
            start_date = end_date.addDays(-days)
            self.start_date.setDate(start_date)

        self.end_date.setDate(QDate.currentDate())

    def run_backtest(self):
        """Run backtest simulation"""
        if not BACKTEST_AVAILABLE:
            QMessageBox.critical(
                self,
                "Backtester Not Available",
                "BTCBacktester not found. Make sure backtest.py is in testing/ folder."
            )
            return

        # Check if model exists
        model_path = ROOT_DIR / 'models' / 'lstm_balanced_full.pth'
        if not model_path.exists():
            QMessageBox.warning(
                self,
                "Model Not Found",
                f"Model file not found:\n{model_path}\n\n"
                "Please train the model first from the Model tab."
            )
            return

        # Prepare configuration
        config = {
            'model_path': str(model_path),
            'scaler_path': str(ROOT_DIR / 'preprocessed_data_lstm_1h' / 'scalers.pkl'),
            'feature_cols_path': str(ROOT_DIR / 'preprocessed_data_lstm_1h' / 'feature_cols.pkl'),
            'sequence_length_path': str(ROOT_DIR / 'preprocessed_data_lstm_1h' / 'sequence_length.pkl'),
            'db_path': str(ROOT_DIR / 'data' / 'btc_ohlcv.db'),
            'initial_capital': self.initial_capital.value(),
            'trading_fee': self.trading_fee.value() / 100,
            'position_size': self.position_size.value() / 100,
            'stop_loss': self.stop_loss.value() / 100,
            'min_confidence': self.min_confidence.value() / 100,
            'max_trades_per_day': self.max_trades_per_day.value(),
            'slippage': self.slippage.value() / 100,
            'min_hold_hours': self.min_hold_hours.value(),
            'start_date': self.start_date.date().toString('yyyy-MM-dd'),
            'end_date': self.end_date.date().toString('yyyy-MM-dd'),
        }

        # Confirm
        reply = QMessageBox.question(
            self,
            "Run Backtest",
            f"<b>Run backtest with the following settings?</b><br><br>"
            f"<b>Period:</b> {config['start_date']} to {config['end_date']}<br>"
            f"<b>Capital:</b> ${config['initial_capital']:,.0f}<br>"
            f"<b>Position Size:</b> {self.position_size.value():.0f}%<br>"
            f"<b>Min Confidence:</b> {self.min_confidence.value():.0f}%<br>"
            f"<b>Max Trades/Day:</b> {config['max_trades_per_day']}<br><br>"
            f"<i>This may take a few minutes...</i>",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Reset UI
        self.progress_bar.setValue(0)
        self.run_backtest_btn.setEnabled(False)
        self.export_trades_btn.setEnabled(False)
        self.save_chart_btn.setEnabled(False)
        self.log_text.clear()
        self.equity_plot.clear()

        # Reset metrics
        for key in self.metric_labels:
            self.metric_labels[key].setText("--")

        # Start worker
        self.worker = BacktestWorker(config)
        self.worker.progress.connect(self.update_progress)
        self.worker.log_message.connect(self.append_log)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.handle_error)
        self.worker.start()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def append_log(self, message):
        """Append log message"""
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def handle_error(self, error_msg):
        """Handle backtest error"""
        self.run_backtest_btn.setEnabled(True)
        self.append_log(f"\n❌ ERROR:\n{error_msg}")

        QMessageBox.critical(
            self,
            "Backtest Error",
            f"An error occurred during backtesting:\n\n{error_msg[:500]}\n\n"
            "Check the log for details."
        )

    def display_results(self, results):
        """Display backtest results"""
        self.backtest_results = results
        metrics = results['metrics']

        # Update metrics
        self.metric_labels['final_capital'].setText(f"${metrics['final_capital']:,.2f}")
        self.metric_labels['total_return'].setText(f"{metrics['total_return']:.2f}%")
        self.metric_labels['annualized_return'].setText(f"{metrics['annualized_return']:.2f}%")
        self.metric_labels['expected_live'].setText(f"{metrics['annualized_return']*0.3:.1f}%")
        self.metric_labels['sharpe'].setText(f"{metrics['sharpe_ratio']:.2f}")
        self.metric_labels['drawdown'].setText(f"{metrics['max_drawdown']:.2f}%")
        self.metric_labels['profit_factor'].setText(f"{metrics['profit_factor']:.2f}")
        self.metric_labels['win_rate'].setText(f"{metrics['win_rate']:.1f}%")
        self.metric_labels['trades'].setText(f"{metrics['total_trades']:,}")
        self.metric_labels['avg_trades_day'].setText(f"{metrics['avg_trades_per_day']:.1f}")
        self.metric_labels['avg_win'].setText(f"${metrics['avg_win']:,.2f}")
        self.metric_labels['avg_loss'].setText(f"${metrics['avg_loss']:,.2f}")
        self.metric_labels['best_trade'].setText(f"${metrics['best_trade']:,.2f}")
        self.metric_labels['worst_trade'].setText(f"${metrics['worst_trade']:,.2f}")
        self.metric_labels['avg_hold'].setText(f"{metrics['avg_hold_hours']:.1f}h")

        # Color code performance
        if metrics['total_return'] > 0:
            self.metric_labels['total_return'].setStyleSheet("font-weight: bold; color: #26A69A; font-size: 12px;")
            self.metric_labels['annualized_return'].setStyleSheet("font-weight: bold; color: #26A69A; font-size: 12px;")
        else:
            self.metric_labels['total_return'].setStyleSheet("font-weight: bold; color: #EF5350; font-size: 12px;")
            self.metric_labels['annualized_return'].setStyleSheet("font-weight: bold; color: #EF5350; font-size: 12px;")

        # Plot equity curve
        equity_df = results['equity_df']
        self.equity_plot.clear()

        times = pd.to_datetime(equity_df['timestamp'])
        time_numeric = (times - times.min()).dt.total_seconds() / 3600

        # Plot equity
        self.equity_plot.plot(
            time_numeric,
            equity_df['equity'].values,
            pen=pg.mkPen("#26A69A", width=2.5),
            name="Portfolio"
        )

        # Add initial capital line
        self.equity_plot.addLine(
            y=self.initial_capital.value(),
            pen=pg.mkPen("#D1D4DC", width=1.5, style=Qt.DashLine),
            label="Initial"
        )

        # Log summary
        self.append_log(f"\n{'='*50}")
        self.append_log(f"✅ BACKTEST COMPLETED")
        self.append_log(f"{'='*50}")
        self.append_log(f"\n💰 RESULTS:")
        self.append_log(f"   Final Capital:    ${metrics['final_capital']:,.2f}")
        self.append_log(f"   Total Return:     {metrics['total_return']:.2f}%")
        self.append_log(f"   Annualized:       {metrics['annualized_return']:.2f}%")
        self.append_log(f"   Expected Live:    ~{metrics['annualized_return']*0.3:.1f}%")
        self.append_log(f"\n📊 PERFORMANCE:")
        self.append_log(f"   Win Rate:         {metrics['win_rate']:.1f}%")
        self.append_log(f"   Profit Factor:    {metrics['profit_factor']:.2f}")
        self.append_log(f"   Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        self.append_log(f"   Max Drawdown:     {metrics['max_drawdown']:.2f}%")
        self.append_log(f"\n🎯 TRADING STATS:")
        self.append_log(f"   Total Trades:     {metrics['total_trades']:,}")
        self.append_log(f"   Trades/Day:       {metrics['avg_trades_per_day']:.1f}")
        self.append_log(f"   Avg Hold:         {metrics['avg_hold_hours']:.1f}h")
        self.append_log(f"   Long Trades:      {metrics['long_trades']:,} ({metrics['long_win_rate']:.1f}% win)")
        self.append_log(f"   Short Trades:     {metrics['short_trades']:,} ({metrics['short_win_rate']:.1f}% win)")
        self.append_log(f"\n🚫 SKIPPED TRADES:")
        self.append_log(f"   Low Confidence:   {metrics['skipped_trades']['confidence']:,}")
        self.append_log(f"   Daily Limit:      {metrics['skipped_trades']['daily_limit']:,}")

        # Performance rating
        if metrics['annualized_return'] > 100:
            self.append_log(f"\n🔥 Rating: EXCEPTIONAL (unrealistic for live)")
        elif metrics['annualized_return'] > 50:
            self.append_log(f"\n✅ Rating: EXCELLENT")
        elif metrics['annualized_return'] > 20:
            self.append_log(f"\n👍 Rating: GOOD")
        elif metrics['annualized_return'] > 0:
            self.append_log(f"\n⚠️  Rating: MODEST")
        else:
            self.append_log(f"\n❌ Rating: LOSS")

        self.append_log(f"\n{'='*50}")
        self.append_log(f"⚠️  Remember: Expect 30-50% of backtest returns in live trading!")

        # Enable export buttons
        self.export_trades_btn.setEnabled(True)
        self.save_chart_btn.setEnabled(True)
        self.run_backtest_btn.setEnabled(True)

        # Show completion dialog
        QMessageBox.information(
            self,
            "✅ Backtest Complete",
            f"<b>Backtest completed successfully!</b><br><br>"
            f"<b>Performance Summary:</b><br>"
            f"• Total Return: <b>{metrics['total_return']:.2f}%</b><br>"
            f"• Annualized: <b>{metrics['annualized_return']:.2f}%</b><br>"
            f"• Win Rate: <b>{metrics['win_rate']:.1f}%</b><br>"
            f"• Profit Factor: <b>{metrics['profit_factor']:.2f}</b><br>"
            f"• Total Trades: <b>{metrics['total_trades']:,}</b><br><br>"
            f"<i>Check the charts and metrics for detailed analysis.</i>"
        )

    def export_trades(self):
        """Export trades to CSV"""
        if self.backtest_results is None:
            QMessageBox.warning(
                self,
                "No Results",
                "No backtest results available to export.\n"
                "Run a backtest first."
            )
            return

        # Get save path
        default_path = str(ROOT_DIR / f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Trades",
            default_path,
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            trades_df = self.backtest_results['trades_df']
            trades_df.to_csv(file_path, index=False)

            QMessageBox.information(
                self,
                "Export Complete",
                f"✅ Trades exported successfully!\n\n"
                f"File: {file_path}\n"
                f"Trades: {len(trades_df):,}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export trades:\n\n{str(e)}"
            )

    def save_chart(self):
        """Save equity curve chart"""
        if self.backtest_results is None:
            QMessageBox.warning(
                self,
                "No Results",
                "No backtest results available.\n"
                "Run a backtest first."
            )
            return

        # Get save path
        default_path = str(ROOT_DIR / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chart",
            default_path,
            "PNG Files (*.png);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Use matplotlib to create detailed chart
            backtester = self.backtest_results['backtester']
            metrics = self.backtest_results['metrics']

            backtester.plot_results(metrics, save_path=file_path)

            QMessageBox.information(
                self,
                "Export Complete",
                f"✅ Chart saved successfully!\n\n{file_path}"
            )
        except Exception as e:
            # Fallback: save pyqtgraph plot
            try:
                exporter = pg.exporters.ImageExporter(self.equity_plot.plotItem)
                exporter.parameters()['width'] = 1920
                exporter.export(file_path)

                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"✅ Chart saved successfully!\n\n{file_path}"
                )
            except Exception as e2:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to save chart:\n\n{str(e)}\n\nFallback also failed:\n{str(e2)}"
                )
