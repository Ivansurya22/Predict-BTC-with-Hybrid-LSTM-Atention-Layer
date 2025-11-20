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
)
from PySide6.QtCore import QDate
from datetime import datetime
import sys
from pathlib import Path

# Import backtesting class
ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

try:
    from testing.backtest import BTCBacktester
except ImportError:
    print("Warning: BTCBacktester not found. Make sure backtest.py is in testing/ folder")
    BTCBacktester = None


# ============================================================================
# BACKGROUND WORKER THREAD
# ============================================================================

class BacktestWorker(QThread):
    """Worker thread for running backtests"""
    progress = Signal(int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            # Initialize backtester
            self.progress.emit(10)

            backtester = BTCBacktester(
                model_path=self.config['model_path'],
                scaler_path=self.config['scaler_path'],
                feature_cols_path=self.config['feature_cols_path'],
                sequence_length_path=self.config['sequence_length_path'],
                db_path=self.config['db_path'],
                initial_capital=self.config['initial_capital'],
                trading_fee=self.config['trading_fee'] / 100,
                position_size=self.config['position_size'] / 100,
                stop_loss=self.config['stop_loss'] / 100,
                min_confidence=self.config['min_confidence'] / 100,
                max_capital_per_trade=self.config.get('max_capital_per_trade'),
                max_trades_per_day=self.config['max_trades_per_day'],
                slippage=self.config['slippage'] / 100,
                min_hold_hours=self.config['min_hold_hours']
            )

            self.progress.emit(30)

            # Fetch data
            df = backtester.fetch_backtest_data(
                start_date=self.config.get('start_date'),
                end_date=self.config.get('end_date')
            )

            self.progress.emit(50)

            # Run backtest
            metrics = backtester.run_backtest(df)

            self.progress.emit(90)

            # Prepare results
            results = {
                'metrics': metrics,
                'backtester': backtester,
                'trades_df': metrics['trades_df'],
                'equity_df': metrics['equity_df']
            }

            self.progress.emit(100)
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


# ============================================================================
# BACKTESTING TAB
# ============================================================================

class BacktestingTab(QWidget):
    """Enhanced Backtesting tab with real integration"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.backtest_results = None
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("📊 Backtesting & Performance Analysis")
        title.setObjectName("headerLabel")
        layout.addWidget(title)

        # Main content splitter
        main_splitter = QSplitter(Qt.Vertical)

        # Top section: Configuration
        config_widget = self.create_config_section()
        main_splitter.addWidget(config_widget)

        # Middle section: Results visualization
        results_widget = self.create_results_section()
        main_splitter.addWidget(results_widget)

        # Bottom section: Trade log
        trade_widget = self.create_trade_section()
        main_splitter.addWidget(trade_widget)

        main_splitter.setSizes([350, 400, 250])
        layout.addWidget(main_splitter)

    def create_config_section(self):
        """Create configuration section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model Configuration
        model_group = QGroupBox("📁 Model Configuration")
        model_layout = QGridLayout()

        model_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_path_label = QLabel("models/lstm_balanced_full.pth")
        self.model_path_label.setStyleSheet("color: #2962FF;")
        model_layout.addWidget(self.model_path_label, 0, 1)

        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.browse_model_btn, 0, 2)

        model_layout.addWidget(QLabel("Database:"), 1, 0)
        self.db_path_label = QLabel("data/btc_ohlcv.db")
        self.db_path_label.setStyleSheet("color: #2962FF;")
        model_layout.addWidget(self.db_path_label, 1, 1)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Date Range Configuration
        date_group = QGroupBox("📅 Date Range")
        date_layout = QGridLayout()

        date_layout.addWidget(QLabel("Start Date:"), 0, 0)
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate(2024, 1, 1))
        self.start_date.setCalendarPopup(True)
        date_layout.addWidget(self.start_date, 0, 1)

        date_layout.addWidget(QLabel("End Date:"), 0, 2)
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        date_layout.addWidget(self.end_date, 0, 3)

        date_group.setLayout(date_layout)
        layout.addWidget(date_group)

        # Trading Parameters
        params_group = QGroupBox("⚙️ Trading Parameters")
        params_layout = QGridLayout()

        # Row 0
        params_layout.addWidget(QLabel("Initial Capital ($):"), 0, 0)
        self.initial_capital = QDoubleSpinBox()
        self.initial_capital.setRange(1000, 1000000)
        self.initial_capital.setValue(10000)
        self.initial_capital.setDecimals(0)
        params_layout.addWidget(self.initial_capital, 0, 1)

        params_layout.addWidget(QLabel("Position Size (%):"), 0, 2)
        self.position_size = QDoubleSpinBox()
        self.position_size.setRange(1, 100)
        self.position_size.setValue(50)
        self.position_size.setDecimals(1)
        params_layout.addWidget(self.position_size, 0, 3)

        # Row 1
        params_layout.addWidget(QLabel("Trading Fee (%):"), 1, 0)
        self.trading_fee = QDoubleSpinBox()
        self.trading_fee.setRange(0, 10)
        self.trading_fee.setValue(0.2)
        self.trading_fee.setDecimals(3)
        params_layout.addWidget(self.trading_fee, 1, 1)

        params_layout.addWidget(QLabel("Slippage (%):"), 1, 2)
        self.slippage = QDoubleSpinBox()
        self.slippage.setRange(0, 5)
        self.slippage.setValue(0.05)
        self.slippage.setDecimals(3)
        params_layout.addWidget(self.slippage, 1, 3)

        # Row 2
        params_layout.addWidget(QLabel("Stop Loss (%):"), 2, 0)
        self.stop_loss = QDoubleSpinBox()
        self.stop_loss.setRange(0, 50)
        self.stop_loss.setValue(3)
        self.stop_loss.setDecimals(1)
        params_layout.addWidget(self.stop_loss, 2, 1)

        params_layout.addWidget(QLabel("Min Confidence (%):"), 2, 2)
        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setRange(0, 100)
        self.min_confidence.setValue(55)
        self.min_confidence.setDecimals(1)
        params_layout.addWidget(self.min_confidence, 2, 3)

        # Row 3
        params_layout.addWidget(QLabel("Max Trades/Day:"), 3, 0)
        self.max_trades_per_day = QSpinBox()
        self.max_trades_per_day.setRange(1, 100)
        self.max_trades_per_day.setValue(10)
        params_layout.addWidget(self.max_trades_per_day, 3, 1)

        params_layout.addWidget(QLabel("Min Hold Hours:"), 3, 2)
        self.min_hold_hours = QDoubleSpinBox()
        self.min_hold_hours.setRange(0, 24)
        self.min_hold_hours.setValue(4)
        self.min_hold_hours.setDecimals(1)
        params_layout.addWidget(self.min_hold_hours, 3, 3)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Action Buttons
        btn_layout = QHBoxLayout()

        self.run_backtest_btn = QPushButton("▶️ Run Backtest")
        self.run_backtest_btn.setStyleSheet("""
            QPushButton {
                background-color: #2962FF;
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1E53E5;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
            }
        """)
        self.run_backtest_btn.clicked.connect(self.run_backtest)
        btn_layout.addWidget(self.run_backtest_btn)

        self.export_btn = QPushButton("📥 Export Results")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        btn_layout.addWidget(self.export_btn)

        self.save_chart_btn = QPushButton("📊 Save Chart")
        self.save_chart_btn.setEnabled(False)
        self.save_chart_btn.clicked.connect(self.save_chart)
        btn_layout.addWidget(self.save_chart_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2A2E39;
                border-radius: 5px;
                text-align: center;
                background-color: #1E222D;
            }
            QProgressBar::chunk {
                background-color: #2962FF;
            }
        """)
        layout.addWidget(self.progress_bar)

        return widget

    def create_results_section(self):
        """Create results visualization section"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left: Metrics
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)

        metrics_title = QLabel("📈 Performance Metrics")
        metrics_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        metrics_layout.addWidget(metrics_title)

        self.metrics_grid = QGridLayout()

        # Create metric labels
        self.metric_labels = {}
        metrics = [
            ("Final Capital", "final_capital", "$"),
            ("Total Return", "total_return", "%"),
            ("Annualized Return", "annualized_return", "%"),
            ("Sharpe Ratio", "sharpe", ""),
            ("Max Drawdown", "drawdown", "%"),
            ("Win Rate", "win_rate", "%"),
            ("Profit Factor", "profit_factor", ""),
            ("Total Trades", "trades", ""),
            ("Avg Trades/Day", "avg_trades_day", ""),
            ("Avg Hold Time", "avg_hold", "h"),
        ]

        for i, (label, key, unit) in enumerate(metrics):
            label_widget = QLabel(f"{label}:")
            value_widget = QLabel("--")
            value_widget.setStyleSheet("font-weight: bold; color: #2962FF;")
            self.metric_labels[key] = (value_widget, unit)

            self.metrics_grid.addWidget(label_widget, i, 0)
            self.metrics_grid.addWidget(value_widget, i, 1)

        metrics_layout.addLayout(self.metrics_grid)
        metrics_layout.addStretch()

        # Right: Equity Curve Chart
        self.equity_plot = pg.PlotWidget()
        self.equity_plot.setBackground("#131722")
        self.equity_plot.showGrid(x=True, y=True, alpha=0.3)
        self.equity_plot.setLabel("left", "Portfolio Value ($)", color="#D1D4DC")
        self.equity_plot.setLabel("bottom", "Time", color="#D1D4DC")
        self.equity_plot.getAxis('left').setPen(pg.mkPen(color='#D1D4DC'))
        self.equity_plot.getAxis('bottom').setPen(pg.mkPen(color='#D1D4DC'))

        layout.addWidget(metrics_widget, stretch=1)
        layout.addWidget(self.equity_plot, stretch=2)

        return widget

    def create_trade_section(self):
        """Create trade log section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        trade_header = QHBoxLayout()
        trade_title = QLabel("📋 Trade Log")
        trade_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        trade_header.addWidget(trade_title)

        self.trade_count_label = QLabel("Total: 0 trades")
        trade_header.addWidget(self.trade_count_label)
        trade_header.addStretch()

        layout.addLayout(trade_header)

        self.trade_table = QTableWidget()
        self.trade_table.setColumnCount(9)
        self.trade_table.setHorizontalHeaderLabels(
            ["Entry Time", "Exit Time", "Position", "Entry Price", "Exit Price", "P&L $", "P&L %", "Hold (h)", "Reason"]
        )

        # Stretch columns appropriately
        header = self.trade_table.horizontalHeader()
        for i in range(9):
            if i in [0, 1]:  # Timestamps
                header.resizeSection(i, 150)
            elif i == 2:  # Position
                header.resizeSection(i, 70)
            elif i in [3, 4]:  # Prices
                header.resizeSection(i, 100)
            elif i in [5, 6]:  # P&L
                header.resizeSection(i, 90)
            elif i == 7:  # Hold time
                header.resizeSection(i, 70)
            else:  # Reason
                header.resizeSection(i, 100)

        # Style the table
        self.trade_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E222D;
                color: #D1D4DC;
                gridline-color: #2A2E39;
                border: 1px solid #2A2E39;
            }
            QTableWidget::item {
                background-color: #1E222D;
                color: #D1D4DC;
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #2962FF;
                color: white;
            }
            QHeaderView::section {
                background-color: #131722;
                color: #D1D4DC;
                padding: 5px;
                border: 1px solid #2A2E39;
                font-weight: bold;
            }
        """)

        layout.addWidget(self.trade_table)
        return widget

    def browse_model(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", str(ROOT_DIR / "models"),
            "Model Files (*.pth);;All Files (*)"
        )
        if file_path:
            self.model_path_label.setText(file_path)

    def run_backtest(self):
        """Run backtest simulation"""
        if BTCBacktester is None:
            QMessageBox.warning(self, "Error", "BTCBacktester not available. Check your installation.")
            return

        # Validate paths
        model_path = self.model_path_label.text()
        if not Path(model_path).exists():
            QMessageBox.warning(self, "Error", f"Model file not found: {model_path}")
            return

        # Prepare configuration
        config = {
            'model_path': model_path,
            'scaler_path': 'preprocessed_data_lstm_1h/scalers.pkl',
            'feature_cols_path': 'preprocessed_data_lstm_1h/feature_cols.pkl',
            'sequence_length_path': 'preprocessed_data_lstm_1h/sequence_length.pkl',
            'db_path': self.db_path_label.text(),
            'initial_capital': self.initial_capital.value(),
            'trading_fee': self.trading_fee.value(),
            'position_size': self.position_size.value(),
            'stop_loss': self.stop_loss.value(),
            'min_confidence': self.min_confidence.value(),
            'max_trades_per_day': self.max_trades_per_day.value(),
            'slippage': self.slippage.value(),
            'min_hold_hours': self.min_hold_hours.value(),
            'start_date': self.start_date.date().toString('yyyy-MM-dd'),
            'end_date': self.end_date.date().toString('yyyy-MM-dd'),
        }

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.run_backtest_btn.setEnabled(False)

        # Create and start worker thread
        self.worker = BacktestWorker(config)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.handle_error)
        self.worker.start()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def handle_error(self, error_msg):
        """Handle backtest error"""
        self.progress_bar.setVisible(False)
        self.run_backtest_btn.setEnabled(True)
        QMessageBox.critical(self, "Backtest Error", f"An error occurred:\n\n{error_msg}")

    def display_results(self, results):
        """Display backtest results"""
        self.progress_bar.setValue(100)
        self.backtest_results = results
        metrics = results['metrics']

        # Update metrics
        self.metric_labels['final_capital'][0].setText(f"${metrics['final_capital']:,.2f}")
        self.metric_labels['total_return'][0].setText(f"{metrics['total_return']:.2f}%")
        self.metric_labels['annualized_return'][0].setText(f"{metrics['annualized_return']:.2f}%")
        self.metric_labels['sharpe'][0].setText(f"{metrics['sharpe_ratio']:.2f}")
        self.metric_labels['drawdown'][0].setText(f"{metrics['max_drawdown']:.2f}%")
        self.metric_labels['win_rate'][0].setText(f"{metrics['win_rate']:.1f}%")
        self.metric_labels['profit_factor'][0].setText(f"{metrics['profit_factor']:.2f}")
        self.metric_labels['trades'][0].setText(f"{metrics['total_trades']:,}")
        self.metric_labels['avg_trades_day'][0].setText(f"{metrics['avg_trades_per_day']:.1f}")
        self.metric_labels['avg_hold'][0].setText(f"{metrics['avg_hold_hours']:.1f}")

        # Color code returns
        if metrics['total_return'] > 0:
            self.metric_labels['total_return'][0].setStyleSheet("font-weight: bold; color: #26A69A;")
            self.metric_labels['annualized_return'][0].setStyleSheet("font-weight: bold; color: #26A69A;")
        else:
            self.metric_labels['total_return'][0].setStyleSheet("font-weight: bold; color: #EF5350;")
            self.metric_labels['annualized_return'][0].setStyleSheet("font-weight: bold; color: #EF5350;")

        # Plot equity curve
        equity_df = results['equity_df']
        self.equity_plot.clear()

        # Convert timestamps to numeric for plotting
        times = pd.to_datetime(equity_df['timestamp'])
        time_numeric = (times - times.min()).dt.total_seconds() / 3600  # Hours since start

        self.equity_plot.plot(
            time_numeric,
            equity_df['equity'].values,
            pen=pg.mkPen("#26A69A", width=2)
        )

        # Add initial capital line
        self.equity_plot.addLine(
            y=self.initial_capital.value(),
            pen=pg.mkPen("#D1D4DC", width=1, style=Qt.DashLine)
        )

        # Display trades in table
        trades_df = results['trades_df']
        self.trade_table.setRowCount(len(trades_df))
        self.trade_count_label.setText(f"Total: {len(trades_df)} trades")

        for row, trade in trades_df.iterrows():
            items = [
                str(trade['entry_time']),
                str(trade['exit_time']),
                trade['position'],
                f"${trade['entry_price']:,.2f}",
                f"${trade['exit_price']:,.2f}",
                f"${trade['pnl_usd']:,.2f}",
                f"{trade['pnl_pct']:.2f}%",
                f"{trade['hold_hours']:.1f}",
                trade['reason']
            ]

            for col, value in enumerate(items):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QColor("#1E222D"))

                # Color P&L columns
                if col in [5, 6]:
                    if trade['pnl_usd'] > 0:
                        item.setForeground(QColor("#26A69A"))
                    else:
                        item.setForeground(QColor("#EF5350"))
                else:
                    item.setForeground(QColor("#D1D4DC"))

                self.trade_table.setItem(row, col, item)

        # Enable export buttons
        self.export_btn.setEnabled(True)
        self.save_chart_btn.setEnabled(True)

        QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
        self.run_backtest_btn.setEnabled(True)

        # Show summary message
        QMessageBox.information(
            self,
            "Backtest Complete",
            f"Backtest completed successfully!\n\n"
            f"Total Return: {metrics['total_return']:.2f}%\n"
            f"Win Rate: {metrics['win_rate']:.1f}%\n"
            f"Total Trades: {metrics['total_trades']:,}"
        )

    def export_results(self):
        """Export results to CSV"""
        if self.backtest_results is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "backtest_results.csv", "CSV Files (*.csv)"
        )

        if file_path:
            trades_df = self.backtest_results['trades_df']
            trades_df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")

    def save_chart(self):
        """Save equity curve chart"""
        if self.backtest_results is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chart", "equity_curve.png", "PNG Files (*.png)"
        )

        if file_path:
            exporter = pg.exporters.ImageExporter(self.equity_plot.plotItem)
            exporter.parameters()['width'] = 1920
            exporter.export(file_path)
            QMessageBox.information(self, "Export Complete", f"Chart saved to:\n{file_path}")
