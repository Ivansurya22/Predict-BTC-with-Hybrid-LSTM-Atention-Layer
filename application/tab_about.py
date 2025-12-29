from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QGroupBox, QHBoxLayout, QFrame, QScrollArea
)


class AboutTab(QWidget):
    """About & Telegram Bot Information Tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup UI components"""
        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
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
        """)

        # Content widget
        content = QWidget()
        content.setStyleSheet("""
            QWidget {
                background-color: #131722;
            }
        """)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        # Header
        header = QLabel("BTC Trading Dashboard")
        header.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 28px;
                font-weight: bold;
                padding: 15px;
            }
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Subtitle
        subtitle = QLabel("Smart Money Concepts & AI Prediction System")
        subtitle.setStyleSheet("""
            QLabel {
                color: #2962FF;
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
                padding-bottom: 15px;
            }
        """)
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # Telegram Bot Section
        telegram_group = self.create_telegram_section()
        layout.addWidget(telegram_group)

        # Features Section
        features_group = self.create_features_section()
        layout.addWidget(features_group)

        # Developer Info Section
        developer_group = self.create_developer_section()
        layout.addWidget(developer_group)

        # Add stretch at bottom
        layout.addStretch()

        scroll.setWidget(content)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def create_telegram_section(self):
        """Create Telegram bot information section"""
        group = QGroupBox("Telegram Alert Bot")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #D1D4DC;
                border: 2px solid #2962FF;
                border-radius: 10px;
                margin-top: 12px;
                padding: 20px;
                background-color: #1E222D;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 20px;
                top: 5px;
                padding: 0 10px;
                color: #D1D4DC;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 30, 20, 20)

        # Description
        description = QLabel(
            "Get real-time trading alerts directly to your Telegram!\n"
            "Receive instant notifications for price movements, trading signals,\n"
            "and AI predictions right on your phone."
        )
        description.setStyleSheet("""
            QLabel {
                color: #D1D4DC;
                font-size: 13px;
                padding: 10px;
            }
        """)
        description.setAlignment(Qt.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)

        # Telegram Button Container
        button_container = QHBoxLayout()
        button_container.addStretch()

        # Telegram Button
        telegram_btn = QPushButton("Open Telegram Bot")
        telegram_btn.setStyleSheet("""
            QPushButton {
                background-color: #0088cc;
                color: white;
                padding: 12px 40px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #006699;
            }
            QPushButton:pressed {
                background-color: #005580;
            }
        """)
        telegram_btn.setMinimumHeight(45)
        telegram_btn.setCursor(Qt.PointingHandCursor)
        telegram_btn.clicked.connect(self.open_telegram_bot)
        button_container.addWidget(telegram_btn)
        button_container.addStretch()

        layout.addLayout(button_container)

        # Bot Username Display
        bot_username = QLabel("@BitcoinSMCAlerts_bot")
        bot_username.setStyleSheet("""
            QLabel {
                color: #0088cc;
                font-size: 13px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                padding: 5px;
            }
        """)
        bot_username.setAlignment(Qt.AlignCenter)
        layout.addWidget(bot_username)

        group.setLayout(layout)
        return group

    def create_features_section(self):
        """Create features information section"""
        group = QGroupBox("Key Features")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #D1D4DC;
                border: 2px solid #2A2E39;
                border-radius: 10px;
                margin-top: 12px;
                padding: 20px;
                background-color: #1E222D;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 20px;
                top: 5px;
                padding: 0 10px;
                color: #D1D4DC;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 30, 20, 20)

        features = [
            "Real-time BTC OHLCV data visualization",
            "AI-powered price prediction with LSTM",
            "Smart Money Concepts (SMC) analysis",
            "Technical indicators (RSI, MACD, EMA, BB)",
            "Advanced backtesting system",
            "Real-time Telegram alerts",
            "Automatic data updates every hour",
            "Professional dark-themed interface",
        ]

        for text in features:
            feature_layout = QHBoxLayout()
            feature_layout.setSpacing(10)

            bullet = QLabel("•")
            bullet.setStyleSheet("""
                QLabel {
                    color: #2962FF;
                    font-size: 16px;
                    font-weight: bold;
                    padding-right: 5px;
                }
            """)
            bullet.setFixedWidth(20)
            feature_layout.addWidget(bullet)

            text_label = QLabel(text)
            text_label.setStyleSheet("""
                QLabel {
                    color: #D1D4DC;
                    font-size: 13px;
                    padding: 3px;
                }
            """)
            text_label.setWordWrap(True)
            feature_layout.addWidget(text_label, 1)

            layout.addLayout(feature_layout)

        group.setLayout(layout)
        return group

    def create_developer_section(self):
        """Create developer information section"""
        group = QGroupBox("Developer Information")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #D1D4DC;
                border: 2px solid #2A2E39;
                border-radius: 10px;
                margin-top: 12px;
                padding: 20px;
                background-color: #1E222D;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 20px;
                top: 5px;
                padding: 0 10px;
                color: #D1D4DC;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 30, 20, 20)

        # Developer name with container
        name_container = QFrame()
        name_container.setStyleSheet("""
            QFrame {
                background-color: #2A2E39;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        name_layout = QVBoxLayout(name_container)
        name_layout.setSpacing(8)
        name_layout.setContentsMargins(15, 15, 15, 15)

        dev_name = QLabel("Much Ivan Surya")
        dev_name.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 22px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        dev_name.setAlignment(Qt.AlignCenter)
        name_layout.addWidget(dev_name)

        year = QLabel("© 2025")
        year.setStyleSheet("""
            QLabel {
                color: #2962FF;
                font-size: 13px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        year.setAlignment(Qt.AlignCenter)
        name_layout.addWidget(year)

        layout.addWidget(name_container)

        # Project description
        description = QLabel(
            "BTC Price Prediction System\n"
            "Multi-Input LSTM with Attention Mechanism\n"
            "Smart Money Concepts Integration"
        )
        description.setStyleSheet("""
            QLabel {
                color: #D1D4DC;
                font-size: 13px;
                padding: 10px;
            }
        """)
        description.setAlignment(Qt.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)

        # Tech stack
        tech_label = QLabel("Built with:")
        tech_label.setStyleSheet("""
            QLabel {
                color: #2962FF;
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
            }
        """)
        tech_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(tech_label)

        tech_stack = QLabel(
            "Python • PyTorch • PySide6 • PyQtGraph\n"
            "Pandas • NumPy • SQLite • TA-Lib\n"
            "SmartMoneyConcepts • python-telegram-bot"
        )
        tech_stack.setStyleSheet("""
            QLabel {
                color: #787B86;
                font-size: 11px;
                font-family: 'Consolas', 'Courier New', monospace;
                padding: 5px;
            }
        """)
        tech_stack.setAlignment(Qt.AlignCenter)
        tech_stack.setWordWrap(True)
        layout.addWidget(tech_stack)

        group.setLayout(layout)
        return group

    def open_telegram_bot(self):
        """Open Telegram bot in browser or Telegram app"""
        # TODO: Replace with your actual bot username
        bot_username = "@BitcoinSMCAlerts_bot"
        telegram_url = f"https://t.me/{bot_username}"
        QDesktopServices.openUrl(QUrl(telegram_url))
