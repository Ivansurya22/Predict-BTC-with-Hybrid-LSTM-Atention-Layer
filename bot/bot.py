import os
import sys
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)

# Add directories to path for imports
BOT_DIR = Path(__file__).parent
ROOT_DIR = BOT_DIR.parent
CONFIG_DIR = ROOT_DIR / 'config'

sys.path.insert(0, str(BOT_DIR))
sys.path.insert(0, str(ROOT_DIR))

# Load .env directly
ENV_PATH = CONFIG_DIR / '.env'
load_dotenv(ENV_PATH)

# Get token from environment
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Import from same directory
import db_handler
import chart_generator

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = TELEGRAM_BOT_TOKEN
CHECK_INTERVAL = 3600  # 1 hour in seconds

# Storage files
USER_SETTINGS_FILE = BOT_DIR / 'user_settings.json'

# Load user settings from file
def load_user_settings():
    """Load user settings from JSON file"""
    if USER_SETTINGS_FILE.exists():
        try:
            with open(USER_SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

# Save user settings to file
def save_user_settings(settings):
    """Save user settings to JSON file"""
    try:
        with open(USER_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving user settings: {e}")
        return False

# Load settings on startup
user_settings = load_user_settings()


class BTCTradingBot:
    """Bitcoin Trading Alert Bot with Model + SMC Validation"""

    def __init__(self, token: str):
        self.token = token
        self.app = None
        self.last_alert_data = None

    def _get_main_keyboard(self):
        """Create main keyboard dengan command buttons"""
        keyboard = [
            [KeyboardButton("📊 Latest"), KeyboardButton("📈 Chart")],
            [KeyboardButton("📉 Stats"), KeyboardButton("📜 History")],
            [KeyboardButton("🔔 Alerts"), KeyboardButton("❓ Help")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk /start command"""
        chat_id = update.effective_chat.id
        user = update.effective_user
        is_new_user = chat_id not in user_settings

        # Initialize user settings
        if is_new_user:
            user_settings[chat_id] = {
                'alerts': False,  # Default OFF, user harus aktifkan manual
                'last_notified': None,
                'first_seen': datetime.now().isoformat(),
                'username': user.username or user.first_name,
                'user_id': user.id
            }
            save_user_settings(user_settings)
            logger.info(f"🆕 New user registered: {user.username or user.first_name} (ID: {chat_id})")

        # Welcome message
        if is_new_user:
            welcome_msg = (
                f"🎉 <b>Selamat Datang, {user.first_name}!</b> 🎉\n\n"
                "Terima kasih telah menggunakan <b>BTC Trading Alert Bot</b>! 🤖\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔍 <b>Apa yang Bot ini lakukan?</b>\n\n"
                "Bot ini memberikan <b>signal trading Bitcoin</b> berdasarkan:\n"
                "  ✅ <b>LSTM AI Model</b> - Prediksi harga\n"
                "  ✅ <b>SMC Analysis</b> - Smart Money Concepts\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🚨 <b>Smart Alert System</b>\n\n"
                "Alert <b>HANYA</b> dikirim jika:\n"
                "  🟢 Model: <b>UP</b> + SMC: <b>Buy</b> = ALIGNED\n"
                "  🔴 Model: <b>DOWN</b> + SMC: <b>Sell</b> = ALIGNED\n\n"
                "Jika tidak aligned = <b>NO ALERT</b> (untuk keamanan) ⚠️\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🎯 <b>Cara Menggunakan:</b>\n\n"
                "Gunakan tombol di bawah atau command:\n"
                "  📊 <b>Latest</b> - Lihat prediksi terbaru\n"
                "  📈 <b>Chart</b> - Grafik harga 24 jam\n"
                "  📉 <b>Stats</b> - Statistik market\n"
                "  📜 <b>History</b> - Riwayat prediksi\n"
                "  🔔 <b>Alerts</b> - Aktifkan/nonaktifkan alert\n"
                "  ❓ <b>Help</b> - Bantuan\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "💡 <b>Tips:</b>\n"
                "  • Aktifkan alert untuk notifikasi otomatis\n"
                "  • Alert dikirim setiap jam (jika ada setup)\n"
                "  • Selalu cek detail sebelum trading!\n\n"
                "⚠️ <b>Disclaimer:</b> Bot ini untuk edukasi.\n"
                "Selalu lakukan riset sendiri sebelum trading.\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🖥️ <b>Ingin Fitur Lebih Lengkap?</b>\n\n"
                "Gunakan <b>Aplikasi Desktop</b> untuk:\n"
                "  📊 <b>SMC Indicators</b> - Order Block, FVG, BOS/CHoCH\n"
                "  📈 <b>Interactive Charts</b> - TradingView style\n"
                "  🔍 <b>Advanced Analysis</b> - Multi-timeframe\n"
                "  📉 <b>Real-time Updates</b> - Live data streaming\n"
                "  💾 <b>Export Data</b> - CSV, Excel, PDF\n"
                "  ⚙️ <b>Custom Settings</b> - Parameter tuning\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🚀 <b>Siap untuk memulai?</b>\n"
                "Tekan tombol di bawah atau gunakan command!\n\n"
                "🙏 Terima kasih telah mempercayai bot kami!"
            )
        else:
            welcome_msg = (
                f"👋 <b>Welcome Back, {user.first_name}!</b>\n\n"
                "🤖 <b>BTC Trading Alert Bot</b>\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "<b>📊 Alert hanya muncul jika:</b>\n"
                "  🟢 Model: UP + SMC: Buy = ALIGNED\n"
                "  🔴 Model: DOWN + SMC: Sell = ALIGNED\n\n"
                "<b>🎯 Gunakan tombol di bawah:</b>\n"
                "  📊 Latest - Prediksi terbaru\n"
                "  📈 Chart - Grafik 24h\n"
                "  📉 Stats - Market stats\n"
                "  📜 History - Riwayat\n"
                "  🔔 Alerts - Toggle alert\n"
                "  ❓ Help - Bantuan\n\n"
                f"🔔 Alert Status: <b>{'ON ✅' if user_settings[chat_id]['alerts'] else 'OFF ❌'}</b>\n\n"
                "⏰ Update otomatis setiap 1 jam\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🖥️ <b>Butuh Analisis Lebih Dalam?</b>\n"
                "Coba <b>Aplikasi Desktop</b> untuk fitur lengkap:\n"
                "SMC Indicators, Interactive Charts, dan lebih banyak lagi!"
            )

        await update.message.reply_text(
            welcome_msg,
            parse_mode='HTML',
            reply_markup=self._get_main_keyboard()
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk /help command"""
        help_msg = (
            "❓ <b>BTC Trading Alert Bot - Help</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>📱 Bot Commands:</b>\n\n"
            "📊 <b>Latest</b> - Prediksi terbaru\n"
            "📈 <b>Chart</b> - Grafik harga 24 jam\n"
            "📉 <b>Stats</b> - Statistik market\n"
            "📜 <b>History</b> - Riwayat prediksi\n"
            "🔔 <b>Alerts</b> - Aktifkan/nonaktifkan notifikasi\n"
            "❓ <b>Help</b> - Tampilkan bantuan ini\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>🎯 Cara Kerja Alert:</b>\n\n"
            "Alert <b>HANYA</b> dikirim jika:\n"
            "  🟢 Model: UP + SMC: Buy\n"
            "  🔴 Model: DOWN + SMC: Sell\n\n"
            "⏰ Pengecekan otomatis setiap 1 jam\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🖥️ <b>Aplikasi Desktop</b>\n\n"
            "Untuk analisis lebih mendalam, gunakan\n"
            "<b>Aplikasi Desktop</b> dengan fitur:\n\n"
            "  📊 <b>SMC Indicators</b>\n"
            "    • Order Blocks (OB)\n"
            "    • Fair Value Gaps (FVG)\n"
            "    • Break of Structure (BOS)\n"
            "    • Change of Character (CHoCH)\n"
            "    • Liquidity Zones\n\n"
            "  📈 <b>Interactive Charts</b>\n"
            "    • Custom indicators\n"
            "    • Candlestick patterns\n\n"
            "  🔍 <b>Advanced Features</b>\n"
            "    • Real-time market scanner\n"
            "    • Backtesting engine\n"
            "    • Custom Models\n"
            "    • Performance analytics\n"
            "    • Export to CSV/Excel/PDF\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "⚠️ <b>Disclaimer:</b>\n"
            "Bot ini untuk tujuan edukasi.\n"
            "Selalu DYOR sebelum trading!\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "💡 <b>Tips:</b>\n"
            "  • Aktifkan alert untuk notifikasi real-time\n"
            "  • Cek history untuk track record\n"
            "  • Gunakan chart untuk konfirmasi visual\n"
            "  • Upgrade ke Desktop untuk fitur pro!\n\n"
            "📞 Butuh bantuan? Hubungi support kami!"
        )

        await update.message.reply_text(
            help_msg,
            parse_mode='HTML',
            reply_markup=self._get_main_keyboard()
        )

    async def latest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk /latest - Show latest prediction"""
        chat_id = update.effective_chat.id

        # Send "processing" message
        msg = await update.message.reply_text("🔍 Fetching latest data...")

        try:
            pred_data = db_handler.get_latest_prediction()

            if not pred_data:
                await msg.edit_text("❌ No prediction data available yet.\n\nRun prediction first!")
                return

            # Format message
            message = self._format_prediction_message(pred_data)

            # Create inline keyboard
            keyboard = [
                [
                    InlineKeyboardButton("📊 Chart", callback_data="chart"),
                    InlineKeyboardButton("📈 Stats", callback_data="stats")
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_latest")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await msg.edit_text(message, parse_mode='HTML', reply_markup=reply_markup)

        except Exception as e:
            logger.error(f"Error in latest_command: {e}")
            await msg.edit_text(f"❌ Error: {str(e)}")

    async def chart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk /chart - Generate and send price chart"""
        chat_id = update.effective_chat.id

        msg = await update.message.reply_text("📊 Generating chart...")

        try:
            # Generate chart
            chart_path = chart_generator.generate_price_chart()

            if not chart_path or not Path(chart_path).exists():
                await msg.edit_text("❌ Failed to generate chart")
                return

            # Get latest price for caption
            price = db_handler.get_latest_price()
            caption = f"📊 <b>BTC/USDT - 24H Chart</b>\n\nCurrent: ${price:,.2f}" if price else "📊 BTC/USDT - 24H Chart"

            # Send photo
            await update.message.reply_photo(
                photo=open(chart_path, 'rb'),
                caption=caption,
                parse_mode='HTML'
            )

            # Delete "generating" message
            await msg.delete()

            # Clean up chart file
            try:
                os.remove(chart_path)
            except:
                pass

        except Exception as e:
            logger.error(f"Error in chart_command: {e}")
            await msg.edit_text(f"❌ Error generating chart: {str(e)}")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk /stats - Market statistics"""
        msg = await update.message.reply_text("📊 Fetching market stats...")

        try:
            stats = db_handler.get_market_stats()

            if not stats:
                await msg.edit_text("❌ No market data available")
                return

            # Format stats message
            change_emoji = "🟢" if stats['price_change_pct_24h'] >= 0 else "🔴"

            stats_msg = (
                f"📊 <b>Market Statistics (24H)</b>\n\n"
                f"💵 Current: ${stats['current_price']:,.2f}\n"
                f"{change_emoji} Change: {stats['price_change_pct_24h']:+.2f}% (${stats['price_change_24h']:+,.2f})\n\n"
                f"📈 High: ${stats['high_24h']:,.2f}\n"
                f"📉 Low: ${stats['low_24h']:,.2f}\n"
                f"📊 Avg Volume: {stats['avg_volume_24h']:,.2f} BTC\n\n"
                f"⏰ Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Keyboard
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_stats")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await msg.edit_text(stats_msg, parse_mode='HTML', reply_markup=reply_markup)

        except Exception as e:
            logger.error(f"Error in stats_command: {e}")
            await msg.edit_text(f"❌ Error: {str(e)}")

    async def history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk /history - Show prediction history"""
        msg = await update.message.reply_text("📜 Fetching history...")

        try:
            df = db_handler.get_prediction_history(limit=10)

            if df is None or df.empty:
                await msg.edit_text("❌ No prediction history available")
                return

            # Count aligned signals
            aligned_count = db_handler.count_aligned_signals(24)

            # Format history message
            history_msg = f"📜 <b>Prediction History (Last 10)</b>\n\n"

            for idx, row in df.iterrows():
                timestamp = row['timestamp']
                model = row['model_signal']
                smc = row['smc_signal'] if row['smc_signal'] else 'N/A'
                aligned = row['aligned']

                icon = "✅" if aligned else "❌"

                history_msg += (
                    f"{icon} <code>{timestamp[:16]}</code>\n"
                    f"   Model: {model} | SMC: {smc}\n\n"
                )

            history_msg += f"\n🎯 Aligned signals (24h): {aligned_count}"

            await msg.edit_text(history_msg, parse_mode='HTML')

        except Exception as e:
            logger.error(f"Error in history_command: {e}")
            await msg.edit_text(f"❌ Error: {str(e)}")

    async def alerts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk /alerts - Toggle alerts"""
        chat_id = update.effective_chat.id
        user = update.effective_user

        if chat_id not in user_settings:
            user_settings[chat_id] = {
                'alerts': False,
                'last_notified': None,
                'first_seen': datetime.now().isoformat(),
                'username': user.username or user.first_name,
                'user_id': user.id
            }

        # Toggle alerts
        old_status = user_settings[chat_id]['alerts']
        user_settings[chat_id]['alerts'] = not old_status
        new_status = user_settings[chat_id]['alerts']

        # Save to file
        save_user_settings(user_settings)

        # Log the change
        if new_status:
            logger.info(f"✅ User {user.username or user.first_name} (ID: {chat_id}) ENABLED alerts")
        else:
            logger.info(f"❌ User {user.username or user.first_name} (ID: {chat_id}) DISABLED alerts")

        # Create message
        if new_status:
            msg = (
                "🔔 <b>Alert Status: ON ✅</b>\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "✅ <b>Alert Telah Diaktifkan!</b>\n\n"
                "Anda akan menerima notifikasi otomatis ketika:\n"
                "  🟢 Model: <b>UP</b> + SMC: <b>Buy</b>\n"
                "  🔴 Model: <b>DOWN</b> + SMC: <b>Sell</b>\n\n"
                "📬 Alert dikirim ke chat ID Anda:\n"
                f"  <code>{chat_id}</code>\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "⏰ <b>Jadwal Pengecekan:</b>\n"
                "  • Setiap 1 jam sekali\n"
                "  • Hanya jika ada setup aligned\n"
                "  • Lengkap dengan chart & detail\n\n"
                "💡 <b>Tips:</b>\n"
                "  • Pastikan notifikasi Telegram aktif\n"
                "  • Cek chat ini secara berkala\n"
                "  • Gunakan /latest untuk cek manual\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔕 Gunakan tombol 🔔 <b>Alerts</b> lagi untuk menonaktifkan"
            )
        else:
            msg = (
                "🔕 <b>Alert Status: OFF ❌</b>\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "❌ <b>Alert Telah Dinonaktifkan</b>\n\n"
                "Anda <b>tidak akan</b> menerima notifikasi otomatis.\n\n"
                "💡 Anda masih bisa:\n"
                "  📊 Cek prediksi manual dengan /latest\n"
                "  📈 Lihat chart dengan /chart\n"
                "  📜 Lihat history dengan /history\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔔 Gunakan tombol 🔔 <b>Alerts</b> lagi untuk mengaktifkan kembali"
            )

        await update.message.reply_text(
            msg,
            parse_mode='HTML',
            reply_markup=self._get_main_keyboard()
        )

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()

        try:
            if query.data == "chart":
                # Generate and send chart
                chart_path = chart_generator.generate_price_chart()

                if chart_path and Path(chart_path).exists():
                    price = db_handler.get_latest_price()
                    caption = f"📊 <b>BTC/USDT - 24H Chart</b>\n\nCurrent: ${price:,.2f}" if price else "📊 BTC/USDT - 24H Chart"

                    await query.message.reply_photo(
                        photo=open(chart_path, 'rb'),
                        caption=caption,
                        parse_mode='HTML'
                    )

                    try:
                        os.remove(chart_path)
                    except:
                        pass
                else:
                    await query.message.reply_text("❌ Failed to generate chart")

            elif query.data == "stats":
                stats = db_handler.get_market_stats()

                if stats:
                    change_emoji = "🟢" if stats['price_change_pct_24h'] >= 0 else "🔴"

                    stats_msg = (
                        f"📊 <b>Market Statistics (24H)</b>\n\n"
                        f"💵 Current: ${stats['current_price']:,.2f}\n"
                        f"{change_emoji} Change: {stats['price_change_pct_24h']:+.2f}%\n\n"
                        f"📈 High: ${stats['high_24h']:,.2f}\n"
                        f"📉 Low: ${stats['low_24h']:,.2f}\n"
                    )

                    await query.message.reply_text(stats_msg, parse_mode='HTML')
                else:
                    await query.message.reply_text("❌ No market data")

            elif query.data == "refresh_latest":
                pred_data = db_handler.get_latest_prediction()

                if pred_data:
                    message = self._format_prediction_message(pred_data)

                    keyboard = [
                        [
                            InlineKeyboardButton("📊 Chart", callback_data="chart"),
                            InlineKeyboardButton("📈 Stats", callback_data="stats")
                        ],
                        [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_latest")]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)

                    await query.edit_message_text(message, parse_mode='HTML', reply_markup=reply_markup)
                else:
                    await query.edit_message_text("❌ No prediction data available")

            elif query.data == "refresh_stats":
                stats = db_handler.get_market_stats()

                if stats:
                    change_emoji = "🟢" if stats['price_change_pct_24h'] >= 0 else "🔴"

                    stats_msg = (
                        f"📊 <b>Market Statistics (24H)</b>\n\n"
                        f"💵 Current: ${stats['current_price']:,.2f}\n"
                        f"{change_emoji} Change: {stats['price_change_pct_24h']:+.2f}%\n\n"
                        f"📈 High: ${stats['high_24h']:,.2f}\n"
                        f"📉 Low: ${stats['low_24h']:,.2f}\n"
                        f"📊 Avg Volume: {stats['avg_volume_24h']:,.2f} BTC\n\n"
                        f"⏰ Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

                    keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_stats")]]
                    reply_markup = InlineKeyboardMarkup(keyboard)

                    await query.edit_message_text(stats_msg, parse_mode='HTML', reply_markup=reply_markup)
                else:
                    await query.edit_message_text("❌ No market data")

        except Exception as e:
            logger.error(f"Error in button_callback: {e}")
            await query.message.reply_text(f"❌ Error: {str(e)}")

    def _format_prediction_message(self, pred_data: dict) -> str:
        """Format prediction data menjadi message"""
        model = pred_data['model']
        smc = pred_data['smc']
        aligned = pred_data['aligned']

        # Emoji based on signal
        if aligned:
            if model['signal'] == 'UP':
                main_emoji = "🟢"
                signal_text = "BUY SIGNAL"
            else:
                main_emoji = "🔴"
                signal_text = "SELL SIGNAL"
        else:
            main_emoji = "⚪"
            signal_text = "NO ALIGNED SETUP"

        # Model section
        model_conf_bar = self._create_confidence_bar(model['confidence'])

        # SMC section
        smc_conf_bar = self._create_confidence_bar(smc['confidence'] / 100)  # SMC confidence is 0-100

        # Risk/Reward emoji
        if smc['risk_reward'] >= 3:
            rr_emoji = "🔥"
        elif smc['risk_reward'] >= 2:
            rr_emoji = "✅"
        else:
            rr_emoji = "⚠️"

        message = (
            f"{main_emoji} <b>{signal_text}</b>\n"
            f"{'='*30}\n\n"

            f"🤖 <b>LSTM Model Prediction</b>\n"
            f"Signal: <b>{model['signal']}</b>\n"
            f"Confidence: {model_conf_bar} {model['confidence']*100:.1f}%\n"
            f"Price: ${model['close_price']:,.2f}\n\n"

            f"📊 Probabilities:\n"
            f"  🔴 DOWN: {model['probabilities']['down']*100:.1f}%\n"
            f"  ⚪ HOLD: {model['probabilities']['hold']*100:.1f}%\n"
            f"  🟢 UP: {model['probabilities']['up']*100:.1f}%\n\n"

            f"{'='*30}\n\n"

            f"🎯 <b>SMC Setup</b>\n"
            f"Signal: <b>{smc['smc_signal']}</b>\n"
            f"Quality: <b>{smc['setup_quality']}</b>\n"
            f"Confidence: {smc_conf_bar} {smc['confidence']:.1f}%\n"
            f"Type: {smc['setup_type']}\n\n"

            f"💰 <b>Trade Levels:</b>\n"
            f"Entry: ${smc['entry_price']:,.2f}\n"
            f"Stop Loss: ${smc['stop_loss']:,.2f}\n"
            f"TP1: ${smc['take_profit_1']:,.2f}\n"
            f"TP2: ${smc['take_profit_2']:,.2f}\n"
            f"TP3: ${smc['take_profit_3']:,.2f}\n"
            f"{rr_emoji} R/R: <b>{smc['risk_reward']:.2f}</b>\n\n"

            f"{'='*30}\n\n"
        )

        # Alignment status
        if aligned:
            message += "✅ <b>ALIGNED SETUP - READY TO TRADE</b>\n\n"
        else:
            message += "❌ <b>NOT ALIGNED - WAIT FOR CONFIRMATION</b>\n\n"

        # Supporting factors (top 3)
        if smc['supporting_factors']:
            factors = smc['supporting_factors'].split('|')[:3]
            message += "✅ <b>Supporting:</b>\n"
            for f in factors:
                message += f"  • {f}\n"

        # Conflicting factors (top 2)
        if smc['conflicting_factors']:
            conflicts = smc['conflicting_factors'].split('|')[:2]
            if conflicts and conflicts[0]:
                message += "\n⚠️ <b>Conflicts:</b>\n"
                for c in conflicts:
                    if c:
                        message += f"  • {c}\n"

        message += f"\n⏰ {model['timestamp'][:19]}"

        return message

    def _create_confidence_bar(self, confidence: float) -> str:
        """Create visual confidence bar"""
        filled = int(confidence * 10)
        return "█" * filled + "░" * (10 - filled)

    async def check_and_alert(self, context: ContextTypes.DEFAULT_TYPE):
        """Periodic task to check for new aligned setups and send alerts"""
        logger.info("🔍 Checking for new aligned setups...")

        try:
            pred_data = db_handler.get_latest_prediction()

            if not pred_data:
                logger.info("No prediction data available")
                return

            # Check if aligned
            if not pred_data['aligned']:
                logger.info("Not aligned - no alert sent")
                return

            # Check if this is a new alert (compare with last alert)
            current_timestamp = pred_data['model']['timestamp']

            if self.last_alert_data and self.last_alert_data == current_timestamp:
                logger.info("Already alerted for this timestamp")
                return

            # Update last alert timestamp
            self.last_alert_data = current_timestamp

            # Send alert to all users with alerts enabled
            message = "🚨 <b>NEW ALIGNED SETUP DETECTED!</b> 🚨\n\n" + self._format_prediction_message(pred_data)

            # Generate chart
            chart_path = chart_generator.generate_price_chart()

            for chat_id, settings in user_settings.items():
                if settings.get('alerts', True):
                    try:
                        # Send message
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode='HTML'
                        )

                        # Send chart if available
                        if chart_path and Path(chart_path).exists():
                            await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=open(chart_path, 'rb'),
                                caption="📊 Current Price Chart"
                            )

                        logger.info(f"✅ Alert sent to {chat_id}")

                    except Exception as e:
                        logger.error(f"Failed to send alert to {chat_id}: {e}")

            # Clean up chart
            if chart_path:
                try:
                    os.remove(chart_path)
                except:
                    pass

            logger.info(f"🚨 Alert sent for {current_timestamp}")

        except Exception as e:
            logger.error(f"Error in check_and_alert: {e}")
            import traceback
            traceback.print_exc()

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")

    async def periodic_check_loop(self):
        """Background task untuk periodic checking"""
        logger.info(f"⏰ Starting periodic check loop (interval: {CHECK_INTERVAL}s)")

        # First check after 10 seconds
        await asyncio.sleep(10)

        while True:
            try:
                # Create a dummy context for the check
                from telegram.ext import ContextTypes

                # This is a workaround since we don't have JobQueue
                class DummyContext:
                    def __init__(self, app):
                        self.bot = app.bot
                        self.application = app

                context = DummyContext(self.app)
                await self.check_and_alert(context)

            except Exception as e:
                logger.error(f"Error in periodic check: {e}")

            # Wait for next interval
            await asyncio.sleep(CHECK_INTERVAL)

    async def message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk text messages dari keyboard buttons"""
        text = update.message.text

        if text == "📊 Latest":
            await self.latest_command(update, context)
        elif text == "📈 Chart":
            await self.chart_command(update, context)
        elif text == "📉 Stats":
            await self.stats_command(update, context)
        elif text == "📜 History":
            await self.history_command(update, context)
        elif text == "🔔 Alerts":
            await self.alerts_command(update, context)
        elif text == "❓ Help":
            await self.help_command(update, context)
        else:
            # Unknown command
            await update.message.reply_text(
                "❓ Perintah tidak dikenali.\n\nGunakan tombol di bawah atau /help untuk bantuan.",
                reply_markup=self._get_main_keyboard()
            )

    def run(self):
        """Start the bot"""
        logger.info("🤖 Starting BTC Trading Alert Bot...")
        logger.info(f"📂 User settings file: {USER_SETTINGS_FILE}")
        logger.info(f"👥 Registered users: {len(user_settings)}")

        # Log active users
        active_users = sum(1 for s in user_settings.values() if s.get('alerts', False))
        logger.info(f"🔔 Users with alerts ON: {active_users}")

        # Create application
        self.app = Application.builder().token(self.token).build()

        # Register handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("latest", self.latest_command))
        self.app.add_handler(CommandHandler("chart", self.chart_command))
        self.app.add_handler(CommandHandler("stats", self.stats_command))
        self.app.add_handler(CommandHandler("history", self.history_command))
        self.app.add_handler(CommandHandler("alerts", self.alerts_command))
        self.app.add_handler(CallbackQueryHandler(self.button_callback))

        # Add message handler for keyboard buttons
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler))

        # Error handler
        self.app.add_error_handler(self.error_handler)

        logger.info("✅ Bot started successfully!")
        logger.info(f"⏰ Alert check interval: {CHECK_INTERVAL}s (1 hour)")
        logger.info("📡 Waiting for messages...")

        # Start periodic check in background
        async def run_bot():
            async with self.app:
                await self.app.initialize()
                await self.app.start()

                # Start periodic check as background task
                check_task = asyncio.create_task(self.periodic_check_loop())

                # Start polling
                await self.app.updater.start_polling(allowed_updates=Update.ALL_TYPES)

                # Keep running
                try:
                    await asyncio.Event().wait()
                except (KeyboardInterrupt, SystemExit):
                    logger.info("\n👋 Shutting down...")
                    check_task.cancel()
                finally:
                    await self.app.updater.stop()
                    await self.app.stop()
                    await self.app.shutdown()

        # Run the bot
        asyncio.run(run_bot())


def main():
    """Main function"""

    # Check if token is provided
    if not BOT_TOKEN:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN not found in config")
        print("\n📝 Please add to config/.env:")
        print("TELEGRAM_BOT_TOKEN=your_bot_token_here")
        print("\n💡 Get bot token from @BotFather on Telegram")
        return

    try:
        bot = BTCTradingBot(BOT_TOKEN)
        bot.run()
    except KeyboardInterrupt:
        logger.info("\n👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
