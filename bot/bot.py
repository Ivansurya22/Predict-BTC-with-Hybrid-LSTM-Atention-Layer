# Fungsi untuk menampilkan harga BTC
async def btc_price(update: Update, context: CallbackContext):
    price = get_latest_price()

    if price:
        formatted_price = format_price(price)
        current_time = datetime.now().strftime("%d %b %Y, %H:%M")

        message = (
            f"📊 <b>HARGA BTC TERKINI</b>\n\n"
            f"💰 <b>Harga:</b> {formatted_price}\n"
            f"🕐 <b>Update:</b> {current_time}\n\n"
            f"💡 <i>Harga diperbarui secara real-time</i>"
        )
    else:
        message = "❌ Data harga tidak tersedia saat ini."

    await update.message.reply_text(message, parse_mode="HTML")

# Fungsi untuk menampilkan hasil prediksiimport os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
import pandas as pd
import json
import time
import asyncio
from datetime import datetime

from db_handler import get_latest_price, get_latest_prediction

# Load Token dari .env
load_dotenv(dotenv_path="config/.env")
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ADMIN USER ID - Ganti dengan user ID Telegram Anda
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0"))

# File untuk menyimpan data user yang sudah subscribe
USERS_DATA_FILE = "bot/users_data.json"

# Global variable untuk menyimpan bot application
bot_app = None

# Global variable untuk menyimpan data terakhir yang dikirim
last_sent_data = {
    'prediction_timestamp': None,
    'price': None,
    'last_alert_time': None
}

# Global variable untuk tracking data monitoring
monitoring_active = True
last_known_prediction_timestamp = None
last_known_price = None

# Global variable untuk mode broadcast admin
admin_broadcast_mode = {}

# Fungsi untuk format harga dengan pemisah ribuan
def format_price(price):
    """Format harga dengan pemisah ribuan untuk mudah dibaca"""
    return f"${price:,.2f}"

# Fungsi untuk load data users
def load_users_data():
    try:
        with open(USERS_DATA_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Fungsi untuk save data users
def save_users_data(users_data):
    os.makedirs(os.path.dirname(USERS_DATA_FILE), exist_ok=True)
    with open(USERS_DATA_FILE, 'w') as f:
        json.dump(users_data, f, indent=2)

# Fungsi untuk menambahkan user ke daftar subscriber
def add_user_to_subscribers(user_id, chat_id):
    users_data = load_users_data()
    users_data[str(user_id)] = {
        'chat_id': chat_id,
        'subscribed_at': datetime.now().isoformat(),
        'active': True,
        'last_alert_sent': None
    }
    save_users_data(users_data)

# Fungsi untuk update waktu alert terakhir user
def update_user_last_alert(user_id, alert_time):
    users_data = load_users_data()
    if str(user_id) in users_data:
        users_data[str(user_id)]['last_alert_sent'] = alert_time
        save_users_data(users_data)

# Fungsi untuk cek apakah user adalah admin
def is_admin(user_id):
    return user_id == ADMIN_USER_ID

# Fungsi untuk mengirim alert selamat datang dengan data terbaru
async def send_welcome_alert(chat_id, user_id):
    """Kirim alert data terbaru ke user baru"""
    try:
        current_price = get_latest_price()
        prediction_data = get_latest_prediction()

        if current_price or prediction_data:
            current_time = datetime.now().strftime("%d %b %Y, %H:%M")
            message_parts = [f"🎉 <b>SELAMAT DATANG!</b> 🎉\n📅 {current_time}\n"]

            # Informasi harga terkini
            if current_price:
                message_parts.append(f"💰 <b>Harga BTC Terkini:</b> {format_price(current_price)}\n")

            # Informasi prediksi terbaru
            if prediction_data:
                model = prediction_data['model']
                signal_emoji = {
                    'down': '🔽',
                    'up': '🔼',
                    'hold': '↔️'
                }.get(model['signal'].lower(), '⚠️')

                confidence_pct = f"{model['confidence']*100:.1f}%"
                probs = model['probabilities']

                message_parts.append(f"🔮 <b>Prediksi Terbaru:</b> {signal_emoji} {model['signal'].upper()}")
                message_parts.append(f"📊 <b>Confidence:</b> {confidence_pct}")
                message_parts.append(f"📈 Up: {probs['up']*100:.0f}% | Down: {probs['down']*100:.0f}% | Hold: {probs['hold']*100:.0f}%")

            message_parts.append("\n🔔 <b>Anda akan menerima alert otomatis untuk update selanjutnya!</b>")

            welcome_message = "\n".join(message_parts)

            await bot_app.bot.send_message(
                chat_id=chat_id,
                text=welcome_message,
                parse_mode="HTML"
            )

            update_user_last_alert(user_id, datetime.now().isoformat())
            print(f"✅ Welcome alert sent to user {user_id}")

    except Exception as e:
        print(f"❌ Error sending welcome alert: {e}")

# Fungsi untuk menampilkan tombol menu utama
async def start(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    add_user_to_subscribers(user_id, chat_id)

    if is_admin(user_id):
        keyboard = [
            ["📊 Cek Harga BTC", "🔮 Hasil Prediksi"],
            ["📊 Signal SMC", "⚙️ Pengaturan Alert"],
            ["ℹ️ Status Subscription"],
            ["📢 Broadcast Message (Admin)", "📊 Stats Admin"]
        ]
    else:
        keyboard = [
            ["📊 Cek Harga BTC", "🔮 Hasil Prediksi"],
            ["📊 Signal SMC", "⚙️ Pengaturan Alert"],
            ["ℹ️ Status Subscription"]
        ]

    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    admin_text = " (Admin Mode)" if is_admin(user_id) else ""
    welcome_message = (
        f"👋 Selamat datang di BTC Trading Bot!{admin_text}\n\n"
        "🔔 Anda sekarang akan menerima alert otomatis untuk:\n"
        "• Data prediksi terbaru (real-time)\n"
        "• Perubahan harga signifikan\n"
        "• Update berkala setiap jam\n\n"
        "Pilih opsi di bawah ini:"
    )

    await update.message.reply_text(welcome_message, reply_markup=reply_markup)
    asyncio.create_task(send_welcome_alert(chat_id, user_id))

# Fungsi untuk broadcast message (Admin only)
async def admin_broadcast(update: Update, context: CallbackContext):
    user_id = update.effective_user.id

    if not is_admin(user_id):
        await update.message.reply_text("❌ Anda tidak memiliki akses admin.")
        return

    admin_broadcast_mode[user_id] = True

    message = (
        "📢 <b>MODE BROADCAST ADMIN</b>\n\n"
        "Kirim pesan atau gambar selanjutnya untuk di-broadcast ke semua subscriber.\n"
        "• Untuk gambar: Kirim foto dengan caption\n"
        "• Untuk teks: Kirim pesan biasa\n"
        "Gunakan /cancel untuk membatalkan.\n\n"
        "⚠️ <b>Peringatan:</b> Pesan akan dikirim ke semua user aktif!"
    )

    await update.message.reply_text(message, parse_mode="HTML")

# Fungsi untuk menampilkan statistik admin
async def admin_stats(update: Update, context: CallbackContext):
    user_id = update.effective_user.id

    if not is_admin(user_id):
        await update.message.reply_text("❌ Anda tidak memiliki akses admin.")
        return

    users_data = load_users_data()

    total_users = len(users_data)
    active_users = sum(1 for user in users_data.values() if user.get('active', True))
    inactive_users = total_users - active_users

    today = datetime.now().date()
    today_users = 0
    for user in users_data.values():
        try:
            join_date = datetime.fromisoformat(user['subscribed_at']).date()
            if join_date == today:
                today_users += 1
        except:
            pass

    current_price = get_latest_price()
    price_text = format_price(current_price) if current_price else "N/A"

    message = (
        "📊 <b>STATISTIK ADMIN</b>\n\n"
        f"👥 <b>Total User:</b> {total_users}\n"
        f"✅ <b>User Aktif:</b> {active_users}\n"
        f"❌ <b>User Tidak Aktif:</b> {inactive_users}\n"
        f"🆕 <b>User Hari Ini:</b> {today_users}\n\n"
        f"💰 <b>Harga BTC Saat Ini:</b> {price_text}\n\n"
        "📢 Gunakan /broadcast untuk mengirim pesan ke semua user"
    )

    await update.message.reply_text(message, parse_mode="HTML")

# Fungsi untuk cancel broadcast mode
async def cancel_broadcast(update: Update, context: CallbackContext):
    user_id = update.effective_user.id

    if user_id in admin_broadcast_mode:
        del admin_broadcast_mode[user_id]
        await update.message.reply_text("✅ Mode broadcast dibatalkan.")
    else:
        await update.message.reply_text("❌ Anda tidak dalam mode broadcast.")

# Fungsi untuk menampilkan status subscription
async def status_subscription(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    users_data = load_users_data()

    if str(user_id) in users_data and users_data[str(user_id)]['active']:
        subscribed_at = users_data[str(user_id)]['subscribed_at']
        subscribe_date = datetime.fromisoformat(subscribed_at).strftime("%d %b %Y, %H:%M")

        last_alert = users_data[str(user_id)].get('last_alert_sent')
        last_alert_text = "Belum ada"
        if last_alert:
            last_alert_date = datetime.fromisoformat(last_alert).strftime("%d %b %Y, %H:%M")
            last_alert_text = last_alert_date

        message = (
            "✅ <b>STATUS SUBSCRIPTION AKTIF</b>\n\n"
            f"📅 <b>Bergabung sejak:</b> {subscribe_date}\n"
            f"🔔 <b>Alert Otomatis:</b> Aktif\n"
            f"⏰ <b>Alert Terakhir:</b> {last_alert_text}\n"
            f"🚀 <b>Alert Real-time:</b> Aktif\n\n"
            "Anda akan menerima notifikasi otomatis untuk:\n"
            "• Prediksi trading terbaru (langsung)\n"
            "• Perubahan harga signifikan (langsung)\n"
            "• Update berkala setiap jam"
        )
    else:
        message = (
            "❌ <b>SUBSCRIPTION TIDAK AKTIF</b>\n\n"
            "Ketik /start untuk mengaktifkan alert otomatis."
        )

    await update.message.reply_text(message, parse_mode="HTML")

# Fungsi untuk pengaturan alert
async def alert_settings(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    users_data = load_users_data()

    if str(user_id) in users_data:
        current_status = "Aktif ✅" if users_data[str(user_id)]['active'] else "Nonaktif ❌"

        message = (
            f"⚙️ <b>PENGATURAN ALERT</b>\n\n"
            f"Status saat ini: {current_status}\n\n"
            "Gunakan perintah berikut:\n"
            "• /enable_alert - Aktifkan alert\n"
            "• /disable_alert - Nonaktifkan alert\n"
            "• /status - Cek status subscription\n"
            "• /force_alert - Dapatkan alert data terbaru sekarang"
        )

        if is_admin(user_id):
            message += "\n\n<b>Admin Commands:</b>\n"
            message += "• /broadcast - Broadcast pesan ke semua user\n"
            message += "• /stats - Lihat statistik user\n"
            message += "• /cancel - Cancel broadcast mode"

    else:
        message = "❌ Anda belum terdaftar. Ketik /start terlebih dahulu."

    await update.message.reply_text(message, parse_mode="HTML")

# Fungsi untuk mengaktifkan alert
async def enable_alert(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    users_data = load_users_data()

    if str(user_id) in users_data:
        users_data[str(user_id)]['active'] = True
        save_users_data(users_data)
        await update.message.reply_text("✅ Alert otomatis telah diaktifkan!")
        asyncio.create_task(send_welcome_alert(chat_id, user_id))
    else:
        await update.message.reply_text("❌ Ketik /start terlebih dahulu untuk mendaftar.")

# Fungsi untuk menonaktifkan alert
async def disable_alert(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    users_data = load_users_data()

    if str(user_id) in users_data:
        users_data[str(user_id)]['active'] = False
        save_users_data(users_data)
        await update.message.reply_text("❌ Alert otomatis telah dinonaktifkan.")
    else:
        await update.message.reply_text("❌ Anda tidak terdaftar dalam sistem.")

# Fungsi untuk force alert
async def force_alert(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    users_data = load_users_data()

    if str(user_id) in users_data and users_data[str(user_id)]['active']:
        await send_welcome_alert(chat_id, user_id)
        await update.message.reply_text("✅ Alert data terbaru telah dikirim!")
    else:
        await update.message.reply_text("❌ Alert tidak aktif atau Anda belum terdaftar.")

# Fungsi untuk menampilkan harga BTC
async def btc_price(update: Update, context: CallbackContext):
    price = get_latest_price()

    if price:
        formatted_price = format_price(price)
        current_time = datetime.now().strftime("%d %b %Y, %H:%M")

        message = (
            f"📊 <b>HARGA BTC TERKINI</b>\n\n"
            f"💰 <b>Harga:</b> {formatted_price}\n"
            f"🕐 <b>Update:</b> {current_time}\n\n"
            f"💡 <i>Harga diperbarui secara real-time</i>"
        )
    else:
        message = "❌ Data harga tidak tersedia saat ini."

    await update.message.reply_text(message, parse_mode="HTML")

# Fungsi untuk menampilkan menu filter alert
async def filter_alert(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    current_mode = get_user_filter_mode(user_id)
    current_config = FILTER_MODES[current_mode]

    keyboard = []
    for mode_key, mode_config in FILTER_MODES.items():
        button_text = mode_config['name']
        if mode_key == current_mode:
            button_text += " ✅"

        keyboard.append([InlineKeyboardButton(
            text=button_text,
            callback_data=f"filter_mode_{mode_key}"
        )])

    keyboard.append([
        InlineKeyboardButton("📊 Perbandingan Mode", callback_data="filter_comparison"),
        InlineKeyboardButton("🧪 Test Filter", callback_data="test_current_filter")
    ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    message = (
        f"🎯 <b>FILTER ALERT STRATEGY</b>\n\n"
        f"<b>Mode Aktif:</b> {current_config['name']}\n"
        f"<b>Deskripsi:</b> {current_config['description']}\n\n"
        f"<b>Kriteria Saat Ini:</b>\n"
        f"• Confidence: >{current_config['min_confidence']*100:.0f}%\n"
        f"• Min Supporting Factors: {current_config['min_supporting_factors']}\n"
        f"• Max Conflicting Factors: {current_config['max_conflicting_factors']}\n\n"
        f"<b>Karakteristik:</b> {current_config['characteristics']}\n\n"
        f"Pilih mode strategy di bawah ini:"
    )

    await update.message.reply_text(message, reply_markup=reply_markup, parse_mode="HTML")

# Fungsi untuk handle callback dari inline keyboard
async def handle_filter_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    await query.answer()

    if data.startswith("filter_mode_"):
        new_mode = data.replace("filter_mode_", "")
        set_user_filter_mode(user_id, new_mode)

        mode_config = FILTER_MODES[new_mode]

        message = (
            f"✅ <b>FILTER MODE BERHASIL DIUBAH</b>\n\n"
            f"<b>Mode Baru:</b> {mode_config['name']}\n"
            f"<b>Deskripsi:</b> {mode_config['description']}\n\n"
            f"<b>Kriteria Baru:</b>\n"
            f"• Confidence: >{mode_config['min_confidence']*100:.0f}%\n"
            f"• Min Supporting Factors: {mode_config['min_supporting_factors']}\n"
            f"• Max Conflicting Factors: {mode_config['max_conflicting_factors']}\n\n"
            f"<b>Karakteristik:</b> {mode_config['characteristics']}\n\n"
            f"🔔 Alert akan disesuaikan dengan filter baru Anda!"
        )

        await query.edit_message_text(message, parse_mode="HTML")

    elif data == "filter_comparison":
        message = (
            f"📊 <b>PERBANDINGAN MODE FILTER</b>\n\n"
            f"<b>🛡️ Safe Mode:</b>\n"
            f"• Sinyal/Hari: 1-3\n"
            f"• Akurasi: Tinggi\n"
            f"• Risk Level: Rendah\n"
            f"• Confidence: >80%\n\n"
            f"<b>⚖️ Balanced Mode:</b>\n"
            f"• Sinyal/Hari: 3-6\n"
            f"• Akurasi: Sedang\n"
            f"• Risk Level: Sedang\n"
            f"• Confidence: >60%\n\n"
            f"<b>⚡ Aggressive Mode:</b>\n"
            f"• Sinyal/Hari: 6-12\n"
            f"• Akurasi: Rendah\n"
            f"• Risk Level: Tinggi\n"
            f"• Confidence: >40%\n\n"
            f"💡 <i>Pilih sesuai dengan risk tolerance Anda!</i>"
        )

        await query.edit_message_text(message, parse_mode="HTML")

    elif data == "test_current_filter":
        current_mode = get_user_filter_mode(user_id)
        prediction_data = get_latest_prediction()

        if prediction_data:
            is_pass = check_filter_criteria(prediction_data, current_mode)
            mode_config = FILTER_MODES[current_mode]

            model = prediction_data['model']
            smc = prediction_data.get('smc', {})

            # Parse supporting dan conflicting factors
            supporting_factors = smc.get('supporting_factors', '')
            conflicting_factors = smc.get('conflicting_factors', '')

            if isinstance(supporting_factors, str):
                supporting_count = len([f for f in supporting_factors.split(',') if f.strip()]) if supporting_factors else 0
            else:
                supporting_count = len(supporting_factors) if supporting_factors else 0

            if isinstance(conflicting_factors, str):
                conflicting_count = len([f for f in conflicting_factors.split(',') if f.strip()]) if conflicting_factors else 0
            else:
                conflicting_count = len(conflicting_factors) if conflicting_factors else 0

            status_emoji = "✅ LOLOS" if is_pass else "❌ TIDAK LOLOS"

            message = (
                f"🧪 <b>TEST FILTER: {mode_config['name']}</b>\n\n"
                f"<b>Status:</b> {status_emoji}\n\n"
                f"<b>Data Prediksi Terbaru:</b>\n"
                f"• Signal: {model['signal'].upper()}\n"
                f"• Model Confidence: {model['confidence']*100:.1f}%\n"
                f"• Supporting Factors: {supporting_count}\n"
                f"• Conflicting Factors: {conflicting_count}\n\n"
                f"<b>Filter Criteria:</b>\n"
                f"• Min Confidence: {mode_config['min_confidence']*100:.0f}%\n"
                f"• Min Supporting: {mode_config['min_supporting_factors']}\n"
                f"• Max Conflicting: {mode_config['max_conflicting_factors']}\n\n"
            )

            if is_pass:
                message += "🎉 Prediksi ini akan dikirim sebagai alert!"
            else:
                message += "⏸️ Prediksi ini tidak akan dikirim sebagai alert."

        else:
            message = "❌ Tidak ada data prediksi terbaru untuk ditest."

        await query.edit_message_text(message, parse_mode="HTML")

# Fungsi untuk menampilkan hasil prediksi
async def btc_prediction(update: Update, context: CallbackContext):
    prediction_data = get_latest_prediction()

    if prediction_data:
        model = prediction_data['model']

        timestamp = pd.to_datetime(model['timestamp'])
        formatted_time = timestamp.strftime("%d %b %Y, %H:%M")

        signal_emoji = {
            'down': '🔽',
            'up': '🔼',
            'hold': '↔️'
        }.get(model['signal'].lower(), '⚠️')

        confidence_pct = f"{model['confidence']*100:.1f}%"

        message = f"🔮 <b>HASIL PREDIKSI BTC</b> 🔮\n\n"
        message += f"📅 <b>Update Terakhir:</b> {formatted_time}\n\n"
        message += f"📊 <b>Prediksi Model:</b> {signal_emoji} {model['signal'].upper()} (Confidence: {confidence_pct})\n\n"

        message += f"<b>Probabilitas:</b>\n"
        message += f"• Up: {model['probabilities']['up']*100:.1f}%\n"
        message += f"• Down: {model['probabilities']['down']*100:.1f}%\n"
        message += f"• Hold: {model['probabilities']['hold']*100:.1f}%\n\n"

        message += f"💰 <b>Close Price:</b> {format_price(model['close_price'])}\n"
        message += f"📏 <b>Sequence:</b> {model['sequence_length']} candles\n"
        message += f"🤖 <b>Model Type:</b> {model['model_type']}"

        await update.message.reply_text(message, parse_mode="HTML")
    else:
        await update.message.reply_text("❌ Belum ada prediksi terbaru yang tersedia.")

# Fungsi untuk menampilkan signal SMC
async def smc_signal(update: Update, context: CallbackContext):
    prediction_data = get_latest_prediction()

    if prediction_data and prediction_data.get('smc'):
        smc = prediction_data['smc']

        timestamp = smc.get('timestamp', 'N/A')
        if timestamp != 'N/A':
            timestamp = pd.to_datetime(timestamp).strftime("%d %b %Y, %H:%M")

        signal_emoji = {
            'buy': '🔼',
            'sell': '🔽',
            'no trade': '⏸️'
        }.get(smc['smc_signal'].lower(), '⚠️')

        message = f"🎯 <b>SIGNAL SMC TERBARU</b> 🎯\n\n"
        message += f"📅 <b>Update:</b> {timestamp}\n\n"
        message += f"📊 <b>Signal:</b> {signal_emoji} {smc['smc_signal'].upper()}\n"
        message += f"💰 <b>Entry Price:</b> {format_price(smc['entry_price'])}\n"
        message += f"🛑 <b>Stop Loss:</b> {format_price(smc['stop_loss'])}\n\n"

        message += f"<b>Take Profit Levels:</b>\n"
        message += f"• TP1: {format_price(smc['take_profit_1'])}\n"
        message += f"• TP2: {format_price(smc['take_profit_2'])}\n"
        message += f"• TP3: {format_price(smc['take_profit_3'])}\n\n"

        message += f"📈 <b>Risk/Reward:</b> {smc['risk_reward']:.2f}\n"
        message += f"⭐ <b>Setup Quality:</b> {smc['setup_quality']}\n"
        message += f"🎯 <b>Confidence:</b> {smc['confidence']:.1f}%\n"
        message += f"📋 <b>Setup Type:</b> {smc['setup_type']}\n\n"

        # Parse supporting factors
        supporting_factors = smc.get('supporting_factors', '')
        if supporting_factors:
            message += f"<b>✅ Supporting Factors:</b>\n"
            if isinstance(supporting_factors, str):
                factors = [f.strip() for f in supporting_factors.split(',') if f.strip()]
                for factor in factors:
                    message += f"• {factor}\n"
            message += "\n"

        # Parse conflicting factors
        conflicting_factors = smc.get('conflicting_factors', '')
        if conflicting_factors:
            message += f"<b>❌ Conflicting Factors:</b>\n"
            if isinstance(conflicting_factors, str):
                factors = [f.strip() for f in conflicting_factors.split(',') if f.strip()]
                for factor in factors:
                    message += f"• {factor}\n"
            message += "\n"

        message += f"⚠️ <b>Invalidation Level:</b> {format_price(smc['invalidation_level'])}"

        await update.message.reply_text(message, parse_mode="HTML")
    else:
        await update.message.reply_text("❌ Belum ada signal SMC yang tersedia.")

# Fungsi untuk membuat pesan alert otomatis
def create_alert_message(prediction_data, current_price, is_periodic=False):
    """Buat pesan alert untuk semua user"""
    global last_sent_data

    message_parts = []
    has_new_data = False

    current_time = datetime.now().strftime("%d %b %Y, %H:%M")
    alert_type = "ALERT BERKALA" if is_periodic else "ALERT REAL-TIME"

    message_parts.append(f"🚨 <b>{alert_type}</b> 🚨")
    message_parts.append(f"📅 {current_time}\n")

    # Informasi harga
    if current_price:
        price_change = ""
        if last_sent_data['price'] and current_price != last_sent_data['price']:
            change = current_price - last_sent_data['price']
            change_pct = (change / last_sent_data['price']) * 100
            if abs(change_pct) >= 0.5:
                price_change = f" ({change:+,.2f}, {change_pct:+.1f}%)"
                has_new_data = True

        message_parts.append(f"💰 <b>Harga BTC:</b> {format_price(current_price)}{price_change}\n")

    # Informasi prediksi jika ada yang baru
    if prediction_data:
        model = prediction_data['model']
        timestamp = pd.to_datetime(model['timestamp'])

        if (not last_sent_data['prediction_timestamp'] or
            timestamp > pd.to_datetime(last_sent_data['prediction_timestamp'])):

            signal_emoji = {
                'down': '🔽',
                'up': '🔼',
                'hold': '↔️'
            }.get(model['signal'].lower(), '⚠️')

            confidence_pct = f"{model['confidence']*100:.1f}%"

            message_parts.append(f"🔮 <b>Prediksi Baru:</b> {signal_emoji} {model['signal'].upper()}")
            message_parts.append(f"📊 <b>Confidence:</b> {confidence_pct}")

            smc = prediction_data.get('smc', {})
            if smc:
                supporting_factors = smc.get('supporting_factors', '')
                conflicting_factors = smc.get('conflicting_factors', '')

                if isinstance(supporting_factors, str):
                    supporting_count = len([f for f in supporting_factors.split(',') if f.strip()]) if supporting_factors else 0
                else:
                    supporting_count = len(supporting_factors) if supporting_factors else 0

                if isinstance(conflicting_factors, str):
                    conflicting_count = len([f for f in conflicting_factors.split(',') if f.strip()]) if conflicting_factors else 0
                else:
                    conflicting_count = len(conflicting_factors) if conflicting_factors else 0

                message_parts.append(f"🎯 SMC: +{supporting_count} -{conflicting_count}")

            probs = model['probabilities']
            message_parts.append(f"📈 Up: {probs['up']*100:.0f}% | Down: {probs['down']*100:.0f}% | Hold: {probs['hold']*100:.0f}%")

            has_new_data = True
            last_sent_data['prediction_timestamp'] = model['timestamp']

    if current_price:
        last_sent_data['price'] = current_price

    # Untuk alert berkala, selalu kirim jika ada data
    if is_periodic and len(message_parts) > 2:
        return "\n".join(message_parts)

    # Untuk alert real-time, hanya kirim jika ada data baru
    return "\n".join(message_parts) if has_new_data else None

# Fungsi untuk mengirim alert ke semua subscriber
async def send_alert_to_subscribers(prediction_data, current_price, alert_type="regular"):
    global bot_app

    if not bot_app:
        return 0

    users_data = load_users_data()

    if not users_data:
        return 0

    # Buat pesan alert
    alert_message = create_alert_message(
        prediction_data,
        current_price,
        is_periodic=(alert_type == "hourly")
    )

    if not alert_message:
        return 0

    active_users = [user_data for user_data in users_data.values() if user_data.get('active', True)]
    sent_count = 0

    for user_data in active_users:
        try:
            await bot_app.bot.send_message(
                chat_id=user_data['chat_id'],
                text=alert_message,
                parse_mode="HTML"
            )
            sent_count += 1
        except Exception as e:
            print(f"Error sending alert to {user_data['chat_id']}: {e}")

    print(f"✅ {alert_type.title()} alert sent to {sent_count} users")
    return sent_count

# Fungsi untuk mengirim broadcast message admin
async def send_broadcast_text(message):
    global bot_app

    if not bot_app:
        return 0

    users_data = load_users_data()

    if not users_data:
        return 0

    active_users = [user_data for user_data in users_data.values() if user_data.get('active', True)]
    sent_count = 0

    for user_data in active_users:
        try:
            await bot_app.bot.send_message(
                chat_id=user_data['chat_id'],
                text=message,
                parse_mode="HTML"
            )
            sent_count += 1
        except Exception as e:
            print(f"Error sending broadcast to {user_data['chat_id']}: {e}")

    print(f"✅ Broadcast sent to {sent_count} users")
    return sent_count

async def send_broadcast_photo(photo_file_id, caption):
    """Kirim foto dengan caption ke semua subscriber"""
    global bot_app

    if not bot_app:
        return 0

    users_data = load_users_data()

    if not users_data:
        return 0

    active_users = [user_data for user_data in users_data.values() if user_data.get('active', True)]
    sent_count = 0

    for user_data in active_users:
        try:
            await bot_app.bot.send_photo(
                chat_id=user_data['chat_id'],
                photo=photo_file_id,
                caption=caption,
                parse_mode="HTML"
            )
            sent_count += 1
        except Exception as e:
            print(f"Error sending photo to {user_data['chat_id']}: {e}")

    print(f"✅ Broadcast photo sent to {sent_count} users")
    return sent_count

# Fungsi untuk mengirim alert berkala setiap jam
async def send_hourly_alert():
    try:
        current_price = get_latest_price()
        prediction_data = get_latest_prediction()

        if current_price or prediction_data:
            await send_alert_to_subscribers(prediction_data, current_price, "hourly")
        else:
            print("ℹ️ No data for hourly alert")

    except Exception as e:
        print(f"❌ Error in hourly alert: {e}")

# Fungsi untuk monitoring data baru secara real-time
async def monitor_new_data():
    """Monitor database untuk data baru dan kirim alert langsung"""
    global last_known_prediction_timestamp, last_known_price, monitoring_active

    while monitoring_active:
        try:
            current_price = get_latest_price()
            prediction_data = get_latest_prediction()

            new_data_detected = False

            if prediction_data:
                current_prediction_timestamp = pd.to_datetime(prediction_data['model']['timestamp'])
                if (last_known_prediction_timestamp is None or
                    current_prediction_timestamp > last_known_prediction_timestamp):
                    last_known_prediction_timestamp = current_prediction_timestamp
                    new_data_detected = True
                    print(f"🔮 New prediction detected: {prediction_data['model']['signal']}")

            if current_price and last_known_price:
                price_change_pct = abs((current_price - last_known_price) / last_known_price) * 100
                if price_change_pct >= 1.0:
                    new_data_detected = True
                    print(f"💰 Significant price change detected: {price_change_pct:.2f}%")

            last_known_price = current_price

            if new_data_detected:
                await send_alert_to_subscribers(prediction_data, current_price, "real-time")

            await asyncio.sleep(30)

        except Exception as e:
            print(f"❌ Error in data monitoring: {e}")
            await asyncio.sleep(60)

async def hourly_alert_loop():
    """Loop untuk alert berkala setiap jam"""
    while monitoring_active:
        try:
            await asyncio.sleep(3600)
            await send_hourly_alert()
        except Exception as e:
            print(f"❌ Error in hourly alert loop: {e}")
            await asyncio.sleep(60)

async def realtime_monitor_loop():
    """Loop untuk monitoring real-time"""
    while monitoring_active:
        try:
            await monitor_new_data()
        except Exception as e:
            print(f"❌ Error in realtime monitor loop: {e}")
            await asyncio.sleep(60)

async def schedule_background_tasks():
    """Schedule background tasks menggunakan asyncio tasks"""
    global bot_app

    if not bot_app:
        return

    print("🔄 Starting background tasks...")

    hourly_task = asyncio.create_task(hourly_alert_loop())
    monitor_task = asyncio.create_task(realtime_monitor_loop())

    await asyncio.gather(hourly_task, monitor_task, return_exceptions=True)

# Fungsi untuk menangani pesan broadcast dari admin
async def handle_admin_broadcast_message(update: Update, context: CallbackContext):
    user_id = update.effective_user.id

    if user_id not in admin_broadcast_mode:
        return False

    del admin_broadcast_mode[user_id]

    try:
        if update.message.photo:
            photo = update.message.photo[-1]
            caption = update.message.caption or ""
            broadcast_caption = f"📢 <b>PESAN DARI ADMIN</b>\n\n{caption}"

            sent_count = await send_broadcast_photo(photo.file_id, broadcast_caption)
            await update.message.reply_text(
                f"✅ Gambar berhasil dikirim ke {sent_count} subscriber!",
                parse_mode="HTML"
            )
        else:
            broadcast_message = f"📢 <b>PESAN DARI ADMIN</b>\n\n{update.message.text}"
            sent_count = await send_broadcast_text(broadcast_message)
            await update.message.reply_text(
                f"✅ Pesan berhasil dikirim ke {sent_count} subscriber!",
                parse_mode="HTML"
            )
    except Exception as e:
        await update.message.reply_text(f"❌ Error mengirim broadcast: {e}")

    return True

# Fungsi untuk menangani semua pesan text
async def handle_message(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    text = update.message.text

    if user_id in admin_broadcast_mode:
        await handle_admin_broadcast_message(update, context)
        return

    if text == "📊 Cek Harga BTC":
        await btc_price(update, context)
    elif text == "🔮 Hasil Prediksi":
        await btc_prediction(update, context)
    elif text == "📊 Signal SMC":
        await smc_signal(update, context)
    elif text == "⚙️ Pengaturan Alert":
        await alert_settings(update, context)
    elif text == "ℹ️ Status Subscription":
        await status_subscription(update, context)
    elif text == "📢 Broadcast Message (Admin)" and is_admin(user_id):
        await admin_broadcast(update, context)
    elif text == "📊 Stats Admin" and is_admin(user_id):
        await admin_stats(update, context)
    else:
        await update.message.reply_text(
            "🤔 Maaf, saya tidak mengerti perintah tersebut.\n"
            "Gunakan menu di bawah atau ketik /help untuk bantuan."
        )

# Fungsi help
async def help_command(update: Update, context: CallbackContext):
    help_text = (
        "📖 <b>BANTUAN BOT BTC TRADING</b>\n\n"
        "<b>Perintah Dasar:</b>\n"
        "/start - Mulai bot dan aktifkan alert\n"
        "/help - Tampilkan bantuan ini\n"
        "/status - Cek status subscription\n"
        "/enable_alert - Aktifkan alert otomatis\n"
        "/disable_alert - Nonaktifkan alert otomatis\n"
        "/force_alert - Dapatkan data terbaru sekarang\n\n"
        "<b>Fitur Alert Otomatis:</b>\n"
        "🔔 Alert real-time untuk prediksi baru\n"
        "📊 Alert perubahan harga signifikan\n"
        "⏰ Alert berkala setiap jam\n\n"
        "<b>Data yang Tersedia:</b>\n"
        "💰 Harga BTC terkini\n"
        "🔮 Prediksi trading (Up/Down/Hold)\n"
        "🎯 Signal SMC dengan entry/SL/TP\n"
        "📈 Analisis teknikal\n"
        "📊 Confidence score\n\n"
        "Gunakan menu di bawah untuk navigasi mudah!"
    )

    user_id = update.effective_user.id
    if is_admin(user_id):
        help_text += (
            "\n<b>Perintah Admin:</b>\n"
            "/broadcast - Kirim pesan ke semua user\n"
            "/stats - Lihat statistik user\n"
            "/cancel - Batalkan mode broadcast"
        )

    await update.message.reply_text(help_text, parse_mode="HTML")

# Fungsi untuk menangani error
async def error_handler(update: object, context: CallbackContext):
    """Log error dan kirim pesan error ke admin jika perlu"""
    import traceback

    error_msg = f"❌ Error occurred: {context.error}"

    print(f"Error: {error_msg}")
    print(f"Traceback: {traceback.format_exc()}")

    if ADMIN_USER_ID and hasattr(update, 'effective_chat'):
        try:
            await context.bot.send_message(
                chat_id=ADMIN_USER_ID,
                text=f"🚨 <b>BOT ERROR</b>\n\n{error_msg}",
                parse_mode="HTML"
            )
        except:
            pass

# Fungsi untuk inisialisasi dan menjalankan bot
def start_bot():
    global bot_app

    if not TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN tidak ditemukan di file .env")
        return

    print("🚀 Starting BTC Trading Bot...")
    print(f"👤 Admin User ID: {ADMIN_USER_ID}")

    bot_app = Application.builder().token(TOKEN).build()

    bot_app.add_handler(CommandHandler("start", start))
    bot_app.add_handler(CommandHandler("help", help_command))
    bot_app.add_handler(CommandHandler("status", status_subscription))
    bot_app.add_handler(CommandHandler("enable_alert", enable_alert))
    bot_app.add_handler(CommandHandler("disable_alert", disable_alert))
    bot_app.add_handler(CommandHandler("force_alert", force_alert))
    bot_app.add_handler(CommandHandler("broadcast", admin_broadcast))
    bot_app.add_handler(CommandHandler("stats", admin_stats))
    bot_app.add_handler(CommandHandler("cancel", cancel_broadcast))

    bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    bot_app.add_handler(MessageHandler(filters.PHOTO, handle_message))

    bot_app.add_error_handler(error_handler)

    print("🔔 Starting background alert services...")

    async def post_init(application):
        asyncio.create_task(schedule_background_tasks())

    bot_app.post_init = post_init

    print("✅ Bot started successfully!")
    print("🔔 Hourly alerts: ACTIVE")
    print("🚀 Real-time monitoring: ACTIVE")
    print("📊 Admin features: ENABLED" if ADMIN_USER_ID else "📊 Admin features: DISABLED")

    try:
        bot_app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        global monitoring_active
        monitoring_active = False
        print("🧹 Cleaning up...")

def stop_bot():
    global monitoring_active, bot_app

    print("🛑 Shutting down bot...")
    monitoring_active = False

    if bot_app:
        try:
            if hasattr(bot_app, 'stop'):
                bot_app.stop()
            if hasattr(bot_app, 'shutdown'):
                bot_app.shutdown()
        except Exception as e:
            print(f"⚠️ Error during shutdown: {e}")

    print("✅ Bot stopped successfully")

if __name__ == "__main__":
    try:
        start_bot()
    except KeyboardInterrupt:
        stop_bot()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        stop_bot()
