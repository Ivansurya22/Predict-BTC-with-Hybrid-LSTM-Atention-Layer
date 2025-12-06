import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime
import io
import os
from pathlib import Path

DB_PATH = "data/btc_ohlcv.db"
CHART_DIR = "temp_charts"


def ensure_chart_dir():
    """Ensure chart directory exists"""
    Path(CHART_DIR).mkdir(exist_ok=True)


def generate_price_chart(hours=24, save_path=None):
    """
    Generate candlestick chart untuk BTC price

    Args:
        hours: Number of hours to display (default: 24)
        save_path: Path to save chart (if None, auto-generate)

    Returns:
        str: Path to saved chart image
    """
    try:
        # Ensure directory exists
        ensure_chart_dir()

        # Load data from database
        with sqlite3.connect(DB_PATH) as conn:
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM btc_1h
                ORDER BY timestamp DESC
                LIMIT {hours}
            """
            df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])

        if df.empty:
            print("❌ No data available for chart")
            return None

        # Reverse to chronological order
        df = df.iloc[::-1].reset_index(drop=True)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(12, 8),
            gridspec_kw={'height_ratios': [3, 1]},
            facecolor='#0E1117'
        )

        # Style configuration
        plt.style.use('dark_background')

        # === CANDLESTICK CHART ===
        width = 0.6
        width2 = 0.05

        up = df[df.close >= df.open]
        down = df[df.close < df.open]

        # Bullish candles (green)
        ax1.bar(up.index, up.close - up.open, width, bottom=up.open, color='#26A69A')
        ax1.bar(up.index, up.high - up.close, width2, bottom=up.close, color='#26A69A')
        ax1.bar(up.index, up.open - up.low, width2, bottom=up.low, color='#26A69A')

        # Bearish candles (red)
        ax1.bar(down.index, down.close - down.open, width, bottom=down.open, color='#EF5350')
        ax1.bar(down.index, down.high - down.open, width2, bottom=down.open, color='#EF5350')
        ax1.bar(down.index, down.close - down.low, width2, bottom=down.low, color='#EF5350')

        # === STYLING ===
        ax1.set_facecolor('#0E1117')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Price range and current price
        current_price = df['close'].iloc[-1]
        price_min = df['low'].min()
        price_max = df['high'].max()

        # Add current price line
        ax1.axhline(y=current_price, color='#FFA500', linestyle='--', linewidth=1, alpha=0.7)
        ax1.text(
            len(df)-1, current_price,
            f' ${current_price:,.2f}',
            verticalalignment='center',
            color='#FFA500',
            fontweight='bold',
            fontsize=10
        )

        # Calculate price change
        price_change = df['close'].iloc[-1] - df['close'].iloc[0]
        price_change_pct = (price_change / df['close'].iloc[0]) * 100
        change_color = '#26A69A' if price_change >= 0 else '#EF5350'
        change_symbol = '+' if price_change >= 0 else ''

        # Title
        ax1.set_title(
            f'BTC/USDT - {hours}H Chart\n'
            f'${current_price:,.2f} ({change_symbol}{price_change_pct:,.2f}%)',
            fontsize=14,
            fontweight='bold',
            color=change_color,
            pad=20
        )

        ax1.set_ylabel('Price (USDT)', fontsize=11, fontweight='bold')
        ax1.set_xlim(-1, len(df))

        # Format y-axis
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # === VOLUME CHART ===
        colors = ['#26A69A' if df['close'].iloc[i] >= df['open'].iloc[i] else '#EF5350'
                  for i in range(len(df))]
        ax2.bar(df.index, df['volume'], color=colors, alpha=0.7)

        ax2.set_facecolor('#0E1117')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.set_ylabel('Volume (BTC)', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Time', fontsize=10, fontweight='bold')

        # Format volume axis
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

        # === X-AXIS LABELS ===
        # Show time labels every N hours
        step = max(1, hours // 6)  # Show ~6 labels
        x_labels = []
        x_positions = []

        for i in range(0, len(df), step):
            x_positions.append(i)
            timestamp = df['timestamp'].iloc[i]
            x_labels.append(timestamp.strftime('%m/%d %H:%M'))

        # Add last timestamp
        if len(df) - 1 not in x_positions:
            x_positions.append(len(df) - 1)
            x_labels.append(df['timestamp'].iloc[-1].strftime('%m/%d %H:%M'))

        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
        ax2.set_xlim(-1, len(df))

        # === ADD STATS BOX ===
        stats_text = (
            f"High: ${price_max:,.2f}\n"
            f"Low: ${price_min:,.2f}\n"
            f"Volume: {df['volume'].sum()/1000:.1f}K BTC"
        )

        ax1.text(
            0.02, 0.98, stats_text,
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            fontsize=9,
            color='white'
        )

        # === TIMESTAMP ===
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        fig.text(
            0.99, 0.01, timestamp_text,
            ha='right', va='bottom',
            fontsize=8, color='gray',
            alpha=0.7
        )

        # Tight layout
        plt.tight_layout()

        # Save chart
        if save_path is None:
            save_path = os.path.join(CHART_DIR, f"btc_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0E1117')
        plt.close()

        print(f"✅ Chart saved: {save_path}")
        return save_path

    except Exception as e:
        print(f"❌ Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_prediction_chart(hours=24, predictions=None):
    """
    Generate chart dengan prediction markers

    Args:
        hours: Number of hours to display
        predictions: DataFrame with predictions to overlay

    Returns:
        str: Path to saved chart
    """
    try:
        ensure_chart_dir()

        # Load OHLCV data
        with sqlite3.connect(DB_PATH) as conn:
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM btc_1h
                ORDER BY timestamp DESC
                LIMIT {hours}
            """
            df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])

        if df.empty:
            return None

        df = df.iloc[::-1].reset_index(drop=True)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0E1117')
        plt.style.use('dark_background')

        # Plot candlesticks (simplified for performance)
        width = 0.6

        up = df[df.close >= df.open]
        down = df[df.close < df.open]

        ax.bar(up.index, up.close - up.open, width, bottom=up.open, color='#26A69A', alpha=0.8)
        ax.bar(down.index, down.close - down.open, width, bottom=down.open, color='#EF5350', alpha=0.8)

        # Add prediction markers if provided
        if predictions is not None and not predictions.empty:
            for _, pred in predictions.iterrows():
                # Find matching index in df
                pred_time = pd.to_datetime(pred['timestamp'])
                matching = df[df['timestamp'] == pred_time]

                if not matching.empty:
                    idx = matching.index[0]
                    price = matching['close'].iloc[0]

                    # Marker style based on signal
                    if pred['aligned']:
                        if pred['model_signal'] == 'UP':
                            marker = '^'
                            color = '#00FF00'
                            label = 'BUY'
                        else:
                            marker = 'v'
                            color = '#FF0000'
                            label = 'SELL'

                        ax.scatter(idx, price, marker=marker, s=200, c=color,
                                  edgecolors='white', linewidths=2, zorder=5)
                        ax.text(idx, price, f' {label}', fontsize=9,
                               fontweight='bold', color=color, va='center')

        # Styling
        ax.set_facecolor('#0E1117')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        current_price = df['close'].iloc[-1]
        ax.set_title(
            f'BTC/USDT with Predictions - {hours}H\nCurrent: ${current_price:,.2f}',
            fontsize=14, fontweight='bold', pad=20
        )

        ax.set_ylabel('Price (USDT)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time', fontsize=11, fontweight='bold')

        # Format axes
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # X-axis labels
        step = max(1, hours // 8)
        x_positions = list(range(0, len(df), step))
        x_labels = [df['timestamp'].iloc[i].strftime('%m/%d %H:%M') for i in x_positions]

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        plt.tight_layout()

        save_path = os.path.join(CHART_DIR, f"btc_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0E1117')
        plt.close()

        return save_path

    except Exception as e:
        print(f"❌ Error generating prediction chart: {e}")
        return None


def cleanup_old_charts(max_age_hours=24):
    """Clean up old chart files"""
    try:
        if not os.path.exists(CHART_DIR):
            return

        current_time = datetime.now()

        for filename in os.listdir(CHART_DIR):
            filepath = os.path.join(CHART_DIR, filename)

            if filename.endswith('.png'):
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                age_hours = (current_time - file_time).total_seconds() / 3600

                if age_hours > max_age_hours:
                    try:
                        os.remove(filepath)
                        print(f"🗑️ Deleted old chart: {filename}")
                    except:
                        pass

    except Exception as e:
        print(f"⚠️ Error cleaning up charts: {e}")


if __name__ == "__main__":
    # Test chart generation
    print("=" * 60)
    print("📊 Testing Chart Generation")
    print("=" * 60)

    print("\n1️⃣ Generating 24h price chart...")
    chart_path = generate_price_chart(hours=24)

    if chart_path:
        print(f"✅ Chart generated: {chart_path}")
    else:
        print("❌ Failed to generate chart")

    print("\n2️⃣ Testing cleanup...")
    cleanup_old_charts(max_age_hours=1)

    print("\n" + "=" * 60)
    print("✅ Chart generation test completed!")
    print("=" * 60)
