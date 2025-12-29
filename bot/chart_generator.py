import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
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
    Generate candlestick chart untuk BTC price (FIXED 24 JAM)

    Args:
        hours: Number of hours to display (default: 24)
        save_path: Path to save chart (if None, auto-generate)

    Returns:
        str: Path to saved chart image
    """
    try:
        # Ensure directory exists
        ensure_chart_dir()

        # Load data from database - ALWAYS GET EXACT 24 HOURS
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

        # Create figure with FIXED SIZE untuk konsistensi
        fig = plt.figure(figsize=(16, 10), facecolor='#0E1117')

        # Create subplots dengan ratio tetap
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        # Style configuration
        plt.style.use('dark_background')

        # === CANDLESTICK CHART ===
        width = 0.7  # Lebih lebar untuk visibility
        width2 = 0.08

        up = df[df.close >= df.open]
        down = df[df.close < df.open]

        # Bullish candles (green)
        ax1.bar(up.index, up.close - up.open, width, bottom=up.open,
                color='#26A69A', edgecolor='#26A69A', linewidth=0)
        ax1.bar(up.index, up.high - up.close, width2, bottom=up.close,
                color='#26A69A', linewidth=0)
        ax1.bar(up.index, up.open - up.low, width2, bottom=up.low,
                color='#26A69A', linewidth=0)

        # Bearish candles (red)
        ax1.bar(down.index, down.close - down.open, width, bottom=down.open,
                color='#EF5350', edgecolor='#EF5350', linewidth=0)
        ax1.bar(down.index, down.high - down.open, width2, bottom=down.open,
                color='#EF5350', linewidth=0)
        ax1.bar(down.index, down.close - down.low, width2, bottom=down.low,
                color='#EF5350', linewidth=0)

        # === STYLING ===
        ax1.set_facecolor('#0E1117')
        ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#333333')
        ax1.set_axisbelow(True)

        # Price range and current price
        current_price = df['close'].iloc[-1]
        price_min = df['low'].min()
        price_max = df['high'].max()

        # Add padding to y-axis untuk lebih readable
        price_range = price_max - price_min
        ax1.set_ylim(price_min - price_range * 0.05, price_max + price_range * 0.05)

        # Add current price line
        ax1.axhline(y=current_price, color='#FFA500', linestyle='--',
                   linewidth=1.5, alpha=0.8, zorder=10)

        # Price label on the right
        ax1.text(
            len(df) - 0.5, current_price,
            f' \${current_price:,.2f}',  # Escaped $ sign
            verticalalignment='center',
            color='#FFA500',
            fontweight='bold',
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0E1117',
                     edgecolor='#FFA500', linewidth=1.5)
        )

        # Calculate price change
        price_change = df['close'].iloc[-1] - df['close'].iloc[0]
        price_change_pct = (price_change / df['close'].iloc[0]) * 100
        change_color = '#26A69A' if price_change >= 0 else '#EF5350'
        change_symbol = '+' if price_change >= 0 else ''

        # Title with better formatting - ESCAPED $ SIGNS
        ax1.set_title(
            rf'BTC/USDT - 24H Chart\n'
            rf'Current: \${current_price:,.2f} ({change_symbol}{price_change_pct:.2f}% | {change_symbol}\${abs(price_change):,.2f})',
            fontsize=16,
            fontweight='bold',
            color=change_color,
            pad=20
        )

        ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold', color='#FFFFFF')
        ax1.set_xlim(-0.5, len(df) - 0.5)

        # Format y-axis dengan thousand separator - ESCAPED $ SIGN
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'\${x:,.0f}'))
        ax1.tick_params(axis='y', labelsize=10, colors='#CCCCCC')
        ax1.tick_params(axis='x', labelbottom=False)  # Hide x labels on top chart

        # === VOLUME CHART ===
        colors = ['#26A69A' if df['close'].iloc[i] >= df['open'].iloc[i] else '#EF5350'
                  for i in range(len(df))]
        ax2.bar(df.index, df['volume'], width=0.8, color=colors, alpha=0.7, linewidth=0)

        ax2.set_facecolor('#0E1117')
        ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#333333')
        ax2.set_axisbelow(True)
        ax2.set_ylabel('Volume (BTC)', fontsize=11, fontweight='bold', color='#FFFFFF')
        ax2.set_xlabel('Time (UTC)', fontsize=11, fontweight='bold', color='#FFFFFF')

        # Format volume axis
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))
        ax2.tick_params(axis='y', labelsize=9, colors='#CCCCCC')
        ax2.tick_params(axis='x', labelsize=9, colors='#CCCCCC')

        # === X-AXIS LABELS (CONSISTENT) ===
        # Always show 8 labels for 24 hours (every 3 hours)
        step = max(1, len(df) // 8)
        x_positions = list(range(0, len(df), step))

        # Make sure we have the last position
        if len(df) - 1 not in x_positions:
            x_positions.append(len(df) - 1)

        x_labels = []
        for pos in x_positions:
            if pos < len(df):
                timestamp = df['timestamp'].iloc[pos]
                # Format: MM/DD HH:MM
                x_labels.append(timestamp.strftime('%m/%d %H:%M'))

        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
        ax2.set_xlim(-0.5, len(df) - 0.5)

        # === ADD STATS BOX ===
        total_volume = df['volume'].sum()
        avg_volume = df['volume'].mean()

        stats_text = (
            f"High: \${price_max:,.2f}\n"  # Escaped $ sign
            f"Low: \${price_min:,.2f}\n"   # Escaped $ sign
            f"Range: \${price_range:,.2f}\n"  # Escaped $ sign
            f"Vol: {total_volume/1000:.1f}K BTC\n"
            f"Avg Vol: {avg_volume/1000:.1f}K"
        )

        ax1.text(
            0.02, 0.98, stats_text,
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a1a',
                     alpha=0.9, edgecolor='#444444', linewidth=1),
            fontsize=10,
            color='#FFFFFF',
            family='monospace'
        )

        # === PERIOD INFO BOX ===
        start_time = df['timestamp'].iloc[0]
        end_time = df['timestamp'].iloc[-1]

        period_text = (
            f"Period: {hours}H\n"
            f"From: {start_time.strftime('%m/%d %H:%M')}\n"
            f"To: {end_time.strftime('%m/%d %H:%M')}"
        )

        ax1.text(
            0.98, 0.98, period_text,
            transform=ax1.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a1a',
                     alpha=0.9, edgecolor='#444444', linewidth=1),
            fontsize=9,
            color='#AAAAAA',
            family='monospace'
        )

        # === TIMESTAMP ===
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        fig.text(
            0.99, 0.01, timestamp_text,
            ha='right', va='bottom',
            fontsize=8, color='#666666',
            alpha=0.8
        )

        # Tight layout
        plt.tight_layout()

        # Save chart
        if save_path is None:
            save_path = os.path.join(CHART_DIR, f"btc_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0E1117', edgecolor='none')
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
        fig, ax = plt.subplots(figsize=(16, 10), facecolor='#0E1117')
        plt.style.use('dark_background')

        # Plot candlesticks
        width = 0.7
        width2 = 0.08

        up = df[df.close >= df.open]
        down = df[df.close < df.open]

        ax.bar(up.index, up.close - up.open, width, bottom=up.open,
               color='#26A69A', alpha=0.8, linewidth=0)
        ax.bar(down.index, down.close - down.open, width, bottom=down.open,
               color='#EF5350', alpha=0.8, linewidth=0)

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

                        ax.scatter(idx, price, marker=marker, s=300, c=color,
                                  edgecolors='white', linewidths=2.5, zorder=10)
                        ax.text(idx, price, f' {label}', fontsize=10,
                               fontweight='bold', color=color, va='center')

        # Styling
        ax.set_facecolor('#0E1117')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#333333')

        current_price = df['close'].iloc[-1]
        ax.set_title(
            rf'BTC/USDT with Predictions - {hours}H\nCurrent: \${current_price:,.2f}',  # Escaped $ sign
            fontsize=16, fontweight='bold', pad=20, color='#FFFFFF'
        )

        ax.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (UTC)', fontsize=12, fontweight='bold')

        # Format axes - ESCAPED $ SIGN
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'\${x:,.0f}'))

        # X-axis labels
        step = max(1, hours // 8)
        x_positions = list(range(0, len(df), step))
        if len(df) - 1 not in x_positions:
            x_positions.append(len(df) - 1)

        x_labels = [df['timestamp'].iloc[i].strftime('%m/%d %H:%M') for i in x_positions if i < len(df)]

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_xlim(-0.5, len(df) - 0.5)

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
