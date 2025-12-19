import ccxt
import asyncio
import sqlite3
import os
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv


class BTCDataFetcher:
    def __init__(self):
        # Load environment variables
        load_dotenv('config/.env')

        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_SECRET_KEY')

        if not api_key or not api_secret:
            raise ValueError("❌ API keys not found in config/.env file!")

        # Gunakan exchange dengan API key untuk data futures
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",  # Spot market
                },
                "timeout": 30000,
            }
        )

        self.db_path = "data/btc_ohlcv.db"
        self.timeframe = "1h"
        self._ensure_directory()
        self._init_db()

    def _ensure_directory(self):
        os.makedirs("data", exist_ok=True)
        os.makedirs("config", exist_ok=True)

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS btc_1h (
                timestamp TEXT PRIMARY KEY,
                open REAL, high REAL, low REAL, close REAL, volume REAL
            )
        """)
        conn.commit()
        conn.close()
        print("✓ Database initialized")

    def test_connection(self):
        try:
            self.exchange.load_markets()
            print("✓ Connection to Binance successful")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def _save_data(self, data):
        if not data:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO btc_1h
            (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            data,
        )
        conn.commit()
        conn.close()

    def _get_last_timestamp(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(timestamp) FROM btc_1h")
        result = cursor.fetchone()[0]
        conn.close()

        if result:
            # Parse timestamp dengan timezone UTC
            dt = datetime.strptime(result, "%Y-%m-%d %H:%M:%S%z")
            return int(dt.timestamp() * 1000)
        return self.exchange.parse8601("2020-01-01T00:00:00Z")

    def _format_candles(self, candles):
        """
        Format candles dengan timezone-aware timestamp.
        Format: YYYY-MM-DD HH:MM:SS+00:00
        """
        return [
            (
                # Gunakan datetime dengan timezone UTC
                datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S%z"
                ),
                c[1],
                c[2],
                c[3],
                c[4],
                c[5],
            )
            for c in candles
        ]

    def fetch_historical(self, symbol="BTC/USDT", force_from_2020=False):
        """
        Fetch historical data. Set force_from_2020=True to redownload all data from 2020.
        """
        if force_from_2020:
            print(f"\n📊 Force fetching ALL data from 2020...")
            since = self.exchange.parse8601("2020-01-01T00:00:00Z")
        else:
            since = self._get_last_timestamp()
            last_date = datetime.fromtimestamp(since / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            )
            print(f"\n📊 Fetching data from last timestamp: {last_date}...")

        current_time = self.exchange.milliseconds()
        total_candles = 0
        consecutive_errors = 0
        max_consecutive_errors = 5

        # Check if we need to fetch
        time_diff_hours = (current_time - since) / 3600000
        if time_diff_hours < 1 and not force_from_2020:
            print("✓ Data is up to date! No historical fetch needed.")
            return

        while since < current_time:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, since, 1000)

                if not ohlcv:
                    print("No more data available")
                    break

                # Filter closed candles only
                tf_ms = 3600000  # 1 hour in ms
                closed = [c for c in ohlcv if c[0] + tf_ms < current_time]

                if not closed:
                    # If no closed candles but we're at current time, we're done
                    if ohlcv and ohlcv[-1][0] + tf_ms >= current_time:
                        print("✓ Reached current time. All historical data fetched.")
                    break

                data = self._format_candles(closed)
                self._save_data(data)

                total_candles += len(data)
                since = closed[-1][0] + 1

                latest_time = datetime.fromtimestamp(
                    closed[-1][0] / 1000, tz=timezone.utc
                )
                print(
                    f"✓ Progress: {latest_time.strftime('%Y-%m-%d %H:%M')} | Total: {total_candles} candles"
                )

                consecutive_errors = 0
                time.sleep(0.5)  # Rate limit protection

            except Exception as e:
                consecutive_errors += 1
                print(
                    f"⚠ Error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}"
                )

                if consecutive_errors >= max_consecutive_errors:
                    print(f"✗ Too many errors. Stopping...")
                    break

                time.sleep(5)

        if total_candles > 0:
            print(f"\n✓ Historical fetch complete. Added {total_candles} new candles")
        else:
            print(f"\n✓ No new historical data to fetch")

    def _wait_for_next_candle(self):
        now = datetime.now(timezone.utc)
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        wait_seconds = (next_hour - now).total_seconds() + 15  # 15s buffer
        return max(wait_seconds, 15)

    async def realtime_monitor(self, symbol="BTC/USDT"):
        print("\n🔴 Starting real-time 1h OHLCV monitoring...")
        print("=" * 70)

        consecutive_errors = 0
        max_consecutive_errors = 10
        last_saved_timestamp = None

        while True:
            try:
                # Fetch latest candles
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=3)

                if not ohlcv:
                    print("⚠ No data received, retrying...")
                    await asyncio.sleep(10)
                    continue

                current_time = self.exchange.milliseconds()
                tf_ms = 3600000  # 1 hour

                # Get closed candles
                closed = [c for c in ohlcv if c[0] + tf_ms < current_time]

                if closed:
                    latest_candle = closed[-1]
                    candle_timestamp = latest_candle[0]

                    # Only save if it's a new candle
                    if candle_timestamp != last_saved_timestamp:
                        data = self._format_candles(closed)
                        self._save_data(data)
                        last_saved_timestamp = candle_timestamp

                        latest = data[-1]
                        print(f"✓ NEW CANDLE SAVED")
                        print(f"  Time   : {latest[0]} UTC")
                        print(f"  Open   : ${latest[1]:,.2f}")
                        print(f"  High   : ${latest[2]:,.2f}")
                        print(f"  Low    : ${latest[3]:,.2f}")
                        print(f"  Close  : ${latest[4]:,.2f}")
                        print(f"  Volume : {latest[5]:,.2f} BTC")
                        print("=" * 70)

                # Show current (incomplete) candle
                if ohlcv:
                    current = ohlcv[-1]
                    current_time_str = datetime.fromtimestamp(
                        current[0] / 1000, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"📊 Current candle (in progress): {current_time_str}")
                    print(
                        f"   O: ${current[1]:,.2f} | H: ${current[2]:,.2f} | L: ${current[3]:,.2f} | C: ${current[4]:,.2f}"
                    )

                consecutive_errors = 0

                # Wait for next update
                wait_time = self._wait_for_next_candle()
                next_update = datetime.now(timezone.utc) + timedelta(seconds=wait_time)
                print(
                    f"⏳ Next check at {next_update.strftime('%H:%M:%S')} UTC ({wait_time:.0f}s)\n"
                )

                await asyncio.sleep(wait_time)

            except Exception as e:
                consecutive_errors += 1
                print(
                    f"⚠ Error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}"
                )

                if consecutive_errors >= max_consecutive_errors:
                    print(f"✗ Too many errors. Stopping monitor...")
                    break

                await asyncio.sleep(10)

    def run(self):
        print("=" * 70)
        print("🚀 BTC 1H OHLCV DATA FETCHER")
        print("=" * 70)

        if not self.test_connection():
            print(
                "\n✗ Cannot connect to Binance. Please check your API keys and internet connection."
            )
            return

        # Fetch historical data
        self.fetch_historical()

        print("\n" + "=" * 70)
        print("📈 Starting real-time monitoring (updates every hour)...")
        print("=" * 70)

        # Start real-time monitoring
        try:
            asyncio.run(self.realtime_monitor())
        except KeyboardInterrupt:
            print("\n\n⏹ Stopped by user")
        except Exception as e:
            print(f"\n✗ Fatal error: {e}")


if __name__ == "__main__":
    try:
        fetcher = BTCDataFetcher()
        fetcher.run()
    except KeyboardInterrupt:
        print("\n\n⏹ Program terminated by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()
