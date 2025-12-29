import os
import sys
import time
import signal
import logging
import threading
import subprocess
import queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import sqlite3

# ============================================================================
# SETUP PATHS & LOGGING
# ============================================================================

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
BOT_DIR = ROOT_DIR / "bot"
APP_DIR = ROOT_DIR / "application"
MODELS_DIR = ROOT_DIR / "models"
STRATEGY_DIR = ROOT_DIR / "strategy"

sys.path.insert(0, str(ROOT_DIR))

# Logging setup
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"orchestrator_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Orchestrator")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for orchestrator"""

    # Database
    DB_PATH = ROOT_DIR / "data" / "btc_ohlcv.db"

    # Check intervals
    DATA_CHECK_INTERVAL = 60  # Check for new data every 60 seconds
    PIPELINE_COOLDOWN = 300   # Wait 5 minutes before next pipeline run

    # Pipeline scripts
    FETCH_DATA_SCRIPT = DATA_DIR / "fetch_data.py"
    FEATURE_ENG_SCRIPT = DATA_DIR / "feature_engineering.py"
    PREPROCESSING_SCRIPT = DATA_DIR / "preprocessing.py"
    PREDICT_SCRIPT = MODELS_DIR / "predict.py"
    SMC_SCRIPT = STRATEGY_DIR / "smc.py"
    BOT_SCRIPT = BOT_DIR / "bot.py"
    APP_SCRIPT = APP_DIR / "app.py"

    # Timeouts (seconds)
    FETCH_TIMEOUT = 600      # 10 minutes
    FEATURE_TIMEOUT = 1800   # 30 minutes
    PREPROCESSING_TIMEOUT = 1800  # 30 minutes
    PREDICT_TIMEOUT = 300    # 5 minutes
    SMC_TIMEOUT = 120        # 2 minutes

    # Delays between pipeline steps (seconds)
    # Memberikan waktu untuk database sync & cleanup
    DELAY_AFTER_NEW_DATA = 30      # 30s wait after new data detected (fetch selesai)
    DELAY_AFTER_FEATURE_ENG = 45   # 45s wait after feature engineering
    DELAY_AFTER_PREPROCESSING = 20  # 20s wait after preprocessing
    DELAY_AFTER_PREDICTION = 10    # 10s wait after prediction


# ============================================================================
# DATABASE MONITOR
# ============================================================================

class DatabaseMonitor:
    """Monitor database for new data"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.last_timestamp = None
        self.last_check = None

    def get_latest_timestamp(self) -> Optional[str]:
        """Get latest OHLCV timestamp from database"""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(timestamp) FROM btc_1h")
            result = cursor.fetchone()
            conn.close()

            return result[0] if result else None

        except Exception as e:
            logger.error(f"Error checking database: {e}")
            return None

    def has_new_data(self) -> bool:
        """Check if new data is available"""
        current_timestamp = self.get_latest_timestamp()

        if current_timestamp is None:
            return False

        # First time checking
        if self.last_timestamp is None:
            self.last_timestamp = current_timestamp
            self.last_check = datetime.now()
            return False

        # Check if timestamp changed
        if current_timestamp != self.last_timestamp:
            logger.info(f"✅ New data detected: {self.last_timestamp} -> {current_timestamp}")
            self.last_timestamp = current_timestamp
            self.last_check = datetime.now()
            return True

        return False

    def update_last_check(self):
        """Update last check time"""
        self.last_check = datetime.now()


# ============================================================================
# PIPELINE EXECUTOR
# ============================================================================

class PipelineExecutor:
    """Execute pipeline steps sequentially"""

    def __init__(self):
        self.is_running = False
        self.last_run = None
        self.run_count = 0

    def run_script(self, script_path: Path, timeout: int, name: str) -> bool:
        """Run a Python script and wait for completion"""

        if not script_path.exists():
            logger.error(f"❌ Script not found: {script_path}")
            return False

        logger.info(f"▶️  Running {name}...")
        start_time = time.time()

        try:
            # Run script as subprocess
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Stream output in real-time
            for line in process.stdout:
                logger.info(f"  [{name}] {line.strip()}")

            # Wait for completion with timeout
            try:
                return_code = process.wait(timeout=timeout)
                elapsed = time.time() - start_time

                if return_code == 0:
                    logger.info(f"✅ {name} completed successfully ({elapsed:.1f}s)")
                    return True
                else:
                    stderr = process.stderr.read()
                    logger.error(f"❌ {name} failed with code {return_code}")
                    logger.error(f"  Error: {stderr}")
                    return False

            except subprocess.TimeoutExpired:
                process.kill()
                logger.error(f"❌ {name} timed out after {timeout}s")
                return False

        except Exception as e:
            logger.error(f"❌ Error running {name}: {e}")
            return False

    def run_pipeline(self) -> bool:
        """Run complete pipeline sequentially"""

        if self.is_running:
            logger.warning("⚠️  Pipeline already running, skipping...")
            return False

        # Check cooldown
        if self.last_run:
            elapsed = (datetime.now() - self.last_run).total_seconds()
            if elapsed < Config.PIPELINE_COOLDOWN:
                logger.info(f"⏳ Pipeline cooldown: {Config.PIPELINE_COOLDOWN - elapsed:.0f}s remaining")
                return False

        self.is_running = True
        self.run_count += 1

        logger.info("=" * 80)
        logger.info(f"🚀 STARTING PIPELINE RUN #{self.run_count}")
        logger.info("=" * 80)

        pipeline_start = time.time()
        success = True

        # Wait after new data detected (fetch_data.py selesai)
        logger.info(f"\n⏳ Waiting {Config.DELAY_AFTER_NEW_DATA}s for database stabilization...")
        time.sleep(Config.DELAY_AFTER_NEW_DATA)

        # Step 1: Feature Engineering
        logger.info("\n📊 STEP 1/4: Feature Engineering")
        logger.info("-" * 80)
        if not self.run_script(
            Config.FEATURE_ENG_SCRIPT,
            Config.FEATURE_TIMEOUT,
            "Feature Engineering"
        ):
            logger.error("❌ Pipeline failed at Feature Engineering")
            success = False

        # Delay after Feature Engineering
        if success:
            logger.info(f"\n⏳ Waiting {Config.DELAY_AFTER_FEATURE_ENG}s before preprocessing...")
            logger.info("   (Allowing database sync & memory cleanup)")
            time.sleep(Config.DELAY_AFTER_FEATURE_ENG)

        # Step 2: Preprocessing (only if step 1 succeeded)
        if success:
            logger.info("\n🔧 STEP 2/4: Preprocessing")
            logger.info("-" * 80)
            if not self.run_script(
                Config.PREPROCESSING_SCRIPT,
                Config.PREPROCESSING_TIMEOUT,
                "Preprocessing"
            ):
                logger.error("❌ Pipeline failed at Preprocessing")
                success = False

        # Delay after Preprocessing
        if success:
            logger.info(f"\n⏳ Waiting {Config.DELAY_AFTER_PREPROCESSING}s before prediction...")
            logger.info("   (Allowing file system sync)")
            time.sleep(Config.DELAY_AFTER_PREPROCESSING)

        # Step 3: Model Prediction (only if step 2 succeeded)
        if success:
            logger.info("\n🤖 STEP 3/4: Model Prediction")
            logger.info("-" * 80)
            if not self.run_script(
                Config.PREDICT_SCRIPT,
                Config.PREDICT_TIMEOUT,
                "Model Prediction"
            ):
                logger.error("❌ Pipeline failed at Model Prediction")
                success = False

        # Delay after Prediction
        if success:
            logger.info(f"\n⏳ Waiting {Config.DELAY_AFTER_PREDICTION}s before SMC strategy...")
            time.sleep(Config.DELAY_AFTER_PREDICTION)

        # Step 4: SMC Strategy (only if step 3 succeeded)
        if success:
            logger.info("\n🎯 STEP 4/4: SMC Strategy")
            logger.info("-" * 80)
            if not self.run_script(
                Config.SMC_SCRIPT,
                Config.SMC_TIMEOUT,
                "SMC Strategy"
            ):
                logger.error("❌ Pipeline failed at SMC Strategy")
                success = False

        # Summary
        pipeline_duration = time.time() - pipeline_start
        logger.info("\n" + "=" * 80)

        if success:
            logger.info(f"✅ PIPELINE COMPLETED SUCCESSFULLY ({pipeline_duration:.1f}s)")
            logger.info(f"   Total delays: {Config.DELAY_AFTER_NEW_DATA + Config.DELAY_AFTER_FEATURE_ENG + Config.DELAY_AFTER_PREPROCESSING + Config.DELAY_AFTER_PREDICTION}s")
        else:
            logger.error(f"❌ PIPELINE FAILED ({pipeline_duration:.1f}s)")

        logger.info("=" * 80 + "\n")

        self.is_running = False
        self.last_run = datetime.now()

        return success


# ============================================================================
# LONG-RUNNING SERVICES
# ============================================================================

class ServiceManager:
    """Manage long-running services"""

    def __init__(self):
        self.services: Dict[str, subprocess.Popen] = {}
        self.should_restart = {}

    def start_service(self, name: str, script_path: Path) -> bool:
        """Start a long-running service"""

        if not script_path.exists():
            logger.error(f"❌ Service script not found: {script_path}")
            return False

        if name in self.services and self.services[name].poll() is None:
            logger.warning(f"⚠️  Service {name} already running")
            return True

        logger.info(f"🚀 Starting service: {name}")

        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            self.services[name] = process
            self.should_restart[name] = True

            logger.info(f"✅ Service {name} started (PID: {process.pid})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start service {name}: {e}")
            return False

    def stop_service(self, name: str):
        """Stop a service"""
        if name not in self.services:
            return

        process = self.services[name]
        if process.poll() is None:
            logger.info(f"🛑 Stopping service: {name}")
            process.terminate()

            try:
                process.wait(timeout=10)
                logger.info(f"✅ Service {name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️  Force killing service {name}")
                process.kill()

        self.should_restart[name] = False

    def check_services(self):
        """Check and restart services if needed"""
        for name, process in self.services.items():
            if process.poll() is not None and self.should_restart.get(name, False):
                logger.warning(f"⚠️  Service {name} died, restarting...")

                # Get script path based on name
                if name == "fetch_data":
                    script = Config.FETCH_DATA_SCRIPT
                elif name == "bot":
                    script = Config.BOT_SCRIPT
                elif name == "app":
                    script = Config.APP_SCRIPT
                else:
                    continue

                self.start_service(name, script)

    def stop_all(self):
        """Stop all services"""
        logger.info("🛑 Stopping all services...")
        for name in list(self.services.keys()):
            self.stop_service(name)


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class Orchestrator:
    """Main orchestrator for BTC Trading System"""

    def __init__(self):
        self.monitor = DatabaseMonitor(Config.DB_PATH)
        self.pipeline = PipelineExecutor()
        self.services = ServiceManager()
        self.running = False
        self.data_queue = queue.Queue()

    def start(self):
        """Start orchestrator"""

        logger.info("=" * 80)
        logger.info("🚀 BTC TRADING SYSTEM ORCHESTRATOR")
        logger.info("=" * 80)
        logger.info(f"📂 Root Directory: {ROOT_DIR}")
        logger.info(f"💾 Database: {Config.DB_PATH}")
        logger.info(f"⏱️  Data Check Interval: {Config.DATA_CHECK_INTERVAL}s")
        logger.info(f"🕐 Pipeline Cooldown: {Config.PIPELINE_COOLDOWN}s")
        logger.info("=" * 80 + "\n")

        # Verify database exists
        if not Config.DB_PATH.exists():
            logger.error(f"❌ Database not found: {Config.DB_PATH}")
            logger.info("💡 Please run fetch_data.py first to create database")
            return

        # Start long-running services
        logger.info("🔧 Starting long-running services...\n")

        # 1. Data Fetcher (always-on)
        if Config.FETCH_DATA_SCRIPT.exists():
            self.services.start_service("fetch_data", Config.FETCH_DATA_SCRIPT)
        else:
            logger.warning(f"⚠️  Fetch data script not found: {Config.FETCH_DATA_SCRIPT}")

        # 2. Telegram Bot (always-on)
        if Config.BOT_SCRIPT.exists():
            self.services.start_service("bot", Config.BOT_SCRIPT)
        else:
            logger.warning(f"⚠️  Bot script not found: {Config.BOT_SCRIPT}")

        # 3. Desktop Application (optional, always-on)
        if Config.APP_SCRIPT.exists():
            # Uncomment to auto-start desktop app
            # self.services.start_service("app", Config.APP_SCRIPT)
            logger.info(f"ℹ️  Desktop app available: {Config.APP_SCRIPT}")
            logger.info("   Run manually with: python application/app.py")

        logger.info("\n" + "=" * 80)
        logger.info("✅ ORCHESTRATOR STARTED")
        logger.info("=" * 80 + "\n")

        self.running = True

        # Start monitoring loop
        try:
            self.monitor_loop()
        except KeyboardInterrupt:
            logger.info("\n\n🛑 Shutdown signal received")
            self.stop()

    def monitor_loop(self):
        """Main monitoring loop"""

        logger.info("👀 Starting data monitoring loop...\n")

        # Initial pipeline run on startup (if data exists)
        latest_ts = self.monitor.get_latest_timestamp()
        if latest_ts:
            logger.info(f"📊 Latest data timestamp: {latest_ts}")
            logger.info("🔄 Running initial pipeline...\n")
            self.pipeline.run_pipeline()

        while self.running:
            try:
                # Check services health
                self.services.check_services()

                # Check for new data
                if self.monitor.has_new_data():
                    logger.info("🔔 New data detected, triggering pipeline...")

                    # Add to queue for sequential processing
                    self.data_queue.put(datetime.now())

                # Process pipeline queue
                if not self.data_queue.empty() and not self.pipeline.is_running:
                    self.data_queue.get()
                    self.pipeline.run_pipeline()

                # Update monitor
                self.monitor.update_last_check()

                # Sleep before next check
                time.sleep(Config.DATA_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"❌ Error in monitor loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)

    def stop(self):
        """Stop orchestrator"""
        logger.info("\n" + "=" * 80)
        logger.info("🛑 STOPPING ORCHESTRATOR")
        logger.info("=" * 80 + "\n")

        self.running = False
        self.services.stop_all()

        logger.info("\n✅ Orchestrator stopped gracefully")
        logger.info("👋 Goodbye!\n")


# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

orchestrator_instance: Optional[Orchestrator] = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global orchestrator_instance

    if orchestrator_instance:
        orchestrator_instance.stop()

    sys.exit(0)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    global orchestrator_instance

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and start orchestrator
    orchestrator_instance = Orchestrator()

    try:
        orchestrator_instance.start()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

        if orchestrator_instance:
            orchestrator_instance.stop()

        sys.exit(1)


if __name__ == "__main__":
    main()
