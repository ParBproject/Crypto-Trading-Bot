"""
bot.py — Main Orchestration Loop
=================================
The CryptoBot class ties together all modules:

  DataManager → PredictorRegistry → StrategyEngine → OrderManager

Main loop:
  1. [Every tick] Fetch latest candles for all configured pairs
  2. [Every tick] Run LSTM inference → get predictions
  3. [Every tick] Run strategy → evaluate signals
  4. [Every tick] Execute approved signals via OrderManager
  5. [Periodic]   Check open positions for stop-loss / take-profit
  6. [Scheduled]  Retrain LSTM models as needed
  7. [Always]     Log state, send notifications, persist on shutdown

Usage:
    python main.py              # start in mode specified by config
    python main.py --mode paper # override mode

Graceful shutdown:
    Ctrl+C → saves positions and model state, then exits cleanly.
"""

import os
import json
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from src.data_fetcher import DataManager
from src.predictor import PredictorRegistry
from src.risk_manager import RiskManager, PortfolioState
from src.strategy import StrategyEngine, SignalType
from src.executor import OrderManager
from src.logger import get_logger, TradeJournal, NotificationDispatcher


# ─────────────────────────────────────────────────────────────
# Configuration Loader
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml", mode_override: Optional[str] = None) -> dict:
    """Load YAML config and overlay environment variables."""
    load_dotenv()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI mode override
    if mode_override:
        config.setdefault("trading", {})["mode"] = mode_override

    # Override log level from env
    env_log_level = os.getenv("LOG_LEVEL")
    if env_log_level:
        config.setdefault("logging", {})["level"] = env_log_level

    return config


# ─────────────────────────────────────────────────────────────
# State Persistence
# ─────────────────────────────────────────────────────────────

STATE_FILE = "data/bot_state.json"


def save_state(portfolio: PortfolioState) -> None:
    """Persist portfolio state to disk for crash recovery."""
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "initial_capital": portfolio.initial_capital,
        "current_capital": portfolio.current_capital,
        "peak_capital": portfolio.peak_capital,
        "open_positions": portfolio.open_positions,
        "trade_history": portfolio.trade_history[-100:],  # Last 100 trades
        "saved_at": datetime.utcnow().isoformat(),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_state(initial_capital: float) -> PortfolioState:
    """Load persisted portfolio state, or create fresh state."""
    if Path(STATE_FILE).exists():
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            portfolio = PortfolioState(
                initial_capital=data.get("initial_capital", initial_capital),
                current_capital=data.get("current_capital", initial_capital),
                peak_capital=data.get("peak_capital", initial_capital),
                open_positions=data.get("open_positions", {}),
                trade_history=data.get("trade_history", []),
            )
            return portfolio
        except Exception:
            pass
    return PortfolioState(
        initial_capital=initial_capital,
        current_capital=initial_capital,
    )


# ─────────────────────────────────────────────────────────────
# CryptoBot
# ─────────────────────────────────────────────────────────────

class CryptoBot:
    """
    Main trading bot class.

    Lifecycle:
        bot = CryptoBot(config)
        bot.start()       # Initialise, train models, begin loop
        bot.stop()        # Graceful shutdown (also called on SIGINT)
    """

    VERSION = "1.0.0"

    def __init__(self, config: dict) -> None:
        self.config = config
        self.logger = get_logger(
            "CryptoBot",
            level=config.get("logging", {}).get("level", "INFO"),
            log_dir=config.get("logging", {}).get("log_dir", "logs/"),
        )
        self._running = False

        # ── Mode ──────────────────────────────────────────────
        self.mode = config.get("trading", {}).get("mode", "paper")
        self.pairs = config.get("trading", {}).get("pairs", ["BTC/USDT"])
        self.loop_interval = config.get("trading", {}).get("loop_interval_sec", 60)

        # ── Portfolio state ───────────────────────────────────
        initial_capital = float(
            config.get("backtest", {}).get("initial_capital", 10_000)
        )
        self.portfolio = load_state(initial_capital)

        # ── Modules ───────────────────────────────────────────
        self.data_manager = DataManager(config)
        self.predictor_registry = PredictorRegistry(config)
        self.risk_manager = RiskManager(config, self.portfolio)
        self.trade_journal = TradeJournal(
            config.get("logging", {}).get("trade_log_file", "logs/trades.csv")
        )
        self.strategy_engine = StrategyEngine(config, self.risk_manager)
        self.order_manager = OrderManager(
            config,
            self.portfolio,
            ccxt_fetcher=self.data_manager.ccxt_fetcher,
            trade_journal=self.trade_journal,
        )
        self.notifier = NotificationDispatcher(config)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self.logger.info(
            f"CryptoBot v{self.VERSION} initialised | Mode: {self.mode.upper()} | "
            f"Pairs: {', '.join(self.pairs)}"
        )

    # ── Startup ───────────────────────────────────────────────

    def start(self) -> None:
        """
        Full startup sequence:
          1. Load / fetch initial data for all pairs
          2. Train LSTM models (or load saved)
          3. Enter the main loop
        """
        self.logger.info("=" * 60)
        self.logger.info(f" Starting CryptoBot [{self.mode.upper()} MODE]")
        self.logger.info("=" * 60)

        if self.mode == "live":
            self.logger.warning(
                "⚠️  LIVE TRADING MODE ACTIVE — Real funds at risk!"
            )
            self.notifier.send("🚀 CryptoBot LIVE mode started", "info")

        # ── Initial data load ──────────────────────────────────
        self.logger.info("Loading initial market data...")
        data_store = {}
        for pair in self.pairs:
            try:
                df = self.data_manager.get_enriched_ohlcv(pair)
                if not df.empty:
                    data_store[pair] = df
                    self.logger.info(
                        f"  ✓ {pair}: {len(df)} candles "
                        f"({df.index[0].strftime('%Y-%m-%d')} → "
                        f"{df.index[-1].strftime('%Y-%m-%d')})"
                    )
                else:
                    self.logger.error(f"  ✗ {pair}: No data received")
            except Exception as e:
                self.logger.error(f"  ✗ {pair}: {e}")

        if not data_store:
            self.logger.critical("No data loaded for any pair. Cannot start.")
            return

        # ── Train / load LSTM models ───────────────────────────
        self.logger.info("Initialising LSTM models...")
        self.predictor_registry.train_all(data_store)

        # ── Enter main loop ────────────────────────────────────
        self._running = True
        self._main_loop()

    def _main_loop(self) -> None:
        """
        Core trading loop. Runs until stop() is called.

        Each iteration:
          1. Update candle data
          2. Check drawdown guard
          3. Generate predictions
          4. Generate signals
          5. Execute signals / manage positions
          6. Log status
          7. Sleep until next interval
        """
        self.logger.info(f"Entering main loop (interval={self.loop_interval}s)")
        tick = 0

        while self._running:
            tick_start = time.monotonic()
            tick += 1
            self.logger.debug(f"─── Tick {tick} ───")

            try:
                # ── 1. Update market data ──────────────────────
                data_store = self._refresh_data()

                # ── 2. Drawdown guard ──────────────────────────
                dd = self.portfolio.drawdown_pct
                if dd >= self.config.get("risk", {}).get("max_drawdown_pct", 12):
                    self.logger.error(
                        f"⛔ Drawdown limit reached ({dd:.1f}%). "
                        f"New trades suspended. Monitor and resume manually."
                    )
                    self.notifier.send(
                        f"⛔ DRAWDOWN HALT: {dd:.1f}% drawdown", "drawdown_alert"
                    )
                    # Still manage existing positions (check stops)
                    self._manage_open_positions(data_store)
                    self._sleep(tick_start)
                    continue

                # ── 3. Periodic model retraining ───────────────
                if tick % max(1, 3600 // self.loop_interval) == 0:
                    self.logger.info("Checking if model retraining is needed...")
                    self.predictor_registry.train_all(data_store)

                # ── 4. Generate predictions ────────────────────
                predictions = self.predictor_registry.predict_all(data_store)

                # ── 5. Generate signals & execute ─────────────
                for pair in self.pairs:
                    df = data_store.get(pair)
                    if df is None or df.empty:
                        continue

                    prediction = predictions.get(pair)
                    existing_pos = self.portfolio.open_positions.get(pair)

                    signal = self.strategy_engine.evaluate(
                        symbol=pair,
                        df=df,
                        prediction=prediction,
                        existing_position=existing_pos,
                    )

                    if signal.signal_type == SignalType.LONG and existing_pos is None:
                        result = self.order_manager.open_trade(signal)
                        if result and result.success:
                            self.notifier.send(
                                f"📈 BUY {pair} @ {result.fill_price:.4f}\n"
                                f"Qty: {result.quantity:.6f}\n"
                                f"Reasons: {', '.join(signal.reasons[:3])}",
                                "trade_open",
                            )

                    elif signal.signal_type == SignalType.SHORT and existing_pos is None:
                        result = self.order_manager.open_trade(signal)
                        if result and result.success:
                            self.notifier.send(
                                f"📉 SELL {pair} @ {result.fill_price:.4f}\n"
                                f"Qty: {result.quantity:.6f}",
                                "trade_open",
                            )

                    elif signal.signal_type.value == "exit" and existing_pos:
                        current_price = self.data_manager.get_current_price(pair) or 0
                        result = self.order_manager.close_trade(
                            symbol=pair,
                            current_price=current_price,
                            reason=", ".join(signal.reasons[:2]),
                        )
                        if result and result.success:
                            self.notifier.send(
                                f"🔒 CLOSED {pair} @ {current_price:.4f}\n"
                                f"Reason: {', '.join(signal.reasons[:2])}",
                                "trade_close",
                            )

                # ── 6. Manage existing positions (stops check) ─
                self._manage_open_positions(data_store)

                # ── 7. Status log ──────────────────────────────
                if tick % max(1, 300 // self.loop_interval) == 0:
                    self._log_status()

                # ── 8. Persist state ───────────────────────────
                if tick % max(1, 600 // self.loop_interval) == 0:
                    save_state(self.portfolio)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Unhandled error in main loop: {e}", exc_info=True)
                time.sleep(5)   # Brief pause before retrying

            self._sleep(tick_start)

    def _refresh_data(self) -> dict:
        """
        Fetch updated candle data for all pairs.
        Returns a dict of {symbol: DataFrame}.
        """
        data_store = {}
        for pair in self.pairs:
            try:
                df = self.data_manager.get_enriched_ohlcv(pair)
                if not df.empty:
                    data_store[pair] = df
            except Exception as e:
                self.logger.warning(f"Data refresh failed for {pair}: {e}")
        return data_store

    def _manage_open_positions(self, data_store: dict) -> None:
        """
        Independently check all open positions for stop-loss / take-profit hits.
        This runs on every tick to ensure stops are never missed between signals.
        """
        for symbol, position in list(self.portfolio.open_positions.items()):
            current_price = self.data_manager.get_current_price(symbol)
            if current_price is None:
                continue

            side = position.get("side", "buy")
            stop_loss = position.get("stop_loss", 0)
            take_profit = position.get("take_profit", float("inf"))

            should_close = False
            reason = ""

            if side == "buy":
                if current_price <= stop_loss:
                    should_close = True
                    reason = f"Stop-loss: {current_price:.4f} ≤ {stop_loss:.4f}"
                elif current_price >= take_profit:
                    should_close = True
                    reason = f"Take-profit: {current_price:.4f} ≥ {take_profit:.4f}"
            else:  # short
                if current_price >= stop_loss:
                    should_close = True
                    reason = f"Stop-loss: {current_price:.4f} ≥ {stop_loss:.4f}"
                elif current_price <= take_profit:
                    should_close = True
                    reason = f"Take-profit: {current_price:.4f} ≤ {take_profit:.4f}"

            # Update trailing stop
            df = data_store.get(symbol)
            if df is not None and not df.empty:
                atr = float(df.iloc[-1].get("atr", current_price * 0.01))
                new_stop = self.risk_manager.calculate_trailing_stop(
                    side=side,
                    current_price=current_price,
                    entry_price=position.get("entry_price", current_price),
                    current_stop=stop_loss,
                    atr=atr,
                )
                if new_stop != stop_loss:
                    self.portfolio.open_positions[symbol]["stop_loss"] = new_stop

            if should_close:
                self.order_manager.close_trade(
                    symbol=symbol,
                    current_price=current_price,
                    reason=reason,
                )
                self.notifier.send(
                    f"🔒 AUTO-CLOSE {symbol} @ {current_price:.4f}\n{reason}",
                    "stop_loss" if "stop" in reason.lower() else "trade_close",
                )

    def _log_status(self) -> None:
        """Print a formatted status summary to the log."""
        summary = self.risk_manager.get_portfolio_summary()
        from tabulate import tabulate

        rows = [[k, v] for k, v in summary.items()]
        table = tabulate(rows, headers=["Metric", "Value"], tablefmt="simple")
        self.logger.info(f"\n📊 Portfolio Status:\n{table}")

        if self.portfolio.open_positions:
            pos_rows = [
                [sym, p["side"].upper(), p["quantity"], p["entry_price"],
                 p.get("stop_loss", 0), p.get("take_profit", 0)]
                for sym, p in self.portfolio.open_positions.items()
            ]
            pos_table = tabulate(
                pos_rows,
                headers=["Pair", "Side", "Qty", "Entry", "Stop", "TP"],
                tablefmt="simple",
            )
            self.logger.info(f"Open Positions:\n{pos_table}")

    def _sleep(self, tick_start: float) -> None:
        """Sleep for the remainder of the loop interval."""
        elapsed = time.monotonic() - tick_start
        sleep_time = max(0, self.loop_interval - elapsed)
        if sleep_time > 0:
            self.logger.debug(f"Sleeping {sleep_time:.1f}s until next tick")
            time.sleep(sleep_time)

    # ── Shutdown ──────────────────────────────────────────────

    def stop(self) -> None:
        """Graceful shutdown: save state, close positions if configured."""
        self.logger.info("CryptoBot shutting down...")
        self._running = False
        save_state(self.portfolio)
        summary = self.risk_manager.get_portfolio_summary()
        self.logger.info(f"Final portfolio state: {summary}")
        self.notifier.send(
            f"🛑 CryptoBot stopped\n"
            f"Capital: ${summary.get('current_capital_usd', 0):.2f}\n"
            f"P&L: ${summary.get('unrealised_pnl_usd', 0):+.2f}",
            "info",
        )

    def _handle_shutdown(self, signum, frame) -> None:
        """SIGINT / SIGTERM handler."""
        self.logger.info(f"Received signal {signum} — initiating graceful shutdown")
        self.stop()
        sys.exit(0)
