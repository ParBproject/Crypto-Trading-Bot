"""
logger.py — Structured Logging & Trade Journal
===============================================
Provides:
  - Colored console + rotating file logger
  - CSV-based trade journal for P&L tracking
  - Optional Telegram / Discord notification dispatch
"""

import csv
import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

try:
    import aiohttp
    import asyncio
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Core Logger Factory
# ─────────────────────────────────────────────────────────────

def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = "logs/",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create (or retrieve) a named logger with:
      - Colored console handler
      - Rotating file handler writing to logs/<name>.log
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ── Console handler ──────────────────────────────────────
    if COLORLOG_AVAILABLE:
        console_fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_fmt)
    else:
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_fmt)

    logger.addHandler(console_handler)

    # ── File handler ─────────────────────────────────────────
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


# ─────────────────────────────────────────────────────────────
# Trade Journal (CSV)
# ─────────────────────────────────────────────────────────────

TRADE_LOG_FIELDS = [
    "timestamp",
    "pair",
    "side",           # buy | sell
    "entry_price",
    "exit_price",
    "quantity",
    "stop_loss",
    "take_profit",
    "pnl_usd",
    "pnl_pct",
    "duration_min",
    "signal_source",  # lstm | rule | combined
    "notes",
]


class TradeJournal:
    """
    Append-only CSV trade journal.
    Thread-safe for single-process use.
    """

    def __init__(self, filepath: str = "logs/trades.csv") -> None:
        self.filepath = filepath
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()
        self.logger = get_logger("TradeJournal")

    def _ensure_header(self) -> None:
        """Write CSV header if file is new/empty."""
        if not Path(self.filepath).exists() or Path(self.filepath).stat().st_size == 0:
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
                writer.writeheader()

    def log_trade(
        self,
        pair: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        entry_time: datetime,
        exit_time: Optional[datetime] = None,
        signal_source: str = "combined",
        notes: str = "",
    ) -> dict:
        """Record a completed trade to the CSV journal."""
        exit_time = exit_time or datetime.utcnow()
        duration_min = (exit_time - entry_time).total_seconds() / 60

        # P&L calculation
        if side == "buy":
            pnl_usd = (exit_price - entry_price) * quantity
        else:
            pnl_usd = (entry_price - exit_price) * quantity

        pnl_pct = (pnl_usd / (entry_price * quantity)) * 100

        record = {
            "timestamp": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "pair": pair,
            "side": side,
            "entry_price": round(entry_price, 8),
            "exit_price": round(exit_price, 8),
            "quantity": round(quantity, 8),
            "stop_loss": round(stop_loss, 8),
            "take_profit": round(take_profit, 8),
            "pnl_usd": round(pnl_usd, 4),
            "pnl_pct": round(pnl_pct, 4),
            "duration_min": round(duration_min, 1),
            "signal_source": signal_source,
            "notes": notes,
        }

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
            writer.writerow(record)

        emoji = "✅" if pnl_usd >= 0 else "❌"
        self.logger.info(
            f"{emoji} TRADE CLOSED | {pair} {side.upper()} | "
            f"Entry: {entry_price:.4f} → Exit: {exit_price:.4f} | "
            f"P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)"
        )
        return record

    def get_summary(self) -> dict:
        """Compute summary statistics from the trade log."""
        try:
            import pandas as pd
            df = pd.read_csv(self.filepath)
            if df.empty:
                return {}
            return {
                "total_trades": len(df),
                "win_rate_pct": round((df["pnl_usd"] > 0).mean() * 100, 2),
                "total_pnl_usd": round(df["pnl_usd"].sum(), 2),
                "avg_pnl_usd": round(df["pnl_usd"].mean(), 2),
                "best_trade_usd": round(df["pnl_usd"].max(), 2),
                "worst_trade_usd": round(df["pnl_usd"].min(), 2),
                "avg_duration_min": round(df["duration_min"].mean(), 1),
            }
        except Exception as e:
            self.logger.warning(f"Could not compute trade summary: {e}")
            return {}


# ─────────────────────────────────────────────────────────────
# Notification Dispatcher
# ─────────────────────────────────────────────────────────────

class NotificationDispatcher:
    """
    Send alerts to Telegram and/or Discord webhooks.
    Designed to be called with asyncio.run() or from an async context.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.logger = get_logger("Notifier")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "")

    def send(self, message: str, event_type: str = "info") -> None:
        """Synchronous wrapper — dispatches to enabled channels."""
        notif_cfg = self.config.get("notifications", {})
        if not notif_cfg.get("telegram_enabled") and not notif_cfg.get("discord_enabled"):
            return

        enabled_events = notif_cfg.get("notify_on", [])
        if event_type not in enabled_events and event_type != "info":
            return

        if notif_cfg.get("telegram_enabled") and self.telegram_token:
            self._send_telegram_sync(message)

        if notif_cfg.get("discord_enabled") and self.discord_webhook:
            self._send_discord_sync(message)

    def _send_telegram_sync(self, message: str) -> None:
        """Send via Telegram Bot API (sync requests fallback)."""
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": f"🤖 CryptoBot\n{message}",
                "parse_mode": "HTML",
            }
            resp = requests.post(url, json=payload, timeout=10)
            if not resp.ok:
                self.logger.warning(f"Telegram notification failed: {resp.text}")
        except Exception as e:
            self.logger.error(f"Telegram error: {e}")

    def _send_discord_sync(self, message: str) -> None:
        """Send via Discord webhook (sync requests)."""
        try:
            import requests
            payload = {"content": f"🤖 **CryptoBot**\n{message}"}
            resp = requests.post(self.discord_webhook, json=payload, timeout=10)
            if resp.status_code not in (200, 204):
                self.logger.warning(f"Discord notification failed: {resp.text}")
        except Exception as e:
            self.logger.error(f"Discord error: {e}")
