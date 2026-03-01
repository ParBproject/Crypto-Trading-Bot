"""
data_fetcher.py — Market Data Acquisition
==========================================
Fetches OHLCV (Open/High/Low/Close/Volume) data from two sources:
  1. CCXT — Exchange-native live order books, trades, and OHLCV candles
  2. CoinGecko — Free, broad market data (price history, market cap, etc.)

Key design decisions:
  - CCXT is the primary source for live candle data during trading
  - CoinGecko supplements with market cap, dominance, on-chain metrics
  - A lightweight file-based cache avoids hammering APIs
  - All DataFrames are standardised: index=datetime, cols=OHLCV + indicators
"""

import os
import time
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Optional dependencies — graceful degradation if not installed
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("ccxt not installed. Exchange data unavailable.")

try:
    from pycoingecko import CoinGeckoAPI
    COINGECKO_AVAILABLE = True
except ImportError:
    COINGECKO_AVAILABLE = False
    logging.warning("pycoingecko not installed. CoinGecko data unavailable.")

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("pandas_ta not installed. Technical indicators will be limited.")

from src.logger import get_logger


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# CoinGecko coin id mapping (symbol → coingecko id)
COINGECKO_ID_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "ADA": "cardano",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "LTC": "litecoin",
}


# ─────────────────────────────────────────────────────────────
# File Cache
# ─────────────────────────────────────────────────────────────

class FileCache:
    """
    Simple JSON-based file cache keyed by query hash.
    Entries expire after `ttl_seconds`.
    """

    def __init__(self, cache_dir: str = "data/cache/") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def get(self, key: str, ttl_seconds: int = 300) -> Optional[dict]:
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            if time.time() - payload["ts"] < ttl_seconds:
                return payload["data"]
        except Exception:
            pass
        return None

    def set(self, key: str, data) -> None:
        try:
            path = self._key_path(key)
            path.write_text(json.dumps({"ts": time.time(), "data": data}))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# CCXT Exchange Wrapper
# ─────────────────────────────────────────────────────────────

class CCXTFetcher:
    """
    Wraps a ccxt exchange instance with:
      - Sandbox/testnet toggle
      - Auto-retry with exponential backoff
      - OHLCV → normalised DataFrame conversion
    """

    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 2.0   # seconds

    def __init__(self, config: dict) -> None:
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt is required for CCXTFetcher. pip install ccxt")

        self.logger = get_logger("CCXTFetcher")
        exc_cfg = config.get("exchange", {})
        exchange_id = exc_cfg.get("name", "binance")
        sandbox = exc_cfg.get("sandbox", True)
        self.rate_limit_ms = exc_cfg.get("rate_limit_ms", 500)

        # Build exchange class dynamically
        exchange_class = getattr(ccxt, exchange_id)

        api_key_env = f"{exchange_id.upper()}_API_KEY"
        secret_env = f"{exchange_id.upper()}_SECRET"

        # If sandbox, try testnet credentials first
        if sandbox:
            api_key_env = f"{exchange_id.upper()}_TESTNET_API_KEY"
            secret_env = f"{exchange_id.upper()}_TESTNET_SECRET"

        self.exchange = exchange_class({
            "apiKey": os.getenv(api_key_env, ""),
            "secret": os.getenv(secret_env, ""),
            "enableRateLimit": True,
            "rateLimit": self.rate_limit_ms,
            "options": {"defaultType": "spot"},
        })

        if sandbox:
            self.exchange.set_sandbox_mode(True)
            self.logger.info(f"Exchange '{exchange_id}' initialised in SANDBOX mode.")
        else:
            self.logger.warning(
                f"Exchange '{exchange_id}' initialised in LIVE mode — real funds at risk!"
            )

    def _retry(self, fn, *args, **kwargs):
        """Execute fn with exponential backoff retries."""
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except ccxt.RateLimitExceeded:
                wait = self.RETRY_DELAY_BASE * (2 ** attempt)
                self.logger.warning(f"Rate limit hit — sleeping {wait:.1f}s")
                time.sleep(wait)
            except ccxt.NetworkError as e:
                self.logger.warning(f"Network error (attempt {attempt+1}): {e}")
                time.sleep(self.RETRY_DELAY_BASE)
            except ccxt.ExchangeError as e:
                self.logger.error(f"Exchange error: {e}")
                raise
        raise RuntimeError(f"Max retries exceeded for {fn.__name__}")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for `symbol`.

        Args:
            symbol:    Trading pair e.g. "BTC/USDT"
            timeframe: CCXT timeframe string e.g. "1h", "4h", "1d"
            limit:     Number of candles to fetch (max depends on exchange)
            since:     Start timestamp in milliseconds (optional)

        Returns:
            DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
        """
        self.logger.debug(f"Fetching {limit} × {timeframe} candles for {symbol}")

        raw = self._retry(
            self.exchange.fetch_ohlcv, symbol, timeframe, since=since, limit=limit
        )

        if not raw:
            self.logger.warning(f"No OHLCV data returned for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        df.sort_index(inplace=True)

        self.logger.debug(
            f"Fetched {len(df)} candles for {symbol} "
            f"({df.index[0]} → {df.index[-1]})"
        )
        return df

    def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker (bid/ask/last/volume) for a symbol."""
        return self._retry(self.exchange.fetch_ticker, symbol)

    def fetch_balance(self) -> dict:
        """Fetch full account balance from the exchange."""
        return self._retry(self.exchange.fetch_balance)

    def fetch_open_orders(self, symbol: str) -> list:
        """Return all open orders for `symbol`."""
        return self._retry(self.exchange.fetch_open_orders, symbol)


# ─────────────────────────────────────────────────────────────
# CoinGecko Wrapper
# ─────────────────────────────────────────────────────────────

class CoinGeckoFetcher:
    """
    Fetches supplemental market data from CoinGecko's free public API.
    Primarily used for:
      - Historical market cap / dominance
      - Fear & Greed index proxy (market cap change)
      - Multi-coin overview for universe ranking
    """

    def __init__(self, cache_dir: str = "data/cache/") -> None:
        if not COINGECKO_AVAILABLE:
            raise ImportError("pycoingecko required. pip install pycoingecko")
        self.cg = CoinGeckoAPI()
        # Inject pro API key if available
        pro_key = os.getenv("COINGECKO_API_KEY", "")
        if pro_key:
            self.cg.api_base_url = "https://pro-api.coingecko.com/api/v3/"
            self.cg.request_timeout = 30
        self.cache = FileCache(cache_dir)
        self.logger = get_logger("CoinGeckoFetcher")

    def get_coin_id(self, symbol: str) -> str:
        """Convert a ticker symbol (e.g. 'BTC') to CoinGecko coin id."""
        base = symbol.split("/")[0].upper()
        coin_id = COINGECKO_ID_MAP.get(base)
        if not coin_id:
            raise ValueError(
                f"Unknown CoinGecko id for symbol '{symbol}'. "
                f"Update COINGECKO_ID_MAP in data_fetcher.py."
            )
        return coin_id

    def fetch_market_chart(
        self,
        symbol: str,
        days: int = 90,
        vs_currency: str = "usd",
    ) -> pd.DataFrame:
        """
        Fetch historical daily OHLC + volume from CoinGecko.

        Returns:
            DataFrame with DatetimeIndex, columns [open, high, low, close, volume]
        """
        coin_id = self.get_coin_id(symbol)
        cache_key = f"cg_chart_{coin_id}_{days}_{vs_currency}"
        cached = self.cache.get(cache_key, ttl_seconds=3600)

        if cached:
            self.logger.debug(f"CoinGecko cache hit: {cache_key}")
            data = cached
        else:
            self.logger.debug(f"Fetching CoinGecko market chart: {coin_id} {days}d")
            try:
                data = self.cg.get_coin_market_chart_by_id(
                    id=coin_id, vs_currency=vs_currency, days=days
                )
                self.cache.set(cache_key, data)
            except Exception as e:
                self.logger.error(f"CoinGecko fetch error: {e}")
                return pd.DataFrame()

        # Build DataFrame from prices and total_volumes
        prices_df = pd.DataFrame(data.get("prices", []), columns=["ts", "close"])
        volumes_df = pd.DataFrame(data.get("total_volumes", []), columns=["ts", "volume"])

        df = prices_df.merge(volumes_df, on="ts", how="inner")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        df.index.name = "timestamp"
        df = df.astype(float)
        df.sort_index(inplace=True)

        self.logger.debug(f"CoinGecko returned {len(df)} rows for {coin_id}")
        return df

    def fetch_global_metrics(self) -> dict:
        """
        Fetch global crypto market data:
          - Total market cap
          - BTC dominance
          - Active cryptocurrencies count
        """
        cache_key = "cg_global"
        cached = self.cache.get(cache_key, ttl_seconds=600)
        if cached:
            return cached
        try:
            data = self.cg.get_global()
            self.cache.set(cache_key, data)
            return data
        except Exception as e:
            self.logger.warning(f"CoinGecko global metrics error: {e}")
            return {}


# ─────────────────────────────────────────────────────────────
# Technical Indicator Calculator
# ─────────────────────────────────────────────────────────────

class IndicatorCalculator:
    """
    Computes technical indicators on top of an OHLCV DataFrame.
    Uses pandas_ta when available; falls back to manual numpy implementations.
    """

    def __init__(self, config: dict) -> None:
        strat_cfg = config.get("strategy", {})
        self.rsi_period = strat_cfg.get("rsi_period", 14)
        self.macd_fast = strat_cfg.get("macd_fast", 12)
        self.macd_slow = strat_cfg.get("macd_slow", 26)
        self.macd_signal = strat_cfg.get("macd_signal", 9)
        self.atr_period = config.get("risk", {}).get("atr_period", 14)
        self.logger = get_logger("IndicatorCalc")

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and append all required indicators to `df`.

        Adds columns:
            rsi, macd, macd_signal, macd_hist,
            atr, bb_upper, bb_lower, bb_mid,
            ema_20, ema_50, sma_200,
            volume_sma_20, volume_ratio
        """
        if df.empty:
            return df

        df = df.copy()

        if TA_AVAILABLE:
            df = self._compute_with_pandas_ta(df)
        else:
            df = self._compute_manual(df)

        # Volume ratio (current vs 20-period average) — always manual
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

        # Drop any rows with all-NaN introduced by indicator lookback periods
        df.dropna(how="all", inplace=True)
        return df

    def _compute_with_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use pandas_ta library for fast vectorised computation."""
        # RSI
        df.ta.rsi(length=self.rsi_period, append=True)
        # MACD
        df.ta.macd(
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
            append=True,
        )
        # ATR
        df.ta.atr(length=self.atr_period, append=True)
        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True)
        # EMAs and SMA
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.sma(length=200, append=True)

        # Normalise column names (pandas_ta uses dynamic names)
        rename_map = {}
        for col in df.columns:
            lc = col.lower()
            if lc.startswith("rsi_"):
                rename_map[col] = "rsi"
            elif "macd_" in lc and "signal" not in lc and "hist" not in lc:
                rename_map[col] = "macd"
            elif "macds_" in lc or ("macd" in lc and "signal" in lc):
                rename_map[col] = "macd_signal"
            elif "macdh_" in lc or ("macd" in lc and "hist" in lc):
                rename_map[col] = "macd_hist"
            elif "atr" in lc and col.startswith("ATR"):
                rename_map[col] = "atr"
            elif lc.startswith("bbu_"):
                rename_map[col] = "bb_upper"
            elif lc.startswith("bbm_"):
                rename_map[col] = "bb_mid"
            elif lc.startswith("bbl_"):
                rename_map[col] = "bb_lower"
            elif lc.startswith("ema_20"):
                rename_map[col] = "ema_20"
            elif lc.startswith("ema_50"):
                rename_map[col] = "ema_50"
            elif lc.startswith("sma_200"):
                rename_map[col] = "sma_200"

        df.rename(columns=rename_map, inplace=True)
        return df

    def _compute_manual(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pure-pandas fallback when pandas_ta is not available.
        Implements RSI, MACD, ATR, Bollinger Bands, EMAs manually.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI (Wilder's smoothing)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ATR (Wilder's smoothing)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(com=self.atr_period - 1, adjust=False).mean()

        # Bollinger Bands (20, 2σ)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_mid"] = sma20
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20

        # EMAs and SMA
        df["ema_20"] = close.ewm(span=20, adjust=False).mean()
        df["ema_50"] = close.ewm(span=50, adjust=False).mean()
        df["sma_200"] = close.rolling(200).mean()

        return df


# ─────────────────────────────────────────────────────────────
# Unified Data Manager
# ─────────────────────────────────────────────────────────────

class DataManager:
    """
    Top-level data facade used by the rest of the bot.

    Usage:
        dm = DataManager(config)
        df = dm.get_enriched_ohlcv("BTC/USDT", timeframe="1h", limit=300)
        current_price = dm.get_current_price("BTC/USDT")
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.logger = get_logger("DataManager")

        # Initialise CCXT exchange fetcher
        if CCXT_AVAILABLE:
            try:
                self.ccxt_fetcher = CCXTFetcher(config)
            except Exception as e:
                self.logger.error(f"CCXT init failed: {e}")
                self.ccxt_fetcher = None
        else:
            self.ccxt_fetcher = None

        # CoinGecko (optional enrichment)
        if COINGECKO_AVAILABLE:
            try:
                self.cg_fetcher = CoinGeckoFetcher(
                    cache_dir=config.get("data", {}).get("cache_dir", "data/cache/")
                )
            except Exception as e:
                self.logger.warning(f"CoinGecko init failed: {e}")
                self.cg_fetcher = None
        else:
            self.cg_fetcher = None

        self.indicator_calc = IndicatorCalculator(config)

        # In-memory candle store {symbol: DataFrame}
        self._candle_store: dict[str, pd.DataFrame] = {}

    def get_enriched_ohlcv(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
        add_indicators: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for `symbol` and optionally enrich with indicators.

        Priority: CCXT (live exchange) → CoinGecko (fallback for historical)

        Args:
            symbol:         Trading pair e.g. "BTC/USDT"
            timeframe:      Override config timeframe (e.g. "1h")
            limit:          Number of candles
            add_indicators: Whether to compute and append TA indicators

        Returns:
            Enriched DataFrame ready for the prediction model
        """
        tf = timeframe or self.config.get("trading", {}).get("timeframe", "1h")
        lim = limit or self.config.get("data", {}).get("lookback_candles", 500)

        df = pd.DataFrame()

        # ── Primary: CCXT ─────────────────────────────────────
        if self.ccxt_fetcher:
            try:
                df = self.ccxt_fetcher.fetch_ohlcv(symbol, timeframe=tf, limit=lim)
            except Exception as e:
                self.logger.warning(f"CCXT fetch failed ({symbol}): {e}")

        # ── Fallback: CoinGecko ───────────────────────────────
        if df.empty and self.cg_fetcher:
            self.logger.info(f"Falling back to CoinGecko for {symbol}")
            try:
                df = self.cg_fetcher.fetch_market_chart(symbol, days=min(lim, 365))
            except Exception as e:
                self.logger.error(f"CoinGecko fallback failed ({symbol}): {e}")

        if df.empty:
            self.logger.error(f"No data obtained for {symbol}")
            return df

        # ── Compute indicators ────────────────────────────────
        if add_indicators:
            df = self.indicator_calc.add_all(df)

        # Cache in memory
        self._candle_store[symbol] = df

        return df

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Return the latest close price for `symbol`."""
        # Try live ticker first
        if self.ccxt_fetcher:
            try:
                ticker = self.ccxt_fetcher.fetch_ticker(symbol)
                return float(ticker.get("last") or ticker.get("close", 0))
            except Exception:
                pass

        # Fall back to last candle in store
        df = self._candle_store.get(symbol)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])

        return None

    def update_candle_store(self, symbol: str, new_candle: dict) -> None:
        """
        Append a single new candle to the in-memory store.
        Used by the live loop to keep the store current without
        fetching the full history on every tick.
        """
        df = self._candle_store.get(symbol)
        if df is None or df.empty:
            return

        new_row = pd.DataFrame([new_candle])
        new_row["timestamp"] = pd.to_datetime(new_row["timestamp"], unit="ms", utc=True)
        new_row.set_index("timestamp", inplace=True)
        new_row = new_row.astype(float)

        # Append and recalculate indicators on tail only (efficient)
        df = pd.concat([df, new_row])
        df = df[~df.index.duplicated(keep="last")]
        df = self.indicator_calc.add_all(df)

        self._candle_store[symbol] = df

    def get_global_market_context(self) -> dict:
        """
        Return BTC dominance and total market cap from CoinGecko.
        Used as macro-regime filter in strategy.
        """
        if self.cg_fetcher:
            return self.cg_fetcher.fetch_global_metrics()
        return {}
