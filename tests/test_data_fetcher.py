"""CoinGecko fallback supplies OHLC, and missing columns fail clearly."""

import pandas as pd
import pytest

from src.data_fetcher import CoinGeckoFetcher, IndicatorCalculator
from src.logger import get_logger


class _Cache:
    def get(self, *_args, **_kwargs):
        return None

    def set(self, *_args, **_kwargs):
        return None


class _CoinGecko:
    def get_coin_ohlc_by_id(self, id, vs_currency, days):
        base = 1_704_067_200_000
        return [
            [base + i * 86_400_000, 100 + i, 110 + i, 90 + i, 105 + i]
            for i in range(40)
        ]


def _fetcher() -> CoinGeckoFetcher:
    fetcher = CoinGeckoFetcher.__new__(CoinGeckoFetcher)
    fetcher.logger = get_logger("test-coingecko")
    fetcher.cache = _Cache()
    fetcher.cg = _CoinGecko()
    return fetcher


def test_coingecko_ohlc_has_columns_indicators_need():
    df = _fetcher().fetch_market_chart("BTC/USDT", days=30)
    assert {"open", "high", "low", "close", "volume"} <= set(df.columns)
    enriched = IndicatorCalculator({}).add_all(df)
    assert "rsi" in enriched.columns
    assert "atr" in enriched.columns


def test_coingecko_empty_ohlc_returns_empty_frame():
    fetcher = _fetcher()

    class _Empty:
        def get_coin_ohlc_by_id(self, id, vs_currency, days):
            return []

    fetcher.cg = _Empty()
    df = fetcher.fetch_market_chart("ETH/USDT", days=30)
    assert df.empty


def test_indicator_calculator_rejects_close_only_frames():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="high"):
        IndicatorCalculator({}).add_all(df)
