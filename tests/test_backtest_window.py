"""The committed backtest window is fetched as a range, not a 500-bar lookback."""

from pathlib import Path

import pandas as pd
import yaml

from backtest import backtest_window, load_backtest_frame
from src.data_fetcher import paginate_ohlcv, parse_binance_kline_csv, binance_vision_month_urls


def test_committed_config_window_is_longer_than_lookback():
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    start, end = backtest_window(config)
    assert start == pd.Timestamp("2023-01-01", tz="UTC")
    assert end == pd.Timestamp("2024-01-01", tz="UTC")
    assert (end - start) > pd.Timedelta(hours=config["data"]["lookback_candles"])


def test_load_backtest_frame_requests_the_config_window():
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    start, end = backtest_window(config)

    class FakeData:
        def __init__(self):
            self.kwargs = None

        def get_enriched_ohlcv(self, pair, **kwargs):
            self.kwargs = kwargs
            index = pd.date_range(start, end, freq="h", tz="UTC")
            return pd.DataFrame(
                {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                index=index,
            )

    data = FakeData()
    frame = load_backtest_frame(data, config, "BTC/USDT")

    assert data.kwargs["since"] == start
    assert data.kwargs["until"] == end
    assert len(frame) >= 200
    assert frame.index[0] >= start
    assert frame.index[-1] <= end


def test_paginate_ohlcv_walks_past_a_single_page():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = start + pd.Timedelta(hours=25)
    since_ms = int(start.timestamp() * 1000)
    until_ms = int(end.timestamp() * 1000)
    calls = []

    def fetch_page(since, limit):
        calls.append(since)
        rows = []
        cursor = since
        for _ in range(limit):
            if cursor > until_ms:
                break
            rows.append([cursor, 1, 2, 0.5, 1.5, 10])
            cursor += 3_600_000
        return rows

    df = paginate_ohlcv(fetch_page, "1h", since_ms, until_ms, page_limit=10)

    assert len(calls) > 1
    assert calls[0] == since_ms
    assert len(df) == 26
    assert df.index[0] == start
    assert df.index[-1] == end
    assert set(df.columns) >= {"open", "high", "low", "close", "volume"}


def test_binance_archive_urls_cover_the_committed_window():
    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-01", tz="UTC")
    urls = binance_vision_month_urls("BTC/USDT", "1h", start, end)
    assert urls[0].endswith("BTCUSDT-1h-2023-01.zip")
    assert urls[-1].endswith("BTCUSDT-1h-2024-01.zip")
    assert len(urls) == 13


def test_parse_binance_kline_csv_reads_ohlc_and_microsecond_timestamps():
    text = (
        "open_time,open,high,low,close,volume,close_time,quote,trades,taker_base,taker_quote,ignore\n"
        "1704067200000000,42000,43000,41000,42500,12.5,0,0,0,0,0,0\n"
    )
    df = parse_binance_kline_csv(text)
    assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert float(df.iloc[0]["high"]) == 43000
    assert float(df.iloc[0]["low"]) == 41000
    assert float(df.iloc[0]["close"]) == 42500
    assert float(df.iloc[0]["volume"]) == 12.5
