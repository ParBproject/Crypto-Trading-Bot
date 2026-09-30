#!/usr/bin/env python3
"""Reproduce the sample backtest published in the README.

Loads the committed Binance spot BTC/USDT 1-hour candles in
``data/sample/btcusdt_1h_2024.csv``, enriches them with the same indicator
code the bot uses, and runs ``BacktestEngine`` with the LSTM turned off.

The rules-only path is deterministic: the same file and the same code produce
the same trades, metrics, and charts. No network access and no TensorFlow
install are required.

Usage (from the repository root):

    python scripts/reproduce_backtest.py

Outputs:

    docs/results/metrics.json
    docs/results/trades.csv
    docs/results/equity_curve.png
    docs/results/price_and_trades.png
    docs/results/signal_gates.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import BacktestEngine, buy_and_hold_final_capital  # noqa: E402
from src.bot import load_config  # noqa: E402
from src.data_fetcher import IndicatorCalculator  # noqa: E402

SAMPLE_CSV = ROOT / "data" / "sample" / "btcusdt_1h_2024.csv"
OUT_DIR = ROOT / "docs" / "results"
INITIAL_CAPITAL = 10_000.0
SYMBOL = "BTC/USDT"

# Dark canvas with an emerald series, matching the portfolio accent.
BG = "#0b1220"
PANEL = "#111827"
GRID = "#1f2937"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
EMERALD = "#10b981"
RED = "#f87171"
AMBER = "#fbbf24"
BLUE = "#93c5fd"


def load_sample(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    if df.empty:
        raise SystemExit(f"Sample file is empty: {path}")
    return df


def buy_and_hold_equity(
    test_df: pd.DataFrame,
    capital: float,
    commission_rate: float,
    slippage_rate: float,
    seq_len: int,
) -> pd.Series:
    """Fully invested long on the engine's entry bar, with the same costs.

    Entry is the open of the first bar a signal can fill (``seq_len`` bars
    into the test window). Each later point is that position liquidated at
    the bar's close, so the last point matches ``buy_and_hold_final_capital``.
    """
    entry_i = seq_len if seq_len < len(test_df) else 0
    entry_open = float(test_df["open"].iloc[entry_i])
    final = buy_and_hold_final_capital(
        capital,
        entry_open,
        float(test_df["close"].iloc[-1]),
        commission_rate,
        slippage_rate,
    )
    entry = entry_open * (1 + slippage_rate)
    if capital <= 0 or entry <= 0:
        return pd.Series(float(capital), index=test_df.index, name="buy_and_hold")
    qty = capital / (entry * (1 + commission_rate))
    values = []
    last = len(test_df) - 1
    for i, close in enumerate(test_df["close"].astype(float)):
        if i < entry_i:
            values.append(float(capital))
        elif i == last:
            values.append(float(final))
        else:
            exit_fill = max(0.0, float(close) * (1 - slippage_rate))
            values.append(qty * exit_fill * (1 - commission_rate))
    return pd.Series(values, index=test_df.index, name="buy_and_hold")


def style_ax(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, loc="left", fontsize=13, pad=12)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_equity(
    equity: pd.Series,
    benchmark: pd.Series,
    path: Path,
    bh_return: float,
    strategy_return: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.4), facecolor=BG)
    style_ax(
        ax,
        f"Mark-to-market equity {strategy_return:+.2f}% vs buy-and-hold {bh_return:+.2f}%",
        "USD",
    )
    ax.plot(benchmark.index, benchmark.values, color=BLUE, linewidth=1.3, label="Buy and hold")
    ax.plot(equity.index, equity.values, color=EMERALD, linewidth=2.0, label="Strategy")
    ax.axhline(INITIAL_CAPITAL, color=MUTED, linewidth=0.8, linestyle="--", label="Starting capital")
    if float(equity.iloc[-1]) == INITIAL_CAPITAL and float(equity.max()) == float(equity.min()):
        ax.text(
            0.02,
            0.08,
            "No round trips, so equity stays at $10,000.",
            transform=ax.transAxes,
            color=EMERALD,
            fontsize=10,
        )
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    save_fig(fig, path)


def plot_price(test_df: pd.DataFrame, trades: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.4), facecolor=BG)
    style_ax(ax, "BTC/USDT close on the test window", "USDT")
    ax.plot(test_df.index, test_df["close"], color=EMERALD, linewidth=1.15)

    longs = [t for t in trades if t["side"] == "buy"]
    shorts = [t for t in trades if t["side"] == "sell"]

    def _scatter(items, column, color, marker, label):
        if not items:
            return
        times = pd.to_datetime([t[column] for t in items], utc=True)
        prices = [t["entry_price"] if column == "entry_time" else t["exit_price"] for t in items]
        ax.scatter(times, prices, s=28, color=color, marker=marker, zorder=3, label=label)

    _scatter(longs, "entry_time", EMERALD, "^", "Long entry")
    _scatter(shorts, "entry_time", RED, "v", "Short entry")
    _scatter(trades, "exit_time", AMBER, "x", "Exit")
    if trades:
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    save_fig(fig, path)


def ta_gate_counts(test_df: pd.DataFrame, config: dict) -> dict:
    """Count the rules-only long gates on the same bars the backtest traded.

    Mirrors ``HybridLSTMStrategy._ta_only_signal``: a long requires RSI below
    the oversold threshold, MACD above its signal, and price above EMA-20,
    all on the same candle. No position is open in this sample, so the
    position check does not remove any bars.
    """
    strat = config.get("strategy", {})
    rsi_max = float(strat.get("rsi_oversold", 35))
    rsi = test_df["rsi"]
    macd_ok = test_df["macd"] > test_df["macd_signal"]
    rsi_ok = rsi < rsi_max
    ema_ok = test_df["close"] > test_df["ema_20"]
    valid = rsi.notna() & test_df["macd"].notna() & test_df["macd_signal"].notna() & test_df["ema_20"].notna()
    all_ok = valid & rsi_ok & macd_ok & ema_ok
    n = int(valid.sum())

    def pct(mask: pd.Series) -> float:
        if n == 0:
            return 0.0
        return round(float((valid & mask).sum()) / n * 100, 2)

    return {
        "valid_bars": n,
        "rsi_oversold_pct": pct(rsi_ok),
        "macd_bullish_pct": pct(macd_ok),
        "price_above_ema20_pct": pct(ema_ok),
        "all_three_bars": int(all_ok.sum()),
        "all_three_pct": pct(rsi_ok & macd_ok & ema_ok),
        "rsi_threshold": rsi_max,
    }


def hourly_move_stats(close: pd.Series) -> dict:
    moves = close.pct_change().dropna() * 100
    return {
        "bars": int(len(moves)),
        "median_abs_pct": round(float(moves.abs().median()), 3),
        "p95_abs_pct": round(float(moves.abs().quantile(0.95)), 3),
        "share_abs_ge_1_5_pct": round(float((moves.abs() >= 1.5).mean() * 100), 2),
    }


def plot_gates(gates: dict, path: Path) -> None:
    labels = ["RSI oversold", "MACD bullish", "Close above EMA-20", "All three"]
    values = [
        gates["rsi_oversold_pct"],
        gates["macd_bullish_pct"],
        gates["price_above_ema20_pct"],
        gates["all_three_pct"],
    ]
    colors = [AMBER, BLUE, EMERALD, RED if values[-1] == 0 else EMERALD]
    fig, ax = plt.subplots(figsize=(11.2, 4.4), facecolor=BG)
    style_ax(ax, "Rules-only long gate, share of test bars", "Percent of bars")
    bars = ax.bar(labels, values, color=colors, width=0.72)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", colors=TEXT)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=10,
        )
    save_fig(fig, path)


def main() -> None:
    if not SAMPLE_CSV.exists():
        raise SystemExit(f"Missing sample candles: {SAMPLE_CSV}")

    config = load_config(str(ROOT / "config" / "config.yaml"))
    raw = load_sample(SAMPLE_CSV)
    enriched = IndicatorCalculator(config).add_all(raw)

    engine = BacktestEngine(config, initial_capital=INITIAL_CAPITAL)
    metrics = engine.run(SYMBOL, enriched, use_ml=False)

    split_idx = int(len(enriched) * 0.70)
    test_df = enriched.iloc[split_idx:]
    seq_len = int(config.get("model", {}).get("sequence_length", 60))
    benchmark = buy_and_hold_equity(
        test_df,
        INITIAL_CAPITAL,
        engine.commission_rate,
        engine.slippage_rate,
        seq_len,
    )
    bh_final = float(metrics["buy_hold_final_capital_usd"])
    if abs(float(benchmark.iloc[-1]) - bh_final) > 0.02:
        raise SystemExit(
            f"Buy-and-hold chart ends at {float(benchmark.iloc[-1]):.2f}, "
            f"engine reports {bh_final:.2f}."
        )
    bh_return = float(metrics["buy_hold_return_pct"])

    equity = pd.Series(
        [point[1] for point in metrics["equity_curve"]],
        index=pd.to_datetime([point[0] for point in metrics["equity_curve"]], utc=True),
        name="equity",
    )
    # The engine records the first equity point twice when the first loop
    # timestamp equals the test start. Keep the last value at each timestamp.
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()

    gates = ta_gate_counts(test_df, config)
    hourly = hourly_move_stats(test_df["close"])

    published = {k: v for k, v in metrics.items() if k not in {"trades", "equity_curve"}}
    if isinstance(published.get("profit_factor"), float) and not np.isfinite(published["profit_factor"]):
        published["profit_factor"] = None
    entry_i = seq_len if seq_len < len(test_df) else 0
    published["buy_and_hold_return_pct"] = metrics["buy_hold_return_pct"]
    published["buy_and_hold_final_usd"] = metrics["buy_hold_final_capital_usd"]
    published["test_start_close"] = round(float(test_df["close"].iloc[0]), 2)
    published["test_end_close"] = round(float(test_df["close"].iloc[-1]), 2)
    published["buy_hold_entry_time"] = test_df.index[entry_i].strftime("%Y-%m-%d %H:%M:%S%z")
    published["buy_hold_entry_open"] = round(float(test_df["open"].iloc[entry_i]), 2)
    published["sample_file"] = str(SAMPLE_CSV.relative_to(ROOT))
    published["sample_rows"] = int(len(raw))
    published["sample_start"] = raw.index[0].strftime("%Y-%m-%d %H:%M:%S%z")
    published["sample_end"] = raw.index[-1].strftime("%Y-%m-%d %H:%M:%S%z")
    published["ml_enabled"] = False
    published["equity_mark_to_market"] = True
    published["ta_gates"] = gates
    published["hourly_moves"] = hourly
    published["notes"] = (
        "Strategy equity is mark-to-market from BacktestEngine with use_ml=False. "
        "A signal at the close fills at the next bar's open. "
        "Buy-and-hold is the engine's fully invested long from that first fillable "
        "open through the last close, with the same commission and slippage. "
        "Sortino is 0 when fewer than two equity returns are negative. "
        "Sharpe and Sortino use the median bar spacing."
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(published, indent=2) + "\n")

    trades = metrics["trades"]
    trade_columns = [
        "entry_time", "exit_time", "symbol", "side", "entry_price",
        "exit_price", "quantity", "pnl_usd", "reason",
    ]
    pd.DataFrame(trades, columns=trade_columns).to_csv(OUT_DIR / "trades.csv", index=False)

    plot_equity(equity, benchmark, OUT_DIR / "equity_curve.png", bh_return, published["total_return_pct"])
    plot_price(test_df, trades, OUT_DIR / "price_and_trades.png")
    plot_gates(gates, OUT_DIR / "signal_gates.png")

    print(json.dumps(published, indent=2))
    print(f"trades={len(trades)} wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
