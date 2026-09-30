"""Next-bar fills, intrabar stops, fees, and the buy-and-hold benchmark."""

from datetime import datetime

import pandas as pd
import pytest

from backtest import (
    BacktestEngine,
    buy_and_hold_final_capital,
    commission_rate_from_config,
    protective_exit,
    slippage_rate_from_config,
)
from src.risk_manager import TradeParameters, infer_periods_per_year
from src.strategy import SignalSource, SignalType, StrategyEngine, TradeSignal

SYMBOL = "BTC/USDT"
ROWS = 400


def _config(**backtest):
    costs = {"commission_pct": 0.0, "slippage_pct": 0.0}
    costs.update(backtest)
    return {
        "backtest": costs,
        "model": {"sequence_length": 5},
        "trading": {"mode": "backtest"},
        "exchange": {"sandbox": True},
        "risk": {
            "max_risk_per_trade_pct": 50,
            "max_single_asset_exposure_pct": 100,
            "max_drawdown_pct": 90,
            "max_open_trades": 5,
        },
    }


def _frame(price_rows=None) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=ROWS, freq="h", tz="UTC")
    open_ = [100.0] * ROWS
    high = [100.0] * ROWS
    low = [100.0] * ROWS
    close = [100.0] * ROWS
    split = int(ROWS * 0.7)
    for test_i, (o, h, l, c) in (price_rows or {}).items():
        abs_i = split + test_i
        open_[abs_i] = o
        high[abs_i] = h
        low[abs_i] = l
        close[abs_i] = c
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=index,
    )


def _params(price: float, side: str, stop: float, take_profit: float) -> TradeParameters:
    return TradeParameters(
        symbol=SYMBOL,
        side=side,
        entry_price=price,
        quantity=1.0,
        stop_loss=stop,
        take_profit=take_profit,
        risk_usd=10.0,
        position_value_usd=price,
        risk_pct_of_account=0.1,
        reward_risk_ratio=2.0,
    )


def _signal(kind: SignalType, price: float, params=None) -> TradeSignal:
    return TradeSignal(
        symbol=SYMBOL,
        signal_type=kind,
        source=SignalSource.RULES_ONLY,
        strength=1.0,
        timestamp=datetime(2024, 1, 1),
        current_price=price,
        trade_params=params,
    )


def _run(monkeypatch, decide, price_rows=None, config=None):
    frame = _frame(price_rows)
    split = int(len(frame) * 0.7)
    test_index = frame.index[split:]
    seen = []
    windows = []

    def evaluate(self, symbol, df, prediction, existing_position=None):
        seen.append(df.index[-1])
        windows.append(df.index)
        kind, stop, take = decide(df.index[-1], test_index, existing_position)
        price = float(df["close"].iloc[-1])
        if kind == "long":
            return _signal(SignalType.LONG, price, _params(price, "buy", stop, take))
        if kind == "short":
            return _signal(SignalType.SHORT, price, _params(price, "sell", stop, take))
        if kind == "exit":
            return _signal(SignalType.EXIT, price)
        return _signal(SignalType.HOLD, price)

    monkeypatch.setattr(StrategyEngine, "evaluate", evaluate)
    engine = BacktestEngine(config or _config(), initial_capital=10_000.0)
    metrics = engine.run(SYMBOL, frame, use_ml=False)
    return metrics, seen, test_index, windows


def _hold(ts, test_index, position):
    return "hold", 0, 0


def test_commission_and_slippage_are_percents():
    assert commission_rate_from_config({"backtest": {"commission_pct": 0.1}}) == pytest.approx(0.001)
    assert commission_rate_from_config({"backtest": {"commission_pct": 1.0}}) == pytest.approx(0.01)
    assert commission_rate_from_config({}) == pytest.approx(0.001)
    assert slippage_rate_from_config({"backtest": {"slippage_pct": 0.05}}) == pytest.approx(0.0005)
    assert slippage_rate_from_config({}) == pytest.approx(0.0005)
    with pytest.raises(ValueError):
        commission_rate_from_config({"backtest": {"commission_pct": -0.1}})


def test_signal_fills_next_open_and_ignores_the_signal_bar_range(monkeypatch):
    def decide(ts, test_index, position):
        if ts == test_index[4] and position is None:
            return "long", 40.0, 10_000.0
        return "hold", 0, 0

    # Bar 4 crashes to 1 and closes at 50. The fill bar opens at 110.
    metrics, seen, test_index, windows = _run(
        monkeypatch,
        decide,
        {4: (100.0, 100.0, 1.0, 50.0), 5: (110.0, 110.0, 110.0, 110.0)},
    )

    assert seen[0] == test_index[4]
    assert test_index[5] not in windows[0]
    assert metrics["fill_model"] == "next_bar_open"
    assert metrics["total_trades"] == 1
    trade = metrics["trades"][0]
    assert trade["entry_price"] == pytest.approx(110.0)
    assert trade["entry_price"] != pytest.approx(50.0)
    assert trade["reason"] == "end_of_backtest"
    assert trade["entry_time"] == str(test_index[5])


def test_stop_uses_the_low_when_the_close_recovers(monkeypatch):
    def decide(ts, test_index, position):
        if ts == test_index[4] and position is None:
            return "long", 90.0, 150.0
        return "hold", 0, 0

    metrics, _, _, _ = _run(
        monkeypatch,
        decide,
        {6: (100.0, 100.0, 70.0, 100.0)},
    )
    trade = metrics["trades"][0]
    assert trade["reason"] == "stop_loss"
    assert trade["exit_price"] == pytest.approx(90.0)
    assert trade["pnl_usd"] == pytest.approx(-10.0)


def test_stop_wins_when_the_bar_touches_stop_and_target(monkeypatch):
    def decide(ts, test_index, position):
        if ts == test_index[4] and position is None:
            return "long", 90.0, 130.0
        return "hold", 0, 0

    metrics, _, _, _ = _run(
        monkeypatch,
        decide,
        {6: (100.0, 160.0, 70.0, 100.0)},
    )
    trade = metrics["trades"][0]
    assert trade["reason"] == "stop_loss"
    assert trade["exit_price"] == pytest.approx(90.0)


def test_gap_through_the_stop_fills_at_the_open(monkeypatch):
    def decide(ts, test_index, position):
        if ts == test_index[4] and position is None:
            return "long", 90.0, 150.0
        return "hold", 0, 0

    metrics, _, _, _ = _run(
        monkeypatch,
        decide,
        {6: (80.0, 80.0, 80.0, 80.0)},
    )
    trade = metrics["trades"][0]
    assert trade["reason"] == "stop_loss"
    assert trade["exit_price"] == pytest.approx(80.0)
    assert trade["pnl_usd"] == pytest.approx(-20.0)


def test_short_stop_uses_the_high(monkeypatch):
    def decide(ts, test_index, position):
        if ts == test_index[4] and position is None:
            return "short", 110.0, 50.0
        return "hold", 0, 0

    metrics, _, _, _ = _run(
        monkeypatch,
        decide,
        {6: (100.0, 120.0, 100.0, 100.0)},
    )
    trade = metrics["trades"][0]
    assert trade["side"] == "sell"
    assert trade["reason"] == "stop_loss"
    assert trade["exit_price"] == pytest.approx(110.0)
    assert trade["pnl_usd"] == pytest.approx(-10.0)


def test_round_trip_charges_entry_and_exit_commission_once(monkeypatch):
    def decide(ts, test_index, position):
        if ts == test_index[4] and position is None:
            return "long", 50.0, 10_000.0
        if ts == test_index[5] and position is not None:
            return "exit", 0, 0
        return "hold", 0, 0

    metrics, _, _, _ = _run(
        monkeypatch,
        decide,
        config=_config(commission_pct=1.0, slippage_pct=0.0),
    )
    trade = metrics["trades"][0]
    assert trade["reason"] == "signal_exit"
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(100.0)
    assert trade["pnl_usd"] == pytest.approx(-1.0)
    assert metrics["final_capital_usd"] == pytest.approx(9998.0)


def test_signal_on_the_last_bar_does_not_fill(monkeypatch):
    def decide(ts, test_index, position):
        if ts == test_index[-1]:
            return "long", 50.0, 10_000.0
        return "hold", 0, 0

    metrics, _, _, _ = _run(monkeypatch, decide)
    assert metrics["total_trades"] == 0
    assert metrics["final_capital_usd"] == pytest.approx(10_000.0)


def test_open_loss_is_marked_before_the_trade_closes(monkeypatch):
    def decide(ts, test_index, position):
        if ts == test_index[4] and position is None:
            return "long", 10.0, 10_000.0
        return "hold", 0, 0

    metrics, _, _, _ = _run(
        monkeypatch,
        decide,
        {6: (100.0, 100.0, 80.0, 80.0)},
    )
    assert len(metrics["equity_curve"]) == 120
    assert metrics["equity_curve"][6][1] == pytest.approx(9980.0)
    assert metrics["max_drawdown_pct"] > 0
    assert metrics["trades"][0]["reason"] == "end_of_backtest"
    assert metrics["final_capital_usd"] == pytest.approx(metrics["equity_curve"][-1][1])


def test_buy_and_hold_when_the_strategy_is_flat(monkeypatch):
    metrics, _, test_index, _ = _run(
        monkeypatch,
        _hold,
        {119: (100.0, 200.0, 100.0, 200.0)},
    )
    expected = buy_and_hold_final_capital(10_000.0, 100.0, 200.0, 0.0, 0.0)
    assert metrics["total_trades"] == 0
    assert metrics["total_return_pct"] == pytest.approx(0.0)
    assert metrics["final_capital_usd"] == pytest.approx(10_000.0)
    assert metrics["buy_hold_final_capital_usd"] == pytest.approx(expected)
    assert metrics["buy_hold_return_pct"] == pytest.approx(100.0)
    assert metrics["sortino_ratio"] == pytest.approx(0.0)
    assert metrics["periods_per_year"] == 365 * 24
    assert test_index[5] is not None


def test_end_of_backtest_charges_the_exit_fee(monkeypatch):
    def decide(ts, test_index, position):
        if ts == test_index[4] and position is None:
            return "long", 10.0, 10_000.0
        return "hold", 0, 0

    metrics, _, _, _ = _run(
        monkeypatch,
        decide,
        config=_config(commission_pct=1.0, slippage_pct=0.0),
    )
    trade = metrics["trades"][0]
    assert trade["reason"] == "end_of_backtest"
    assert trade["pnl_usd"] == pytest.approx(-1.0)
    assert metrics["final_capital_usd"] == pytest.approx(9998.0)


def test_protective_exit_helpers():
    assert protective_exit("buy", 100, 160, 70, 90, 130) == (90, "stop_loss")
    assert protective_exit("buy", 80, 80, 80, 90, 130) == (80, "stop_loss")
    assert protective_exit("buy", 140, 140, 140, 90, 130) == (140, "take_profit")
    assert protective_exit("sell", 100, 120, 40, 110, 50) == (110, "stop_loss")
    assert protective_exit("buy", 100, 105, 95, 90, 130) == (None, "")


def test_bar_spacing_sets_the_annualization_factor():
    hourly = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    daily = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    assert infer_periods_per_year(hourly) == 365 * 24
    assert infer_periods_per_year(daily) == 365


def test_buy_and_hold_helper_applies_fees_once_each_side():
    final = buy_and_hold_final_capital(10_000.0, 100.0, 200.0, 0.01, 0.0)
    qty = 10_000.0 / (100.0 * 1.01)
    assert final == pytest.approx(qty * 200.0 * 0.99)
