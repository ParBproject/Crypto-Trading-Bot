"""Backtest fees: config rate, charged once on entry and once on exit."""

from datetime import datetime

import pandas as pd
import pytest

from backtest import BacktestEngine, commission_rate_from_config, slippage_rate_from_config
from src.risk_manager import TradeParameters
from src.strategy import SignalSource, SignalType, StrategyEngine, TradeSignal

SYMBOL = "BTC/USDT"


def _frame(rows: int = 400) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame({"close": 100.0}, index=index)


def _signal(symbol: str, price: float, kind: SignalType, params=None) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        signal_type=kind,
        source=SignalSource.RULES_ONLY,
        strength=1.0,
        timestamp=datetime(2024, 1, 1),
        current_price=price,
        trade_params=params,
    )


def _long(symbol: str, price: float) -> TradeSignal:
    params = TradeParameters(
        symbol=symbol,
        side="buy",
        entry_price=price,
        quantity=1.0,
        stop_loss=90.0,
        take_profit=130.0,
        risk_usd=10.0,
        position_value_usd=price,
        risk_pct_of_account=0.1,
        reward_risk_ratio=2.0,
    )
    return _signal(symbol, price, SignalType.LONG, params)


def _short(symbol: str, price: float) -> TradeSignal:
    params = TradeParameters(
        symbol=symbol,
        side="sell",
        entry_price=price,
        quantity=1.0,
        stop_loss=110.0,
        take_profit=70.0,
        risk_usd=10.0,
        position_value_usd=price,
        risk_pct_of_account=0.1,
        reward_risk_ratio=2.0,
    )
    return _signal(symbol, price, SignalType.SHORT, params)


def _run(monkeypatch, config, signals):
    state = {"n": 0}

    def evaluate(self, symbol, df, prediction, existing_position=None):
        price = float(df["close"].iloc[-1])
        kind = signals[state["n"]] if state["n"] < len(signals) else "hold"
        state["n"] += 1
        if kind == "long":
            return _long(symbol, price)
        if kind == "short":
            return _short(symbol, price)
        if kind == "exit":
            return _signal(symbol, price, SignalType.EXIT)
        return _signal(symbol, price, SignalType.HOLD)

    monkeypatch.setattr(StrategyEngine, "evaluate", evaluate)
    engine = BacktestEngine(config, initial_capital=10_000.0)
    return engine, engine.run(SYMBOL, _frame(), use_ml=False)


def test_commission_pct_is_a_percent():
    assert commission_rate_from_config({"backtest": {"commission_pct": 0.1}}) == pytest.approx(0.001)
    assert commission_rate_from_config({"backtest": {"commission_pct": 1.0}}) == pytest.approx(0.01)
    assert commission_rate_from_config({}) == pytest.approx(0.001)


def test_long_round_trip_charges_entry_and_exit_once(monkeypatch):
    config = {
        "backtest": {"commission_pct": 1.0},
        "model": {"sequence_length": 60},
        "trading": {"mode": "backtest"},
        "exchange": {"sandbox": True},
    }
    engine, metrics = _run(monkeypatch, config, ["long", "exit"])

    slip = slippage_rate_from_config(config)
    rate = 0.01
    entry = 100.0 * (1 + slip)
    exit_px = 100.0 * (1 - slip)
    entry_fee = entry * rate
    exit_fee = exit_px * rate
    pnl = (exit_px - entry) - exit_fee

    assert metrics["total_trades"] == 1
    assert metrics["trades"][0]["pnl_usd"] == pytest.approx(round(pnl, 4))
    assert metrics["final_capital_usd"] == pytest.approx(round(10_000.0 - entry_fee + pnl, 2))

    # The previous close formula subtracted two entry-notional fees.
    double_charged = (exit_px - entry) - 2 * entry * rate
    assert metrics["trades"][0]["pnl_usd"] != pytest.approx(round(double_charged, 4))
    assert engine.commission_rate == pytest.approx(rate)


def test_short_round_trip_charges_entry_and_exit_once(monkeypatch):
    config = {
        "backtest": {"commission_pct": 0.1},
        "model": {"sequence_length": 60},
        "trading": {"mode": "backtest"},
        "exchange": {"sandbox": True},
    }
    _engine, metrics = _run(monkeypatch, config, ["short", "exit"])

    slip = slippage_rate_from_config(config)
    rate = 0.001
    entry = 100.0 * (1 - slip)
    exit_px = 100.0 * (1 + slip)
    entry_fee = entry * rate
    exit_fee = exit_px * rate
    pnl = (entry - exit_px) - exit_fee

    assert metrics["total_trades"] == 1
    assert metrics["trades"][0]["side"] == "sell"
    assert metrics["trades"][0]["pnl_usd"] == pytest.approx(round(pnl, 4))
    assert metrics["final_capital_usd"] == pytest.approx(round(10_000.0 - entry_fee + pnl, 2))


def test_end_of_backtest_close_charges_exit_fee_once(monkeypatch):
    config = {
        "backtest": {"commission_pct": 0.1},
        "model": {"sequence_length": 60},
        "trading": {"mode": "backtest"},
        "exchange": {"sandbox": True},
    }
    frame = _frame()
    seq_len = 60
    test_len = len(frame) - int(len(frame) * 0.70)
    last_call = test_len - seq_len
    state = {"n": 0}

    def evaluate(self, symbol, df, prediction, existing_position=None):
        state["n"] += 1
        price = float(df["close"].iloc[-1])
        if existing_position is None and state["n"] == last_call:
            return _long(symbol, price)
        return _signal(symbol, price, SignalType.HOLD)

    monkeypatch.setattr(StrategyEngine, "evaluate", evaluate)
    engine = BacktestEngine(config, initial_capital=10_000.0)
    metrics = engine.run(SYMBOL, frame, use_ml=False)

    slip = slippage_rate_from_config(config)
    rate = engine.commission_rate
    entry = 100.0 * (1 + slip)
    exit_px = 100.0 * (1 - slip)
    entry_fee = entry * rate
    exit_fee = exit_px * rate
    pnl = (exit_px - entry) - exit_fee

    assert metrics["total_trades"] == 1
    assert metrics["trades"][0]["reason"] == "end_of_backtest"
    assert metrics["trades"][0]["pnl_usd"] == pytest.approx(round(pnl, 4))
    assert metrics["final_capital_usd"] == pytest.approx(round(10_000.0 - entry_fee + pnl, 2))
