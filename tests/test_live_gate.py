"""Live orders stay off unless the process is explicitly armed."""

import pytest

from src.executor import LiveExecutor, OrderManager, ensure_live_trading_allowed
from src.risk_manager import PortfolioState


def _portfolio() -> PortfolioState:
    return PortfolioState(initial_capital=10_000.0, current_capital=10_000.0)


def test_paper_mode_does_not_require_the_arming_flag(monkeypatch):
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    ensure_live_trading_allowed({"trading": {"mode": "paper"}, "exchange": {"sandbox": False}})
    manager = OrderManager(
        {"trading": {"mode": "paper"}, "backtest": {"initial_capital": 10_000.0}},
        _portfolio(),
    )
    assert manager.mode == "paper"


def test_live_mode_is_refused_without_the_arming_flag(monkeypatch):
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    config = {
        "trading": {"mode": "live"},
        "exchange": {"sandbox": False},
        "backtest": {"initial_capital": 10_000.0},
    }
    with pytest.raises(RuntimeError, match="ALLOW_LIVE_TRADING"):
        ensure_live_trading_allowed(config)
    with pytest.raises(RuntimeError, match="ALLOW_LIVE_TRADING"):
        OrderManager(config, _portfolio(), ccxt_fetcher=object())


def test_live_mode_constructs_when_armed(monkeypatch):
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "1")
    config = {
        "trading": {"mode": "live"},
        "exchange": {"sandbox": True},
        "backtest": {"initial_capital": 10_000.0},
    }
    ensure_live_trading_allowed(config)
    manager = OrderManager(config, _portfolio(), ccxt_fetcher=object())
    assert isinstance(manager.engine, LiveExecutor)
