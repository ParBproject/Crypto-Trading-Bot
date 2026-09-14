from datetime import datetime

import pytest

from src.executor import OrderManager, OrderResult
from src.risk_manager import PortfolioState


SYMBOL = "BTC/USDT"


def _make_manager(side: str, current_capital: float = 9_999.0) -> OrderManager:
    portfolio = PortfolioState(
        initial_capital=10_000.0,
        current_capital=current_capital,
        open_positions={
            SYMBOL: {
                "side": side,
                "quantity": 10.0,
                "entry_price": 100.0,
                "stop_loss": 90.0 if side == "buy" else 110.0,
                "take_profit": 120.0 if side == "buy" else 80.0,
                "entry_time": datetime(2026, 1, 1),
                "order_id": "OPEN-1",
                "value_usd": 1_000.0,
            }
        },
    )
    config = {
        "trading": {"mode": "paper"},
        "backtest": {"initial_capital": 10_000.0},
    }
    return OrderManager(config, portfolio)


def _fill(side: str, fill_price: float, commission: float) -> OrderResult:
    return OrderResult(
        success=True,
        order_id="CLOSE-1",
        symbol=SYMBOL,
        side=side,
        quantity=10.0,
        fill_price=fill_price,
        fill_time=datetime(2026, 1, 2),
        cost_usd=10.0 * fill_price,
        commission_usd=commission,
        mode="paper",
    )


def test_flat_long_close_does_not_add_exit_notional(monkeypatch):
    manager = _make_manager("buy")
    monkeypatch.setattr(
        manager,
        "_place",
        lambda *args, **kwargs: _fill("sell", fill_price=100.0, commission=1.0),
    )

    manager.close_trade(SYMBOL, current_price=100.0, reason="flat exit")

    # 9,999 already reflects the $1 entry fee. A flat close should only deduct
    # the $1 exit fee, not add the $1,000 sale notional to account equity.
    assert manager.portfolio.current_capital == pytest.approx(9_998.0)
    assert manager.portfolio.trade_history[-1]["pnl_usd"] == pytest.approx(-1.0)
    assert SYMBOL not in manager.portfolio.open_positions


def test_profitable_long_close_adds_only_net_realized_pnl(monkeypatch):
    manager = _make_manager("buy")
    monkeypatch.setattr(
        manager,
        "_place",
        lambda *args, **kwargs: _fill("sell", fill_price=110.0, commission=1.1),
    )

    manager.close_trade(SYMBOL, current_price=110.0)

    expected_pnl = (110.0 - 100.0) * 10.0 - 1.1
    assert manager.portfolio.current_capital == pytest.approx(9_999.0 + expected_pnl)
    assert manager.portfolio.trade_history[-1]["pnl_usd"] == pytest.approx(expected_pnl)


def test_profitable_short_close_adds_only_net_realized_pnl(monkeypatch):
    manager = _make_manager("sell")
    monkeypatch.setattr(
        manager,
        "_place",
        lambda *args, **kwargs: _fill("buy", fill_price=90.0, commission=0.9),
    )

    manager.close_trade(SYMBOL, current_price=90.0)

    expected_pnl = (100.0 - 90.0) * 10.0 - 0.9
    assert manager.portfolio.current_capital == pytest.approx(9_999.0 + expected_pnl)
    assert manager.portfolio.trade_history[-1]["pnl_usd"] == pytest.approx(expected_pnl)
