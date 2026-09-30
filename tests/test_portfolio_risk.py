"""Mark-to-market drawdown, position caps, and Sortino on thin samples."""

import pandas as pd
import pytest

from src.risk_manager import PortfolioState, RiskManager, marks_from_ohlcv


def _portfolio(capital=10_000.0) -> PortfolioState:
    return PortfolioState(initial_capital=capital, current_capital=capital)


def _risk(portfolio, **risk) -> RiskManager:
    config = {
        "risk": {
            "max_risk_per_trade_pct": 1.5,
            "atr_stop_multiplier": 2.5,
            "max_single_asset_exposure_pct": 25.0,
            "max_drawdown_pct": 12.0,
            **risk,
        }
    }
    return RiskManager(config, portfolio)


def test_open_loss_is_invisible_until_the_position_is_marked():
    portfolio = _portfolio()
    portfolio.open_positions["BTC/USDT"] = {
        "side": "buy",
        "quantity": 50.0,
        "entry_price": 100.0,
        "value_usd": 5_000.0,
    }
    risk = _risk(portfolio)
    assert portfolio.drawdown_pct == pytest.approx(0.0)
    assert risk._check_drawdown() is True

    equity = portfolio.update_marks({"BTC/USDT": 70.0})
    # (70 - 100) * 50 = -1,500. Equity 8,500 is 15% below the 10,000 peak.
    assert equity == pytest.approx(8_500.0)
    assert portfolio.drawdown_pct == pytest.approx(15.0)
    assert risk._check_drawdown() is False


def test_short_mark_and_peak_are_kept_after_a_giveback():
    portfolio = _portfolio()
    portfolio.open_positions["BTC/USDT"] = {
        "side": "sell",
        "quantity": 10.0,
        "entry_price": 100.0,
        "value_usd": 1_000.0,
    }
    assert portfolio.update_marks({"BTC/USDT": 80.0}) == pytest.approx(10_200.0)
    assert portfolio.update_marks({"BTC/USDT": 110.0}) == pytest.approx(9_900.0)
    assert portfolio.peak_capital == pytest.approx(10_200.0)
    assert portfolio.drawdown_pct == pytest.approx((1 - 9_900.0 / 10_200.0) * 100)


def test_size_is_capped_at_cash_and_exposure_instead_of_levering():
    portfolio = _portfolio()
    risk = _risk(portfolio)
    # Stop distance = 2.5 * 0.4 = 1. Risk budget $150 buys 150 units, notional $15,000.
    capped = risk.calculate_trade_parameters("BTC/USDT", "buy", 100.0, atr=0.4)
    assert capped is not None
    assert capped.position_value_usd == pytest.approx(2_500.0)
    assert capped.quantity == pytest.approx(25.0)
    assert capped.risk_usd == pytest.approx(25.0)
    assert "capped" in capped.notes
    assert risk.is_trade_allowed(capped) is True

    # Wider stop: risk budget already fits under the 25% cap, so size is unchanged.
    uncapped = risk.calculate_trade_parameters("ETH/USDT", "buy", 100.0, atr=4.0)
    assert uncapped is not None
    assert uncapped.quantity == pytest.approx(15.0)
    assert uncapped.notes == ""


def test_sortino_is_zero_when_downside_deviation_is_undefined():
    assert RiskManager.compute_sortino_ratio(pd.Series(dtype=float)) == 0.0
    assert RiskManager.compute_sortino_ratio(pd.Series([0.01, 0.02, 0.0])) == 0.0
    assert RiskManager.compute_sortino_ratio(pd.Series([0.01, -0.02, 0.03])) == 0.0

    value = RiskManager.compute_sortino_ratio(
        pd.Series([0.01, -0.02, 0.01, -0.03]),
        risk_free_rate=0.0,
        periods_per_year=365,
    )
    assert value == pytest.approx(value)
    assert abs(value) < 100


def test_marks_from_ohlcv_uses_the_last_positive_close():
    index = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    frames = {
        "BTC/USDT": pd.DataFrame({"close": [1.0, 2.5]}, index=index),
        "ETH/USDT": pd.DataFrame({"close": [1.0, 0.0]}, index=index),
        "SOL/USDT": pd.DataFrame(),
    }
    assert marks_from_ohlcv(frames) == {"BTC/USDT": 2.5}
