"""Paper-engine short accounting.

A short sale must record a negative position. Covering that short has to
reduce or close it. Treating the cover as a fresh buy leaves a ghost long
and inflates equity.
"""

from datetime import datetime

import pytest

from src.executor import OrderManager, PaperTradeEngine
from src.risk_manager import PortfolioState, TradeParameters
from src.strategy import SignalSource, SignalType, TradeSignal


SYMBOL = "BTC/USDT"
INITIAL = 10_000.0


def _fill_price(side: str, price: float) -> float:
    slip = PaperTradeEngine.SLIPPAGE_PCT
    if side == "buy":
        return price * (1 + slip)
    return price * (1 - slip)


def _fee(quantity: float, fill_price: float) -> float:
    return quantity * fill_price * PaperTradeEngine.COMMISSION_PCT


def _equity(engine: PaperTradeEngine, mark: float) -> float:
    """Cash plus signed inventory marked at `mark`. A short subtracts cover value."""
    quantity = engine.positions.get(SYMBOL, {}).get("quantity", 0.0)
    return engine.cash + quantity * mark


def test_sell_with_no_long_opens_a_short():
    engine = PaperTradeEngine(INITIAL)
    result = engine.fill_order(SYMBOL, "sell", 2.0, 100.0)

    assert result.success
    fill = _fill_price("sell", 100.0)
    fee = _fee(2.0, fill)
    position = engine.positions[SYMBOL]

    assert position["quantity"] == pytest.approx(-2.0)
    assert position["side"] == "sell"
    assert position["entry_price"] == pytest.approx(fill)
    assert engine.get_balance()["positions"][SYMBOL]["qty"] == pytest.approx(-2.0)
    # Short-sale proceeds are credited, net of commission.
    assert engine.cash == pytest.approx(INITIAL + 2.0 * fill - fee)
    # At the fill, the short's mark cancels the proceeds. Only the fee is spent.
    assert _equity(engine, fill) == pytest.approx(INITIAL - fee)


def test_full_cover_flattens_short_and_realizes_pnl():
    engine = PaperTradeEngine(INITIAL)
    engine.fill_order(SYMBOL, "sell", 2.0, 100.0)
    cover = engine.fill_order(SYMBOL, "buy", 2.0, 90.0)

    assert cover.success
    assert SYMBOL not in engine.positions
    assert engine.get_balance()["positions"] == {}

    short_fill = _fill_price("sell", 100.0)
    cover_fill = _fill_price("buy", 90.0)
    fees = _fee(2.0, short_fill) + _fee(2.0, cover_fill)
    expected_pnl = (short_fill - cover_fill) * 2.0 - fees

    assert engine.cash == pytest.approx(INITIAL + expected_pnl)
    assert engine.cash - engine.initial_capital == pytest.approx(expected_pnl)
    # Flat book: equity is cash. A ghost long would add the cover notional on top.
    assert _equity(engine, cover_fill) == pytest.approx(INITIAL + expected_pnl)


def test_partial_cover_reduces_short_without_opening_a_long():
    engine = PaperTradeEngine(INITIAL)
    engine.fill_order(SYMBOL, "sell", 10.0, 100.0)
    engine.fill_order(SYMBOL, "buy", 4.0, 80.0)

    short_fill = _fill_price("sell", 100.0)
    cover_fill = _fill_price("buy", 80.0)
    position = engine.positions[SYMBOL]

    assert position["quantity"] == pytest.approx(-6.0)
    assert position["side"] == "sell"
    assert position["entry_price"] == pytest.approx(short_fill)

    open_fee = _fee(10.0, short_fill)
    cover_fee = _fee(4.0, cover_fill)
    assert engine.cash == pytest.approx(
        INITIAL + 10.0 * short_fill - open_fee - 4.0 * cover_fill - cover_fee
    )
    realized = (short_fill - cover_fill) * 4.0 - cover_fee
    unrealized = (short_fill - cover_fill) * 6.0
    assert _equity(engine, cover_fill) == pytest.approx(
        INITIAL - open_fee + realized + unrealized
    )


def test_adding_to_a_short_averages_the_entry():
    engine = PaperTradeEngine(INITIAL)
    engine.fill_order(SYMBOL, "sell", 4.0, 100.0)
    engine.fill_order(SYMBOL, "sell", 6.0, 80.0)

    first = _fill_price("sell", 100.0)
    second = _fill_price("sell", 80.0)
    position = engine.positions[SYMBOL]

    assert position["quantity"] == pytest.approx(-10.0)
    assert position["side"] == "sell"
    assert position["entry_price"] == pytest.approx((4.0 * first + 6.0 * second) / 10.0)
    open_fee = _fee(4.0, first) + _fee(6.0, second)
    assert engine.cash == pytest.approx(INITIAL + 4.0 * first + 6.0 * second - open_fee)
    # Marked at the averaged entry, only commissions have been spent.
    assert _equity(engine, position["entry_price"]) == pytest.approx(INITIAL - open_fee)


def test_buy_larger_than_short_flips_to_long():
    engine = PaperTradeEngine(INITIAL)
    engine.fill_order(SYMBOL, "sell", 10.0, 100.0)
    result = engine.fill_order(SYMBOL, "buy", 16.0, 90.0)

    assert result.success
    short_fill = _fill_price("sell", 100.0)
    buy_fill = _fill_price("buy", 90.0)
    position = engine.positions[SYMBOL]

    assert position["quantity"] == pytest.approx(6.0)
    assert position["side"] == "buy"
    assert position["entry_price"] == pytest.approx(buy_fill)

    realized = (short_fill - buy_fill) * 10.0
    fees = _fee(10.0, short_fill) + _fee(16.0, buy_fill)
    # Residual long was opened at this fill, so it has no unrealized PnL here.
    assert _equity(engine, buy_fill) == pytest.approx(INITIAL + realized - fees)
    assert engine.cash == pytest.approx(
        INITIAL
        + 10.0 * short_fill
        - _fee(10.0, short_fill)
        - 16.0 * buy_fill
        - _fee(16.0, buy_fill)
    )


def test_sell_larger_than_long_flips_to_short():
    engine = PaperTradeEngine(INITIAL)
    engine.fill_order(SYMBOL, "buy", 10.0, 100.0)
    result = engine.fill_order(SYMBOL, "sell", 14.0, 110.0)

    assert result.success
    long_fill = _fill_price("buy", 100.0)
    sell_fill = _fill_price("sell", 110.0)
    position = engine.positions[SYMBOL]

    assert position["quantity"] == pytest.approx(-4.0)
    assert position["side"] == "sell"
    assert position["entry_price"] == pytest.approx(sell_fill)

    realized = (sell_fill - long_fill) * 10.0
    fees = _fee(10.0, long_fill) + _fee(14.0, sell_fill)
    assert _equity(engine, sell_fill) == pytest.approx(INITIAL + realized - fees)


def test_order_manager_short_round_trip_does_not_leave_a_ghost_long():
    portfolio = PortfolioState(initial_capital=INITIAL, current_capital=INITIAL)
    manager = OrderManager(
        {"trading": {"mode": "paper"}, "backtest": {"initial_capital": INITIAL}},
        portfolio,
    )
    opened = manager.open_trade(
        TradeSignal(
            symbol=SYMBOL,
            signal_type=SignalType.SHORT,
            source=SignalSource.RULES_ONLY,
            strength=0.8,
            timestamp=datetime(2026, 1, 1),
            current_price=100.0,
            trade_params=TradeParameters(
                symbol=SYMBOL,
                side="sell",
                entry_price=100.0,
                quantity=2.0,
                stop_loss=110.0,
                take_profit=80.0,
                risk_usd=20.0,
                position_value_usd=200.0,
                risk_pct_of_account=0.2,
                reward_risk_ratio=2.0,
            ),
        )
    )
    assert opened is not None and opened.success
    assert manager.engine.positions[SYMBOL]["quantity"] == pytest.approx(-2.0)
    assert manager.engine.positions[SYMBOL]["side"] == "sell"

    closed = manager.close_trade(SYMBOL, current_price=90.0, reason="cover")
    assert closed is not None and closed.success
    assert SYMBOL not in manager.portfolio.open_positions
    assert SYMBOL not in manager.engine.positions

    net_pnl = (
        (opened.fill_price - closed.fill_price) * 2.0
        - opened.commission_usd
        - closed.commission_usd
    )
    assert manager.engine.cash == pytest.approx(INITIAL + net_pnl)
    assert manager.portfolio.current_capital == pytest.approx(INITIAL + net_pnl)
