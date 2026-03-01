"""
executor.py — Order Execution via CCXT
=======================================
Handles the translation from trade signals and parameters into
actual (or simulated) exchange orders.

Three execution modes:
  1. paper  — Simulates fills at current market price (no real orders)
  2. live   — Places real orders via CCXT (requires exchange credentials)
  3. backtest — Called by backtest.py for historical simulation

Safety features:
  - Requires explicit config mode = "live" for real orders
  - Confirmation prompt before first live trade
  - Rate-limit-aware retry logic
  - All fills written to TradeJournal and portfolio state
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from src.risk_manager import TradeParameters, PortfolioState
from src.strategy import TradeSignal, SignalType
from src.logger import get_logger, TradeJournal


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    """Represents the outcome of an order placement attempt."""
    success: bool
    order_id: Optional[str]
    symbol: str
    side: str
    quantity: float
    fill_price: float
    fill_time: datetime
    cost_usd: float
    commission_usd: float
    mode: str          # "paper" | "live" | "backtest"
    raw_response: Optional[dict] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Paper Trading Engine
# ─────────────────────────────────────────────────────────────

class PaperTradeEngine:
    """
    Simulates order fills at current market price.
    Tracks open positions and cash in-memory.
    Includes simulated commission and slippage.
    """

    COMMISSION_PCT = 0.001   # 0.1% per trade
    SLIPPAGE_PCT = 0.0005    # 0.05% market impact

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: dict[str, dict] = {}
        self.order_counter = 0
        self.logger = get_logger("PaperTrade")

    def fill_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
    ) -> OrderResult:
        """
        Simulate a market order fill.

        Applies slippage: buys fill slightly above, sells slightly below.
        """
        slip = self.SLIPPAGE_PCT
        if side == "buy":
            fill_price = current_price * (1 + slip)
        else:
            fill_price = current_price * (1 - slip)

        cost = quantity * fill_price
        commission = cost * self.COMMISSION_PCT

        self.order_counter += 1
        order_id = f"PAPER-{self.order_counter:06d}"

        if side == "buy":
            if cost + commission > self.cash:
                return OrderResult(
                    success=False,
                    order_id=None,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    fill_price=fill_price,
                    fill_time=datetime.utcnow(),
                    cost_usd=cost,
                    commission_usd=commission,
                    mode="paper",
                    error=f"Insufficient cash: need ${cost+commission:.2f}, have ${self.cash:.2f}",
                )
            self.cash -= (cost + commission)
            self.positions[symbol] = {
                "quantity": quantity,
                "entry_price": fill_price,
                "entry_time": datetime.utcnow(),
                "side": side,
            }
            self.logger.info(
                f"📄 PAPER BUY  {quantity:.6f} {symbol} @ {fill_price:.4f} "
                f"| Cost: ${cost:.2f} | Cash left: ${self.cash:.2f}"
            )

        else:  # sell
            existing = self.positions.get(symbol, {})
            sell_qty = min(quantity, existing.get("quantity", 0))
            if sell_qty <= 0:
                # Short selling (if exchange supports)
                sell_qty = quantity

            proceeds = sell_qty * fill_price - commission
            self.cash += proceeds
            if symbol in self.positions:
                del self.positions[symbol]
            self.logger.info(
                f"📄 PAPER SELL {sell_qty:.6f} {symbol} @ {fill_price:.4f} "
                f"| Proceeds: ${proceeds:.2f} | Cash: ${self.cash:.2f}"
            )

        return OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            fill_time=datetime.utcnow(),
            cost_usd=cost,
            commission_usd=commission,
            mode="paper",
        )

    def get_balance(self) -> dict:
        """Return current portfolio value breakdown."""
        return {
            "cash_usd": round(self.cash, 2),
            "positions": {
                sym: {
                    "qty": pos["quantity"],
                    "entry": pos["entry_price"],
                }
                for sym, pos in self.positions.items()
            },
        }


# ─────────────────────────────────────────────────────────────
# Live Executor
# ─────────────────────────────────────────────────────────────

class LiveExecutor:
    """
    Places real orders on the exchange via CCXT.

    ⚠️  This class operates with REAL FUNDS when config.mode = "live".
    Double-check all parameters before enabling live mode.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(self, ccxt_fetcher) -> None:
        self.fetcher = ccxt_fetcher
        self.logger = get_logger("LiveExecutor")
        self._confirmed = False

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> OrderResult:
        """
        Place a market order on the exchange.

        For the first live order, requires interactive confirmation
        (unless running in automated headless mode with CONFIRM_LIVE=1 env var).
        """
        import os
        if not self._confirmed and os.getenv("CONFIRM_LIVE", "0") != "1":
            confirm = input(
                f"\n⚠️  LIVE ORDER: {side.upper()} {quantity:.6f} {symbol}\n"
                f"Type 'yes' to confirm, anything else to abort: "
            )
            if confirm.strip().lower() != "yes":
                self.logger.warning("Live order aborted by user.")
                return OrderResult(
                    success=False, order_id=None, symbol=symbol, side=side,
                    quantity=quantity, fill_price=0, fill_time=datetime.utcnow(),
                    cost_usd=0, commission_usd=0, mode="live",
                    error="Aborted by user"
                )
            self._confirmed = True

        for attempt in range(self.MAX_RETRIES):
            try:
                order = self.fetcher.exchange.create_market_order(
                    symbol, side, quantity
                )
                fill_price = float(
                    order.get("average") or order.get("price") or
                    order.get("info", {}).get("avgPrice", 0)
                )
                cost = float(order.get("cost", quantity * fill_price))
                fee = float(order.get("fee", {}).get("cost", 0))

                self.logger.info(
                    f"✅ LIVE {side.upper()} {quantity:.6f} {symbol} "
                    f"@ {fill_price:.4f} | Cost: ${cost:.2f} | Fee: ${fee:.4f}"
                )
                return OrderResult(
                    success=True,
                    order_id=str(order.get("id")),
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    fill_price=fill_price,
                    fill_time=datetime.utcnow(),
                    cost_usd=cost,
                    commission_usd=fee,
                    mode="live",
                    raw_response=order,
                )

            except ccxt.InsufficientFunds as e:
                self.logger.error(f"Insufficient funds: {e}")
                return OrderResult(
                    success=False, order_id=None, symbol=symbol, side=side,
                    quantity=quantity, fill_price=0, fill_time=datetime.utcnow(),
                    cost_usd=0, commission_usd=0, mode="live",
                    error=str(e)
                )
            except ccxt.RateLimitExceeded:
                wait = self.RETRY_DELAY * (2 ** attempt)
                self.logger.warning(f"Rate limit hit — waiting {wait:.1f}s")
                time.sleep(wait)
            except ccxt.ExchangeError as e:
                self.logger.error(f"Exchange error (attempt {attempt+1}): {e}")
                if attempt == self.MAX_RETRIES - 1:
                    return OrderResult(
                        success=False, order_id=None, symbol=symbol, side=side,
                        quantity=quantity, fill_price=0, fill_time=datetime.utcnow(),
                        cost_usd=0, commission_usd=0, mode="live", error=str(e)
                    )
                time.sleep(self.RETRY_DELAY)


# ─────────────────────────────────────────────────────────────
# Order Manager — top-level facade
# ─────────────────────────────────────────────────────────────

class OrderManager:
    """
    Unified order management layer.

    Routes to PaperTradeEngine or LiveExecutor based on config mode.
    Maintains open position state and calls TradeJournal on close.
    """

    def __init__(
        self,
        config: dict,
        portfolio: PortfolioState,
        ccxt_fetcher=None,
        trade_journal: Optional[TradeJournal] = None,
    ) -> None:
        self.config = config
        self.portfolio = portfolio
        self.journal = trade_journal
        self.mode = config.get("trading", {}).get("mode", "paper")
        self.logger = get_logger("OrderManager")

        if self.mode == "paper" or self.mode == "backtest":
            initial = config.get("backtest", {}).get("initial_capital", 10_000)
            self.engine = PaperTradeEngine(initial_capital=float(initial))
            self.logger.info(f"Order manager: PAPER mode (capital=${initial:,.0f})")
        elif self.mode == "live":
            if ccxt_fetcher is None:
                raise RuntimeError("CCXTFetcher required for live mode.")
            self.engine = LiveExecutor(ccxt_fetcher)
            self.logger.warning(
                "Order manager: ⚠️  LIVE mode — real funds at risk!"
            )
        else:
            raise ValueError(f"Unknown trading mode: {self.mode}")

    def open_trade(self, signal: TradeSignal) -> Optional[OrderResult]:
        """
        Open a new position based on a trade signal.

        Updates portfolio state and notifies the trade journal.
        """
        if not signal.is_actionable() or signal.signal_type == SignalType.EXIT:
            return None

        params = signal.trade_params
        if params is None:
            self.logger.warning(f"Signal has no trade_params: {signal.symbol}")
            return None

        side = "buy" if signal.signal_type == SignalType.LONG else "sell"
        current_price = signal.current_price

        result = self._place(params.symbol, side, params.quantity, current_price)

        if result.success:
            # Update portfolio state
            self.portfolio.open_positions[params.symbol] = {
                "side": side,
                "quantity": params.quantity,
                "entry_price": result.fill_price,
                "stop_loss": params.stop_loss,
                "take_profit": params.take_profit,
                "entry_time": result.fill_time,
                "order_id": result.order_id,
                "value_usd": result.cost_usd,
            }
            self.portfolio.current_capital -= result.commission_usd
            self.logger.info(
                f"Position opened: {side.upper()} {params.quantity:.6f} {params.symbol} "
                f"@ {result.fill_price:.4f} | SL={params.stop_loss:.4f} | TP={params.take_profit:.4f}"
            )

        return result

    def close_trade(
        self,
        symbol: str,
        current_price: float,
        reason: str = "",
    ) -> Optional[OrderResult]:
        """
        Close an open position for `symbol` at current market price.
        Records the completed trade in the journal.
        """
        position = self.portfolio.open_positions.get(symbol)
        if not position:
            self.logger.warning(f"No open position to close for {symbol}")
            return None

        # Close = opposite side of the open
        close_side = "sell" if position["side"] == "buy" else "buy"
        quantity = position["quantity"]

        result = self._place(symbol, close_side, quantity, current_price)

        if result.success:
            # Realise P&L
            entry = position["entry_price"]
            qty = position["quantity"]
            if position["side"] == "buy":
                pnl = (result.fill_price - entry) * qty - result.commission_usd
            else:
                pnl = (entry - result.fill_price) * qty - result.commission_usd

            self.portfolio.current_capital += result.cost_usd - result.commission_usd
            self.portfolio.current_capital += pnl if position["side"] == "buy" else 0
            self.portfolio.update_peak()

            # Append to history
            self.portfolio.trade_history.append({
                "symbol": symbol,
                "pnl_usd": pnl,
                "entry": entry,
                "exit": result.fill_price,
                "side": position["side"],
            })

            # Remove from open positions
            del self.portfolio.open_positions[symbol]

            # Journal entry
            if self.journal:
                self.journal.log_trade(
                    pair=symbol,
                    side=position["side"],
                    entry_price=entry,
                    exit_price=result.fill_price,
                    quantity=qty,
                    stop_loss=position.get("stop_loss", 0),
                    take_profit=position.get("take_profit", 0),
                    entry_time=position.get("entry_time", result.fill_time),
                    exit_time=result.fill_time,
                    notes=reason,
                )

            self.logger.info(
                f"Position closed: {symbol} | P&L: ${pnl:+.2f} | Reason: {reason}"
            )

        return result

    def _place(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> OrderResult:
        """Internal dispatch to the underlying engine."""
        if isinstance(self.engine, PaperTradeEngine):
            return self.engine.fill_order(symbol, side, quantity, price)
        elif isinstance(self.engine, LiveExecutor):
            return self.engine.place_market_order(symbol, side, quantity)
        else:
            raise RuntimeError("Unknown execution engine type")

    def get_open_positions(self) -> dict:
        return dict(self.portfolio.open_positions)

    def is_position_open(self, symbol: str) -> bool:
        return symbol in self.portfolio.open_positions
