"""
risk_manager.py — Position Sizing, Stop-Loss & Drawdown Controls
================================================================
Adapted from Portfolio-Risk-Analysis-Credit-Risk-Modeling principles.

Implements:
  1. Position sizing  — Kelly Criterion or fixed % risk model
  2. Stop-loss        — Volatility-based (ATR multiplier)
  3. Take-profit      — Reward:risk ratio or predicted price target
  4. Drawdown guard   — Halt trading if account drawdown exceeds limit
  5. Exposure cap     — Prevent over-concentration in a single asset
  6. Risk metrics     — Sharpe ratio, Sortino ratio, Max Drawdown helpers

Design philosophy:
  "Size positions to survive the worst; let winners run within reason."
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.logger import get_logger


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────

@dataclass
class TradeParameters:
    """
    Complete parameters for a proposed trade, calculated by RiskManager.
    The executor uses these values to size and place orders.
    """
    symbol: str
    side: str                  # "buy" | "sell"
    entry_price: float
    quantity: float            # Amount of base asset to trade
    stop_loss: float           # Price at which to exit for loss
    take_profit: float         # Price at which to exit for gain
    risk_usd: float            # USD amount at risk on this trade
    position_value_usd: float  # Total position value in USD
    risk_pct_of_account: float # % of account being risked
    reward_risk_ratio: float   # Realised R:R (may differ from target)
    sizing_method: str = "fixed_pct"  # "fixed_pct" | "kelly"
    notes: str = ""


@dataclass
class PortfolioState:
    """
    Snapshot of current portfolio used for drawdown and exposure calculations.
    Updated by the bot's main loop after every fill.
    """
    initial_capital: float
    current_capital: float
    open_positions: dict = field(default_factory=dict)   # {symbol: {"side", "qty", "entry", "value_usd"}}
    peak_capital: float = 0.0
    trade_history: list = field(default_factory=list)    # List of completed trade PnLs

    def __post_init__(self):
        if self.peak_capital == 0.0:
            self.peak_capital = self.current_capital

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak capital (positive = drawdown)."""
        if self.peak_capital == 0:
            return 0.0
        return (1 - self.current_capital / self.peak_capital) * 100

    @property
    def total_open_exposure_usd(self) -> float:
        """Sum of all open position values."""
        return sum(p["value_usd"] for p in self.open_positions.values())

    def exposure_for(self, symbol: str) -> float:
        """USD value of open position in a specific symbol."""
        pos = self.open_positions.get(symbol, {})
        return pos.get("value_usd", 0.0)

    def update_peak(self) -> None:
        """Call after every P&L update."""
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital


# ─────────────────────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────────────────────

class RiskManager:
    """
    Central risk gatekeeper for the trading bot.

    Usage:
        risk = RiskManager(config, portfolio_state)
        params = risk.calculate_trade_parameters(
            symbol="BTC/USDT",
            side="buy",
            entry_price=65000,
            atr=800,
            predicted_pct_change=2.1
        )
        if risk.is_trade_allowed(params):
            executor.place_order(params)
    """

    def __init__(self, config: dict, portfolio: PortfolioState) -> None:
        self.config = config
        self.portfolio = portfolio
        self.logger = get_logger("RiskManager")

        risk_cfg = config.get("risk", {})
        self.max_risk_per_trade_pct = risk_cfg.get("max_risk_per_trade_pct", 1.5)
        self.atr_stop_multiplier = risk_cfg.get("atr_stop_multiplier", 2.5)
        self.reward_risk_ratio = risk_cfg.get("reward_risk_ratio", 2.5)
        self.max_drawdown_pct = risk_cfg.get("max_drawdown_pct", 12.0)
        self.max_single_asset_pct = risk_cfg.get("max_single_asset_exposure_pct", 25.0)
        self.max_open_trades = risk_cfg.get("max_open_trades", 3)
        self.use_kelly = risk_cfg.get("use_kelly_criterion", False)
        self.kelly_fraction = risk_cfg.get("kelly_fraction", 0.25)

    # ── Position Sizing ───────────────────────────────────────

    def calculate_trade_parameters(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        atr: float,
        predicted_pct_change: Optional[float] = None,
        win_rate: Optional[float] = None,   # Required for Kelly
    ) -> Optional[TradeParameters]:
        """
        Calculate position size, stop-loss, and take-profit for a potential trade.

        Stop-loss is volatility-based: entry ± (atr_stop_multiplier × ATR).
        Take-profit is derived from reward:risk ratio or predicted price target.
        Position size keeps dollar risk ≤ max_risk_per_trade_pct of capital.

        Args:
            symbol:               Trading pair
            side:                 "buy" or "sell"
            entry_price:          Proposed entry price
            atr:                  Current ATR value (same timeframe as candle data)
            predicted_pct_change: LSTM forecast (optional, used to cap TP)
            win_rate:             Historical win rate (needed for Kelly)

        Returns:
            TradeParameters if viable, None if position is too small to be worth taking
        """
        if entry_price <= 0 or atr <= 0:
            self.logger.warning(f"Invalid price/ATR: entry={entry_price}, atr={atr}")
            return None

        capital = self.portfolio.current_capital

        # ── Stop distance ──────────────────────────────────────
        stop_distance = self.atr_stop_multiplier * atr

        if side == "buy":
            stop_loss = entry_price - stop_distance
            if stop_loss <= 0:
                stop_loss = entry_price * 0.95    # Hard floor: 5% below entry
        else:  # sell/short
            stop_loss = entry_price + stop_distance

        # ── Position sizing ────────────────────────────────────
        if self.use_kelly and win_rate is not None:
            quantity = self._kelly_size(
                capital, entry_price, stop_distance, win_rate
            )
        else:
            quantity = self._fixed_pct_size(capital, entry_price, stop_distance)

        if quantity <= 0:
            self.logger.debug(f"Position size too small — skipping {symbol}")
            return None

        position_value = quantity * entry_price
        risk_usd = quantity * stop_distance
        risk_pct = (risk_usd / capital) * 100

        # ── Take-profit ────────────────────────────────────────
        tp_from_rr = stop_distance * self.reward_risk_ratio

        if predicted_pct_change is not None:
            # Use the smaller of: RR-based TP or LSTM prediction (conservative)
            predicted_distance = abs(predicted_pct_change / 100) * entry_price
            tp_distance = min(tp_from_rr, predicted_distance * 1.1)
        else:
            tp_distance = tp_from_rr

        if side == "buy":
            take_profit = entry_price + tp_distance
        else:
            take_profit = entry_price - tp_distance

        realised_rr = tp_distance / stop_distance if stop_distance > 0 else 0

        params = TradeParameters(
            symbol=symbol,
            side=side,
            entry_price=round(entry_price, 8),
            quantity=round(quantity, 8),
            stop_loss=round(stop_loss, 8),
            take_profit=round(take_profit, 8),
            risk_usd=round(risk_usd, 4),
            position_value_usd=round(position_value, 4),
            risk_pct_of_account=round(risk_pct, 4),
            reward_risk_ratio=round(realised_rr, 3),
            sizing_method="kelly" if self.use_kelly else "fixed_pct",
        )

        self.logger.info(
            f"Trade params [{symbol} {side.upper()}]: "
            f"qty={quantity:.6f} @ {entry_price:.4f} | "
            f"SL={stop_loss:.4f} | TP={take_profit:.4f} | "
            f"Risk=${risk_usd:.2f} ({risk_pct:.2f}%) | R:R={realised_rr:.2f}"
        )
        return params

    def _fixed_pct_size(
        self, capital: float, entry_price: float, stop_distance: float
    ) -> float:
        """
        Fixed-fraction position sizing.
        Risk exactly max_risk_per_trade_pct% of account on the stop distance.

        qty = (capital × risk%) / stop_distance
        """
        max_risk_usd = capital * (self.max_risk_per_trade_pct / 100)
        if stop_distance <= 0:
            return 0.0
        return max_risk_usd / stop_distance

    def _kelly_size(
        self,
        capital: float,
        entry_price: float,
        stop_distance: float,
        win_rate: float,
    ) -> float:
        """
        Fractional Kelly Criterion position sizing.

        Full Kelly: f* = W - (1-W)/R
          W = win probability
          R = avg_win / avg_loss (assumed to be reward_risk_ratio here)
        Fractional Kelly = kelly_fraction × f*

        Caps at max_risk_per_trade_pct for safety.
        """
        R = self.reward_risk_ratio
        full_kelly = win_rate - (1 - win_rate) / R
        full_kelly = max(0.0, full_kelly)          # Never negative
        fractional_kelly = self.kelly_fraction * full_kelly

        # Cap at max risk per trade
        capped_kelly = min(fractional_kelly, self.max_risk_per_trade_pct / 100)

        max_risk_usd = capital * capped_kelly
        if stop_distance <= 0:
            return 0.0
        qty = max_risk_usd / stop_distance

        self.logger.debug(
            f"Kelly sizing: full_kelly={full_kelly:.3f}, "
            f"fractional={fractional_kelly:.3f}, "
            f"capped={capped_kelly:.3f}"
        )
        return qty

    # ── Trade Approval Gates ──────────────────────────────────

    def is_trade_allowed(self, params: TradeParameters) -> bool:
        """
        Run all risk checks. Returns True only if ALL pass.

        Checks:
          1. Drawdown guard — account drawdown within limit
          2. Max open trades — not holding too many positions
          3. Single-asset exposure cap
          4. Minimum position value (avoid dust trades)
          5. Risk per trade cap
        """
        if not self._check_drawdown():
            return False

        if not self._check_max_open_trades(params.symbol):
            return False

        if not self._check_single_asset_exposure(params):
            return False

        if not self._check_min_position_value(params):
            return False

        if not self._check_risk_per_trade(params):
            return False

        return True

    def _check_drawdown(self) -> bool:
        """Halt trading if current drawdown exceeds max_drawdown_pct."""
        dd = self.portfolio.drawdown_pct
        if dd >= self.max_drawdown_pct:
            self.logger.error(
                f"⛔ DRAWDOWN HALT: Current drawdown {dd:.2f}% "
                f"≥ limit {self.max_drawdown_pct:.2f}%. "
                f"Bot halted until manual review."
            )
            return False
        if dd >= self.max_drawdown_pct * 0.75:
            self.logger.warning(
                f"⚠️  Approaching drawdown limit: {dd:.2f}% "
                f"(limit {self.max_drawdown_pct:.2f}%)"
            )
        return True

    def _check_max_open_trades(self, symbol: str) -> bool:
        """Reject new trades if max open trades is reached (unless same symbol)."""
        open_positions = self.portfolio.open_positions
        # Allow adding to existing position (handled in strategy layer)
        if symbol in open_positions:
            return True
        if len(open_positions) >= self.max_open_trades:
            self.logger.info(
                f"Max open trades reached ({self.max_open_trades}). "
                f"Skipping new position in {symbol}."
            )
            return False
        return True

    def _check_single_asset_exposure(self, params: TradeParameters) -> bool:
        """Ensure a single asset doesn't exceed max exposure cap."""
        capital = self.portfolio.current_capital
        if capital <= 0:
            return False

        existing_exposure = self.portfolio.exposure_for(params.symbol)
        total_new_exposure = existing_exposure + params.position_value_usd
        new_exposure_pct = (total_new_exposure / capital) * 100

        if new_exposure_pct > self.max_single_asset_pct:
            self.logger.info(
                f"Exposure cap for {params.symbol}: "
                f"{new_exposure_pct:.1f}% > {self.max_single_asset_pct:.1f}%"
            )
            return False
        return True

    def _check_min_position_value(self, params: TradeParameters) -> bool:
        """Minimum trade size: $10 USD equivalent."""
        if params.position_value_usd < 10:
            self.logger.debug(
                f"Position value too small: ${params.position_value_usd:.2f}"
            )
            return False
        return True

    def _check_risk_per_trade(self, params: TradeParameters) -> bool:
        """Risk per trade must not exceed 2× the configured limit (safety net)."""
        hard_limit = self.max_risk_per_trade_pct * 2
        if params.risk_pct_of_account > hard_limit:
            self.logger.warning(
                f"Risk {params.risk_pct_of_account:.2f}% exceeds hard limit "
                f"{hard_limit:.2f}%"
            )
            return False
        return True

    # ── Dynamic Stop Management ───────────────────────────────

    def calculate_trailing_stop(
        self,
        side: str,
        current_price: float,
        entry_price: float,
        current_stop: float,
        atr: float,
        activation_pct: float = 1.0,
    ) -> float:
        """
        Calculate a trailing stop price.

        Activates once trade is profitable by activation_pct%.
        Trails at atr_stop_multiplier × ATR behind current price.

        Args:
            side:            "buy" or "sell"
            current_price:   Latest market price
            entry_price:     Original entry price
            current_stop:    Current stop-loss price
            atr:             Current ATR
            activation_pct:  % profit before trailing activates

        Returns:
            New stop-loss price (may be same as current if not yet trailing)
        """
        trail_distance = self.atr_stop_multiplier * atr

        if side == "buy":
            profit_pct = (current_price - entry_price) / entry_price * 100
            if profit_pct < activation_pct:
                return current_stop   # Not yet in profit enough to trail
            new_stop = current_price - trail_distance
            return max(new_stop, current_stop)   # Only move stop UP
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct < activation_pct:
                return current_stop
            new_stop = current_price + trail_distance
            return min(new_stop, current_stop)   # Only move stop DOWN

    # ── Portfolio Metrics ─────────────────────────────────────

    @staticmethod
    def compute_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 365 * 24,  # Hourly candles, 24/7
    ) -> float:
        """
        Compute annualised Sharpe Ratio from a series of period returns.

        Args:
            returns:          Series of per-period decimal returns
            risk_free_rate:   Annual risk-free rate (default 5%)
            periods_per_year: For hourly crypto data: 365 × 24

        Returns:
            Annualised Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        rf_per_period = risk_free_rate / periods_per_year
        excess = returns - rf_per_period
        return float((excess.mean() / excess.std()) * np.sqrt(periods_per_year))

    @staticmethod
    def compute_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 365 * 24,
    ) -> float:
        """
        Sortino Ratio — like Sharpe but only penalises downside volatility.
        Better suited to crypto's asymmetric return distribution.
        """
        if returns.empty:
            return 0.0
        rf_per_period = risk_free_rate / periods_per_year
        excess = returns - rf_per_period
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 1 else 1e-9
        return float((excess.mean() / downside_std) * np.sqrt(periods_per_year))

    @staticmethod
    def compute_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Compute maximum drawdown percentage from an equity curve.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Maximum drawdown as a positive percentage (e.g., 15.3 = -15.3%)
        """
        if equity_curve.empty:
            return 0.0
        roll_max = equity_curve.cummax()
        drawdown = (equity_curve - roll_max) / roll_max * 100
        return float(abs(drawdown.min()))

    @staticmethod
    def compute_calmar_ratio(
        annualised_return_pct: float, max_drawdown_pct: float
    ) -> float:
        """
        Calmar Ratio = Annualised return / Max drawdown.
        Higher is better. Used to evaluate risk-adjusted backtest performance.
        """
        if max_drawdown_pct == 0:
            return 0.0
        return annualised_return_pct / max_drawdown_pct

    def get_portfolio_summary(self) -> dict:
        """Return a human-readable summary of current portfolio state."""
        p = self.portfolio
        trade_pnls = [t.get("pnl_usd", 0) for t in p.trade_history]
        wins = sum(1 for pnl in trade_pnls if pnl > 0)
        total = len(trade_pnls)
        win_rate = (wins / total * 100) if total > 0 else 0.0

        return {
            "initial_capital_usd": p.initial_capital,
            "current_capital_usd": round(p.current_capital, 2),
            "unrealised_pnl_usd": round(p.current_capital - p.initial_capital, 2),
            "peak_capital_usd": round(p.peak_capital, 2),
            "drawdown_pct": round(p.drawdown_pct, 2),
            "open_positions": len(p.open_positions),
            "open_exposure_usd": round(p.total_open_exposure_usd, 2),
            "total_trades": total,
            "win_rate_pct": round(win_rate, 2),
            "total_realised_pnl": round(sum(trade_pnls), 2),
        }
