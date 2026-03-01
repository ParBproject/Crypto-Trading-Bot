"""
strategy.py — Hybrid Signal Generation
=======================================
Combines LSTM predictions with classical technical analysis filters
to produce high-conviction trade signals.

Signal logic (default):
  LONG  if: LSTM predicts +pct ≥ long_threshold
            AND RSI < rsi_overbought (not overbought)
            AND (MACD > signal OR MACD crossing upward)
            AND volume_ratio > volume_surge_multiplier
            AND price > ema_20 (short-term uptrend)

  SHORT if: LSTM predicts -pct ≤ short_threshold
            AND RSI > rsi_oversold (not oversold)
            AND MACD < signal
            AND price < ema_20

  EXIT  if: Trailing stop hit, take-profit hit, or signal reversal

Extensible: Subclass BaseStrategy to implement alternative strategies
without modifying the orchestration code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np

from src.predictor import PredictionResult
from src.risk_manager import RiskManager, TradeParameters, PortfolioState
from src.logger import get_logger


# ─────────────────────────────────────────────────────────────
# Signal Enums & Data Classes
# ─────────────────────────────────────────────────────────────

class SignalType(Enum):
    LONG  = "long"
    SHORT = "short"
    EXIT  = "exit"
    HOLD  = "hold"


class SignalSource(Enum):
    LSTM_ONLY   = "lstm_only"
    RULES_ONLY  = "rules_only"
    COMBINED    = "combined"
    EXIT_SIGNAL = "exit"


@dataclass
class TradeSignal:
    """
    Structured output from the strategy layer.
    Consumed by the executor to place or close orders.
    """
    symbol: str
    signal_type: SignalType
    source: SignalSource
    strength: float              # 0–1, overall signal conviction
    timestamp: datetime
    current_price: float
    predicted_pct_change: Optional[float] = None
    prediction_confidence: Optional[float] = None
    trade_params: Optional[TradeParameters] = None
    reasons: list = field(default_factory=list)  # Human-readable justifications
    warnings: list = field(default_factory=list)

    def is_actionable(self) -> bool:
        return self.signal_type in (SignalType.LONG, SignalType.SHORT, SignalType.EXIT)

    def __str__(self) -> str:
        return (
            f"[{self.signal_type.value.upper()}] {self.symbol} @ {self.current_price:.4f} "
            f"| strength={self.strength:.2f} | src={self.source.value} "
            f"| pred={self.predicted_pct_change:+.2f}%"
            if self.predicted_pct_change is not None
            else f"[{self.signal_type.value.upper()}] {self.symbol} @ {self.current_price:.4f}"
        )


# ─────────────────────────────────────────────────────────────
# Base Strategy Interface
# ─────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """Abstract base — implement generate_signal() in subclasses."""

    def __init__(self, config: dict, risk_manager: RiskManager) -> None:
        self.config = config
        self.risk = risk_manager
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        prediction: Optional[PredictionResult],
        existing_position: Optional[dict],
    ) -> TradeSignal:
        """
        Evaluate market data and prediction to produce a trade signal.

        Args:
            symbol:            Trading pair (e.g. "BTC/USDT")
            df:                Enriched OHLCV DataFrame (with indicators)
            prediction:        LSTM PredictionResult, or None if unavailable
            existing_position: Current open position dict, or None

        Returns:
            TradeSignal
        """
        ...


# ─────────────────────────────────────────────────────────────
# Hybrid LSTM + Technical Analysis Strategy
# ─────────────────────────────────────────────────────────────

class HybridLSTMStrategy(BaseStrategy):
    """
    Primary strategy: ML predictions filtered through TA confirmation gates.

    The philosophy is:
      1. LSTM provides the *directional bias* (long / short / neutral)
      2. Technical filters *confirm* the bias and reduce false signals
      3. Risk manager *sizes* the trade if approved

    Strength scoring: Each satisfied condition adds to a 0–1 score.
    Trade is only generated if score ≥ min_strength threshold.
    """

    MIN_SIGNAL_STRENGTH = 0.55   # At least 55% of conditions satisfied

    def __init__(self, config: dict, risk_manager: RiskManager) -> None:
        super().__init__(config, risk_manager)
        strat_cfg = config.get("strategy", {})

        self.long_threshold = strat_cfg.get("long_threshold_pct", 1.5)
        self.short_threshold = strat_cfg.get("short_threshold_pct", -1.5)
        self.rsi_oversold = strat_cfg.get("rsi_oversold", 35)
        self.rsi_overbought = strat_cfg.get("rsi_overbought", 65)
        self.vol_surge = strat_cfg.get("volume_surge_multiplier", 1.5)

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        prediction: Optional[PredictionResult],
        existing_position: Optional[dict] = None,
    ) -> TradeSignal:
        """
        Full signal generation pipeline.

        Step 1: Check for exit conditions on open positions
        Step 2: Check long entry conditions
        Step 3: Check short entry conditions
        Step 4: Default to HOLD
        """
        if df.empty or len(df) < 2:
            return self._hold(symbol, 0.0, "Insufficient data")

        latest = df.iloc[-1]
        current_price = float(latest.get("close", 0))
        if current_price <= 0:
            return self._hold(symbol, 0.0, "Invalid price")

        # ── Step 1: Exit existing position? ───────────────────
        if existing_position:
            exit_signal = self._check_exit(
                symbol, current_price, df, existing_position, prediction
            )
            if exit_signal:
                return exit_signal

        # ── If we have no prediction, fall back to TA-only ────
        if prediction is None:
            return self._ta_only_signal(symbol, df, existing_position)

        predicted_pct = prediction.predicted_pct_change
        confidence = prediction.confidence

        # ── Step 2: Long entry ────────────────────────────────
        if predicted_pct >= self.long_threshold and existing_position is None:
            signal = self._evaluate_long(
                symbol, df, latest, current_price, predicted_pct, confidence
            )
            if signal:
                return signal

        # ── Step 3: Short entry ───────────────────────────────
        if predicted_pct <= self.short_threshold and existing_position is None:
            signal = self._evaluate_short(
                symbol, df, latest, current_price, predicted_pct, confidence
            )
            if signal:
                return signal

        return self._hold(
            symbol, current_price,
            f"No entry: pred={predicted_pct:+.2f}%, conf={confidence:.2f}"
        )

    # ── Long Logic ────────────────────────────────────────────

    def _evaluate_long(
        self,
        symbol: str,
        df: pd.DataFrame,
        latest: pd.Series,
        price: float,
        predicted_pct: float,
        confidence: float,
    ) -> Optional[TradeSignal]:
        """Score long entry conditions. Return signal if strength ≥ threshold."""
        score = 0.0
        total_checks = 6
        reasons = []
        warnings = []

        # 1. LSTM prediction strength
        lstm_score = min(abs(predicted_pct) / (self.long_threshold * 3), 1.0) * confidence
        score += lstm_score
        reasons.append(f"LSTM: +{predicted_pct:.2f}% (conf={confidence:.2f})")

        # 2. RSI not overbought
        rsi = self._get_val(latest, "rsi")
        if rsi is not None:
            if rsi < self.rsi_overbought:
                score += 1
                reasons.append(f"RSI={rsi:.1f} (< {self.rsi_overbought} ✓)")
            else:
                warnings.append(f"RSI={rsi:.1f} overbought")
                score += 0.3   # Partial credit — not a dealbreaker

        # 3. MACD bullish
        macd = self._get_val(latest, "macd")
        macd_sig = self._get_val(latest, "macd_signal")
        if macd is not None and macd_sig is not None:
            if macd > macd_sig:
                score += 1
                reasons.append(f"MACD bullish ({macd:.4f} > {macd_sig:.4f})")
            else:
                # Check for recent crossover (last 2 candles)
                if len(df) >= 2:
                    prev = df.iloc[-2]
                    prev_macd = self._get_val(prev, "macd")
                    prev_sig = self._get_val(prev, "macd_signal")
                    if prev_macd is not None and prev_sig is not None:
                        if prev_macd < prev_sig and macd > macd_sig:
                            score += 1
                            reasons.append("MACD bullish crossover ✓")
                        else:
                            warnings.append("MACD bearish")

        # 4. Volume surge
        vol_ratio = self._get_val(latest, "volume_ratio")
        if vol_ratio is not None:
            if vol_ratio >= self.vol_surge:
                score += 1
                reasons.append(f"Volume surge: {vol_ratio:.1f}x avg ✓")
            else:
                warnings.append(f"Low volume: {vol_ratio:.1f}x avg")

        # 5. Price above EMA-20 (short-term uptrend)
        ema20 = self._get_val(latest, "ema_20")
        if ema20 is not None:
            if price > ema20:
                score += 1
                reasons.append(f"Price > EMA-20 ({price:.2f} > {ema20:.2f}) ✓")
            else:
                warnings.append(f"Price below EMA-20")

        # 6. Price above EMA-50 (medium-term trend)
        ema50 = self._get_val(latest, "ema_50")
        if ema50 is not None:
            if price > ema50:
                score += 1
                reasons.append(f"Price > EMA-50 ✓")

        strength = score / total_checks
        if strength < self.MIN_SIGNAL_STRENGTH:
            self.logger.debug(
                f"{symbol} LONG rejected: strength={strength:.2f} < {self.MIN_SIGNAL_STRENGTH}"
            )
            return None

        # Calculate trade parameters
        atr = self._get_val(latest, "atr") or price * 0.01
        trade_params = self.risk.calculate_trade_parameters(
            symbol=symbol,
            side="buy",
            entry_price=price,
            atr=atr,
            predicted_pct_change=predicted_pct,
        )

        if trade_params is None or not self.risk.is_trade_allowed(trade_params):
            return self._hold(symbol, price, "Risk check failed for long")

        return TradeSignal(
            symbol=symbol,
            signal_type=SignalType.LONG,
            source=SignalSource.COMBINED,
            strength=round(strength, 3),
            timestamp=datetime.utcnow(),
            current_price=price,
            predicted_pct_change=predicted_pct,
            prediction_confidence=confidence,
            trade_params=trade_params,
            reasons=reasons,
            warnings=warnings,
        )

    # ── Short Logic ───────────────────────────────────────────

    def _evaluate_short(
        self,
        symbol: str,
        df: pd.DataFrame,
        latest: pd.Series,
        price: float,
        predicted_pct: float,
        confidence: float,
    ) -> Optional[TradeSignal]:
        """Mirror of _evaluate_long for short/sell signals."""
        score = 0.0
        total_checks = 6
        reasons = []
        warnings = []

        # 1. LSTM prediction
        lstm_score = min(abs(predicted_pct) / (abs(self.short_threshold) * 3), 1.0) * confidence
        score += lstm_score
        reasons.append(f"LSTM: {predicted_pct:.2f}% (conf={confidence:.2f})")

        # 2. RSI not oversold
        rsi = self._get_val(latest, "rsi")
        if rsi is not None:
            if rsi > self.rsi_oversold:
                score += 1
                reasons.append(f"RSI={rsi:.1f} (> {self.rsi_oversold} ✓)")
            else:
                warnings.append(f"RSI={rsi:.1f} oversold — risky short")
                score += 0.2

        # 3. MACD bearish
        macd = self._get_val(latest, "macd")
        macd_sig = self._get_val(latest, "macd_signal")
        if macd is not None and macd_sig is not None:
            if macd < macd_sig:
                score += 1
                reasons.append(f"MACD bearish ✓")
            else:
                warnings.append("MACD still bullish")

        # 4. Volume
        vol_ratio = self._get_val(latest, "volume_ratio")
        if vol_ratio is not None:
            if vol_ratio >= self.vol_surge:
                score += 1
                reasons.append(f"Volume surge: {vol_ratio:.1f}x avg ✓")

        # 5 & 6. Price relative to EMAs
        ema20 = self._get_val(latest, "ema_20")
        ema50 = self._get_val(latest, "ema_50")
        if ema20 is not None and price < ema20:
            score += 1
            reasons.append(f"Price < EMA-20 ✓")
        if ema50 is not None and price < ema50:
            score += 1
            reasons.append(f"Price < EMA-50 ✓")

        strength = score / total_checks
        if strength < self.MIN_SIGNAL_STRENGTH:
            return None

        atr = self._get_val(latest, "atr") or price * 0.01
        trade_params = self.risk.calculate_trade_parameters(
            symbol=symbol,
            side="sell",
            entry_price=price,
            atr=atr,
            predicted_pct_change=predicted_pct,
        )

        if trade_params is None or not self.risk.is_trade_allowed(trade_params):
            return self._hold(symbol, price, "Risk check failed for short")

        return TradeSignal(
            symbol=symbol,
            signal_type=SignalType.SHORT,
            source=SignalSource.COMBINED,
            strength=round(strength, 3),
            timestamp=datetime.utcnow(),
            current_price=price,
            predicted_pct_change=predicted_pct,
            prediction_confidence=confidence,
            trade_params=trade_params,
            reasons=reasons,
            warnings=warnings,
        )

    # ── Exit Logic ────────────────────────────────────────────

    def _check_exit(
        self,
        symbol: str,
        current_price: float,
        df: pd.DataFrame,
        position: dict,
        prediction: Optional[PredictionResult],
    ) -> Optional[TradeSignal]:
        """
        Evaluate whether an open position should be exited.

        Exit conditions:
          - Stop-loss hit
          - Take-profit hit
          - Signal reversal (LSTM now forecasts opposite direction)
          - Trailing stop hit (if applicable)
        """
        side = position.get("side", "buy")
        entry_price = position.get("entry_price", current_price)
        stop_loss = position.get("stop_loss", 0)
        take_profit = position.get("take_profit", float("inf"))
        atr = float(df.iloc[-1].get("atr", current_price * 0.01))

        reasons = []

        # ── Stop-loss ──────────────────────────────────────────
        if side == "buy" and current_price <= stop_loss:
            reasons.append(f"Stop-loss hit: {current_price:.4f} ≤ {stop_loss:.4f}")
        elif side == "sell" and current_price >= stop_loss:
            reasons.append(f"Stop-loss hit: {current_price:.4f} ≥ {stop_loss:.4f}")

        # ── Take-profit ────────────────────────────────────────
        if side == "buy" and current_price >= take_profit:
            reasons.append(f"Take-profit hit: {current_price:.4f} ≥ {take_profit:.4f}")
        elif side == "sell" and current_price <= take_profit:
            reasons.append(f"Take-profit hit: {current_price:.4f} ≤ {take_profit:.4f}")

        # ── Trailing stop ──────────────────────────────────────
        trail_stop = self.risk.calculate_trailing_stop(
            side=side,
            current_price=current_price,
            entry_price=entry_price,
            current_stop=stop_loss,
            atr=atr,
        )
        if trail_stop != stop_loss:
            # Update position's stop loss
            position["stop_loss"] = trail_stop
            self.logger.debug(
                f"Trailing stop updated for {symbol}: {stop_loss:.4f} → {trail_stop:.4f}"
            )
            if side == "buy" and current_price <= trail_stop:
                reasons.append(f"Trailing stop hit: {trail_stop:.4f}")
            elif side == "sell" and current_price >= trail_stop:
                reasons.append(f"Trailing stop hit: {trail_stop:.4f}")

        # ── Signal reversal ────────────────────────────────────
        if prediction is not None:
            pct = prediction.predicted_pct_change
            if side == "buy" and pct <= self.short_threshold:
                reasons.append(f"Signal reversal: LSTM now predicts {pct:.2f}%")
            elif side == "sell" and pct >= self.long_threshold:
                reasons.append(f"Signal reversal: LSTM now predicts {pct:.2f}%")

        if reasons:
            return TradeSignal(
                symbol=symbol,
                signal_type=SignalType.EXIT,
                source=SignalSource.EXIT_SIGNAL,
                strength=1.0,
                timestamp=datetime.utcnow(),
                current_price=current_price,
                predicted_pct_change=(
                    prediction.predicted_pct_change if prediction else None
                ),
                reasons=reasons,
            )
        return None

    # ── TA-Only Fallback ──────────────────────────────────────

    def _ta_only_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        existing_position: Optional[dict],
    ) -> TradeSignal:
        """
        Simpler rule-based signal when LSTM prediction is unavailable.
        Uses RSI + MACD + EMA crossover only.
        """
        latest = df.iloc[-1]
        price = float(latest.get("close", 0))

        rsi = self._get_val(latest, "rsi")
        macd = self._get_val(latest, "macd")
        macd_sig = self._get_val(latest, "macd_signal")
        ema20 = self._get_val(latest, "ema_20")

        if existing_position is None:
            # Long: RSI oversold + MACD bullish + above EMA
            if (rsi is not None and rsi < self.rsi_oversold and
                macd is not None and macd_sig is not None and macd > macd_sig and
                ema20 is not None and price > ema20):

                atr = self._get_val(latest, "atr") or price * 0.01
                trade_params = self.risk.calculate_trade_parameters(
                    symbol=symbol, side="buy", entry_price=price, atr=atr
                )
                if trade_params and self.risk.is_trade_allowed(trade_params):
                    return TradeSignal(
                        symbol=symbol,
                        signal_type=SignalType.LONG,
                        source=SignalSource.RULES_ONLY,
                        strength=0.65,
                        timestamp=datetime.utcnow(),
                        current_price=price,
                        trade_params=trade_params,
                        reasons=[f"TA-only: RSI={rsi:.1f}, MACD bullish"],
                    )

        return self._hold(symbol, price, "No ML prediction available; TA-only HOLD")

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _get_val(row: pd.Series, col: str) -> Optional[float]:
        """Safely extract a float from a Series row."""
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)

    def _hold(self, symbol: str, price: float, reason: str = "") -> TradeSignal:
        return TradeSignal(
            symbol=symbol,
            signal_type=SignalType.HOLD,
            source=SignalSource.COMBINED,
            strength=0.0,
            timestamp=datetime.utcnow(),
            current_price=price,
            reasons=[reason] if reason else [],
        )


# ─────────────────────────────────────────────────────────────
# Signal Registry — one strategy per pair
# ─────────────────────────────────────────────────────────────

class StrategyEngine:
    """
    Manages strategy instances and dispatches signal generation
    for all configured trading pairs.
    """

    def __init__(self, config: dict, risk_manager: RiskManager) -> None:
        self.config = config
        self.risk = risk_manager
        self.logger = get_logger("StrategyEngine")
        self._strategy = HybridLSTMStrategy(config, risk_manager)

    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        prediction: Optional[PredictionResult],
        existing_position: Optional[dict] = None,
    ) -> TradeSignal:
        """
        Run the strategy and return a TradeSignal for `symbol`.
        Logs signal details at INFO level.
        """
        signal = self._strategy.generate_signal(
            symbol=symbol,
            df=df,
            prediction=prediction,
            existing_position=existing_position,
        )

        if signal.is_actionable():
            self.logger.info(
                f"📊 SIGNAL: {signal}"
                + (f" | Reasons: {', '.join(signal.reasons)}" if signal.reasons else "")
            )
        else:
            self.logger.debug(f"HOLD {symbol}: {', '.join(signal.reasons)}")

        return signal
