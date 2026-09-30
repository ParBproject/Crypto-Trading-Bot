#!/usr/bin/env python3
"""
backtest.py — Historical Strategy Backtester
=============================================
Simulates the trading strategy on historical OHLCV data to evaluate:
  - Total return, against a fully invested buy-and-hold
  - Sharpe ratio, Sortino ratio, Calmar ratio
  - Maximum drawdown on mark-to-market equity
  - Win rate, average win/loss
  - Equity curve

Fills are not same-bar closes. A signal is decided at the bar close and
filled at the next bar's open. Resting stops and targets fill from that
bar's high and low; if both are touched, the stop fills first.

The backtest uses the same DataManager, LSTMPredictor, StrategyEngine,
and RiskManager as the live bot — ensuring consistency between
backtested and live behaviour.

Usage:
    python backtest.py
    python backtest.py --pair BTC/USDT --start 2023-01-01 --end 2024-01-01
    python backtest.py --capital 50000 --no-ml  (rule-based only)

Output:
    - Printed metrics table
    - logs/backtest_<pair>_<date>.json  (full trade log)
    - logs/equity_<pair>.csv            (equity curve)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))

from src.bot import load_config
from src.data_fetcher import DataManager
from src.predictor import LSTMPredictor
from src.risk_manager import RiskManager, PortfolioState, infer_periods_per_year
from src.strategy import StrategyEngine, SignalType
from src.logger import get_logger


# ─────────────────────────────────────────────────────────────
# Costs and fills
# ─────────────────────────────────────────────────────────────

def _percent_rate(config: dict, key: str, default_percent: float) -> float:
    """Read a backtest cost written as a percent. 0.1 means 0.1%, not 10%."""
    pct = float((config.get("backtest") or {}).get(key, default_percent))
    if pct < 0:
        raise ValueError(f"backtest.{key} must be >= 0")
    return pct / 100.0


def commission_rate_from_config(config: dict) -> float:
    """Per-fill commission fraction. Entry and exit each pay it once."""
    return _percent_rate(config, "commission_pct", 0.1)


def slippage_rate_from_config(config: dict) -> float:
    """Adverse slippage fraction applied to each fill."""
    return _percent_rate(config, "slippage_pct", 0.05)


def ohlc_values(row: pd.Series) -> tuple:
    """Open, high, low, close. Missing OHLC falls back to the close."""
    close = float(row["close"])

    def _num(key: str, default: float) -> float:
        if key not in row.index:
            return default
        val = row[key]
        if val is None or pd.isna(val):
            return default
        return float(val)

    open_ = _num("open", close)
    high = max(_num("high", max(open_, close)), open_, close)
    low = min(_num("low", min(open_, close)), open_, close)
    return open_, high, low, close


def protective_exit(side: str, open_: float, high: float, low: float, stop: float, take_profit: float):
    """Raw price of a resting stop or target, before slippage.

    The bar's high and low are the path. A close that recovers does not
    cancel a level the range already traded. If both levels are touched
    and the open is still between them, the stop fills first. A gap
    through a level fills at the open, not at the level on the wrong side
    of the gap.
    """
    if side == "buy":
        stop_hit = low <= stop
        target_hit = high >= take_profit
        gapped_stop = open_ <= stop
        gapped_target = open_ >= take_profit
    else:
        stop_hit = high >= stop
        target_hit = low <= take_profit
        gapped_stop = open_ >= stop
        gapped_target = open_ <= take_profit

    if gapped_stop and (stop_hit or gapped_stop):
        return open_, "stop_loss"
    if gapped_target and target_hit:
        return open_, "take_profit"
    if stop_hit and target_hit:
        return stop, "stop_loss"
    if stop_hit:
        return stop, "stop_loss"
    if target_hit:
        return take_profit, "take_profit"
    return None, ""


def buy_and_hold_final_capital(
    initial: float,
    entry_open: float,
    exit_close: float,
    commission_rate: float,
    slippage_rate: float,
) -> float:
    """Fully invested spot long, same commission and slippage as the strategy."""
    entry = float(entry_open) * (1 + slippage_rate)
    if initial <= 0 or entry <= 0:
        return float(initial)
    qty = initial / (entry * (1 + commission_rate))
    exit_fill = max(0.0, float(exit_close) * (1 - slippage_rate))
    return qty * exit_fill * (1 - commission_rate)


# ─────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Walk-forward simulation on historical OHLCV data.

    Methodology:
      1. Split data: first 70% for model training, last 30% for testing
      2. Decide at the close of bar t using only bars through t
      3. Fill that decision at the open of bar t+1
      4. Check the resting stop and target against bar t+1 high/low
      5. Mark equity to the close, including open P&L
    """

    def __init__(self, config: dict, initial_capital: float = 10_000.0) -> None:
        self.config = config
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate_from_config(config)
        self.slippage_rate = slippage_rate_from_config(config)
        self.logger = get_logger("BacktestEngine")
        self.dm = DataManager(config)

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        use_ml: bool = True,
        train_split: float = 0.70,
    ) -> dict:
        """
        Execute a backtest on `df` for `symbol`.

        Args:
            symbol:      Trading pair
            df:          Full enriched OHLCV + indicator DataFrame
            use_ml:      If False, run rule-based strategy only
            train_split: Fraction of data used for LSTM training

        Returns:
            Metrics dict with Sharpe, Sortino, drawdown, win rate, etc.
        """
        self.logger.info(
            f"Backtesting {symbol} | "
            f"{'With LSTM' if use_ml else 'Rules only'} | "
            f"{len(df)} candles | Train split: {train_split:.0%}"
        )

        split_idx = int(len(df) * train_split)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:].copy()

        if len(test_df) < 100:
            raise ValueError(
                f"Test set too small ({len(test_df)} rows). Need ≥ 100."
            )

        self.logger.info(
            f"Train: {len(train_df)} candles | Test: {len(test_df)} candles "
            f"({test_df.index[0]} → {test_df.index[-1]})"
        )

        # ── Train LSTM ─────────────────────────────────────────
        predictor = None
        if use_ml:
            try:
                predictor = LSTMPredictor(self.config, symbol)
                if predictor.needs_retraining():
                    self.logger.info("Training LSTM on historical data...")
                    predictor.train(train_df)
                    predictor.save()
            except Exception as e:
                self.logger.warning(f"LSTM training failed: {e}. Falling back to rules-only.")
                predictor = None

        # ── Initialise portfolio ───────────────────────────────
        portfolio = PortfolioState(
            initial_capital=self.initial_capital,
            current_capital=self.initial_capital,
        )
        risk_mgr = RiskManager(self.config, portfolio)
        strategy = StrategyEngine(self.config, risk_mgr)

        # ── Tracking ───────────────────────────────────────────
        equity_curve = []
        equity_times = []
        trades = []
        open_position = None
        pending_entry = None
        pending_exit = False

        seq_len = int(self.config.get("model", {}).get("sequence_length", 60))
        self.logger.info(
            "Execution: signal at bar close, fill at next open; "
            f"commission={self.commission_rate:.4%}, slippage={self.slippage_rate:.4%}"
        )

        def mark(close_price: float) -> float:
            if open_position is None:
                return portfolio.current_capital
            qty = open_position["quantity"]
            entry = open_position["entry_price"]
            if open_position["side"] == "buy":
                unrealized = (close_price - entry) * qty
            else:
                unrealized = (entry - close_price) * qty
            return portfolio.current_capital + unrealized

        def liquidate(raw_price: float, reason: str, when) -> None:
            nonlocal open_position
            position = open_position
            side = position["side"]
            fill = self._slip_exit(side, raw_price)
            qty = position["quantity"]
            exit_fee = qty * fill * self.commission_rate
            if side == "buy":
                pnl = (fill - position["entry_price"]) * qty - exit_fee
            else:
                pnl = (position["entry_price"] - fill) * qty - exit_fee
            portfolio.current_capital += pnl
            portfolio.update_peak()
            portfolio.trade_history.append({"pnl_usd": pnl})
            trades.append({
                "entry_time": str(position["entry_time"]),
                "exit_time": str(when),
                "symbol": symbol,
                "side": side,
                "entry_price": position["entry_price"],
                "exit_price": fill,
                "quantity": qty,
                "pnl_usd": round(pnl, 4),
                "reason": reason,
            })
            open_position = None
            portfolio.open_positions.pop(symbol, None)

        # ── Walk-forward loop ──────────────────────────────────
        # Bar i is processed with information known during that bar.
        # Decisions made at the close wait for the next open.
        for i in range(len(test_df)):
            open_, high, low, close = ohlc_values(test_df.iloc[i])
            when = test_df.index[i]

            if open_position is not None and pending_exit:
                liquidate(open_, "signal_exit", when)
                pending_exit = False

            if open_position is None and pending_entry is not None:
                open_position = self._open_position(
                    pending_entry, open_, when, portfolio, symbol
                )
                pending_entry = None

            if open_position is not None:
                raw, reason = protective_exit(
                    open_position["side"],
                    open_,
                    high,
                    low,
                    open_position["stop_loss"],
                    open_position["take_profit"],
                )
                if raw is not None:
                    liquidate(raw, reason, when)
                    pending_exit = False

            equity_curve.append(mark(close))
            equity_times.append(when)

            if i < seq_len - 1:
                continue

            window_df = pd.concat([train_df.tail(seq_len), test_df.iloc[: i + 1]])
            prediction = None
            if predictor is not None:
                try:
                    prediction = predictor.predict(window_df)
                except Exception:
                    prediction = None

            signal = strategy.evaluate(
                symbol=symbol,
                df=window_df,
                prediction=prediction,
                existing_position=open_position,
            )
            # A decision at this close can fill on a later bar only.
            if i >= len(test_df) - 1:
                continue
            if open_position is not None and signal.signal_type == SignalType.EXIT:
                pending_exit = True
            elif (
                open_position is None
                and signal.is_actionable()
                and signal.signal_type != SignalType.EXIT
                and signal.trade_params is not None
                and risk_mgr.is_trade_allowed(signal.trade_params)
            ):
                pending_entry = signal

        # The sample has ended, so an open position is liquidated at the
        # last close. There is no next open to trade.
        if open_position is not None:
            _last_open, _last_high, _last_low, last_close = ohlc_values(test_df.iloc[-1])
            liquidate(last_close, "end_of_backtest", test_df.index[-1])
            equity_curve[-1] = portfolio.current_capital

        # ── Compute metrics ────────────────────────────────────
        equity_series = pd.Series(equity_curve, index=equity_times, name="equity")
        returns = equity_series.pct_change().dropna()
        periods_per_year = infer_periods_per_year(test_df.index)

        total_return_pct = (portfolio.current_capital / self.initial_capital - 1) * 100
        n_days = (test_df.index[-1] - test_df.index[0]).days or 1
        annualised_return = (
            (portfolio.current_capital / self.initial_capital) ** (365 / n_days) - 1
        ) * 100

        entry_i = seq_len if seq_len < len(test_df) else 0
        bh_open, _, _, _ = ohlc_values(test_df.iloc[entry_i])
        _, _, _, bh_close = ohlc_values(test_df.iloc[-1])
        bh_final = buy_and_hold_final_capital(
            self.initial_capital,
            bh_open,
            bh_close,
            self.commission_rate,
            self.slippage_rate,
        )
        bh_return_pct = (bh_final / self.initial_capital - 1) * 100

        pnls = [t["pnl_usd"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        if losses and sum(losses) != 0:
            profit_factor = round(abs(sum(wins) / sum(losses)), 3)
        else:
            profit_factor = None

        sharpe = RiskManager.compute_sharpe_ratio(returns, periods_per_year=periods_per_year)
        sortino = RiskManager.compute_sortino_ratio(returns, periods_per_year=periods_per_year)
        max_dd = RiskManager.compute_max_drawdown(equity_series)
        calmar = RiskManager.compute_calmar_ratio(annualised_return, max_dd)

        metrics = {
            "symbol": symbol,
            "period": f"{test_df.index[0].date()} → {test_df.index[-1].date()}",
            "candles_tested": len(test_df),
            "fill_model": "next_bar_open",
            "stop_model": "intrabar_ohlc_stop_first",
            "periods_per_year": periods_per_year,
            "initial_capital_usd": self.initial_capital,
            "final_capital_usd": round(portfolio.current_capital, 2),
            "total_return_pct": round(total_return_pct, 2),
            "buy_hold_final_capital_usd": round(bh_final, 2),
            "buy_hold_return_pct": round(bh_return_pct, 2),
            "excess_return_pct": round(total_return_pct - bh_return_pct, 2),
            "annualised_return_pct": round(annualised_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": len(trades),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0,
            "avg_win_usd": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss_usd": round(sum(losses) / len(losses), 2) if losses else 0,
            "profit_factor": profit_factor,
            "strategy": "LSTM+TA" if use_ml and predictor is not None else "TA-only",
            "trades": trades,
            "equity_curve": list(zip([str(t) for t in equity_times], equity_curve)),
        }
        return metrics

    def _slip_entry(self, side: str, raw: float) -> float:
        if side == "buy":
            return raw * (1 + self.slippage_rate)
        return raw * (1 - self.slippage_rate)

    def _slip_exit(self, side: str, raw: float) -> float:
        """Adverse slippage: sells receive less, covers pay more."""
        if side == "buy":
            return raw * (1 - self.slippage_rate)
        return raw * (1 + self.slippage_rate)

    def _open_position(self, signal, raw_open: float, when, portfolio: PortfolioState, symbol: str):
        params = signal.trade_params
        if params is None:
            return None
        side = "buy" if signal.signal_type == SignalType.LONG else "sell"
        fill = self._slip_entry(side, raw_open)
        qty = params.quantity
        entry_fee = qty * fill * self.commission_rate
        portfolio.current_capital -= entry_fee
        position = {
            "side": side,
            "entry_price": fill,
            "entry_time": when,
            "quantity": qty,
            "stop_loss": params.stop_loss,
            "take_profit": params.take_profit,
            "value_usd": qty * fill,
        }
        portfolio.open_positions[symbol] = position
        return position


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest the trading strategy")
    parser.add_argument("--pair", default=None, help="Single pair (e.g. BTC/USDT)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-ml", action="store_true", help="Rule-based only")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    return parser.parse_args()


def print_metrics(metrics: dict) -> None:
    skip = {"trades", "equity_curve", "symbol"}
    rows = [[k, v] for k, v in metrics.items() if k not in skip]
    print(f"\n{'='*55}")
    print(f"  Backtest Results: {metrics['symbol']}")
    print(f"{'='*55}")
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_outline"))

    # Trade sample
    trades = metrics.get("trades", [])
    if trades:
        print(f"\nSample trades (last 5):")
        sample = trades[-5:]
        print(tabulate(
            [[t["side"].upper(), t["entry_price"], t["exit_price"],
              f"${t['pnl_usd']:+.2f}", t["reason"]] for t in sample],
            headers=["Side", "Entry", "Exit", "P&L", "Reason"],
            tablefmt="simple"
        ))


def main():
    args = parse_args()
    config = load_config(args.config, mode_override="backtest")

    # Apply CLI overrides
    if args.capital:
        config.setdefault("backtest", {})["initial_capital"] = args.capital
    if args.start:
        config.setdefault("backtest", {})["start_date"] = args.start
    if args.end:
        config.setdefault("backtest", {})["end_date"] = args.end

    pairs = [args.pair] if args.pair else config.get("trading", {}).get("pairs", ["BTC/USDT"])
    initial_capital = float(config.get("backtest", {}).get("initial_capital", 10_000))

    logger = get_logger("Backtest")
    dm = DataManager(config)

    all_results = []

    for pair in pairs:
        logger.info(f"\nFetching data for {pair}...")
        df = dm.get_enriched_ohlcv(pair)

        if df.empty:
            logger.error(f"No data for {pair}")
            continue

        # Filter by date range if specified
        start = config.get("backtest", {}).get("start_date")
        end = config.get("backtest", {}).get("end_date")
        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]

        if len(df) < 200:
            logger.error(f"Insufficient data after date filter: {len(df)} rows")
            continue

        engine = BacktestEngine(config, initial_capital=initial_capital)
        metrics = engine.run(pair, df, use_ml=not args.no_ml)
        print_metrics(metrics)
        all_results.append(metrics)

        # Save results
        if args.save:
            out_path = Path("logs") / f"backtest_{pair.replace('/', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            out_path.parent.mkdir(exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({k: v for k, v in metrics.items() if k != "trades"}, f, indent=2)
            logger.info(f"Results saved to {out_path}")

            # Equity curve CSV
            eq_path = Path("logs") / f"equity_{pair.replace('/', '-')}.csv"
            pd.DataFrame(
                metrics["equity_curve"], columns=["timestamp", "equity"]
            ).to_csv(eq_path, index=False)

    logger.info("Backtest complete.")


if __name__ == "__main__":
    main()
