#!/usr/bin/env python3
"""
backtest.py — Historical Strategy Backtester
=============================================
Simulates the trading strategy on historical OHLCV data to evaluate:
  - Total return
  - Sharpe ratio, Sortino ratio, Calmar ratio
  - Maximum drawdown
  - Win rate, average win/loss
  - Equity curve

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

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))

from src.bot import load_config
from src.data_fetcher import DataManager
from src.predictor import LSTMPredictor, PredictorRegistry
from src.risk_manager import RiskManager, PortfolioState
from src.strategy import StrategyEngine, SignalType
from src.executor import PaperTradeEngine, OrderResult
from src.logger import get_logger, TradeJournal


# ─────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Walk-forward simulation on historical OHLCV data.

    Methodology:
      1. Split data: first 70% for model training, last 30% for testing
      2. Walk forward candle-by-candle through test period
      3. At each candle, run prediction → signal → (simulated) execution
      4. Track equity, positions, and trade outcomes
    """

    COMMISSION_PCT = 0.001   # 0.1%
    SLIPPAGE_PCT = 0.0003    # 0.03%

    def __init__(self, config: dict, initial_capital: float = 10_000.0) -> None:
        self.config = config
        self.initial_capital = initial_capital
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
        equity_curve = [self.initial_capital]
        equity_times = [test_df.index[0]]
        trades = []
        open_position = None

        seq_len = self.config.get("model", {}).get("sequence_length", 60)

        # ── Walk-forward loop ──────────────────────────────────
        for i in range(seq_len, len(test_df)):
            # Build look-back window for inference
            window_df = pd.concat([train_df.tail(seq_len), test_df.iloc[:i]])
            current_row = test_df.iloc[i]
            current_price = float(current_row["close"])
            current_time = test_df.index[i]

            # LSTM prediction
            prediction = None
            if predictor is not None:
                try:
                    prediction = predictor.predict(window_df)
                except Exception:
                    pass

            # Strategy signal
            signal = strategy.evaluate(
                symbol=symbol,
                df=window_df,
                prediction=prediction,
                existing_position=open_position,
            )

            # ── Manage open position ───────────────────────────
            if open_position:
                entry_p = open_position["entry_price"]
                stop = open_position["stop_loss"]
                tp = open_position["take_profit"]
                side = open_position["side"]

                should_close = False
                close_reason = ""

                if side == "buy":
                    if current_price <= stop:
                        should_close, close_reason = True, "stop_loss"
                    elif current_price >= tp:
                        should_close, close_reason = True, "take_profit"
                else:
                    if current_price >= stop:
                        should_close, close_reason = True, "stop_loss"
                    elif current_price <= tp:
                        should_close, close_reason = True, "take_profit"

                if signal.signal_type.value == "exit":
                    should_close, close_reason = True, "signal_exit"

                if should_close:
                    fill = current_price * (1 - self.SLIPPAGE_PCT if side == "buy" else 1 + self.SLIPPAGE_PCT)
                    qty = open_position["quantity"]
                    cost = qty * fill
                    commission = cost * self.COMMISSION_PCT

                    if side == "buy":
                        pnl = (fill - entry_p) * qty - 2 * entry_p * qty * self.COMMISSION_PCT
                    else:
                        pnl = (entry_p - fill) * qty - 2 * entry_p * qty * self.COMMISSION_PCT

                    portfolio.current_capital += pnl
                    portfolio.update_peak()
                    portfolio.trade_history.append({"pnl_usd": pnl})

                    trades.append({
                        "entry_time": str(open_position["entry_time"]),
                        "exit_time": str(current_time),
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry_p,
                        "exit_price": fill,
                        "quantity": qty,
                        "pnl_usd": round(pnl, 4),
                        "reason": close_reason,
                    })
                    open_position = None
                    del portfolio.open_positions[symbol]

            # ── Open new position ──────────────────────────────
            if open_position is None and signal.is_actionable() and signal.signal_type.value != "exit":
                params = signal.trade_params
                if params and risk_mgr.is_trade_allowed(params):
                    side = "buy" if signal.signal_type == SignalType.LONG else "sell"
                    slip = self.SLIPPAGE_PCT
                    fill = current_price * (1 + slip if side == "buy" else 1 - slip)
                    qty = params.quantity
                    commission = qty * fill * self.COMMISSION_PCT

                    portfolio.current_capital -= commission
                    open_position = {
                        "side": side,
                        "entry_price": fill,
                        "entry_time": current_time,
                        "quantity": qty,
                        "stop_loss": params.stop_loss,
                        "take_profit": params.take_profit,
                        "value_usd": qty * fill,
                    }
                    portfolio.open_positions[symbol] = open_position

            equity_curve.append(portfolio.current_capital)
            equity_times.append(current_time)

        # ── Force-close any open position at end ───────────────
        if open_position:
            final_price = float(test_df.iloc[-1]["close"])
            side = open_position["side"]
            pnl = (
                (final_price - open_position["entry_price"]) * open_position["quantity"]
                if side == "buy"
                else (open_position["entry_price"] - final_price) * open_position["quantity"]
            )
            portfolio.current_capital += pnl
            portfolio.update_peak()
            trades.append({
                "entry_time": str(open_position["entry_time"]),
                "exit_time": str(test_df.index[-1]),
                "symbol": symbol,
                "side": side,
                "entry_price": open_position["entry_price"],
                "exit_price": final_price,
                "quantity": open_position["quantity"],
                "pnl_usd": round(pnl, 4),
                "reason": "end_of_backtest",
            })

        # ── Compute metrics ────────────────────────────────────
        equity_series = pd.Series(equity_curve, index=equity_times, name="equity")
        returns = equity_series.pct_change().dropna()

        total_return_pct = (portfolio.current_capital / self.initial_capital - 1) * 100
        n_days = (test_df.index[-1] - test_df.index[0]).days or 1
        annualised_return = (
            (portfolio.current_capital / self.initial_capital) ** (365 / n_days) - 1
        ) * 100

        pnls = [t["pnl_usd"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        sharpe = RiskManager.compute_sharpe_ratio(returns)
        sortino = RiskManager.compute_sortino_ratio(returns)
        max_dd = RiskManager.compute_max_drawdown(equity_series)
        calmar = RiskManager.compute_calmar_ratio(annualised_return, max_dd)

        metrics = {
            "symbol": symbol,
            "period": f"{test_df.index[0].date()} → {test_df.index[-1].date()}",
            "candles_tested": len(test_df),
            "initial_capital_usd": self.initial_capital,
            "final_capital_usd": round(portfolio.current_capital, 2),
            "total_return_pct": round(total_return_pct, 2),
            "annualised_return_pct": round(annualised_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": len(trades),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0,
            "avg_win_usd": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss_usd": round(sum(losses) / len(losses), 2) if losses else 0,
            "profit_factor": round(abs(sum(wins) / sum(losses)), 3) if losses and sum(losses) else float("inf"),
            "strategy": "LSTM+TA" if use_ml else "TA-only",
            "trades": trades,
            "equity_curve": list(zip([str(t) for t in equity_times], equity_curve)),
        }
        return metrics


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
