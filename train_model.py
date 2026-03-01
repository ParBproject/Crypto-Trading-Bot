#!/usr/bin/env python3
"""
train_model.py — Standalone LSTM Training Script
==================================================
Fetches historical data and trains/retrains LSTM models
for all configured trading pairs. Saves weights to models/.

Usage:
    python train_model.py                         # Train all pairs
    python train_model.py --pair BTC/USDT         # Single pair
    python train_model.py --pair BTC/USDT --epochs 200
    python train_model.py --evaluate              # Show metrics only

This script is designed to be run independently before starting
the bot, or scheduled (e.g. via cron) for periodic retraining.
"""

import argparse
import sys
from pathlib import Path
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))

from src.bot import load_config
from src.data_fetcher import DataManager
from src.predictor import LSTMPredictor
from src.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM model(s)")
    parser.add_argument("--pair", default=None, help="Single pair to train (e.g. BTC/USDT)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Force retrain even if model is current")
    parser.add_argument("--evaluate", action="store_true", help="Print model eval metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = get_logger("TrainScript")

    # Apply epoch override
    if args.epochs:
        config.setdefault("model", {})["epochs"] = args.epochs

    pairs = [args.pair] if args.pair else config.get("trading", {}).get("pairs", ["BTC/USDT"])

    logger.info(f"Training LSTM models for: {', '.join(pairs)}")

    dm = DataManager(config)
    results = []

    for pair in pairs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {pair}")
        logger.info(f"{'='*50}")

        # Fetch data
        logger.info("Fetching historical data...")
        df = dm.get_enriched_ohlcv(pair)
        if df.empty:
            logger.error(f"No data for {pair}. Skipping.")
            continue

        logger.info(f"Data: {len(df)} candles | Features: {df.columns.tolist()}")

        # Train
        predictor = LSTMPredictor(config, pair)

        if not args.force and not predictor.needs_retraining():
            logger.info(f"Model for {pair} is up to date. Use --force to retrain.")
            continue

        metrics = predictor.train(df)
        predictor.save()

        # Quick inference test
        test_result = predictor.predict(df)
        pred_str = (
            f"{test_result.predicted_pct_change:+.2f}% ({test_result.direction})"
            if test_result else "N/A"
        )

        results.append([
            pair,
            f"{metrics.final_train_loss:.4f}",
            f"{metrics.final_val_loss:.4f}",
            metrics.epochs_trained,
            f"{metrics.training_time_sec:.1f}s",
            pred_str,
        ])

    if results:
        print("\n" + "="*60)
        print("Training Results:")
        print(tabulate(
            results,
            headers=["Pair", "Train Loss", "Val Loss", "Epochs", "Time", "Latest Pred"],
            tablefmt="rounded_outline"
        ))


if __name__ == "__main__":
    main()
