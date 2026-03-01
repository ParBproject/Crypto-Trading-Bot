#!/usr/bin/env python3
"""
main.py — Entry Point
======================
Start the crypto trading bot.

Usage:
    python main.py                    # Use config.yaml mode
    python main.py --mode paper       # Paper trading (simulated)
    python main.py --mode live        # Live trading (real funds ⚠️)
    python main.py --mode backtest    # Redirect to backtest.py
    python main.py --config path/to/config.yaml
    python main.py --pairs BTC/USDT ETH/USDT --mode paper
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from src.bot import CryptoBot, load_config
from src.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Cryptocurrency Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode paper
  python main.py --mode live --config config/config.yaml
  python main.py --pairs BTC/USDT --mode paper
  python backtest.py  (for backtesting)
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest"],
        default=None,
        help="Trading mode (overrides config.yaml)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Trading pairs to trade (e.g. BTC/USDT ETH/USDT)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Redirect to backtest script
    if args.mode == "backtest":
        print("For backtesting, run: python backtest.py")
        sys.exit(0)

    # Load configuration
    try:
        config = load_config(args.config, mode_override=args.mode)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.config}")
        print("Copy config/config.yaml.example to config/config.yaml and edit it.")
        sys.exit(1)

    # Override pairs from CLI
    if args.pairs:
        config.setdefault("trading", {})["pairs"] = args.pairs

    # Safety gate for live mode
    mode = config.get("trading", {}).get("mode", "paper")
    if mode == "live":
        print("\n" + "=" * 60)
        print("  ⚠️  WARNING: LIVE TRADING MODE")
        print("  Real funds will be used for trading.")
        print("  Make sure you have thoroughly tested in paper mode first.")
        print("=" * 60)
        confirm = input("\nType 'I understand the risks' to continue: ")
        if confirm.strip() != "I understand the risks":
            print("Aborted. Switch to --mode paper for safe testing.")
            sys.exit(0)

    # Launch bot
    logger = get_logger("main", level=config.get("logging", {}).get("level", "INFO"))
    logger.info(f"Starting in {mode.upper()} mode with pairs: {config['trading']['pairs']}")

    try:
        bot = CryptoBot(config)
        bot.start()
    except Exception as e:
        logger.critical(f"Bot crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
