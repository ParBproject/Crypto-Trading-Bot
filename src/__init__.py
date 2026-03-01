"""
Crypto Trading Bot — Source Package
=====================================
Modular, production-ready automated trading system for cryptocurrency markets.

Modules:
    data_fetcher  — Historical & real-time OHLCV data (CoinGecko + CCXT)
    predictor     — LSTM-based price direction forecasting
    strategy      — Signal generation (ML + rule-based hybrid)
    risk_manager  — Position sizing, stop-loss, drawdown controls
    executor      — Order placement via CCXT
    logger        — Structured logging & trade journaling
    bot           — Main orchestration loop
"""

__version__ = "1.0.0"
__author__ = "Crypto Trading Bot"
