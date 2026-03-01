# 🤖 Automated Cryptocurrency Trading Bot

A modular, production-ready Python trading bot that combines **LSTM price prediction** with **technical analysis** and **rigorous risk management** for automated crypto trading.

---

## ⚠️ IMPORTANT DISCLAIMERS

> **Trading bots carry significant financial risk.** This software is provided for educational purposes. Never trade with funds you cannot afford to lose.

- **Always start in paper/sandbox mode** — real-money losses can be catastrophic
- No strategy guarantees profit — crypto markets are unpredictable
- Comply with your exchange's Terms of Service and local regulations
- Rate limits, API key security, and proper testing are critical
- Past backtest performance does not guarantee future results

---

## ✨ Features

| Feature | Detail |
|---|---|
| **LSTM Prediction** | Multi-layer LSTM forecasts next-candle % price change |
| **Monte Carlo Dropout** | Uncertainty estimation via MC dropout inference |
| **Hybrid Strategy** | ML signals + RSI, MACD, EMA, volume filters |
| **Risk Management** | ATR-based stops, Kelly/fixed sizing, drawdown guard |
| **Paper Trading** | Full simulation with slippage & commission |
| **Backtesting** | Walk-forward with Sharpe, Sortino, Calmar metrics |
| **Multi-Exchange** | CCXT: Binance, Bybit, Kraken, Coinbase (100+ total) |
| **Market Data** | CCXT (live) + CoinGecko (historical/market cap) |
| **Notifications** | Telegram + Discord webhook alerts |
| **Trade Journal** | CSV log with P&L, duration, signal source |
| **Auto-Retrain** | Scheduled LSTM retraining (configurable interval) |

---

## 📁 Project Structure

```
crypto_trading_bot/
├── config/
│   └── config.yaml          # All configuration (non-secret)
├── .env.example             # API key template → copy to .env
├── src/
│   ├── data_fetcher.py      # CCXT + CoinGecko data + TA indicators
│   ├── predictor.py         # LSTM training, inference, MC dropout
│   ├── strategy.py          # Hybrid signal generation
│   ├── risk_manager.py      # Position sizing, stops, drawdown
│   ├── executor.py          # Paper/live order execution
│   ├── logger.py            # Structured logging + trade journal
│   └── bot.py               # Main orchestration loop
├── models/                  # Saved LSTM weights (auto-created)
├── logs/                    # Trade logs, equity curves (auto-created)
├── data/                    # Candle cache, state persistence
├── backtest.py              # Walk-forward backtesting script
├── train_model.py           # Standalone LSTM training
├── main.py                  # Entry point
├── run_bot.sh               # Shell launcher
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Setup environment

```bash
# Clone / create directory
cd crypto_trading_bot

# Create virtual environment
python3 -m python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy and edit environment variables
cp .env.example .env
# Fill in BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET

# Review config (sandbox=true by default — safe)
cat config/config.yaml
```

Get Binance testnet keys: https://testnet.binance.vision/

### 3. Train LSTM models

```bash
# Train on all configured pairs (fetches historical data automatically)
python train_model.py

# Or train a specific pair with more epochs
python train_model.py --pair BTC/USDT --epochs 150
```

### 4. Run backtest

```bash
# Evaluate strategy on historical data before going live
python backtest.py --pair BTC/USDT --save

# Compare ML vs rule-based
python backtest.py --pair BTC/USDT --no-ml
```

### 5. Start paper trading

```bash
python main.py --mode paper

# Or use the shell script
chmod +x run_bot.sh
./run_bot.sh --mode paper
```

### 6. Live trading (after extensive testing)

```bash
# Edit config.yaml: sandbox: false
# Edit .env: fill in real API keys
python main.py --mode live
```

---

## ⚙️ Configuration Reference

Key settings in `config/config.yaml`:

```yaml
exchange:
  name: "binance"    # ccxt exchange id
  sandbox: true      # ALWAYS start with true

trading:
  pairs: ["BTC/USDT", "ETH/USDT"]
  timeframe: "1h"    # Candle timeframe
  mode: "paper"      # paper | live | backtest

model:
  sequence_length: 60   # LSTM input window (candles)
  lstm_units: [64, 64, 32]
  epochs: 100

risk:
  max_risk_per_trade_pct: 1.5    # % of account per trade
  atr_stop_multiplier: 2.5       # Stop = entry ± 2.5 × ATR
  reward_risk_ratio: 2.5         # Take-profit target
  max_drawdown_pct: 12.0         # Halt if drawdown > 12%
  max_open_trades: 3
```

---

## 📊 Signal Logic

The `HybridLSTMStrategy` generates signals by scoring 6 conditions (0–1 each):

**Long signal** (requires strength ≥ 0.55):
1. ✅ LSTM predicts ≥ +1.5% price increase (weighted by confidence)
2. ✅ RSI < 65 (not overbought)
3. ✅ MACD > Signal line (or bullish crossover)
4. ✅ Volume > 1.5× 20-period average
5. ✅ Price above EMA-20
6. ✅ Price above EMA-50

**Short signal**: Inverse conditions apply.

**Exit signal** triggers on:
- Stop-loss hit (ATR-based, with trailing)
- Take-profit hit
- Signal reversal (LSTM flips direction)

---

## 🛡️ Risk Management

Based on Portfolio-Risk-Analysis principles:

| Control | Method |
|---|---|
| Position size | Fixed % risk (1.5% default) or fractional Kelly |
| Stop-loss | `entry ± 2.5 × ATR(14)` |
| Take-profit | `stop_distance × 2.5` (or LSTM predicted target) |
| Trailing stop | Activates at +1% profit; trails 2.5× ATR |
| Max drawdown | Bot halts if portfolio drops >12% from peak |
| Exposure cap | Max 25% of capital in one asset |
| Max open trades | 3 simultaneous positions |

---

## 📈 Performance Metrics (Backtest Output)

```
Metric                    Value
──────────────────────    ──────
Total Return             +18.4%
Annualised Return        +22.1%
Sharpe Ratio              1.34
Sortino Ratio             1.89
Max Drawdown              8.2%
Calmar Ratio              2.70
Win Rate                  54.2%
Avg Win                  $124
Avg Loss                  -$68
Profit Factor             1.82
```

*Sample metrics — actual performance varies by market conditions*

---

## 🔧 Extending the Bot

### Add a new strategy

```python
# src/strategy.py
class MomentumStrategy(BaseStrategy):
    def generate_signal(self, symbol, df, prediction, existing_position):
        # Your logic here
        ...
```

### Add a new indicator

```python
# src/data_fetcher.py — IndicatorCalculator._compute_manual()
df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
```

### Add a new exchange

```bash
# Just change config.yaml
exchange:
  name: "bybit"    # Any ccxt-supported exchange
```

---

## 🗺️ Roadmap / Next Improvements

1. **Reinforcement Learning** — Replace LSTM with PPO/DQN agent (Stable-Baselines3)
2. **Sentiment Analysis** — Reddit/Twitter/news sentiment via BERT/FinBERT
3. **Multi-pair portfolio** — Correlation-adjusted position sizing across pairs
4. **On-chain metrics** — Glassnode/CryptoQuant API integration (NVT, MVRV, etc.)
5. **Order types** — Limit orders, iceberg orders for reduced slippage
6. **Dashboard** — Streamlit/Dash real-time monitoring UI
7. **Walk-forward optimisation** — Auto-tune strategy parameters
8. **Regime detection** — HMM or clustering to classify bull/bear/sideways markets

---

## 🔐 Security Best Practices

- **Never commit `.env`** — it's in `.gitignore` by default
- Use **API keys with trade-only permissions** (no withdrawal access)
- Use **IP whitelisting** on your exchange API keys
- Rotate API keys periodically
- Run on a VPS with firewall rules, not your personal machine
- Consider hardware 2FA for exchange accounts

---

## 📦 Dependencies

- `ccxt` — Exchange API (100+ exchanges)
- `pycoingecko` — Market data
- `tensorflow` — LSTM model
- `pandas-ta` — Technical indicators
- `scikit-learn` — Data preprocessing
- `python-dotenv` — Secrets management
- `PyYAML` — Configuration
- `colorlog` — Coloured logging
- `tabulate` — Pretty output tables

---

## 📄 License

MIT License — use at your own risk.
