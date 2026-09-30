# Crypto Trading Bot

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-10b981?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://github.com/ParBproject/Crypto-Trading-Bot/actions/workflows/ci.yml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/ParBproject/Crypto-Trading-Bot/ci.yml?style=flat-square&label=tests"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-10b981?style=flat-square"></a>
  <a href="config/config.yaml"><img alt="Default mode is paper trading" src="https://img.shields.io/badge/default-paper%20trading-10b981?style=flat-square"></a>
</p>

Paper-trades crypto pairs from hourly candles, with technical filters, an optional LSTM forecast, and hard limits on size, stops, exposure, and drawdown.

## Overview

The bot is a research loop for one decision: given the latest candles, open a paper long, open a paper short, or do nothing, and if it trades, how large the order should be.

On each pass it loads OHLCV for the pairs in `config/config.yaml`, then adds RSI, MACD, ATR, Bollinger Bands, and moving averages. An optional per-pair LSTM forecasts the next candle's percent change. `HybridLSTMStrategy` turns that forecast, or a technical fallback, into a signal. `RiskManager` sizes the order from ATR and rejects it when drawdown, the open-trade count, or single-asset exposure is past the cap in config. `OrderManager` fills in paper mode. Live orders are refused unless `ALLOW_LIVE_TRADING=1`, and `main.py` still asks for a confirmation phrase.

The charts below come from one rules-only replay of committed Binance spot candles. Read them as the record of that run.

## Features

- CCXT candles for any exchange id in config, with paged history for the backtest dates, a Binance archive fallback, and a CoinGecko OHLC fallback
- Indicator pipeline that uses `pandas_ta` when it is installed and a pure-pandas implementation otherwise
- Optional stacked LSTM that outputs a percent-change forecast, with Monte Carlo dropout used as a confidence score
- Hybrid entries (forecast plus RSI, MACD, volume, and EMA gates) and a rules-only fallback when no forecast is available
- Fixed-fraction sizing, or fractional Kelly when a win rate is supplied, with ATR stops and a reward-to-risk target
- Drawdown halt, a cap on open trades, and a per-asset exposure cap
- Paper fills with slippage and commission. Live orders are refused unless `ALLOW_LIVE_TRADING=1`, and `main.py` still asks for a confirmation phrase
- CSV trade journal, rotating logs, and optional Telegram or Discord alerts
- Walk-forward backtest that calls the same strategy and risk objects as the bot. A signal at the close fills at the next bar's open, and open positions are marked to market

## Architecture

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'primaryColor':'#064e3b','primaryTextColor':'#ecfdf5','primaryBorderColor':'#10b981','lineColor':'#6ee7b7','secondaryColor':'#111827','tertiaryColor':'#0b1220','background':'#0b1220','fontFamily':'ui-sans-serif, system-ui, sans-serif'}}}%%
flowchart LR
    A[Candles<br/>CCXT or sample CSV] --> B[Indicators<br/>RSI MACD ATR EMA]
    B --> C[LSTM forecast<br/>optional]
    B --> D[Strategy]
    C --> D
    D --> E[Risk limits<br/>size stop exposure]
    E --> F[Paper or live fill]
    F --> G[Journal and equity]
```

## Sample backtest

`scripts/reproduce_backtest.py` loads the committed 2024 hourly file, builds indicators with `IndicatorCalculator`, and runs `BacktestEngine` with the LSTM off. Fills are next-bar opens, equity is marked to market, and buy-and-hold uses the same commission and slippage as the engine. The same file and the same code produce the same trades and the same charts. It runs offline from that CSV.

| | |
|---|---|
| Sample | Binance spot BTC/USDT, 1-hour, 1 Jan 2024 00:00 UTC through 31 Dec 2024 23:00 UTC |
| File | [`data/sample/btcusdt_1h_2024.csv`](data/sample/btcusdt_1h_2024.csv) — 8,784 bars, no gaps, from the public monthly archive on data.binance.vision |
| Bars the engine traded | Last 30% of the file (the built-in split): 13 Sep 2024 04:00 UTC through 31 Dec 2024 23:00 UTC, 2,636 bars |
| Starting capital | $10,000 |
| Costs | 0.1% commission and 0.05% slippage, charged on entry and on exit |
| Rules for a long | RSI below 35, MACD above its signal, and close above EMA-20, all on the same bar |
| Trades | 0 |
| Ending equity | $10,000.00 (0.00%), marked to market |
| Buy and hold | $15,462.84 (+54.63%) |
| Max drawdown | 0.00% |
| Sharpe | 0.0 |
| Sortino | 0.0 |

Buy-and-hold is a fully invested long from the 15 Sep 2024 16:00 UTC open at 60,335.41 (the first bar a next-bar fill can use; the model sequence length is 60) through the 31 Dec close at 93,576, with those costs on both sides. The strategy sizes from the risk budget, so a traded result would not be the same bet.

The rule set stayed in cash. On the 2,636 test bars, RSI was oversold 6.68% of the time, MACD was bullish 47.57% of the time, and the close was above EMA-20 56.68% of the time. All three were true together on **0** bars, so the risk manager never sized an order.

The LSTM threshold is a high bar on this same window, and this run did not train a model. The median absolute hourly move was 0.242%, the 95th percentile was 1.122%, and 2.24% of hours moved 1.5% or more. Config asks for a predicted move of at least +1.5% before a long is even considered.

<p align="center"><img src="docs/results/equity_curve.png" alt="Mark-to-market equity flat at 10000 dollars against a buy-and-hold line that finishes at 15463 dollars" width="100%"></p>

<p align="center"><img src="docs/results/price_and_trades.png" alt="BTC/USDT hourly close from 13 September 2024 to 31 December 2024" width="100%"></p>

<p align="center"><img src="docs/results/signal_gates.png" alt="Share of test bars passing each rules-only long gate, with the joint gate at zero" width="100%"></p>

Regenerate the table and the charts from a checkout:

```bash
python scripts/reproduce_backtest.py
```

[`docs/results/metrics.json`](docs/results/metrics.json) is the raw record. [`docs/results/trades.csv`](docs/results/trades.csv) is the closed-trade log. For this run the log is header-only.

## Quickstart

These commands are enough to regenerate the sample backtest and run the tests from a fresh clone.

```bash
git clone https://github.com/ParBproject/Crypto-Trading-Bot.git
cd Crypto-Trading-Bot
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas PyYAML python-dotenv scikit-learn tabulate matplotlib pytest
python scripts/reproduce_backtest.py
python -m pytest
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### Run the paper loop

The full dependency list adds the exchange client, TensorFlow, and the optional alert packages:

```bash
pip install -r requirements.txt
cp .env.example .env
# config/config.yaml is already in the repo (paper mode, sandbox).
# If it is missing, main.py asks you to copy the template:
# cp config/config.yaml.example config/config.yaml
python main.py --mode paper
```

`main.py` repeats until you stop the process. Live orders start only after `--mode live` and the confirmation prompt. Keep the sandbox flag in `config/config.yaml` until the exchange path has been checked with testnet credentials.

Review `config/config.yaml` before running. `trading.mode` defaults to `paper`. Live orders are refused unless `ALLOW_LIVE_TRADING=1`, and `main.py` still asks for a confirmation phrase. Real-money keys also require `exchange.sandbox: false`.

`python backtest.py` loads the committed window, 1 Jan 2023 through 1 Jan 2024, instead of the 500-bar live lookback. It reads public historical candles for that range. When the exchange REST call fails, Binance symbols fall back to the monthly archive on data.binance.vision. `backtest.commission_pct` and `backtest.slippage_pct` are percents (`0.1` means 0.1%), charged once on entry and once on exit. A signal at the close fills at the next bar's open. If TensorFlow is not installed, that command continues with the rules-only strategy.

## Tests

```bash
python -m pytest
```

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs that suite on every push and pull request, on Python 3.11, after a Ruff syntax check and `compileall`. The tests cover paper closes, next-bar fills, entry and exit fees, the configured history window, CoinGecko OHLC columns, Sortino on a flat curve, the live-trading gate, and the example config.

## Project structure

```text
Crypto-Trading-Bot/
├── main.py                      # paper and live loop
├── backtest.py                  # walk-forward backtest
├── train_model.py               # standalone LSTM training
├── scripts/reproduce_backtest.py
├── config/config.yaml
├── config/config.yaml.example   # paper/sandbox template, no secrets
├── data/sample/                 # 2024 BTC/USDT hourly candles
├── docs/results/                # metrics, trades, charts
├── src/
│   ├── bot.py
│   ├── data_fetcher.py
│   ├── predictor.py
│   ├── strategy.py
│   ├── risk_manager.py
│   ├── executor.py
│   └── logger.py
├── tests/
├── .github/workflows/ci.yml
├── requirements.txt
└── LICENSE
```

## Limitations and next steps

- The rules-only long requires three conditions on the same bar: RSI below 35, MACD above its signal, and close above EMA-20. On this test window those three never occurred together (0 of 2,636 bars), so the backtest opened no trades.
- The published run is one pair, hourly spot candles for 2024, rules only, and only the last 30% of the file. Entry and exit rules were left as they are.
- Equity is marked to market, including open profit and loss. This run never opened a position, so the curve stays at $10,000.
- Sharpe and Sortino are both 0.0 on this run. Sortino stays 0 unless at least two equity returns are negative. Annualization uses the median bar spacing, which is hourly on this file.
- Live orders need `ALLOW_LIVE_TRADING=1` as well as `trading.mode: live`. The backtest history request uses public candles, with the Binance monthly archive as a fallback when the REST call fails.

## Backtest assumptions

`python backtest.py` walks one bar at a time:

- A signal is computed at the close and filled at the next bar's open. It does not trade that same bar's close.
- Stop-loss and take-profit are resting orders. They fill from the bar's high and low. If both are touched, the stop fills first. A gap through the level fills at the open.
- `backtest.commission_pct` and `backtest.slippage_pct` are percents (`0.1` means 0.1%). Each is charged on entry and on exit.
- The equity curve marks open positions to market, so drawdown includes unrealized losses. Sharpe and Sortino use the bar spacing in the data rather than assuming every series is hourly.
- `buy_hold_return_pct` is a fully invested long from the first bar the strategy could have traded through the last close, with the same fees and slippage. The strategy sizes from the risk budget, so the two returns are not the same bet size.

## Risk & Security Notice

This project is educational and is not financial advice. Automated trading can create rapid losses. Never commit API credentials, never enable withdrawals on a trading key, and do not use real capital without independent testing, monitoring, compliance review, and a verified emergency-stop procedure.

This is a research project. Paper mode is the path the tests and the sample run cover. Nothing here is a recommendation to trade.

## License

MIT. See [LICENSE](LICENSE).
