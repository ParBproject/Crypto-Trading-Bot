#!/usr/bin/env bash
# run_bot.sh — Convenience launcher with environment checks
# Usage: ./run_bot.sh [--mode paper|live|backtest]

set -e

# ── Colour output ─────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════╗"
echo "║   Automated Cryptocurrency Trading Bot   ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Check Python ──────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo -e "${RED}ERROR: Python not found. Install Python 3.10+${RESET}"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo -e "Python: ${GREEN}$($PYTHON --version)${RESET}"

# ── Check .env ────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}WARNING: .env not found. Copying from .env.example${RESET}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env with your API keys before live trading.${RESET}"
fi

# ── Check config ──────────────────────────────────────────────
if [ ! -f "config/config.yaml" ]; then
    echo -e "${RED}ERROR: config/config.yaml not found.${RESET}"
    exit 1
fi

# ── Check virtual environment ─────────────────────────────────
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "Virtual env: ${GREEN}activated${RESET}"
fi

# ── Install dependencies if needed ───────────────────────────
if ! $PYTHON -c "import ccxt" &>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${RESET}"
    $PYTHON -m pip install -r requirements.txt -q
fi

# ── Parse mode ────────────────────────────────────────────────
MODE=${1:---mode paper}
echo -e "Mode: ${GREEN}${MODE}${RESET}"
echo ""

# ── Launch ────────────────────────────────────────────────────
$PYTHON main.py $MODE

echo -e "\n${GREEN}Bot exited.${RESET}"
