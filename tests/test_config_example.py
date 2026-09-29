"""main.py points at a real template with paper/sandbox defaults and no secrets."""

from pathlib import Path

import yaml


def test_config_example_matches_safe_defaults():
    path = Path("config/config.yaml.example")
    text = path.read_text()
    config = yaml.safe_load(text)

    assert config["trading"]["mode"] == "paper"
    assert config["exchange"]["sandbox"] is True
    assert config["backtest"]["start_date"] == "2023-01-01"
    assert config["backtest"]["end_date"] == "2024-01-01"
    assert config["backtest"]["commission_pct"] == 0.1
    assert "BINANCE_API_KEY=" not in text
    assert "BINANCE_SECRET=" not in text


def test_main_points_at_the_example_file():
    source = Path("main.py").read_text()
    assert "config/config.yaml.example" in source
