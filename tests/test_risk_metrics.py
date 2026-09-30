"""Sortino is defined only when there is a real downside sample."""

import numpy as np
import pandas as pd
import pytest

from src.risk_manager import RiskManager


def test_sortino_flat_curve_is_zero():
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert RiskManager.compute_sortino_ratio(returns) == 0.0


def test_sortino_with_no_negative_returns_is_zero():
    returns = pd.Series([0.01, 0.02, 0.0, 0.015])
    assert RiskManager.compute_sortino_ratio(returns) == 0.0


def test_sortino_with_one_negative_return_is_zero():
    returns = pd.Series([0.02, -0.01, 0.03, 0.01])
    assert RiskManager.compute_sortino_ratio(returns) == 0.0


def test_sortino_with_downside_matches_the_formula():
    returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
    periods = 365 * 24
    rf = 0.05 / periods
    excess = returns - rf
    downside_std = float(returns[returns < 0].std())
    expected = float((excess.mean() / downside_std) * np.sqrt(periods))

    value = RiskManager.compute_sortino_ratio(returns)

    assert np.isfinite(value)
    assert abs(value) < 1_000
    assert value == pytest.approx(expected)
