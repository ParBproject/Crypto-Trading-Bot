"""
predictor.py — LSTM Price Direction Forecasting
================================================
Adapts stock-price-predictor LSTM concepts for crypto markets.

Architecture:
  - Input: sliding window of OHLCV + technical indicators (sequence_length candles)
  - Model: Multi-layer stacked LSTM with dropout + Dense output head
  - Output: Predicted % price change over forecast_horizon candles
  - Training: Adam optimiser, MSE loss, early stopping on val_loss

The predictor outputs a *relative price change forecast* (e.g. +1.8% or -0.6%).
The strategy module interprets this signal against configured thresholds.

Usage:
    predictor = LSTMPredictor(config)
    predictor.train(df)          # or load saved weights
    signal = predictor.predict(df)  # returns PredictionResult
"""

import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.logger import get_logger


# ─────────────────────────────────────────────────────────────
# Feature Configuration
# ─────────────────────────────────────────────────────────────

# Features used as model inputs (must exist in enriched DataFrame)
INPUT_FEATURES = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd", "macd_signal", "macd_hist",
    "atr", "ema_20", "ema_50", "volume_ratio",
    "bb_upper", "bb_lower",
]

# Allow graceful degradation if some indicators aren't available
REQUIRED_FEATURES = ["open", "high", "low", "close", "volume"]


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Structured output from the LSTM predictor."""
    symbol: str
    timestamp: datetime
    predicted_pct_change: float    # e.g. +1.8 means +1.8% predicted rise
    current_price: float
    predicted_price: float
    confidence: float              # 0–1, derived from prediction ensemble
    direction: str                 # "up" | "down" | "neutral"
    model_version: str = ""
    features_used: list = field(default_factory=list)


@dataclass
class TrainingMetrics:
    """Returned after a model training run."""
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    epochs_trained: int
    training_time_sec: float
    feature_count: int


# ─────────────────────────────────────────────────────────────
# Sequence Builder
# ─────────────────────────────────────────────────────────────

class SequenceBuilder:
    """
    Transforms an OHLCV + indicator DataFrame into (X, y) numpy arrays
    suitable for LSTM training/inference.

    Scaling: RobustScaler per feature (resistant to outliers / pump-dumps).
    Target:  % change in close price over forecast_horizon candles.
    """

    def __init__(self, sequence_length: int = 60, forecast_horizon: int = 1) -> None:
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = RobustScaler()
        self.feature_columns: list = []
        self.logger = get_logger("SequenceBuilder")
        self._is_fitted = False

    def _select_features(self, df: pd.DataFrame) -> list:
        """Return intersection of desired features and available columns."""
        available = set(df.columns.tolist())
        features = [f for f in INPUT_FEATURES if f in available]
        missing = set(INPUT_FEATURES) - available
        if missing:
            self.logger.debug(f"Features not available (skipping): {missing}")
        if not all(f in features for f in REQUIRED_FEATURES):
            raise ValueError(
                f"Required features {REQUIRED_FEATURES} missing from DataFrame."
            )
        return features

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler on `df` and return (X, y) training arrays.

        X shape: (n_samples, sequence_length, n_features)
        y shape: (n_samples,)  — % change in close over forecast_horizon
        """
        self.feature_columns = self._select_features(df)
        data = df[self.feature_columns].values.astype(np.float32)

        # Remove any rows with NaN (indicator warm-up period)
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]
        close_prices = df["close"].values[valid_mask]

        if len(data) < self.sequence_length + self.forecast_horizon + 1:
            raise ValueError(
                f"Insufficient data: need ≥ {self.sequence_length + self.forecast_horizon + 1} "
                f"valid rows, got {len(data)}"
            )

        # Fit scaler
        self.scaler.fit(data)
        scaled = self.scaler.transform(data).astype(np.float32)
        self._is_fitted = True

        return self._build_sequences(scaled, close_prices)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform `df` for inference (scaler must already be fitted).
        Returns X shape: (1, sequence_length, n_features) for single prediction.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")

        data = df[self.feature_columns].values.astype(np.float32)
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]

        if len(data) < self.sequence_length:
            raise ValueError(
                f"Not enough data for inference: need {self.sequence_length}, "
                f"got {len(data)}"
            )

        scaled = self.scaler.transform(data).astype(np.float32)
        # Take the last `sequence_length` rows as the inference window
        window = scaled[-self.sequence_length:]
        return window[np.newaxis, :, :]   # (1, seq_len, n_features)

    def _build_sequences(
        self, scaled: np.ndarray, close_prices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build sliding-window sequences and percentage-change targets."""
        X, y = [], []
        n = len(scaled)
        for i in range(self.sequence_length, n - self.forecast_horizon + 1):
            X.append(scaled[i - self.sequence_length : i])
            # Target: % change from current close to future close
            current_close = close_prices[i - 1]
            future_close = close_prices[i + self.forecast_horizon - 1]
            pct_change = ((future_close - current_close) / current_close) * 100
            y.append(pct_change)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# LSTM Model Builder
# ─────────────────────────────────────────────────────────────

def build_lstm_model(
    sequence_length: int,
    n_features: int,
    lstm_units: list,
    dropout: float,
    learning_rate: float,
) -> "tf.keras.Model":
    """
    Construct a stacked LSTM regression model.

    Architecture:
        Input → LSTM(units[0], return_sequences) → Dropout → BatchNorm
             → LSTM(units[1], return_sequences) → Dropout → BatchNorm
             → LSTM(units[2])                   → Dropout
             → Dense(32, relu) → Dense(1)       (predicted % change)

    Args:
        sequence_length: Input time steps
        n_features:      Number of input features per timestep
        lstm_units:      List of unit counts per LSTM layer
        dropout:         Dropout rate (0–1)
        learning_rate:   Adam optimizer learning rate

    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required. pip install tensorflow")

    model = Sequential(name="CryptoLSTM")
    model.add(Input(shape=(sequence_length, n_features)))

    # Stack LSTM layers
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        model.add(LSTM(
            units,
            return_sequences=return_seq,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        ))
        model.add(Dropout(dropout))
        if return_seq:
            model.add(BatchNormalization())

    # Dense head
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(dropout / 2))
    model.add(Dense(1, name="pct_change_output"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="huber",   # Huber loss is more robust to outliers than MSE
        metrics=["mae"],
    )

    return model


# ─────────────────────────────────────────────────────────────
# Main LSTM Predictor Class
# ─────────────────────────────────────────────────────────────

class LSTMPredictor:
    """
    Manages the full ML lifecycle for a single trading pair:
      - Data preprocessing & sequence building
      - Model construction, training, and serialisation
      - Inference with confidence estimation

    One instance per trading pair (e.g., BTC/USDT).
    """

    MODEL_VERSION = "1.0"

    def __init__(self, config: dict, symbol: str = "BTC_USDT") -> None:
        self.config = config
        self.symbol = symbol.replace("/", "_")
        self.logger = get_logger(f"LSTMPredictor[{self.symbol}]")

        model_cfg = config.get("model", {})
        self.seq_len = model_cfg.get("sequence_length", 60)
        self.horizon = model_cfg.get("forecast_horizon", 1)
        self.lstm_units = model_cfg.get("lstm_units", [64, 64, 32])
        self.dropout = model_cfg.get("dropout", 0.2)
        self.epochs = model_cfg.get("epochs", 100)
        self.batch_size = model_cfg.get("batch_size", 32)
        self.val_split = model_cfg.get("validation_split", 0.15)
        self.patience = model_cfg.get("early_stopping_patience", 10)
        self.lr = model_cfg.get("learning_rate", 0.001)
        self.model_dir = Path(model_cfg.get("model_dir", "models/"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.seq_builder = SequenceBuilder(self.seq_len, self.horizon)
        self.model: Optional["tf.keras.Model"] = None
        self._last_trained: Optional[datetime] = None

        # Try to load saved model
        self._try_load()

    # ── Persistence ───────────────────────────────────────────

    def _model_path(self) -> Path:
        return self.model_dir / f"{self.symbol}_lstm.keras"

    def _scaler_path(self) -> Path:
        return self.model_dir / f"{self.symbol}_scaler.pkl"

    def save(self) -> None:
        """Persist model weights and scaler to disk."""
        if self.model is None:
            return
        try:
            self.model.save(str(self._model_path()))
            import pickle
            with open(self._scaler_path(), "wb") as f:
                pickle.dump(self.seq_builder, f)
            self.logger.info(f"Model saved → {self._model_path()}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    def _try_load(self) -> bool:
        """Load saved model + scaler if they exist."""
        if not self._model_path().exists() or not self._scaler_path().exists():
            return False
        if not TF_AVAILABLE:
            return False
        try:
            self.model = load_model(str(self._model_path()))
            import pickle
            with open(self._scaler_path(), "rb") as f:
                self.seq_builder = pickle.load(f)
            self._last_trained = datetime.fromtimestamp(
                self._model_path().stat().st_mtime
            )
            self.logger.info(
                f"Loaded saved model for {self.symbol} "
                f"(trained {self._last_trained.strftime('%Y-%m-%d %H:%M')})"
            )
            return True
        except Exception as e:
            self.logger.warning(f"Could not load saved model: {e}")
            return False

    # ── Training ──────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> TrainingMetrics:
        """
        Train the LSTM on historical OHLCV + indicator data.

        Args:
            df: Enriched DataFrame from DataManager.get_enriched_ohlcv()

        Returns:
            TrainingMetrics with loss history and timing info
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for training.")

        self.logger.info(
            f"Starting LSTM training for {self.symbol} "
            f"({len(df)} rows, seq_len={self.seq_len})"
        )
        t0 = time.time()

        # Build sequences
        X, y = self.seq_builder.fit_transform(df)
        n_features = X.shape[2]
        self.logger.info(
            f"Training data: X={X.shape}, y={y.shape}, features={n_features}"
        )

        # Split: last val_split fraction for validation (temporal, no shuffle)
        split_idx = int(len(X) * (1 - self.val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Build fresh model
        self.model = build_lstm_model(
            sequence_length=self.seq_len,
            n_features=n_features,
            lstm_units=self.lstm_units,
            dropout=self.dropout,
            learning_rate=self.lr,
        )
        self.model.summary(print_fn=self.logger.debug)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(3, self.patience // 3),
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(self._model_path()),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False,   # Preserve time order
        )

        self._last_trained = datetime.utcnow()
        elapsed = time.time() - t0

        # Save scaler separately (model checkpoint saves keras model)
        import pickle
        with open(self._scaler_path(), "wb") as f:
            pickle.dump(self.seq_builder, f)

        train_losses = history.history.get("loss", [])
        val_losses = history.history.get("val_loss", [])

        metrics = TrainingMetrics(
            final_train_loss=float(train_losses[-1]) if train_losses else 0.0,
            final_val_loss=float(val_losses[-1]) if val_losses else 0.0,
            best_val_loss=float(min(val_losses)) if val_losses else 0.0,
            epochs_trained=len(train_losses),
            training_time_sec=elapsed,
            feature_count=n_features,
        )

        self.logger.info(
            f"Training complete in {elapsed:.1f}s | "
            f"Train loss: {metrics.final_train_loss:.4f} | "
            f"Val loss: {metrics.final_val_loss:.4f} | "
            f"Epochs: {metrics.epochs_trained}"
        )
        return metrics

    # ── Inference ─────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> Optional[PredictionResult]:
        """
        Run inference on the most recent window in `df`.

        Args:
            df: Enriched DataFrame (must have ≥ seq_len valid rows)

        Returns:
            PredictionResult, or None if model unavailable / data insufficient
        """
        if self.model is None:
            self.logger.warning("No trained model available. Train first.")
            return None

        try:
            X = self.seq_builder.transform(df)
        except (ValueError, RuntimeError) as e:
            self.logger.warning(f"Cannot prepare inference input: {e}")
            return None

        # Monte Carlo Dropout inference: run N passes with dropout enabled
        # to estimate prediction uncertainty
        n_passes = 20
        preds = []
        for _ in range(n_passes):
            pred = self.model(X, training=True)   # training=True keeps dropout active
            preds.append(float(pred.numpy()[0, 0]))

        predicted_pct = float(np.mean(preds))
        std_pct = float(np.std(preds))

        # Confidence: higher when MC std is low relative to the predicted change
        confidence = float(np.clip(1.0 - (std_pct / (abs(predicted_pct) + 1e-6 + 1.0)), 0, 1))

        current_price = float(df["close"].iloc[-1])
        predicted_price = current_price * (1 + predicted_pct / 100)

        # Determine direction
        thresholds = self.config.get("strategy", {})
        long_thr = thresholds.get("long_threshold_pct", 1.5)
        short_thr = thresholds.get("short_threshold_pct", -1.5)

        if predicted_pct >= long_thr:
            direction = "up"
        elif predicted_pct <= short_thr:
            direction = "down"
        else:
            direction = "neutral"

        result = PredictionResult(
            symbol=self.symbol.replace("_", "/"),
            timestamp=datetime.utcnow(),
            predicted_pct_change=round(predicted_pct, 4),
            current_price=round(current_price, 8),
            predicted_price=round(predicted_price, 8),
            confidence=round(confidence, 4),
            direction=direction,
            model_version=self.MODEL_VERSION,
            features_used=self.seq_builder.feature_columns,
        )

        self.logger.info(
            f"Prediction [{self.symbol}]: {predicted_pct:+.2f}% "
            f"({direction.upper()}) | conf={confidence:.2f} | "
            f"MC std={std_pct:.3f}"
        )
        return result

    def needs_retraining(self) -> bool:
        """Returns True if model is untrained or past retraining interval."""
        if self._last_trained is None:
            return True
        retrain_interval_h = self.config.get("model", {}).get(
            "retrain_interval_hours", 24
        )
        elapsed_h = (datetime.utcnow() - self._last_trained).total_seconds() / 3600
        return elapsed_h >= retrain_interval_h


# ─────────────────────────────────────────────────────────────
# Multi-Pair Predictor Registry
# ─────────────────────────────────────────────────────────────

class PredictorRegistry:
    """
    Manages one LSTMPredictor instance per trading pair.
    Provides a unified interface for the bot's main loop.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.logger = get_logger("PredictorRegistry")
        self._predictors: dict[str, LSTMPredictor] = {}

    def get(self, symbol: str) -> LSTMPredictor:
        """Return (or create) the predictor for `symbol`."""
        if symbol not in self._predictors:
            self._predictors[symbol] = LSTMPredictor(self.config, symbol)
        return self._predictors[symbol]

    def train_all(self, data_store: dict[str, pd.DataFrame]) -> None:
        """Train / retrain models for all symbols that need it."""
        for symbol, df in data_store.items():
            predictor = self.get(symbol)
            if predictor.needs_retraining():
                self.logger.info(f"Retraining model for {symbol}...")
                try:
                    predictor.train(df)
                    predictor.save()
                except Exception as e:
                    self.logger.error(f"Training failed for {symbol}: {e}")
            else:
                self.logger.debug(f"Model for {symbol} is up to date — skipping retrain")

    def predict_all(
        self, data_store: dict[str, pd.DataFrame]
    ) -> dict[str, Optional[PredictionResult]]:
        """Run inference for all symbols, return dict of results."""
        results = {}
        for symbol, df in data_store.items():
            predictor = self.get(symbol)
            results[symbol] = predictor.predict(df)
        return results
