"""
train_model.py
Trains two models per coin:
  1. Classifier  — price direction (Up / Down) next day
  2. Regressor   — price % change next day
Features: OHLCV + 20 technical indicators + market data + lag features
Validation: walk-forward time-series cross-validation (no data leakage)
Tracking: MLflow experiment logging
Output: models/model.pkl
"""

import os
import warnings
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, r2_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ta  # technical analysis library

warnings.filterwarnings("ignore")

DATA_FILE   = "data/crypto_data.csv"
MODEL_DIR   = "models"
MODEL_FILE  = os.path.join(MODEL_DIR, "model.pkl")
N_SPLITS    = 5     # walk-forward CV folds
RANDOM_STATE = 42


# ─── Feature Engineering ──────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v_col = df.get("total_volume_usd", pd.Series(np.nan, index=df.index)).astype(float)

    # Trend
    df["ema_12"]   = ta.trend.EMAIndicator(c, window=12).ema_indicator()
    df["ema_26"]   = ta.trend.EMAIndicator(c, window=26).ema_indicator()
    df["macd"]     = ta.trend.MACD(c).macd()
    df["macd_sig"] = ta.trend.MACD(c).macd_signal()
    df["macd_diff"]= ta.trend.MACD(c).macd_diff()
    df["sma_20"]   = ta.trend.SMAIndicator(c, window=20).sma_indicator()
    df["sma_50"]   = ta.trend.SMAIndicator(c, window=50).sma_indicator()

    # Momentum
    df["rsi_14"]   = ta.momentum.RSIIndicator(c, window=14).rsi()
    df["stoch_k"]  = ta.momentum.StochasticOscillator(h, l, c).stoch()
    df["stoch_d"]  = ta.momentum.StochasticOscillator(h, l, c).stoch_signal()
    df["williams_r"]= ta.momentum.WilliamsRIndicator(h, l, c).williams_r()

    # Volatility
    bb = ta.volatility.BollingerBands(c, window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"]   = bb.bollinger_pband()
    df["atr_14"]   = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()

    # Volume
    df["obv"]      = ta.volume.OnBalanceVolumeIndicator(c, v_col).on_balance_volume()

    # Price-derived
    df["price_range"]    = (h - l) / c
    df["close_vs_sma20"] = (c - df["sma_20"]) / df["sma_20"]
    df["close_vs_sma50"] = (c - df["sma_50"]) / df["sma_50"]

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"return_lag_{lag}"] = df["close"].pct_change(lag)
    df["rolling_vol_7"]  = df["close"].pct_change().rolling(7).std()
    df["rolling_vol_30"] = df["close"].pct_change().rolling(30).std()
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["next_close"]   = df["close"].shift(-1)
    df["pct_change"]   = (df["next_close"] - df["close"]) / df["close"] * 100   # regression target
    df["direction"]    = (df["pct_change"] > 0).astype(int)                      # classification target (1=Up, 0=Down)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = build_targets(df)
    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "open", "high", "low", "close",
    "ema_12", "ema_26", "macd", "macd_sig", "macd_diff",
    "sma_20", "sma_50",
    "rsi_14", "stoch_k", "stoch_d", "williams_r",
    "bb_upper", "bb_lower", "bb_width", "bb_pct", "atr_14",
    "obv", "price_range", "close_vs_sma20", "close_vs_sma50",
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_7", "close_lag_14", "close_lag_30",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_7", "return_lag_14", "return_lag_30",
    "rolling_vol_7", "rolling_vol_30",
    "market_cap_rank", "sentiment_votes_up_pct",
    "price_change_pct_7d", "price_change_pct_30d",
    "fear_greed_index",
]


# ─── Walk-Forward CV ──────────────────────────────────────────────────────────

def walk_forward_score(pipeline, X: np.ndarray, y: np.ndarray, task: str) -> dict:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_te)
        if task == "classification":
            scores.append(accuracy_score(y_te, preds))
        else:
            scores.append(r2_score(y_te, preds))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "folds": scores}


# ─── Train one coin ───────────────────────────────────────────────────────────

def train_coin(coin_id: str, df: pd.DataFrame) -> dict:
    print(f"\n{'─'*55}")
    print(f"  Training: {coin_id}  ({len(df)} rows)")
    print(f"{'─'*55}")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].values
    y_cls = df["direction"].values
    y_reg = df["pct_change"].values

    cls_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE
        ))
    ])

    reg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE
        ))
    ])

    cls_scores = walk_forward_score(cls_pipe, X, y_cls, "classification")
    reg_scores = walk_forward_score(reg_pipe, X, y_reg, "regression")

    # Final fit on all data
    cls_pipe.fit(X, y_cls)
    reg_pipe.fit(X, y_reg)

    print(f"  Direction accuracy : {cls_scores['mean']:.2%} ± {cls_scores['std']:.2%}")
    print(f"  Pct-change R²      : {reg_scores['mean']:.4f} ± {reg_scores['std']:.4f}")

    return {
        "classifier": cls_pipe,
        "regressor": reg_pipe,
        "feature_cols": feat_cols,
        "cls_accuracy": cls_scores["mean"],
        "reg_r2": reg_scores["mean"],
        "last_row": df.iloc[-1].to_dict(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def train_all():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment("crypto-prediction")

    # ── Load CSV and handle different column formats ──────────────────────────
    df_raw = pd.read_csv(DATA_FILE)

    # Fix timestamp column — could be unix ms (int) or a date string
    if "timestamp" in df_raw.columns:
        try:
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms")
        except Exception:
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    elif "date" in df_raw.columns:
        df_raw["timestamp"] = pd.to_datetime(df_raw["date"])
    else:
        raise ValueError("CSV must have a 'timestamp' or 'date' column")

    # Rename volume column if needed
    if "volume" in df_raw.columns and "total_volume_usd" not in df_raw.columns:
        df_raw["total_volume_usd"] = df_raw["volume"]

    print(f"Loaded {len(df_raw):,} rows from {DATA_FILE}")
    print(f"Columns: {list(df_raw.columns)}")
    print(f"Coins found: {df_raw['coin'].unique().tolist()}")

    all_models = {}
    coins = df_raw["coin"].unique().tolist()

    with mlflow.start_run(run_name="full_retrain"):
        mlflow.log_param("coins", coins)
        mlflow.log_param("n_cv_splits", N_SPLITS)
        mlflow.log_param("features", len(FEATURE_COLS))

        for coin in coins:
            coin_df = df_raw[df_raw["coin"] == coin].copy()
            if len(coin_df) < 30:
                print(f"  ⚠ Skipping {coin}: not enough data ({len(coin_df)} rows)")
                continue

            prepared = prepare_features(coin_df)
            if len(prepared) < 30:
                print(f"  ⚠ Skipping {coin}: not enough rows after feature engineering ({len(prepared)} rows)")
                continue

            result = train_coin(coin, prepared)
            all_models[coin] = result

            mlflow.log_metric(f"{coin}_cls_accuracy", result["cls_accuracy"])
            mlflow.log_metric(f"{coin}_reg_r2", result["reg_r2"])

        joblib.dump(all_models, MODEL_FILE)
        mlflow.log_artifact(MODEL_FILE)
        print(f"\n✅ Models saved → {MODEL_FILE}")

    return all_models


if __name__ == "__main__":
    train_all()