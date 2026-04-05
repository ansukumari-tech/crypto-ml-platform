"""
train_model.py  — fixed for crypto_data.csv columns:
date, open, high, low, close, volume, coin, timestamp, price
"""

import os
import warnings
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ta

warnings.filterwarnings("ignore")

DATA_FILE    = "data/crypto_data.csv"
MODEL_DIR    = "models"
MODEL_FILE   = os.path.join(MODEL_DIR, "model.pkl")
N_SPLITS     = 5
RANDOM_STATE = 42


def add_technical_indicators(df):
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    df["ema_12"]    = ta.trend.EMAIndicator(c, window=12).ema_indicator()
    df["ema_26"]    = ta.trend.EMAIndicator(c, window=26).ema_indicator()
    df["sma_20"]    = ta.trend.SMAIndicator(c, window=20).sma_indicator()
    df["sma_50"]    = ta.trend.SMAIndicator(c, window=50).sma_indicator()
    macd            = ta.trend.MACD(c)
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df["rsi_14"]    = ta.momentum.RSIIndicator(c, window=14).rsi()
    stoch           = ta.momentum.StochasticOscillator(h, l, c, window=14)
    df["stoch_k"]   = stoch.stoch()
    df["stoch_d"]   = stoch.stoch_signal()
    bb              = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_width"]  = bb.bollinger_wband()
    df["bb_pct"]    = bb.bollinger_pband()
    df["atr_14"]    = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["obv"]       = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["price_range"]    = h - l
    df["close_vs_sma20"] = c / df["sma_20"] - 1
    df["close_vs_sma50"] = c / df["sma_50"] - 1
    return df

def add_lag_features(df):
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"close_lag_{lag}"]  = df["close"].shift(lag)
        df[f"return_lag_{lag}"] = df["close"].pct_change(lag)
    for w in [7, 30]:
        df[f"rolling_vol_{w}"] = df["close"].pct_change().rolling(w).std()
    return df

def build_targets(df):
    df["next_close"] = df["close"].shift(-1)
    df["pct_change"] = (df["next_close"] - df["close"]) / df["close"] * 100
    df["direction"]  = (df["pct_change"] > 0).astype(int)
    return df

FEATURE_COLS = [
    "open","high","low","close","volume",
    "ema_12","ema_26","sma_20","sma_50",
    "macd","macd_sig","macd_diff",
    "rsi_14","stoch_k","stoch_d",
    "bb_upper","bb_lower","bb_width","bb_pct","atr_14",
    "obv","price_range","close_vs_sma20","close_vs_sma50",
    "close_lag_1","close_lag_2","close_lag_3","close_lag_7","close_lag_14","close_lag_30",
    "return_lag_1","return_lag_2","return_lag_3","return_lag_7","return_lag_14","return_lag_30",
    "rolling_vol_7","rolling_vol_30",
]

def prepare_features(df):
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = build_targets(df)
    df = df.dropna()
    return df

def walk_forward_score(pipeline, X, y, task):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []
    for tr, te in tscv.split(X):
        pipeline.fit(X[tr], y[tr])
        preds = pipeline.predict(X[te])
        scores.append(accuracy_score(y[te], preds) if task=="classification" else r2_score(y[te], preds))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}

def train_coin(coin_id, df):
    print(f"\n  Training: {coin_id}  ({len(df)} rows)")
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X     = df[feat_cols].values
    y_cls = df["direction"].values
    y_reg = df["pct_change"].values
    cls_pipe = Pipeline([("scaler", StandardScaler()), ("model", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE))])
    reg_pipe = Pipeline([("scaler", StandardScaler()), ("model", GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE))])
    cls_s = walk_forward_score(cls_pipe, X, y_cls, "classification")
    reg_s = walk_forward_score(reg_pipe, X, y_reg, "regression")
    cls_pipe.fit(X, y_cls)
    reg_pipe.fit(X, y_reg)
    print(f"    Accuracy : {cls_s['mean']:.2%}  |  R² : {reg_s['mean']:.4f}")
    return {"classifier": cls_pipe, "regressor": reg_pipe, "feature_cols": feat_cols, "cls_accuracy": cls_s["mean"], "reg_r2": reg_s["mean"], "last_row": df.iloc[-1].to_dict()}

def train_all():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df_raw = pd.read_csv(DATA_FILE)
    df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.strftime("%Y-%m-%d")
    print(f"Loaded {len(df_raw):,} rows | Coins: {sorted(df_raw['coin'].unique().tolist())}")
    mlflow.set_experiment("crypto-prediction")
    all_models = {}
    with mlflow.start_run(run_name="full_retrain"):
        for coin in sorted(df_raw["coin"].unique()):
            coin_df  = df_raw[df_raw["coin"] == coin].copy()
            prepared = prepare_features(coin_df)
            if len(prepared) < 50:
                print(f"  Skipping {coin}: only {len(prepared)} rows after features")
                continue
            result = train_coin(coin, prepared)
            all_models[coin] = result
            mlflow.log_metric(f"{coin}_accuracy", result["cls_accuracy"])
        joblib.dump(all_models, MODEL_FILE)
        mlflow.log_artifact(MODEL_FILE)
    print(f"\n✅ Trained {len(all_models)} coins → {MODEL_FILE}")
    for coin, r in all_models.items():
        print(f"  {coin:20s}  acc={r['cls_accuracy']:.2%}  r2={r['reg_r2']:.4f}")

if __name__ == "__main__":
    train_all()