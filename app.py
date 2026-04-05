"""
app.py
Flask backend — serves predictions, triggers auto-retrain via APScheduler.
Endpoints:
  GET  /                        → dashboard (index.html)
  GET  /api/coins               → list of available coins
  GET  /api/predict/<coin>      → direction + pct_change prediction
  GET  /api/history/<coin>      → last 90 days OHLC from CSV
  GET  /api/model/stats         → accuracy/R² for all coins
  POST /api/retrain             → manually trigger retrain
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from flask import Flask, jsonify, render_template, request
from apscheduler.schedulers.background import BackgroundScheduler

# ── Local modules ──────────────────────────────────────────────────────────────
import data_collector
import train_model as tm

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_FILE = "models/model.pkl"
DATA_FILE  = "data/crypto_data.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Model cache ────────────────────────────────────────────────────────────────
_models: dict = {}
_loaded_at: str = ""


def load_models():
    global _models, _loaded_at
    if os.path.exists(MODEL_FILE):
        _models = joblib.load(MODEL_FILE)
        _loaded_at = datetime.utcnow().isoformat()
        log.info(f"Models loaded — {len(_models)} coins")
    else:
        log.warning("model.pkl not found. Run train_model.py first.")


def retrain_pipeline():
    """Full retrain: collect fresh data → retrain models → reload."""
    log.info("⏰ Scheduled retrain started")
    try:
        data_collector.collect_all()
        tm.train_all()
        load_models()
        log.info("✅ Retrain complete")
    except Exception as e:
        log.error(f"Retrain failed: {e}")


# ── Scheduler ──────────────────────────────────────────────────────────────────
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(retrain_pipeline, "cron", hour=1, minute=0, id="daily_retrain")


# ── Prediction helper ──────────────────────────────────────────────────────────

def predict_coin(coin_id: str) -> dict:
    if coin_id not in _models:
        return {"error": f"No model found for '{coin_id}'. Available: {list(_models.keys())}"}

    m = _models[coin_id]
    last = m["last_row"]
    feat_cols = m["feature_cols"]

    row = pd.DataFrame([last])[feat_cols].fillna(0)
    X = row.values

    direction_prob = m["classifier"].predict_proba(X)[0]
    direction_pred = int(m["classifier"].predict(X)[0])
    pct_pred = float(m["regressor"].predict(X)[0])

    return {
        "coin": coin_id,
        "predicted_direction": "UP" if direction_pred == 1 else "DOWN",
        "confidence_pct": round(float(max(direction_prob)) * 100, 1),
        "predicted_pct_change": round(pct_pred, 3),
        "current_price": last.get("close"),
        "predicted_price": round(last.get("close", 0) * (1 + pct_pred / 100), 4),
        "model_accuracy": round(m["cls_accuracy"] * 100, 1),
        "model_r2": round(m["reg_r2"], 4),
        "predicted_at": datetime.utcnow().isoformat(),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/coins")
def api_coins():
    return jsonify({"coins": list(_models.keys()), "loaded_at": _loaded_at})


@app.route("/api/predict/<coin_id>")
def api_predict(coin_id: str):
    result = predict_coin(coin_id.lower())
    status = 400 if "error" in result else 200
    return jsonify(result), status


@app.route("/api/predict/all")
def api_predict_all():
    results = {coin: predict_coin(coin) for coin in _models}
    return jsonify(results)


@app.route("/api/history/<coin_id>")
def api_history(coin_id: str):
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "Data file not found. Run data_collector.py"}), 404
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    coin_df = df[df["coin"] == coin_id.lower()].tail(90)[
        ["timestamp", "open", "high", "low", "close"]
    ]
    if coin_df.empty:
        return jsonify({"error": f"No history for '{coin_id}'"}), 404
    records = coin_df.to_dict(orient="records")
    for r in records:
        r["timestamp"] = str(r["timestamp"])
    return jsonify({"coin": coin_id, "history": records})


@app.route("/api/model/stats")
def api_model_stats():
    stats = []
    for coin, m in _models.items():
        stats.append({
            "coin": coin,
            "cls_accuracy_pct": round(m["cls_accuracy"] * 100, 1),
            "reg_r2": round(m["reg_r2"], 4),
            "last_price": m["last_row"].get("close"),
        })
    return jsonify({"models": stats, "loaded_at": _loaded_at})


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    import threading
    t = threading.Thread(target=retrain_pipeline, daemon=True)
    t.start()
    return jsonify({"message": "Retrain started in background. Check logs for progress."}), 202


# ── App entry ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_models()
    scheduler.start()
    log.info("🚀 Starting server — http://127.0.0.1:5000")
    try:
        app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
    finally:
        scheduler.shutdown()