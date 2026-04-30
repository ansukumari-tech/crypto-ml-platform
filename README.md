# 🔮 CryptoOracle — ML Prediction Platform

A machine learning platform that predicts short-term price direction and percentage change for the **top 10 cryptocurrencies** using Gradient Boosting models, 20+ technical indicators, and live CoinGecko data.

---

## 📁 Project Structure

```
crypto-ml-platform/
├── data/
│   └── crypto_data.csv          # Raw OHLCV + market data (git-ignored)
├── models/
│   └── model.pkl                # Trained models per coin (git-ignored)
├── templates/
│   └── index.html               # Frontend dashboard (Chart.js)
├── app.py                       # Flask API + APScheduler
├── data_collector.py            # CoinGecko data fetcher
├── train_model.py               # Model training + MLflow logging
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

```
data_collector.py  →  crypto_data.csv  →  train_model.py  →  model.pkl  →  app.py  →  Dashboard
```

1. **Data Collection** — `data_collector.py` pulls 365 days of OHLC candles, market data (volume, market cap, sentiment), and the Fear & Greed Index from free APIs.
2. **Feature Engineering** — `train_model.py` computes 40+ features: EMA, MACD, RSI, Bollinger Bands, ATR, OBV, lag returns, rolling volatility.
3. **Model Training** — Two scikit-learn `GradientBoosting` pipelines per coin: one classifier (UP/DOWN) and one regressor (% change). Validated with walk-forward time-series cross-validation.
4. **Serving** — Flask exposes REST endpoints consumed by the Chart.js dashboard. APScheduler retrains daily at 01:00 UTC.

---

## 🚀 Quickstart

### 1. Clone & set up environment

```bash
git clone https://github.com/<your-username>/crypto-ml-platform.git
cd crypto-ml-platform

python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell) — run this first if activation fails
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Collect data

```bash
python data_collector.py
```

This fetches ~3,650 rows (365 days × 10 coins) and saves them to `data/crypto_data.csv`.

> ⏳ Takes ~2 minutes due to CoinGecko free-tier rate limits. If you see `429 Too Many Requests` errors for some coins, wait a few minutes and re-run — already-fetched coins will be overwritten safely.

### 3. Train models

```bash
python train_model.py
```

Trains classifier + regressor for each coin using 5-fold walk-forward CV and saves `models/model.pkl`. MLflow logs metrics to `mlruns/`.

What you should see:
```
Loaded 3,650 rows from data/crypto_data.csv
Columns: ['date', 'open', 'high', 'low', 'close', 'volume', 'coin', 'timestamp', ...]
Coins found: ['bitcoin', 'ethereum', ...]

───────────────────────────────────────────────────────
  Training: bitcoin  (335 rows)
───────────────────────────────────────────────────────
  Direction accuracy : 58.00% ± 3.00%
  Pct-change R²      : 0.0312 ± 0.0100
...
✅ Models saved → models\model.pkl
```

> ⏳ Expect 2–5 minutes depending on your machine. Requires at least 30 rows per coin after feature engineering.

### 4. Run the server

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard UI |
| `GET` | `/api/coins` | List of trained coins + last loaded time |
| `GET` | `/api/predict/<coin>` | Prediction for a single coin |
| `GET` | `/api/predict/all` | Predictions for all coins |
| `GET` | `/api/history/<coin>` | Last 90 days OHLC data |
| `GET` | `/api/model/stats` | Accuracy + R² for all models |
| `POST` | `/api/retrain` | Trigger background retrain |

### Example response — `/api/predict/bitcoin`

```json
{
  "coin": "bitcoin",
  "predicted_direction": "UP",
  "confidence_pct": 67.3,
  "predicted_pct_change": 1.842,
  "current_price": 68421.5,
  "predicted_price": 69681.6,
  "model_accuracy": 58.4,
  "model_r2": 0.0312,
  "predicted_at": "2025-01-01T12:00:00"
}
```

---

## 🧠 Models & Features

### Features (40+)

| Category | Features |
|----------|----------|
| OHLCV | open, high, low, close |
| Trend | EMA 12/26, MACD, MACD signal/diff, SMA 20/50 |
| Momentum | RSI 14, Stochastic K/D, Williams %R |
| Volatility | Bollinger Bands (upper/lower/width/pct), ATR 14 |
| Volume | OBV (uses `volume` column, mapped to `total_volume_usd`) |
| Price-derived | price_range, close vs SMA20/50 |
| Lag features | close & return lags: 1, 2, 3, 7, 14, 30 days |
| Volatility | Rolling vol 7d / 30d |
| Market | market_cap_rank, sentiment_votes_up_pct |
| External | price_change_pct_7d/30d, fear_greed_index |

### Model Architecture

```
GradientBoostingClassifier   →  Direction (UP / DOWN)
  n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8

GradientBoostingRegressor    →  % Price Change
  n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8

Both wrapped in:  StandardScaler → GradientBoosting Pipeline
Validation:       TimeSeriesSplit (5 folds, no data leakage)
Minimum rows:     30 per coin after feature engineering
```

### CSV Format Compatibility

`train_model.py` automatically handles different CSV formats:

- `timestamp` column can be **Unix milliseconds** (int) or a **date string**
- Falls back to `date` column if `timestamp` is missing
- `volume` column is automatically mapped to `total_volume_usd` for OBV calculation

---

## 📊 MLflow Tracking

View experiment runs locally:

```bash
mlflow ui --port 5001
# Open http://127.0.0.1:5001
```

Metrics logged per coin:
- `<coin>_cls_accuracy` — direction accuracy across CV folds
- `<coin>_reg_r2` — R² of price % change regressor

---

## 🔄 Auto-Retrain

The scheduler in `app.py` retrains all models every day at **01:00 UTC** automatically. You can also trigger a manual retrain:

- **UI**: Click the `⟳ Retrain` button in the dashboard header
- **API**: `POST /api/retrain`
- **CLI**: `python data_collector.py && python train_model.py`

---

## 🛠️ Supported Coins

| Symbol | CoinGecko ID |
|--------|-------------|
| BTC | bitcoin |
| ETH | ethereum |
| USDT | tether |
| BNB | binancecoin |
| SOL | solana |
| XRP | ripple |
| USDC | usd-coin |
| DOGE | dogecoin |
| ADA | cardano |
| AVAX | avalanche-2 |

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `model.pkl not found` | Run `python train_model.py` first |
| `No history for coin` | Run `python data_collector.py` first |
| All coins skipped during training | Your CSV has fewer than 30 rows per coin — re-run `data_collector.py` |
| CoinGecko 429 error | Free-tier rate limit hit — wait 60s and retry, or increase `time.sleep()` to `15` in `data_collector.py` |
| Dashboard shows `—` everywhere | Models not loaded — stop server, retrain, restart `app.py` |
| PowerShell activation error | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first |
| Port 5000 already in use | Change `port=5000` to `port=5001` in `app.py` |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Cryptocurrency markets are highly volatile. These model predictions are **not financial advice** and should never be used as the sole basis for trading decisions.
