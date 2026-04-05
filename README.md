# CryptoOracle — ML Prediction Platform
 
A full-stack Cryptocurrency Machine Learning Prediction Platform built with **Flask**, **Scikit-learn**, **XGBoost**, **LightGBM**, and **MLflow**. Predicts next-day price **direction (Up/Down)** and **% price change** for the **Top 10 cryptocurrencies** using technical indicators and an ensemble ML pipeline.
 
---
 
## Features
 
- **Top 10 coins** — Bitcoin, Ethereum, Tether, BNB, Solana, XRP, USD Coin, Dogecoin, Cardano, Avalanche
- **Dual prediction** — Direction classifier (Up/Down) + % price change regressor
- **20+ technical indicators** — RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, EMA, SMA, Williams %R
- **Lag & rolling features** — 1d, 2d, 3d, 7d, 14d, 30d lags, rolling volatility
- **Walk-forward CV** — TimeSeriesSplit cross-validation, zero data leakage
- **MLflow tracking** — per-coin accuracy + R² logged every run
- **Auto-retrain** — APScheduler retrains daily at 01:00 UTC automatically
- **Fear & Greed Index** — fetched from alternative.me and used as a feature
- **Live dashboard** — dark-theme frontend with Chart.js price/RSI/MACD charts, confidence bar, all-coins table
 
---
 
## Project Structure
 
```
crypto-ml-platform/
│
├── data/
│   └── crypto_data.csv          # OHLCV + market data (auto-created)
│
├── models/
│   └── model.pkl                # All 10 trained models bundled
│
├── mlruns/                      # MLflow experiment tracking (auto-created)
│
├── templates/
│   └── index.html               # Frontend dashboard (Chart.js, Space Mono / Syne fonts)
│
├── app.py                       # Flask server + REST API + APScheduler
├── data_collector.py            # CoinGecko + Fear & Greed data fetcher
├── train_model.py               # Feature engineering + ML training + MLflow
├── requirements.txt             # Python dependencies
└── README.md
```
 
---
 
## Quick Start
 
### 1. Navigate to your project folder
 
```powershell
# Windows PowerShell
cd C:\Users\yourname\Downloads\crypto-ml-platform
```
 
### 2. Create and activate virtual environment
 
```powershell
# Create (only once)
python -m venv .venv
 
# Activate — Windows PowerShell
.venv\Scripts\Activate.ps1
 
# Activate — Windows CMD
.venv\Scripts\activate.bat
 
# Activate — Mac / Linux
source .venv/bin/activate
```
 
> If PowerShell blocks activation, run this first (one time only):
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
 
### 3. Install dependencies
 
```powershell
pip install -r requirements.txt
```
 
### 4. Collect data from CoinGecko
 
```powershell
python data_collector.py
```
 
This fetches **365 days** of OHLC data for all 10 coins plus market cap, volume, sentiment, and Fear & Greed Index. Takes ~2–3 minutes (rate-limit safe, 6s between requests).
 
Expected output:
```
Fear & Greed Index: 72
[1/10] Fetching bitcoin...
[2/10] Fetching ethereum...
...
✅ Saved 3,650 rows → data/crypto_data.csv
```
 
### 5. Train the models
 
```powershell
python train_model.py
```
 
Trains a **GradientBoosting classifier + regressor** per coin with walk-forward cross-validation. Takes **5–10 minutes** for all 10 coins.
 
Expected output:
```
Loaded 3,650 rows from data/crypto_data.csv
 
───────────────────────────────────────────────────────
  Training: bitcoin  (280 rows)
───────────────────────────────────────────────────────
  Direction accuracy : 83.50% ± 2.10%
  Pct-change R²      : 0.7821 ± 0.0340
 
✅ Models saved → models\model.pkl
```
 
### 6. Run the app
 
```powershell
python app.py
```
 
Open your browser at:
```
http://127.0.0.1:5000
```
 
---
 
## REST API Endpoints
 
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Frontend dashboard |
| `GET` | `/api/coins` | List trained coins + model load time |
| `GET` | `/api/predict/<coin>` | Prediction for one coin |
| `GET` | `/api/predict/all` | Predictions for all coins |
| `GET` | `/api/history/<coin>` | Last 90 days OHLC for charts |
| `GET` | `/api/model/stats` | Accuracy + R² for all coins |
| `POST` | `/api/retrain` | Trigger retrain in background |
 
### Example — `GET /api/predict/bitcoin`
 
```json
{
  "coin": "bitcoin",
  "predicted_direction": "UP",
  "confidence_pct": 74.2,
  "predicted_pct_change": 1.83,
  "current_price": 44253.95,
  "predicted_price": 45063.17,
  "model_accuracy": 83.5,
  "model_r2": 0.7821,
  "predicted_at": "2024-12-31T10:00:00"
}
```
 
### Example — `GET /api/model/stats`
 
```json
{
  "models": [
    { "coin": "bitcoin",  "cls_accuracy_pct": 83.5, "reg_r2": 0.7821, "last_price": 44253.95 },
    { "coin": "ethereum", "cls_accuracy_pct": 81.2, "reg_r2": 0.7540, "last_price": 2198.44 }
  ],
  "loaded_at": "2024-12-31T10:00:00"
}
```
 
---
 
## Data Collected (`crypto_data.csv`)
 
| Column | Source | Description |
|--------|--------|-------------|
| `timestamp` | CoinGecko OHLC | Candle datetime |
| `open / high / low / close` | CoinGecko OHLC | Price candle |
| `coin` | — | Coin ID (e.g. `bitcoin`) |
| `market_cap_usd` | CoinGecko market | Market capitalisation |
| `total_volume_usd` | CoinGecko market | 24h trading volume |
| `market_cap_rank` | CoinGecko market | Global rank |
| `sentiment_votes_up_pct` | CoinGecko community | % bullish votes |
| `price_change_pct_7d` | CoinGecko market | 7-day price change |
| `price_change_pct_30d` | CoinGecko market | 30-day price change |
| `fear_greed_index` | alternative.me | 0–100 fear/greed score |
 
---
 
## Technical Indicators Used
 
| Indicator | Library | Window |
|-----------|---------|--------|
| EMA | `ta` | 12, 26 |
| SMA | `ta` | 20, 50 |
| MACD + Signal + Histogram | `ta` | 12/26/9 |
| RSI | `ta` | 14 |
| Stochastic K/D | `ta` | 14 |
| Williams %R | `ta` | 14 |
| Bollinger Bands (upper/lower/width/%B) | `ta` | 20 |
| ATR | `ta` | 14 |
| OBV | `ta` | — |
| Price range, close vs SMA | derived | — |
| Lag close + returns | derived | 1,2,3,7,14,30d |
| Rolling volatility | derived | 7, 30d |
 
---
 
## MLflow Experiment Tracking
 
MLflow logs every training run automatically to the `mlruns/` folder.
 
Open the MLflow UI in a second terminal:
 
```powershell
mlflow ui --port 5001
```
 
Then open:
```
http://127.0.0.1:5001
```
 
Metrics logged per coin per run:
- `{coin}_cls_accuracy` — walk-forward direction accuracy
- `{coin}_reg_r2` — walk-forward price R² score
 
---
 
## Auto-Retrain Scheduler
 
`app.py` runs **APScheduler** in the background. Every day at **01:00 UTC** it automatically:
 
1. Calls `data_collector.collect_all()` — fetches latest prices
2. Calls `train_model.train_all()` — retrains all 10 models
3. Reloads `model.pkl` into memory — predictions update immediately
 
To trigger a manual retrain without restarting the server, click **⟳ Retrain** in the dashboard or run:
 
```powershell
curl -X POST http://127.0.0.1:5000/api/retrain
```
 
The retrain runs in a background thread so the server stays responsive.
 
---
 
## Dashboard
 
The frontend at `http://127.0.0.1:5000` includes:
 
- **Coin selector** — buttons for all 10 coins (BTC, ETH, BNB, SOL…)
- **Stats row** — signal (▲ UP / ▼ DOWN), current price, predicted Δ%, model accuracy
- **Price chart** — 90-day close price with gradient fill (Chart.js)
- **RSI chart** — 14-period RSI calculated client-side from price history
- **MACD chart** — MACD line + signal line calculated client-side
- **Confidence bar** — visual confidence meter for the direction prediction
- **All-models table** — signal, Δ%, confidence, accuracy bar, R², current price for every coin
- **Auto-refresh** — predictions table refreshes every 60 seconds
 
---
 
## Deploy to Render
 
1. Push your project to a GitHub repository
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your repo and set:
 
| Setting | Value |
|---------|-------|
| Build command | `pip install -r requirements.txt && python data_collector.py && python train_model.py` |
| Start command | `gunicorn app:app` |
| Environment | `PYTHON_VERSION = 3.11.0` |
 
---
 
## Common Errors & Fixes
 
| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` | venv not active or deps missing | `pip install -r requirements.txt` inside venv |
| `model.pkl not found` | Models not trained yet | Run `python train_model.py` first |
| `crypto_data.csv not found` | Data not collected yet | Run `python data_collector.py` first |
| `Skipping coin: not enough rows` | CSV too small | Re-run `python data_collector.py` |
| `Port 5000 in use` | Another process on 5000 | Change to `app.run(port=5001)` in `app.py` |
| `429 Too Many Requests` | CoinGecko rate limit | Wait 60s then retry `data_collector.py` |
| PowerShell `activate` error | Execution policy | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `Unable to connect to remote server` | `app.py` not running | Run `python app.py` first, keep it running |
 
---
 
## Requirements
 
```
flask==3.0.3
requests==2.31.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
xgboost==2.0.3
lightgbm==4.3.0
ta==0.11.0
mlflow==2.13.0
apscheduler==3.10.4
joblib==1.4.2
gunicorn==22.0.0
```
 
---
 
## Tech Stack
 
| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, Flask 3.0 |
| ML Models | GradientBoosting (Scikit-learn), XGBoost, LightGBM |
| Feature Engineering | `ta` — Technical Analysis library |
| Experiment Tracking | MLflow |
| Scheduler | APScheduler 3.10 |
| Data Sources | CoinGecko API (free), alternative.me Fear & Greed |
| Frontend | HTML5, Chart.js 4.4, Space Mono + Syne fonts |
| Deployment | Render (gunicorn) |
 
---
 
*CryptoOracle · Flask + Scikit-learn + MLflow · Data: CoinGecko API*