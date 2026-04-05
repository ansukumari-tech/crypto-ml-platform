"""
data_collector.py
Fetches OHLCV + market data for the top 10 coins from CoinGecko (free tier).
Saves to data/crypto_data.csv  — run this daily for best results.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime

TOP_10_COINS = [
    "bitcoin", "ethereum", "tether", "binancecoin",
    "solana", "ripple", "usd-coin", "dogecoin",
    "cardano", "avalanche-2"
]

DAYS = 365          # history to pull (max free tier)
VS_CURRENCY = "usd"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "crypto_data.csv")


def fetch_ohlc(coin_id: str, days: int = DAYS) -> pd.DataFrame:
    """Fetch OHLC candles from CoinGecko (free, no key required)."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": VS_CURRENCY, "days": days}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["coin"] = coin_id
    return df


def fetch_market_data(coin_id: str) -> dict:
    """Fetch current market data (volume, market cap, rank, sentiment)."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "community_data": "true",
        "developer_data": "false",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    d = resp.json()
    md = d.get("market_data", {})
    return {
        "coin": coin_id,
        "market_cap_usd": md.get("market_cap", {}).get("usd"),
        "total_volume_usd": md.get("total_volume", {}).get("usd"),
        "market_cap_rank": d.get("market_cap_rank"),
        "sentiment_votes_up_pct": d.get("sentiment_votes_up_percentage"),
        "price_change_pct_7d": md.get("price_change_percentage_7d"),
        "price_change_pct_30d": md.get("price_change_percentage_30d"),
        "ath_change_pct": md.get("ath_change_percentage", {}).get("usd"),
        "circulating_supply": md.get("circulating_supply"),
        "total_supply": md.get("total_supply"),
    }


def fetch_fear_greed() -> int:
    """Fetch the Crypto Fear & Greed Index (alternative.me, free)."""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        resp.raise_for_status()
        return int(resp.json()["data"][0]["value"])
    except Exception:
        return 50   # neutral fallback


def collect_all() -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fear_greed = fetch_fear_greed()
    print(f"Fear & Greed Index: {fear_greed}")

    all_ohlc = []
    all_market = []

    for i, coin in enumerate(TOP_10_COINS):
        print(f"[{i+1}/{len(TOP_10_COINS)}] Fetching {coin}...")
        try:
            ohlc = fetch_ohlc(coin)
            all_ohlc.append(ohlc)

            market = fetch_market_data(coin)
            all_market.append(market)
        except Exception as e:
            print(f"  ⚠ Skipped {coin}: {e}")

        # CoinGecko free tier: max ~10-15 req/min
        if i < len(TOP_10_COINS) - 1:
            time.sleep(6)

    ohlc_df = pd.concat(all_ohlc, ignore_index=True)
    market_df = pd.DataFrame(all_market)

    merged = ohlc_df.merge(market_df, on="coin", how="left")
    merged["fear_greed_index"] = fear_greed
    merged["collected_at"] = datetime.utcnow().isoformat()

    # Sort and save
    merged.sort_values(["coin", "timestamp"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved {len(merged):,} rows → {OUTPUT_FILE}")
    return merged


if __name__ == "__main__":
    collect_all()