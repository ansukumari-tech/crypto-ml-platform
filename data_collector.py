import requests
import pandas as pd

url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

params = {
    "vs_currency": "usd",
    "days": "365",
    "interval": "daily"
}

response = requests.get(url, params=params)
data = response.json()

prices = data["prices"]

records = []

for item in prices:
    record = {
        "timestamp": item[0],
        "price": item[1]
    }
    records.append(record)

df = pd.DataFrame(records)

df.to_csv("data/crypto_data.csv", index=False)

print("Crypto data collected successfully")