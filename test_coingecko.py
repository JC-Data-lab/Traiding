
import requests
import pandas as pd

URL = "https://api.coingecko.com/api/v3/coins/markets"
PARAMS = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": 10,
    "page": 1,
    "sparkline": "false",
}

resp = requests.get(URL, params=PARAMS, timeout=30)
resp.raise_for_status()
data = resp.json()

df = pd.DataFrame(data)[["market_cap_rank", "name", "symbol", "current_price", "market_cap", "total_volume"]]
df["symbol"] = df["symbol"].str.upper()

print(df.to_string(index=False))
