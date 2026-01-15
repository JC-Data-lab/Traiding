import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
BINANCE_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER_PRICE = "https://api.binance.com/api/v3/ticker/price"

INTERVAL = "1d"


def fetch_json_with_retries(url: str, params: dict, timeout: int = 30, tries: int = 3) -> list | dict:
    """
    Requêtes HTTP robustes (utile sur Streamlit Cloud).
    - Retry sur 429/5xx et certaines erreurs réseau.
    - Backoff progressif.
    """
    last_exc = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={
                    # Certains endpoints sont plus stables avec un UA explicite
                    "User-Agent": "Mozilla/5.0 (StreamlitApp; +https://streamlit.io)"
                },
            )
            # Retry sur rate limit / erreurs serveurs
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)

            r.raise_for_status()
            return r.json()

        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_exc = e
            # backoff: 1s, 2s, 4s ...
            time.sleep(2 ** (attempt - 1))

    raise last_exc


@st.cache_data(ttl=1800)  # 30 min
def get_top_coins(vs_currency: str, top_n: int) -> pd.DataFrame:
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": top_n,
        "page": 1,
        "sparkline": "false",
    }

    try:
        data = fetch_json_with_retries(COINGECKO_URL, params=params, timeout=30, tries=3)
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        st.error(
            "Erreur CoinGecko. "
            f"Statut HTTP: {code}. "
            "Sur Streamlit Cloud, CoinGecko peut limiter/bloquer certaines IP. "
            "Réessaie plus tard ou utilise un autre provider."
        )
        st.stop()
    except Exception as e:
        st.error(
            "Erreur réseau lors de l’appel CoinGecko. "
            "Réessaie. Si ça persiste, il faudra passer sur un autre provider (CoinMarketCap/CMC, CryptoCompare, etc.)."
        )
        st.stop()

    df = pd.DataFrame(data)[["market_cap_rank", "name", "symbol"]]
    df["symbol"] = df["symbol"].str.upper()
    return df.sort_values("market_cap_rank")


@st.cache_data(ttl=3600)  # 1h
def get_binance_symbols_set() -> set[str]:
    r = requests.get(BINANCE_EXCHANGE_INFO, timeout=30)
    r.raise_for_status()
    info = r.json()
    return {s["symbol"] for s in info["symbols"] if s.get("status") == "TRADING"}


@st.cache_data(ttl=30)  # 30s : prix live
def get_live_price(symbol: str) -> float:
    r = requests.get(BINANCE_TICKER_PRICE, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    data = r.json()
    return float(data["price"])


@st.cache_data(ttl=300)  # 5 min
def get_klines(symbol: str, interval: str = "1d", limit: int = 400) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_KLINES, params=params, timeout=30)
    r.raise_for_status()
    klines = r.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for c in ["close", "volume", "quote_asset_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date_utc"] = df["open_time"].dt.date
    return df[["date_utc", "open_time", "close_time", "close", "volume", "quote_asset_volume"]].dropna()


def get_close_on_or_before(df: pd.DataFrame, ref_date):
    if df.empty:
        return np.nan, None, None

    available_dates = sorted(df["date_utc"].unique())
    candidates = [x for x in available_dates if x <= ref_date]
    if not candidates:
        return np.nan, None, None

    d = candidates[-1]
    row = df.loc[df["date_utc"] == d].sort_values("open_time").iloc[-1]
    close_price = float(row["close"])
    close_time_utc = row["close_time"]
    return close_price, d, close_time_utc


def compute_metrics(df: pd.DataFrame, ref_date, window_days: int) -> tuple[float, float, object]:
    if df.empty:
        return np.nan, np.nan, None

    available_dates = sorted(df["date_utc"].unique())
    candidates = [x for x in available_dates if x <= ref_date]
    if not candidates:
        return np.nan, np.nan, None
    d = candidates[-1]

    idx = available_dates.index(d)
    start_idx = max(0, idx - (window_days - 1))
    window_dates = available_dates[start_idx: idx + 1]

    w = df[df["date_utc"].isin(window_dates)].sort_values("date_utc")
    if len(w) < 2:
        return np.nan, np.nan, d

    closes = w["close"].to_numpy()
    log_returns = np.log(closes[1:] / closes[:-1])
    vol = float(np.std(log_returns, ddof=1))

    liq = float(w["quote_asset_volume"].mean())
    return vol, liq, d


def format_table(out: pd.DataFrame) -> pd.DataFrame:
    out2 = out.copy()

    def r2(x): return None if pd.isna(x) else round(float(x), 2)
    def r6(x): return None if pd.isna(x) else round(float(x), 6)

    out2["Price (Live)"] = out2["Price (Live)"].map(r2)
    out2["Price (Close 1D)"] = out2["Price (Close 1D)"].map(r2)
    out2["Volatility"] = out2["Volatility"].map(r6)
    out2["Liquidity"] = out2["Liquidity"].map(lambda x: None if pd.isna(x) else round(float(x), 2))
    out2["Score"] = out2["Score"].map(lambda x: None if pd.isna(x) else round(float(x), 2))
    out2["%"] = out2["%"].map(lambda x: None if pd.isna(x) else round(float(x), 2))

    cols = [
        "Rank (MC)", "Crypto", "Symbol",
        "Price (Live)", "Price (Close 1D)",
        "Volatility", "Liquidity", "Score", "%"
    ]
    return out2[cols]


def main():
    st.set_page_config(page_title="Crypto Volatility x Liquidity", layout="wide")
    st.title("Top Crypto — Volatilité & Liquidité (CoinGecko + Binance)")

    with st.sidebar:
        st.subheader("Paramètres")
        vs_currency = st.selectbox("Devise (CoinGecko)", ["usd", "eur"], index=0)
        top_n = st.number_input("Top N", min_value=5, max_value=50, value=10, step=1)
        window_days = st.slider("Fenêtre (N jours)", min_value=7, max_value=120, value=30, step=1)
        quote = st.selectbox("Paire Binance (quote)", ["USDT"], index=0)

        ref_date = st.date_input("Date de référence (Close 1D + métriques)", value=datetime.utcnow().date())
        run = st.button("Calculer / Rafraîchir")

    st.caption(
        "Prix : Price (Live) = dernier prix échangé Binance (comme TradingView Last). "
        "Price (Close 1D) = clôture daily Binance (UTC). Daily = 00:00 → 23:59 UTC."
    )

    if not run:
        st.info("Clique sur « Calculer / Rafraîchir » pour générer le tableau.")
        return

    top = get_top_coins(vs_currency, int(top_n))
    tradable = get_binance_symbols_set()

    rows = []
    skipped = []
    used_dates = set()
    used_close_times = []

    progress = st.progress(0)
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        sym = r["symbol"]
        pair = f"{sym}{quote}"

        if pair not in tradable:
            skipped.append(pair)
            progress.progress(i / len(top))
            continue

        try:
            price_live = get_live_price(pair)
        except Exception:
            price_live = np.nan

        k = get_klines(pair, INTERVAL, limit=400)
        price_close_1d, used_date_price, close_time_utc = get_close_on_or_before(k, ref_date)
        vol, liq, used_date_metrics = compute_metrics(k, ref_date, int(window_days))

        used_date = used_date_metrics if used_date_metrics is not None else used_date_price
        if used_date:
            used_dates.add(used_date)
        if close_time_utc is not None:
            used_close_times.append(close_time_utc)

        rows.append({
            "Rank (MC)": int(r["market_cap_rank"]),
            "Crypto": r["name"],
            "Symbol": sym,
            "Price (Live)": price_live,
            "Price (Close 1D)": price_close_1d,
            "Volatility": vol,
            "Liquidity": liq,
        })
        progress.progress(i / len(top))

    progress.empty()

    if not rows:
        st.error("Aucune donnée récupérée depuis Binance pour les coins sélectionnés.")
        if skipped:
            st.write("Paires ignorées :", ", ".join(skipped))
        return

    out = pd.DataFrame(rows)
    out["Score"] = out["Volatility"] * out["Liquidity"]
    total_score = out["Score"].sum(skipna=True)
    out["%"] = (out["Score"] / total_score * 100) if total_score and total_score > 0 else np.nan
    out = out.sort_values("Score", ascending=False)

    effective_date = max(used_dates) if used_dates else ref_date
    if used_close_times:
        effective_close_time_utc = max(used_close_times)
        st.caption(
            f"Date utilisée : {effective_date} | Fenêtre : {window_days} jours | Quote : {quote} | "
            f"Clôture daily (UTC) : {effective_close_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
    else:
        st.caption(f"Date utilisée : {effective_date} | Fenêtre : {window_days} jours | Quote : {quote}")

    st.dataframe(format_table(out), use_container_width=True)

    if skipped:
        st.warning("Paires absentes sur Binance (ignorées) : " + ", ".join(skipped))


if __name__ == "__main__":
    main()
