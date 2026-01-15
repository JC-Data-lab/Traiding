import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date, time, timedelta, timezone
from zoneinfo import ZoneInfo

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
BINANCE_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER_PRICE = "https://api.binance.com/api/v3/ticker/price"

INTERVAL = "1d"
PARIS_TZ = ZoneInfo("Europe/Paris")


@st.cache_data(ttl=1800)  # 30 min
def get_top_coins(vs_currency: str, top_n: int) -> pd.DataFrame:
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": top_n,
        "page": 1,
        "sparkline": "false",
    }
    r = requests.get(COINGECKO_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
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
    """
    Binance ticker/price = LAST PRICE (dernier prix échangé) -> comme TradingView (Last)
    """
    r = requests.get(BINANCE_TICKER_PRICE, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    data = r.json()
    return float(data["price"])


@st.cache_data(ttl=3600)  # historique intraday (15:00) : cache long
def get_price_at_paris_time(symbol: str, ref_date: date, hour: int = 15, minute: int = 0, interval: str = "1m"):
    """
    Récupère le prix à ~15:00 heure de Paris pour une date donnée.
    Méthode: prend le CLOSE de la bougie '1m' dont l'open_time correspond à 15:00 Europe/Paris (converti UTC).
    Retourne: (price, open_time_utc_used)
    """
    # 15:00 Europe/Paris -> UTC (DST auto)
    target_paris = datetime.combine(ref_date, time(hour=hour, minute=minute), tzinfo=PARIS_TZ)
    target_utc = target_paris.astimezone(timezone.utc)
    target_ms = int(target_utc.timestamp() * 1000)

    # On fetch une petite fenêtre autour (±3 minutes) pour robustesse
    start_ms = target_ms - 3 * 60_000
    end_ms = target_ms + 3 * 60_000

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 10,
    }
    r = requests.get(BINANCE_KLINES, params=params, timeout=20)
    r.raise_for_status()
    klines = r.json()

    if not klines:
        return np.nan, None

    # kline format: [open_time, open, high, low, close, volume, close_time, ...]
    # On cherche la bougie dont open_time == target_ms ; sinon on prend la plus proche.
    best = None
    best_diff = None
    for k in klines:
        ot = int(k[0])
        diff = abs(ot - target_ms)
        if best is None or diff < best_diff:
            best = k
            best_diff = diff

    if best is None:
        return np.nan, None

    open_time_utc = datetime.fromtimestamp(int(best[0]) / 1000, tz=timezone.utc)
    price_close = float(best[4])  # close
    return price_close, open_time_utc


@st.cache_data(ttl=300)  # 5 min : historique daily
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

    # Binance timestamps are UTC
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for c in ["close", "volume", "quote_asset_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date_utc"] = df["open_time"].dt.date
    return df[["date_utc", "open_time", "close_time", "close", "volume", "quote_asset_volume"]].dropna()


def get_close_on_or_before(df: pd.DataFrame, ref_date: date):
    """
    Retourne (close_price, used_date, used_close_time_utc)
    - used_date = dernière date dispo <= ref_date
    - used_close_time_utc = timestamp UTC de clôture de la bougie daily
    """
    if df.empty:
        return np.nan, None, None

    available_dates = sorted(df["date_utc"].unique())
    candidates = [x for x in available_dates if x <= ref_date]
    if not candidates:
        return np.nan, None, None

    d = candidates[-1]
    row = df.loc[df["date_utc"] == d].sort_values("open_time").iloc[-1]
    return float(row["close"]), d, row["close_time"]


def compute_metrics(df: pd.DataFrame, ref_date: date, window_days: int):
    """
    Volatility = std des rendements log sur N jours (clôtures daily)
    Liquidity  = moyenne du quote_asset_volume sur N jours (daily)
    """
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
    out2["Price (15:00 Paris)"] = out2["Price (15:00 Paris)"].map(r2)
    out2["Price (Close 1D)"] = out2["Price (Close 1D)"].map(r2)

    out2["Volatility"] = out2["Volatility"].map(r6)
    out2["Liquidity"] = out2["Liquidity"].map(lambda x: None if pd.isna(x) else round(float(x), 2))
    out2["Score"] = out2["Score"].map(lambda x: None if pd.isna(x) else round(float(x), 2))
    out2["%"] = out2["%"].map(lambda x: None if pd.isna(x) else round(float(x), 2))

    cols = [
        "Rank (MC)", "Crypto", "Symbol",
        "Price (Live)", "Price (15:00 Paris)", "Price (Close 1D)",
        "Volatility", "Liquidity", "Score", "%"
    ]
    return out2[cols]


def main():
    st.set_page_config(page_title="Crypto Volatility x Liquidity", layout="wide")
    st.title("Top 10 Crypto — Volatilité & Liquidité (CoinGecko + Binance)")

    with st.sidebar:
        st.subheader("Paramètres")
        vs_currency = st.selectbox("Devise (CoinGecko)", ["usd", "eur"], index=0)
        top_n = st.number_input("Top N", min_value=5, max_value=50, value=10, step=1)
        window_days = st.slider("Fenêtre (N jours)", min_value=7, max_value=120, value=30, step=1)
        quote = st.selectbox("Paire Binance (quote)", ["USDT"], index=0)

        ref_date = st.date_input("Date de référence (Close 1D + métriques)", value=datetime.utcnow().date())
        show_15h = st.checkbox("Afficher Price (15:00 Paris)", value=True)

        run = st.button("Calculer / Rafraîchir")

    st.caption(
        "Prix : "
        "**Price (Live)** = dernier prix échangé (Binance) = TradingView (Last). "
        "**Price (15:00 Paris)** = close de la bougie 1m autour de 15:00 Europe/Paris (DST auto). "
        "**Price (Close 1D)** = clôture daily Binance en UTC. "
        "Daily Binance : 00:00 → 23:59 UTC."
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
    used_15h_times = []

    progress = st.progress(0)
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        sym = r["symbol"]
        pair = f"{sym}{quote}"

        if pair not in tradable:
            skipped.append(pair)
            progress.progress(i / len(top))
            continue

        # 1) prix live (LAST)
        try:
            price_live = get_live_price(pair)
        except Exception:
            price_live = np.nan

        # 2) prix à 15:00 Paris (optionnel)
        price_15h = np.nan
        time_15h_utc = None
        if show_15h:
            try:
                price_15h, time_15h_utc = get_price_at_paris_time(pair, ref_date, hour=15, minute=0, interval="1m")
            except Exception:
                price_15h, time_15h_utc = np.nan, None
            if time_15h_utc is not None:
                used_15h_times.append(time_15h_utc)

        # 3) klines daily pour close + metrics
        k = get_klines(pair, INTERVAL, limit=400)

        price_close_1d, used_date_price, close_time_utc = get_close_on_or_before(k, ref_date)
        vol, liq, used_date_metrics = compute_metrics(k, ref_date, int(window_days))

        used_date = used_date_metrics if used_date_metrics is not None else used_date_price
        if used_date:
            used_dates.add(used_date)
        if close_time_utc is not None:
            used_close_times.append(close_time_utc)

        row_out = {
            "Rank (MC)": int(r["market_cap_rank"]),
            "Crypto": r["name"],
            "Symbol": sym,
            "Price (Live)": price_live,
            "Price (15:00 Paris)": price_15h,
            "Price (Close 1D)": price_close_1d,
            "Volatility": vol,
            "Liquidity": liq,
        }
        rows.append(row_out)
        progress.progress(i / len(top))

    progress.empty()

    if not rows:
        st.error("Aucune donnée récupérée depuis Binance pour les coins sélectionnés.")
        if skipped:
            st.write("Paires ignorées :", ", ".join(skipped))
        return

    out = pd.DataFrame(rows)

    # Si l'option 15h est OFF, on enlève la colonne pour ne pas polluer l'UI
    if not show_15h and "Price (15:00 Paris)" in out.columns:
        out = out.drop(columns=["Price (15:00 Paris)"])

    out["Score"] = out["Volatility"] * out["Liquidity"]
    total_score = out["Score"].sum(skipna=True)
    out["%"] = (out["Score"] / total_score * 100) if total_score and total_score > 0 else np.nan
    out = out.sort_values("Score", ascending=False)

    effective_date = max(used_dates) if used_dates else ref_date

    info_parts = [f"Date utilisée (daily/métriques) : {effective_date}", f"Fenêtre : {window_days} jours", f"Quote : {quote}"]
    if used_close_times:
        info_parts.append(f"Clôture daily (UTC) : {max(used_close_times).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if show_15h and used_15h_times:
        info_parts.append(f"Point 15:00 Paris (UTC utilisé) : {max(used_15h_times).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    st.caption(" | ".join(info_parts))

    st.dataframe(format_table(out) if show_15h else format_table(out.drop(columns=["Price (15:00 Paris)"], errors="ignore")),
                 use_container_width=True)

    if skipped:
        st.warning("Paires absentes sur Binance (ignorées) : " + ", ".join(skipped))


if __name__ == "__main__":
    main()
