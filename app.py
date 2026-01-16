import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date, time as dtime, timezone, timedelta
from zoneinfo import ZoneInfo

# ---------- CONFIG ----------
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

# CryptoCompare (min-api)
CC_PRICE = "https://min-api.cryptocompare.com/data/price"
CC_HISTODAY = "https://min-api.cryptocompare.com/data/v2/histoday"
CC_HISTOHOUR = "https://min-api.cryptocompare.com/data/v2/histohour"

INTERVAL = "1d"
PARIS_TZ = ZoneInfo("Europe/Paris")


# ---------- HELPERS ----------
def _cc_headers() -> dict:
    # CryptoCompare recommande l'auth en header: Authorization: Apikey <key>
    api_key = st.secrets.get("CRYPTOCOMPARE_API_KEY", "")
    return {"Authorization": f"Apikey {api_key}", "Accept": "application/json", "User-Agent": "Mozilla/5.0"}


def safe_get_json(url: str, params: dict | None = None, timeout: int = 30, tries: int = 4) -> dict:
    """
    GET robuste (retry + backoff) — utile sur Cloud.
    """
    last_exc = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=_cc_headers())
            # Retry sur rate limit / erreurs temporaires
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_exc = e
            time.sleep(2 ** (attempt - 1))
    raise last_exc


def mondays_of_year(year: int) -> list[date]:
    """Retourne tous les lundis de l'année."""
    d = date(year, 1, 1)
    while d.weekday() != 0:  # Monday=0
        d += timedelta(days=1)
    out = []
    while d.year == year:
        out.append(d)
        d += timedelta(days=7)
    return out


# ---------- COINGECKO ----------
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


# ---------- CRYPTOCOMPARE ----------
@st.cache_data(ttl=30)  # 30s
def get_live_price_cc(fsym: str, tsym: str) -> float:
    """
    Prix live (dernier prix agrégé CryptoCompare).
    """
    try:
        data = safe_get_json(CC_PRICE, params={"fsym": fsym, "tsyms": tsym}, timeout=15, tries=3)
        return float(data.get(tsym, np.nan))
    except Exception:
        return np.nan


@st.cache_data(ttl=600)  # 10 min
def get_histoday_cc(fsym: str, tsym: str, limit: int = 400, to_ts: int | None = None) -> pd.DataFrame:
    """
    Daily OHLCV (CryptoCompare). Retourne close + volumeto (proxy liquidité en monnaie tsym).
    """
    params = {"fsym": fsym, "tsym": tsym, "limit": int(limit)}
    if to_ts is not None:
        params["toTs"] = int(to_ts)

    j = safe_get_json(CC_HISTODAY, params=params, timeout=30, tries=4)

    if j.get("Response") != "Success":
        raise requests.HTTPError(f"CryptoCompare histoday error: {j.get('Message', 'Unknown')}")

    rows = j["Data"]["Data"]
    df = pd.DataFrame(rows)

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["date_utc"] = df["time"].dt.date

    for c in ["close", "volumeto", "volumefrom"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[["date_utc", "time", "close", "volumeto", "volumefrom"]].dropna()


@st.cache_data(ttl=3600)  # 1h
def get_price_at_paris_15h_cc(
    fsym: str, tsym: str, ref_date: date, hour: int = 15
) -> tuple[float, datetime | None]:
    """
    Prix au plus proche de 15:00 Europe/Paris via histohour.
    (On prend le CLOSE 1H de l'heure la plus proche.)
    """
    target_paris = datetime.combine(ref_date, dtime(hour=hour, minute=0), tzinfo=PARIS_TZ)
    target_utc = target_paris.astimezone(timezone.utc)
    target_ts = int(target_utc.timestamp())

    params = {
        "fsym": fsym,
        "tsym": tsym,
        "limit": 6,
        "toTs": target_ts,
    }

    try:
        j = safe_get_json(CC_HISTOHOUR, params=params, timeout=30, tries=4)
        if j.get("Response") != "Success":
            return np.nan, None

        rows = j["Data"]["Data"]
        if not rows:
            return np.nan, None

        best = None
        best_diff = None
        for r in rows:
            t = int(r["time"])
            diff = abs(t - target_ts)
            if best is None or diff < best_diff:
                best = r
                best_diff = diff

        if best is None:
            return np.nan, None

        used_time_utc = datetime.fromtimestamp(int(best["time"]), tz=timezone.utc)
        price_close = float(best.get("close", np.nan))
        return price_close, used_time_utc

    except Exception:
        return np.nan, None


@st.cache_data(ttl=3600)  # 1h
def build_monday_15h_index_2025(fsym: str, tsym: str, hour: int = 15) -> pd.DataFrame:
    """
    Construit un indice base 100 sur tous les lundis 2025 à 15:00 Paris,
    basé sur le Close 1H (CryptoCompare histohour).
    """
    mondays = mondays_of_year(2025)

    rows = []
    for d in mondays:
        p, used_utc = get_price_at_paris_15h_cc(fsym, tsym, d, hour=hour)
        rows.append(
            {
                "Monday": d,
                "Price (Close 1H @ 15:00 Paris)": p,
                "UTC used": used_utc,
            }
        )

    df = pd.DataFrame(rows)

    # Nettoyage
    df["Monday"] = pd.to_datetime(df["Monday"])
    df["Price (Close 1H @ 15:00 Paris)"] = pd.to_numeric(df["Price (Close 1H @ 15:00 Paris)"], errors="coerce")
    df = df.dropna(subset=["Price (Close 1H @ 15:00 Paris)"]).sort_values("Monday")

    if df.empty or len(df) < 2:
        df["Index (Base 100)"] = np.nan
        return df

    # Indice base 100
    prices = df["Price (Close 1H @ 15:00 Paris)"].to_numpy()
    idx = np.zeros(len(prices), dtype=float)
    idx[0] = 100.0
    for i in range(1, len(prices)):
        idx[i] = idx[i - 1] * (prices[i] / prices[i - 1])

    df["Index (Base 100)"] = idx
    return df


# ---------- METRICS ----------
def compute_metrics(df: pd.DataFrame, ref_date: date, window_days: int) -> tuple[float, float, date | None]:
    """
    Volatilité = std des rendements log sur la fenêtre
    Liquidité (proxy) = moyenne de volumeto sur la fenêtre (volume en monnaie tsym)
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

    liq = float(w["volumeto"].mean())
    return vol, liq, d


def get_close_on_or_before(df: pd.DataFrame, ref_date: date) -> tuple[float, date | None]:
    if df.empty:
        return np.nan, None
    available_dates = sorted(df["date_utc"].unique())
    candidates = [x for x in available_dates if x <= ref_date]
    if not candidates:
        return np.nan, None
    d = candidates[-1]
    close_price = float(df.loc[df["date_utc"] == d, "close"].iloc[-1])
    return close_price, d


def format_table(out: pd.DataFrame, show_15h: bool) -> pd.DataFrame:
    out2 = out.copy()

    def r2(x): return None if pd.isna(x) else round(float(x), 2)
    def r6(x): return None if pd.isna(x) else round(float(x), 6)

    out2["Price (Live)"] = out2["Price (Live)"].map(r2)
    out2["Price (Close 1D)"] = out2["Price (Close 1D)"].map(r2)
    if show_15h and "Price (15:00 Paris)" in out2.columns:
        out2["Price (15:00 Paris)"] = out2["Price (15:00 Paris)"].map(r2)

    out2["Volatility"] = out2["Volatility"].map(r6)
    out2["Liquidity"] = out2["Liquidity"].map(lambda x: None if pd.isna(x) else round(float(x), 2))
    out2["Score"] = out2["Score"].map(lambda x: None if pd.isna(x) else round(float(x), 2))
    out2["%"] = out2["%"].map(lambda x: None if pd.isna(x) else round(float(x), 2))

    cols = ["Rank (MC)", "Crypto", "Symbol", "Price (Live)"]
    if show_15h:
        cols += ["Price (15:00 Paris)"]
    cols += ["Price (Close 1D)", "Volatility", "Liquidity", "Score", "%"]
    return out2[cols]


# ---------- APP ----------
def main():
    st.set_page_config(page_title="Crypto Volatility x Liquidity", layout="wide")
    st.title("Top Crypto — Volatilité & Liquidité (CoinGecko + CryptoCompare)")

    with st.sidebar:
        st.subheader("Paramètres")

        vs_currency = st.selectbox("Devise (CoinGecko)", ["usd", "eur"], index=0)
        top_n = st.number_input("Top N", min_value=5, max_value=50, value=10, step=1)
        window_days = st.slider("Fenêtre (N jours)", min_value=7, max_value=120, value=30, step=1)
        ref_date = st.date_input("Date de référence (daily)", value=datetime.utcnow().date())
        show_15h = st.checkbox("Afficher Price (15:00 Paris)", value=True)

        # Graph (lundis 2025)
        st.divider()
        show_graph = st.checkbox("Afficher le graphique (lundis 15h 2025)", value=True)

        run = st.button("Calculer / Rafraîchir")

    tsym = "USD" if vs_currency.lower() == "usd" else "EUR"

    st.caption(
        "Sources : CoinGecko (Top coins) + CryptoCompare (prix + OHLCV + volumes). "
        "Interprétation : Volatility ↑ = ça bouge plus ; Liquidity ↑ = volume plus élevé ; "
        "Score = Volatility × Liquidity ; % = part du score."
    )

    if not run:
        st.info("Clique sur « Calculer / Rafraîchir » pour générer le tableau.")
        return

    if not st.secrets.get("CRYPTOCOMPARE_API_KEY"):
        st.error("Secret manquant : CRYPTOCOMPARE_API_KEY. Va dans Streamlit Cloud → Settings → Secrets.")
        st.stop()

    top = get_top_coins(vs_currency, int(top_n))

    rows = []
    skipped = []
    used_dates = set()
    used_15h_times = []

    progress = st.progress(0.0)

    for i, (_, r) in enumerate(top.iterrows(), start=1):
        sym = r["symbol"]

        try:
            price_live = get_live_price_cc(sym, tsym)

            to_ts = int(datetime(ref_date.year, ref_date.month, ref_date.day, tzinfo=timezone.utc).timestamp())
            dfd = get_histoday_cc(sym, tsym, limit=400, to_ts=to_ts)

            price_close_1d, used_date_close = get_close_on_or_before(dfd, ref_date)
            vol, liq, used_date_metrics = compute_metrics(dfd, ref_date, int(window_days))

            used_date = used_date_metrics if used_date_metrics is not None else used_date_close
            if used_date:
                used_dates.add(used_date)

            price_15h = np.nan
            if show_15h:
                price_15h, used_15h_utc = get_price_at_paris_15h_cc(sym, tsym, ref_date, hour=15)
                if used_15h_utc is not None:
                    used_15h_times.append(used_15h_utc)

            row_out = {
                "Rank (MC)": int(r["market_cap_rank"]),
                "Crypto": r["name"],
                "Symbol": sym,
                "Price (Live)": price_live,
                "Price (Close 1D)": price_close_1d,
                "Volatility": vol,
                "Liquidity": liq,
            }
            if show_15h:
                row_out["Price (15:00 Paris)"] = price_15h

            rows.append(row_out)

        except Exception:
            skipped.append(sym)

        progress.progress(i / len(top))

    progress.empty()

    if not rows:
        st.error("Aucune donnée exploitable. Vérifie la clé CryptoCompare et réessaie.")
        if skipped:
            st.write("Symboles ignorés :", ", ".join(skipped))
        return

    out = pd.DataFrame(rows)

    out["Score"] = out["Volatility"] * out["Liquidity"]
    total_score = out["Score"].sum(skipna=True)
    out["%"] = (out["Score"] / total_score * 100) if total_score and total_score > 0 else np.nan
    out = out.sort_values("Score", ascending=False)

    effective_date = max(used_dates) if used_dates else ref_date

    info_parts = [
        f"Devise : {tsym}",
        f"Date utilisée (daily) : {effective_date}",
        f"Fenêtre : {window_days} jours",
    ]
    if show_15h and used_15h_times:
        info_parts.append(f"Point 15:00 Paris (UTC utilisé) : {max(used_15h_times).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    st.caption(" | ".join(info_parts))

    # --- TABLE ---
    st.dataframe(format_table(out, show_15h=show_15h), use_container_width=True)

    if skipped:
        st.warning("Symboles ignorés (pas dispo / erreur provider) : " + ", ".join(skipped))

    # --- GRAPH AFTER TABLE ---
    if show_graph and not out.empty:
        st.subheader("Graphique — Indice Base 100 (tous les lundis 2025 à 15:00 Paris, Close 1H)")

        # Sélection coin pour le graphique (basé sur la table actuelle)
        symbols = out["Symbol"].dropna().unique().tolist()
        default_symbol = symbols[0] if symbols else "BTC"

        col1, col2 = st.columns([1, 2])
        with col1:
            graph_symbol = st.selectbox("Crypto (graph)", options=symbols, index=0)

        # Construction série (cache 1h)
        serie = build_monday_15h_index_2025(graph_symbol, tsym, hour=15)

        if serie.empty or serie["Index (Base 100)"].dropna().empty:
            st.warning("Impossible de construire la série (données manquantes). Essaie un autre symbole.")
        else:
            # Petite table de contrôle
            with col2:
                st.caption("Interprétation : l’indice démarre à 100 au 1er lundi 2025 (15:00 Paris). "
                           "S’il monte, la crypto a progressé entre les lundis ; s’il baisse, elle a reculé.")

            st.dataframe(
                serie[["Monday", "Price (Close 1H @ 15:00 Paris)", "Index (Base 100)"]].copy(),
                use_container_width=True
            )

            chart_df = serie[["Monday", "Index (Base 100)"]].set_index("Monday")
            st.line_chart(chart_df)

    # fin main


if __name__ == "__main__":
    main()
