"""
Crypto Volatility x Liquidity Dashboard
Sources : CoinGecko (Top coins) + CryptoCompare (prix, OHLCV, volumes)

Lancer en local :
    streamlit run app.py
"""

import time
from datetime import datetime, date, time as dtime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------- CONFIG ----------
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

CC_PRICE = "https://min-api.cryptocompare.com/data/price"
CC_HISTODAY = "https://min-api.cryptocompare.com/data/v2/histoday"
CC_HISTOHOUR = "https://min-api.cryptocompare.com/data/v2/histohour"

PARIS_TZ = ZoneInfo("Europe/Paris")


# ---------- HELPERS HTTP ----------
def _cc_headers() -> dict:
    """Headers pour CryptoCompare (avec clé API si présente)."""
    api_key = st.secrets.get("CRYPTOCOMPARE_API_KEY", "")
    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
    if api_key:
        headers["Authorization"] = f"Apikey {api_key}"
    return headers


def safe_get_json(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    tries: int = 4,
) -> dict:
    """GET robuste avec retry exponentiel sur erreurs temporaires."""
    last_exc = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_exc = e
            if attempt < tries:
                time.sleep(2 ** (attempt - 1))
    raise last_exc


def mondays_of_year(year: int) -> list[date]:
    """Retourne tous les lundis d'une année donnée."""
    d = date(year, 1, 1)
    while d.weekday() != 0:  # 0 = lundi
        d += timedelta(days=1)
    out = []
    while d.year == year:
        out.append(d)
        d += timedelta(days=7)
    return out


def utc_now_date() -> date:
    """Date UTC d'aujourd'hui (remplace datetime.utcnow() deprecated)."""
    return datetime.now(timezone.utc).date()


# ---------- COINGECKO ----------
@st.cache_data(ttl=1800, show_spinner=False)  # 30 min
def get_top_coins(vs_currency: str, top_n: int) -> pd.DataFrame:
    """Top N cryptos par market cap (CoinGecko) — SNAPSHOT ACTUEL uniquement.

    NOTE : CoinGecko ne permet pas de récupérer le Top N à une date passée
    (pas sur le plan gratuit). Le classement utilisé est celui du moment
    où tu cliques sur « Rafraîchir ».
    """
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": top_n,
        "page": 1,
        "sparkline": "false",
    }
    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
    data = safe_get_json(COINGECKO_URL, params=params, headers=headers, timeout=30, tries=4)
    df = pd.DataFrame(data)[["market_cap_rank", "name", "symbol"]]
    df["symbol"] = df["symbol"].str.upper()
    return df.sort_values("market_cap_rank").reset_index(drop=True)


# ---------- CRYPTOCOMPARE ----------
@st.cache_data(ttl=30, show_spinner=False)  # 30s
def get_live_price_cc(fsym: str, tsym: str) -> float:
    """Prix live agrégé CryptoCompare."""
    try:
        data = safe_get_json(
            CC_PRICE,
            params={"fsym": fsym, "tsyms": tsym},
            headers=_cc_headers(),
            timeout=15,
            tries=3,
        )
        return float(data.get(tsym, np.nan))
    except Exception:
        return np.nan


@st.cache_data(ttl=600, show_spinner=False)  # 10 min
def get_histoday_cc(fsym: str, tsym: str, limit: int = 2000, to_ts: int | None = None) -> pd.DataFrame:
    """Historique journalier OHLCV.

    On récupère l'historique le plus large possible (limit=2000 jours ≈ 5 ans).
    Ça permet ensuite de calculer les fenêtres glissantes sur n'importe quelle
    date historique sans refaire d'appels API.
    """
    params = {"fsym": fsym, "tsym": tsym, "limit": int(limit)}
    if to_ts is not None:
        params["toTs"] = int(to_ts)

    j = safe_get_json(CC_HISTODAY, params=params, headers=_cc_headers(), timeout=30, tries=4)

    if j.get("Response") != "Success":
        raise requests.HTTPError(f"CryptoCompare histoday error: {j.get('Message', 'Unknown')}")

    rows = j.get("Data", {}).get("Data", [])
    if not rows:
        return pd.DataFrame(columns=["date_utc", "time", "close", "volumeto", "volumefrom"])

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["date_utc"] = df["time"].dt.date

    for c in ["close", "volumeto", "volumefrom"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[["date_utc", "time", "close", "volumeto", "volumefrom"]].dropna()


@st.cache_data(ttl=3600, show_spinner=False)  # 1h
def get_price_at_paris_hour_cc(
    fsym: str, tsym: str, ref_date: date, hour: int = 15
) -> tuple[float, datetime | None]:
    """Prix au plus proche d'une heure Paris donnée, via histohour."""
    target_paris = datetime.combine(ref_date, dtime(hour=hour, minute=0), tzinfo=PARIS_TZ)
    target_utc = target_paris.astimezone(timezone.utc)
    target_ts = int(target_utc.timestamp())

    params = {"fsym": fsym, "tsym": tsym, "limit": 6, "toTs": target_ts}

    try:
        j = safe_get_json(CC_HISTOHOUR, params=params, headers=_cc_headers(), timeout=30, tries=4)
        if j.get("Response") != "Success":
            return np.nan, None

        rows = j.get("Data", {}).get("Data", [])
        if not rows:
            return np.nan, None

        best = min(rows, key=lambda r: abs(int(r["time"]) - target_ts))
        used_time_utc = datetime.fromtimestamp(int(best["time"]), tz=timezone.utc)
        return float(best.get("close", np.nan)), used_time_utc
    except Exception:
        return np.nan, None


# ---------- METRICS ----------
def compute_metrics_on_date(
    df: pd.DataFrame, ref_date: date, window_days: int
) -> tuple[float, float, date | None]:
    """Calcule vol/liq à une date donnée sur une fenêtre glissante de N jours.

    Cette fonction est le coeur de l'analyse : elle peut être appelée pour
    la date actuelle (tableau) OU pour n'importe quelle date historique
    (graphique).
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
    window_dates = available_dates[start_idx : idx + 1]
    w = df[df["date_utc"].isin(window_dates)].sort_values("date_utc")

    if len(w) < 2:
        return np.nan, np.nan, d

    closes = w["close"].to_numpy()
    log_returns = np.log(closes[1:] / closes[:-1])
    vol = float(np.std(log_returns, ddof=1))
    liq = float(w["volumeto"].mean())
    return vol, liq, d


def get_close_on_or_before(df: pd.DataFrame, ref_date: date) -> tuple[float, date | None]:
    """Prix de clôture du jour ref_date ou du jour précédent disponible."""
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
    """Formatage des colonnes pour affichage."""
    out2 = out.copy()

    def r2(x):
        return None if pd.isna(x) else round(float(x), 2)

    def r6(x):
        return None if pd.isna(x) else round(float(x), 6)

    out2["Price (Live)"] = out2["Price (Live)"].map(r2)
    out2["Price (Close 1D)"] = out2["Price (Close 1D)"].map(r2)
    if show_15h and "Price (15:00 Paris)" in out2.columns:
        out2["Price (15:00 Paris)"] = out2["Price (15:00 Paris)"].map(r2)

    out2["Volatility"] = out2["Volatility"].map(r6)
    out2["Liquidity"] = out2["Liquidity"].map(r2)
    out2["Score"] = out2["Score"].map(r2)
    out2["%"] = out2["%"].map(r2)

    cols = ["Rank (MC)", "Crypto", "Symbol", "Price (Live)"]
    if show_15h:
        cols += ["Price (15:00 Paris)"]
    cols += ["Price (Close 1D)", "Volatility", "Liquidity", "Score", "%"]
    return out2[cols]


# ---------- GRAPHIQUE : CALCUL HISTORIQUE ----------
#
# ==============================================================================
# WARNING — Périmètre du graphique (Top N fixe)
# ==============================================================================
# Le graphique historique utilise le TOP N ACTUEL de CoinGecko comme "panier"
# figé pour toute l'année. Autrement dit :
#
#   - Le % score affiché au survol = part du score de la crypto dans la somme
#     des scores du Top N ACTUEL, recalculée à chaque lundi avec une fenêtre
#     glissante de N jours.
#
#   - Si une crypto est entrée dans le Top 10 en milieu d'année (ex: elle
#     n'était pas là en janvier 2025), elle apparaît quand même dans le
#     graphique pour janvier 2025. Son % est calculé comme si elle faisait
#     partie du panier depuis le début.
#
#   - Inversement, une crypto qui était dans le Top 10 en janvier 2025 mais
#     qui n'y est plus aujourd'hui n'apparaîtra pas dans le graphique.
#
# POURQUOI CETTE LIMITE ?
# L'endpoint /coins/markets de CoinGecko ne permet pas de requêter le Top N
# à une date passée (pas sur le plan gratuit). Reconstituer le Top N
# historique nécessiterait de requêter le market cap historique de ~17 000
# cryptos pour chaque lundi, ce qui est infaisable (rate limit, coût).
#
# IMPACT EN PRATIQUE ?
# Le Top 10 crypto est assez stable dans le temps (BTC, ETH, SOL, XRP, BNB
# sont là depuis longtemps). L'impact réel est donc limité, mais il faut
# en être conscient pour interpréter correctement le graphique.
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)  # 1h
def build_full_history_for_symbols(
    symbols: tuple[str, ...], tsym: str, year: int, hour: int, window_days: int
) -> pd.DataFrame:
    """Construit toutes les données historiques pour le graphique d'une année.

    Retourne un DataFrame long avec colonnes :
        - Monday (datetime)
        - Symbol (str)
        - Price (float) : prix à hour:00 Paris (Close 1H)
        - Index (float) : base 100 au 1er lundi disponible
        - Volatility (float) : std log-returns sur fenêtre glissante N jours
        - Liquidity (float) : moyenne volumeto sur fenêtre glissante N jours
        - Score (float) : Volatility × Liquidity
        - Pct (float) : part du Score dans la somme des scores du Top N à ce lundi

    Arguments :
        symbols : tuple de symboles (tuple pour être hashable -> cacheable)
        tsym : devise cible (USD/EUR)
        year : année à traiter
        hour : heure Paris pour le prix de référence des lundis
        window_days : taille de la fenêtre glissante
    """
    # 1) Lister les lundis passés de l'année
    mondays = mondays_of_year(year)
    today = utc_now_date()
    mondays = [d for d in mondays if d <= today]

    if not mondays:
        return pd.DataFrame()

    # 2) Pour chaque symbole, récupérer UNE FOIS tout l'historique daily
    #    (limit=2000 -> ~5 ans, largement assez)
    histoday_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df_hist = get_histoday_cc(sym, tsym, limit=2000)
            histoday_by_sym[sym] = df_hist
        except Exception:
            histoday_by_sym[sym] = pd.DataFrame()

    # 3) Pour chaque lundi x chaque symbole, calculer :
    #    - Prix Close 1H a hour:00 Paris (via histohour)
    #    - Vol/Liq/Score sur fenetre glissante (via histoday deja en cache)
    rows = []
    for d in mondays:
        for sym in symbols:
            price, _ = get_price_at_paris_hour_cc(sym, tsym, d, hour=hour)

            df_hist = histoday_by_sym.get(sym, pd.DataFrame())
            vol, liq, _ = compute_metrics_on_date(df_hist, d, window_days)

            score = vol * liq if (not np.isnan(vol) and not np.isnan(liq)) else np.nan

            rows.append({
                "Monday": d,
                "Symbol": sym,
                "Price": price,
                "Volatility": vol,
                "Liquidity": liq,
                "Score": score,
            })

    df = pd.DataFrame(rows)
    df["Monday"] = pd.to_datetime(df["Monday"])

    # 4) Calcul du % score par lundi (part du score dans le total du lundi)
    total_scores = df.groupby("Monday")["Score"].transform("sum")
    df["Pct"] = np.where(total_scores > 0, df["Score"] / total_scores * 100, np.nan)

    # 5) Calcul de l'indice base 100 par symbole
    #    Le 1er lundi avec un prix valide = base 100
    df = df.sort_values(["Symbol", "Monday"]).copy()
    df["Index"] = np.nan

    for sym in df["Symbol"].unique():
        mask = df["Symbol"] == sym
        sub = df.loc[mask].sort_values("Monday")
        valid_prices = sub["Price"].dropna()
        if len(valid_prices) >= 1:
            base_price = valid_prices.iloc[0]
            df.loc[mask, "Index"] = 100.0 * df.loc[mask, "Price"] / base_price

    return df.reset_index(drop=True)


def build_plotly_chart(
    df_long: pd.DataFrame, year: int, hour: int, window_days: int
) -> go.Figure:
    """Construit la figure Plotly avec hover unifié.

    Hover : pour un lundi donné, affiche toutes les cryptos + leur Index + leur %.
    """
    fig = go.Figure()

    symbols = df_long["Symbol"].unique().tolist()

    for sym in symbols:
        sub = df_long[df_long["Symbol"] == sym].sort_values("Monday")

        # customdata pour le hover : [Index, Pct, Price, Vol, Liq, Score]
        customdata = np.column_stack([
            sub["Index"].to_numpy(),
            sub["Pct"].to_numpy(),
            sub["Price"].to_numpy(),
            sub["Volatility"].to_numpy(),
            sub["Liquidity"].to_numpy(),
            sub["Score"].to_numpy(),
        ])

        fig.add_trace(go.Scatter(
            x=sub["Monday"],
            y=sub["Index"],
            mode="lines+markers",
            name=sym,
            customdata=customdata,
            hovertemplate=(
                f"<b>{sym}</b><br>"
                "Index : %{customdata[0]:.2f}<br>"
                "Part Score : %{customdata[1]:.2f}%<br>"
                "Prix : %{customdata[2]:,.2f}<br>"
                "Volatilite : %{customdata[3]:.6f}<br>"
                "Liquidite : %{customdata[4]:,.0f}<br>"
                "Score : %{customdata[5]:,.2f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"Indice Base 100 — lundis {year} à {hour}:00 Paris (fenêtre glissante {window_days}j)",
        xaxis_title="Lundi",
        yaxis_title="Indice (base 100)",
        hovermode="x unified",
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    return fig


# ---------- APP ----------
def main():
    st.set_page_config(page_title="Crypto Volatility x Liquidity", layout="wide")
    st.title("📊 Top Crypto — Volatilité & Liquidité")
    st.caption("Sources : CoinGecko (Top coins) + CryptoCompare (prix, OHLCV, volumes)")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.subheader("Paramètres")

        vs_currency = st.selectbox("Devise", ["usd", "eur"], index=0)
        top_n = st.number_input("Top N", min_value=5, max_value=50, value=10, step=1)
        window_days = st.slider("Fenêtre (jours)", min_value=7, max_value=120, value=30, step=1)
        ref_date = st.date_input("Date de référence", value=utc_now_date())
        show_15h = st.checkbox("Afficher Price (15:00 Paris)", value=True)

        st.divider()
        st.subheader("Graphique historique")
        show_graph = st.checkbox("Afficher le graphique", value=True)
        graph_year = st.number_input(
            "Année", min_value=2020, max_value=utc_now_date().year, value=utc_now_date().year, step=1
        )
        graph_hour = st.slider("Heure (Paris)", min_value=0, max_value=23, value=15, step=1)

        st.divider()
        run = st.button("🚀 Calculer / Rafraîchir", type="primary", use_container_width=True)

    tsym = "USD" if vs_currency.lower() == "usd" else "EUR"

    with st.expander("ℹ️ Comment lire ce dashboard ?"):
        st.markdown(
            """
            **Tableau (date de référence = aujourd'hui ou date choisie)**
            - **Volatility** : écart-type des rendements log journaliers sur la fenêtre. ↑ = ça bouge plus.
            - **Liquidity** : volume moyen journalier (en USD/EUR). ↑ = plus liquide.
            - **Score** = Volatility × Liquidity.
            - **%** : part du score d'une crypto dans le total du Top N à cette date.

            **Graphique historique (tous les lundis d'une année)**
            - **Indice base 100** : 100 au 1er lundi affiché, évolue selon le prix à l'heure Paris choisie.
            - Au survol d'un point : Index + Part Score + Prix + Vol + Liq + Score à cette date précise.
            - Vol/Liq/Score sont recalculés à chaque lundi avec une **fenêtre glissante** (standard finance).
            """
        )

    if not run:
        st.info("👈 Configure les paramètres puis clique sur **Calculer / Rafraîchir**.")
        return

    # ---------- VÉRIF SECRET ----------
    if not st.secrets.get("CRYPTOCOMPARE_API_KEY"):
        st.warning(
            "⚠️ Clé API CryptoCompare absente (fonctionne quand même en mode limité). "
            "Pour + de stabilité : Settings → Secrets → `CRYPTOCOMPARE_API_KEY`"
        )

    # ---------- FETCH TOP COINS ----------
    try:
        top = get_top_coins(vs_currency, int(top_n))
    except Exception as e:
        st.error(f"❌ Erreur CoinGecko : {e}")
        st.stop()

    # ---------- BOUCLE DE CALCUL TABLEAU ----------
    rows = []
    skipped = []
    used_dates = set()
    used_15h_times = []

    progress = st.progress(0.0, text="Récupération des données...")

    for i, (_, r) in enumerate(top.iterrows(), start=1):
        sym = r["symbol"]
        progress.progress(i / len(top), text=f"Traitement de {sym} ({i}/{len(top)})")

        try:
            price_live = get_live_price_cc(sym, tsym)

            to_ts = int(
                datetime(ref_date.year, ref_date.month, ref_date.day, tzinfo=timezone.utc).timestamp()
            )
            dfd = get_histoday_cc(sym, tsym, limit=400, to_ts=to_ts)

            price_close_1d, used_date_close = get_close_on_or_before(dfd, ref_date)
            vol, liq, used_date_metrics = compute_metrics_on_date(dfd, ref_date, int(window_days))

            used_date = used_date_metrics or used_date_close
            if used_date:
                used_dates.add(used_date)

            price_15h = np.nan
            if show_15h:
                price_15h, used_15h_utc = get_price_at_paris_hour_cc(sym, tsym, ref_date, hour=15)
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

    progress.empty()

    if not rows:
        st.error("❌ Aucune donnée exploitable.")
        if skipped:
            st.write("Symboles ignorés :", ", ".join(skipped))
        return

    # ---------- TABLEAU ----------
    out = pd.DataFrame(rows)
    out["Score"] = out["Volatility"] * out["Liquidity"]
    total_score = out["Score"].sum(skipna=True)
    out["%"] = (out["Score"] / total_score * 100) if total_score and total_score > 0 else np.nan
    out = out.sort_values("Score", ascending=False)

    effective_date = max(used_dates) if used_dates else ref_date
    info_parts = [
        f"💱 {tsym}",
        f"📅 Date daily : {effective_date}",
        f"📏 Fenêtre : {window_days} j",
    ]
    if show_15h and used_15h_times:
        info_parts.append(
            f"⏰ 15:00 Paris : {max(used_15h_times).strftime('%Y-%m-%d %H:%M UTC')}"
        )
    st.caption(" | ".join(info_parts))

    st.dataframe(format_table(out, show_15h=show_15h), use_container_width=True, hide_index=True)

    if skipped:
        st.warning(f"⚠️ Symboles ignorés : {', '.join(skipped)}")

    # ---------- GRAPHIQUE ----------
    if show_graph and not out.empty:
        st.divider()
        st.subheader(f"📈 Indice Base 100 — lundis {graph_year} à {graph_hour}:00 Paris")

        # WARNING visible dans l'UI pour l'utilisateur
        st.warning(
            f"⚠️ **Périmètre du graphique** : le panier de cryptos utilisé est le "
            f"**Top {top_n} ACTUEL** de CoinGecko (celui du tableau ci-dessus), "
            f"appliqué à toute l'année {graph_year}. "
            f"Une crypto entrée/sortie du Top {top_n} en cours d'année n'est "
            f"**pas** prise en compte de façon dynamique. "
            f"Explication technique dans les commentaires du code."
        )

        # Liste des symboles du Top N (figé)
        all_symbols = out["Symbol"].dropna().unique().tolist()

        col1, col2 = st.columns([1, 3])
        with col1:
            selected_symbols = st.multiselect(
                "Cryptos à afficher",
                options=all_symbols,
                default=all_symbols[:3] if len(all_symbols) >= 3 else all_symbols,
                help="Sélectionne les cryptos que tu veux voir sur le graphique.",
            )

        if not selected_symbols:
            st.info("👆 Sélectionne au moins une crypto.")
            return

        # NOTE IMPORTANTE : on passe TOUJOURS all_symbols à la fonction de calcul
        # (pas seulement selected_symbols) pour que le calcul du % score soit
        # fait sur le Top N complet. Sinon le % serait biaisé vers 100% si on
        # ne sélectionne qu'une crypto.
        with col2:
            st.caption(
                f"💡 Base 100 au 1er lundi {graph_year} disponible. "
                f"Le **%** au survol = part du score dans le Top {top_n} complet (pas seulement sélection)."
            )

        with st.spinner("Construction de l'historique (peut prendre ~30s au 1er run, puis instant avec cache)..."):
            df_long = build_full_history_for_symbols(
                symbols=tuple(all_symbols),
                tsym=tsym,
                year=int(graph_year),
                hour=int(graph_hour),
                window_days=int(window_days),
            )

        if df_long.empty:
            st.warning("⚠️ Pas de données disponibles pour cette période.")
            return

        # Filtrer pour affichage (mais les % ont été calculés sur TOUT le Top N)
        df_display = df_long[df_long["Symbol"].isin(selected_symbols)].copy()

        # Plotly chart
        fig = build_plotly_chart(df_display, int(graph_year), int(graph_hour), int(window_days))
        st.plotly_chart(fig, use_container_width=True)

        # Données brutes
        with st.expander("📊 Voir les données brutes"):
            display_df = df_display[[
                "Monday", "Symbol", "Price", "Index", "Volatility", "Liquidity", "Score", "Pct"
            ]].copy()
            display_df["Monday"] = display_df["Monday"].dt.strftime("%Y-%m-%d")
            display_df = display_df.round({
                "Price": 2, "Index": 2, "Volatility": 6,
                "Liquidity": 2, "Score": 2, "Pct": 2,
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()