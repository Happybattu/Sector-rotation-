"""
Sector Rotation Dashboard - NSE
=================================
Pehchano ki abhi konsa sector "chal raha" hai aur institutional money kaha
rotate ho raha hai - Relative Strength + RRG (Relative Rotation Graph) style
methodology ke through.

Run:
    pip install streamlit yfinance pandas numpy plotly --break-system-packages
    streamlit run sector_rotation_dashboard.py

Methodology:
    1. RS-Ratio: Har sector index ki price ko Nifty50 se divide karke
       relative strength nikalte hai, phir usko rolling z-score se
       normalize karte hai (mean=100). >100 = outperforming.
    2. RS-Momentum: RS-Ratio ka short-term rate of change, normalize
       karke (mean=100). >100 = momentum accelerating (badh raha hai).
    3. In dono ko cross karke 4 quadrant classification milti hai
       (jaise RRG charts pro terminals pe dikhate hain):
         - LEADING    (RS>100, Mom>100)  -> sector strong hai aur
                                              momentum bhi badh raha
         - WEAKENING  (RS>100, Mom<100)  -> abhi strong hai par
                                              momentum thak raha - profit booking zone
         - LAGGING    (RS<100, Mom<100)  -> weak sector, avoid
         - IMPROVING  (RS<100, Mom>100)  -> weak tha par turn kar raha -
                                              early entry zone
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Sector Rotation Dashboard - NSE", layout="wide")

# ---------------------------------------------------------------------------
# Sector universe - NSE sectoral indices available on Yahoo Finance
# ---------------------------------------------------------------------------
BENCHMARK = "^NSEI"  # Nifty 50

SECTOR_TICKERS = {
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty Auto": "^CNXAUTO",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Realty": "^CNXREALTY",
    "Nifty Energy": "^CNXENERGY",
    "Nifty PSU Bank": "^CNXPSUBANK",
    "Nifty Media": "^CNXMEDIA",
    "Nifty Infra": "^CNXINFRA",
    "Nifty Commodities": "^CNXCOMMODITIES",
    "Nifty PSE": "^CNXPSE",
    "Nifty Service Sector": "^CNXSERVICE",
    "Nifty MNC": "^CNXMNC",
}

RS_WINDOW = 20      # rolling window for RS-Ratio z-score normalization
MOM_WINDOW = 5       # window for RS-Momentum rate-of-change


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(tickers: dict, benchmark: str, period="1y"):
    """Download adjusted close prices for benchmark + all sector tickers.
    Skips any ticker that fails to download instead of crashing the app."""
    all_tickers = {**{"Nifty 50": benchmark}, **tickers}
    data = {}
    failed = []
    for name, tk in all_tickers.items():
        try:
            hist = yf.download(tk, period=period, interval="1d",
                                progress=False, auto_adjust=True)
            if hist is None or hist.empty:
                failed.append(name)
                continue

            # yfinance sometimes returns MultiIndex columns even for a
            # single ticker (e.g. ('Close', 'TICKER')). Normalize to a
            # plain 1-D Series in every case.
            if isinstance(hist.columns, pd.MultiIndex):
                if "Close" not in hist.columns.get_level_values(0):
                    failed.append(name)
                    continue
                close = hist["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            else:
                if "Close" not in hist.columns:
                    failed.append(name)
                    continue
                close = hist["Close"]

            close = pd.Series(close).astype(float).dropna()
            if close.empty:
                failed.append(name)
                continue
            data[name] = close
        except Exception:
            failed.append(name)

    if not data:
        return None, failed

    df = pd.concat(data, axis=1)
    df.columns = list(data.keys())
    df = df.ffill().dropna(how="all")
    return df, failed


def compute_returns(df: pd.DataFrame):
    """1D / 1W / 1M / 3M % returns for every column, latest value."""
    out = {}
    periods = {"1D": 1, "1W": 5, "1M": 21, "3M": 63}
    for name in df.columns:
        row = {}
        series = df[name].dropna()
        for label, n in periods.items():
            if len(series) > n:
                row[label] = (series.iloc[-1] / series.iloc[-1 - n] - 1) * 100
            else:
                row[label] = np.nan
        out[name] = row
    return pd.DataFrame(out).T


def compute_rrg(df: pd.DataFrame, benchmark_col="Nifty 50",
                 rs_window=RS_WINDOW, mom_window=MOM_WINDOW):
    """Compute RS-Ratio and RS-Momentum (RRG style) for each sector."""
    rs_ratio = pd.DataFrame(index=df.index)
    rs_mom = pd.DataFrame(index=df.index)

    bench = df[benchmark_col]
    for col in df.columns:
        if col == benchmark_col:
            continue
        relative = df[col] / bench
        roll_mean = relative.rolling(rs_window).mean()
        roll_std = relative.rolling(rs_window).std()
        z = (relative - roll_mean) / roll_std.replace(0, np.nan)
        rs_ratio[col] = 100 + z * 3  # scaled z-score around 100

        mom = rs_ratio[col].diff(mom_window)
        mom_mean = mom.rolling(rs_window).mean()
        mom_std = mom.rolling(rs_window).std()
        mz = (mom - mom_mean) / mom_std.replace(0, np.nan)
        rs_mom[col] = 100 + mz * 3

    return rs_ratio.dropna(how="all"), rs_mom.dropna(how="all")


def classify_quadrant(rs, mom):
    if pd.isna(rs) or pd.isna(mom):
        return "N/A"
    if rs >= 100 and mom >= 100:
        return "Leading"
    if rs >= 100 and mom < 100:
        return "Weakening"
    if rs < 100 and mom < 100:
        return "Lagging"
    return "Improving"


QUADRANT_COLOR = {
    "Leading": "#2ecc71",
    "Weakening": "#f1c40f",
    "Lagging": "#e74c3c",
    "Improving": "#3498db",
    "N/A": "#95a5a6",
}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📊 Sector Rotation Dashboard - NSE")
st.caption("RS-Ratio + RS-Momentum (RRG-style) - konsa sector chal raha hai, "
           "yeh trend se pata chalta hai, ek din ke move se nahi.")

with st.sidebar:
    st.header("Settings")
    period = st.selectbox("History window", ["6mo", "1y", "2y"], index=1)
    rs_window = st.slider("RS-Ratio smoothing (days)", 10, 40, RS_WINDOW)
    mom_window = st.slider("RS-Momentum window (days)", 3, 15, MOM_WINDOW)
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()

with st.spinner("Fetching NSE sector index data..."):
    prices, failed = fetch_prices(SECTOR_TICKERS, BENCHMARK, period=period)

if prices is None:
    st.error("Data fetch fail ho gaya. Internet connection ya yfinance check karo.")
    st.stop()

if failed:
    st.warning(f"Yeh tickers fetch nahi hue, skip kar diya: {', '.join(failed)}")

st.caption(f"Last data point: {prices.index[-1].strftime('%d %b %Y')} | "
           f"{len(prices.columns)-1} sectors loaded")

# --- Section 1: Returns heatmap / ranking table -----------------------------
st.subheader("1️⃣ Sector Performance Ranking")
returns_df = compute_returns(prices)
returns_df = returns_df.drop(index="Nifty 50", errors="ignore")
returns_df = returns_df.sort_values("1M", ascending=False)

styled = returns_df.style.background_gradient(
    cmap="RdYlGn", axis=0, vmin=-15, vmax=15
).format("{:.2f}%")
st.dataframe(styled, use_container_width=True)

# --- Section 2: RRG Quadrant chart ------------------------------------------
st.subheader("2️⃣ Sector Rotation Quadrant (RRG-style)")
st.caption("Leading = chal raha hai | Weakening = thak raha, profit booking zone | "
           "Improving = turn kar raha, early entry | Lagging = avoid")

rs_ratio_df, rs_mom_df = compute_rrg(prices, "Nifty 50", rs_window, mom_window)

if not rs_ratio_df.empty:
    latest_rs = rs_ratio_df.iloc[-1]
    latest_mom = rs_mom_df.iloc[-1]

    tail_len = 5  # show short trailing tail for each sector to see direction
    fig = go.Figure()

    for sector in rs_ratio_df.columns:
        rs_tail = rs_ratio_df[sector].dropna().iloc[-tail_len:]
        mom_tail = rs_mom_df[sector].dropna().iloc[-tail_len:]
        n = min(len(rs_tail), len(mom_tail))
        if n == 0:
            continue
        rs_tail, mom_tail = rs_tail.iloc[-n:], mom_tail.iloc[-n:]
        quad = classify_quadrant(rs_tail.iloc[-1], mom_tail.iloc[-1])
        color = QUADRANT_COLOR[quad]

        fig.add_trace(go.Scatter(
            x=rs_tail, y=mom_tail, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=[6]*(n-1) + [12], color=color),
            name=f"{sector} ({quad})",
            hovertemplate=f"{sector}<br>RS: %{{x:.1f}}<br>Mom: %{{y:.1f}}<extra></extra>"
        ))
        fig.add_annotation(x=rs_tail.iloc[-1], y=mom_tail.iloc[-1],
                            text=sector.replace("Nifty ", ""), showarrow=False,
                            yshift=14, font=dict(size=10, color=color))

    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.add_vline(x=100, line_dash="dot", line_color="gray")
    fig.update_layout(
        xaxis_title="RS-Ratio (Strength →)",
        yaxis_title="RS-Momentum (Momentum →)",
        height=650,
        showlegend=True,
        legend=dict(font=dict(size=9)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Section 3: Quadrant summary table ---
    st.subheader("3️⃣ Current Quadrant - Kaun kis zone mein hai")
    quad_rows = []
    for sector in rs_ratio_df.columns:
        rs_val = latest_rs.get(sector, np.nan)
        mom_val = latest_mom.get(sector, np.nan)
        quad_rows.append({
            "Sector": sector,
            "RS-Ratio": round(rs_val, 1) if pd.notna(rs_val) else None,
            "RS-Momentum": round(mom_val, 1) if pd.notna(mom_val) else None,
            "Zone": classify_quadrant(rs_val, mom_val),
        })
    quad_df = pd.DataFrame(quad_rows).sort_values(
        ["Zone", "RS-Ratio"], ascending=[True, False]
    )

    def color_zone(val):
        c = QUADRANT_COLOR.get(val, "#fff")
        return f"background-color: {c}33; color: {c}; font-weight: bold"

    st.dataframe(
        quad_df.style.applymap(color_zone, subset=["Zone"]),
        use_container_width=True, hide_index=True
    )

    leading = quad_df[quad_df["Zone"] == "Leading"]["Sector"].tolist()
    improving = quad_df[quad_df["Zone"] == "Improving"]["Sector"].tolist()
    if leading:
        st.success(f"🟢 **Abhi chal raha hai (Leading):** {', '.join(leading)}")
    if improving:
        st.info(f"🔵 **Turn kar raha hai (Improving - early entry candidates):** {', '.join(improving)}")
else:
    st.warning("RRG calculate karne ke liye enough data points nahi hai. History window badhao.")

# --- Section 4: RS trend lines for top sectors ------------------------------
st.subheader("4️⃣ RS-Ratio Trend - Top 5 Sectors")
top5 = returns_df.head(5).index.tolist()
if top5:
    fig2 = go.Figure()
    for sector in top5:
        if sector in rs_ratio_df.columns:
            fig2.add_trace(go.Scatter(
                x=rs_ratio_df.index[-90:], y=rs_ratio_df[sector].iloc[-90:],
                mode="lines", name=sector.replace("Nifty ", "")
            ))
    fig2.add_hline(y=100, line_dash="dot", line_color="gray")
    fig2.update_layout(height=400, yaxis_title="RS-Ratio (100 = neutral)")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption(
    "⚠️ Yeh tool sirf price-based relative strength dikhata hai. FII/DII "
    "cash + F&O flow, sector earnings, aur macro triggers (rate cuts, crude, "
    "budget) ke saath cross-check karke hi decision lo. Investment advice nahi hai."
)
