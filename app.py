"""
app.py - point-and-click momentum dashboard.

Run:   pip install streamlit
       streamlit run app.py

Click a row in the screen to load that stock's price chart and an AI news
brief. Put your API keys in the sidebar, or as env vars / Streamlit secrets
(NEWS_API_KEY, OPENAI_API_KEY).

This is a research aid: the screen ranks what has ALREADY moved, and the news
brief digests headlines. Neither is a buy signal -- verify the primary sources,
size positions, use a stop.
"""

import os

import streamlit as st

from equity_backtester import (
    load_prices, screen_universe, breakout_scan,
    sp500_tickers, sp400_tickers, sp600_tickers,
)
from pipeline import news_brief


def _secret(key, default=""):
    """Resolve a key from Streamlit secrets first, then env vars."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

st.set_page_config(page_title="Momentum screener", layout="wide")


# ---------------- cached helpers (avoid re-downloading / re-billing) ----------
@st.cache_data(ttl=3600, show_spinner="Loading prices...")
def get_prices(tickers, start):
    return load_prices(list(tickers), start=start)


@st.cache_data(ttl=86400, show_spinner="Fetching index constituents...")
def get_index(which):
    if which == "sp1500":
        names = set(sp500_tickers()) | set(sp400_tickers()) | set(sp600_tickers())
        return tuple(sorted(names))
    return tuple({"sp500": sp500_tickers, "sp400": sp400_tickers,
                  "sp600": sp600_tickers}[which]())


@st.cache_data(ttl=900, show_spinner="Scanning the news...")
def get_brief(ticker, provider, model, news_key, openai_key):
    return news_brief(ticker, provider=provider, news_api_key=news_key or None,
                      model=model, openai_api_key=openai_key or None)


# ---------------- sidebar ----------------
st.sidebar.header("Universe")
mode = st.sidebar.radio(
    "Source",
    ["Custom list", "S&P 500 (large cap)", "S&P 400 (mid cap)",
     "S&P 600 (small cap)", "All (S&P 1500)"],
    index=0,
)
if mode == "Custom list":
    txt = st.sidebar.text_area(
        "Tickers", "NVDA MU SNDK AVGO SMCI MRVL AMD TSM ASML ANET WDC STX")
    tickers = tuple(t.strip().upper() for t in txt.replace(",", " ").split() if t.strip())
else:
    which = {"S&P 500 (large cap)": "sp500", "S&P 400 (mid cap)": "sp400",
             "S&P 600 (small cap)": "sp600", "All (S&P 1500)": "sp1500"}[mode]
    try:
        tickers = get_index(which)
        if which == "sp1500":
            st.sidebar.caption(f"{len(tickers)} names (large+mid+small). First load "
                               "takes a few minutes on the free tier, then caches.")
        else:
            st.sidebar.caption(f"{len(tickers)} names. First price load is slow; it caches after.")
    except Exception as e:
        st.sidebar.error(f"Couldn't load the index list: {e}. Use Custom list meanwhile.")
        tickers = ()
start = st.sidebar.text_input("History start", "2024-01-01")

st.sidebar.header("View")
view = st.sidebar.radio("Show", ["Full screen (who's rising)", "Breakouts only"])
near_high = st.sidebar.slider("Within % of 52w high (breakouts)", 0.0, 0.25, 0.03, 0.01)
min_mom = st.sidebar.slider("Min 12-1 momentum (breakouts)", 0.0, 1.0, 0.30, 0.05)

st.sidebar.header("News brief")
provider = st.sidebar.selectbox("Provider", ["fmp", "finnhub", "alphavantage", "tiingo"])
st.sidebar.caption("FMP news works on a paid plan. finnhub & alphavantage are "
                   "free alternatives if you ever need one.")
model = st.sidebar.text_input("LLM model", "gpt-4o-mini")
# Default the news key to a provider-specific secret (e.g. FINNHUB_API_KEY) so
# it persists; fall back to NEWS_API_KEY. You can always paste a key here too.
_default_news_key = _secret(f"{provider.upper()}_API_KEY") or _secret("NEWS_API_KEY")
news_key = st.sidebar.text_input("News API key", _default_news_key, type="password")
openai_key = st.sidebar.text_input("OpenAI key", _secret("OPENAI_API_KEY"), type="password")


# ---------------- main ----------------
st.title("Momentum screen + AI news brief")
st.caption("Click a row to load its chart and news brief. Trailing returns: "
           "positive = rising. This is a research tool, not a buy signal.")

if not tickers:
    st.warning("Add some tickers in the sidebar.")
    st.stop()

try:
    prices = get_prices(tickers, start)
except Exception as e:
    st.error(f"Couldn't load prices: {e}")
    st.stop()

if view.startswith("Breakouts"):
    table = breakout_scan(prices, near_high_pct=near_high, min_mom_12_1=min_mom)
else:
    table = screen_universe(prices, sort_by="Ret_3M")

if table.empty:
    st.info("No names matched the current filters. Loosen the sliders.")
    st.stop()

display = table.reset_index().rename(columns={"index": "Ticker"})

event = st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

try:
    selected_rows = list(event.selection.rows)
except Exception:
    selected_rows = []

if not selected_rows:
    st.info("Click a row above to load that stock's price chart and AI news brief.")
    st.stop()

ticker = display.iloc[selected_rows[0]]["Ticker"]
left, right = st.columns(2)

with left:
    st.subheader(f"{ticker} - price")
    series = prices[ticker].dropna()
    st.line_chart(series)
    if len(series) > 252:
        st.metric("12-month change", f"{series.iloc[-1] / series.iloc[-253] - 1:.1%}")

with right:
    st.subheader(f"{ticker} - AI news brief")
    if not (news_key and openai_key):
        st.warning("Enter your News API key and OpenAI key in the sidebar to generate briefs.")
    else:
        try:
            st.markdown(get_brief(ticker, provider, model, news_key, openai_key))
        except Exception as e:
            st.error(f"News brief failed: {e}")
