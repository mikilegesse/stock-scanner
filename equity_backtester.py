"""
equity_backtester.py
====================

A lookahead-safe backtesting framework for long-term / position-trading US
equity strategies.

Design goals
------------
1. No lookahead bias. A strategy emits *target weights* using only data
   available up to and including date t. The engine executes them on t+1.
2. Honest costs. Commission + slippage are charged on the dollar value
   actually traded (turnover), not ignored.
3. Real metrics. CAGR, Sharpe, Sortino, max drawdown, Calmar, turnover.
4. Reusable. The engine is strategy-agnostic; strategies are just functions
   that return a (dates x tickers) weight matrix.

Convention for the weight matrix
--------------------------------
- A numeric row  -> rebalance to those target weights on that date.
- An all-NaN row -> HOLD (do nothing; let the existing position drift).
This lets you express buy-and-hold (weights once, NaN forever after),
monthly rebalancing (weights on month-ends, NaN in between), or daily
timing (weights only on days the signal changes) without spurious turnover.

The engine + strategies + metrics depend ONLY on pandas/numpy, so they run
anywhere. Live data loading uses yfinance (network required).

Honest caveats you should keep in mind
--------------------------------------
- Prices are forward-filled; a delisted ticker holds flat forever rather than
  going to zero. For a real universe, handle delistings / use a survivorship-
  bias-free dataset.
- Cash earns 0% here. In risk-off periods (e.g. SMA fully in cash) this
  slightly understates returns vs. holding T-bills. Set rf in metrics() to
  benchmark, or extend the engine to accrue a cash yield.
- auto_adjust=True in the loader approximates total return (splits + dividends
  folded into price). It is not identical to a true total-return index.
- A backtest is a hypothesis test, not a promise. Beware overfitting: the more
  parameters you tune to one history, the less the result means out of sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------
class Backtest:
    """Target-weight portfolio backtester with transaction costs."""

    def __init__(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        commission_bps: float = 2.0,
        slippage_bps: float = 3.0,
        periods_per_year: int = 252,
    ):
        # Forward-fill so a missing print on a single day doesn't force a sale.
        self.prices = prices.sort_index().ffill().astype(float)
        self.initial_capital = float(initial_capital)
        self.cost_rate = (commission_bps + slippage_bps) / 10_000.0
        self.ppy = periods_per_year
        self._equity: pd.Series | None = None
        self._turnover: pd.Series | None = None
        self._weights: pd.DataFrame | None = None

    def run(self, target_weights: pd.DataFrame, lag: int = 1) -> "Backtest":
        """Simulate the strategy.

        target_weights : (dates x tickers) per the convention above.
        lag            : bars between decision and execution (1 = decide on
                         close of t, trade on close of t+1). Keep >= 1.
        """
        prices = self.prices
        cols = list(prices.columns)

        tw = target_weights.reindex(index=prices.index, columns=cols).shift(lag)
        px_vals = prices.values.astype(float)          # (T, N)
        tw_vals = tw.values.astype(float)              # (T, N), NaN = hold
        T, N = px_vals.shape
        hold_row = np.all(np.isnan(tw_vals), axis=1)

        cash = self.initial_capital
        shares = np.zeros(N)
        equity = np.empty(T)
        turnover = np.zeros(T)
        weights_rec = np.zeros((T, N))
        prev_val = self.initial_capital
        cost_rate = self.cost_rate

        for i in range(T):
            px = px_vals[i]
            total = cash + np.nansum(shares * px)      # mark to market

            if hold_row[i]:
                equity[i] = total
                if total > 0:
                    weights_rec[i] = np.nan_to_num(shares * px) / total
                prev_val = total
                continue

            w = np.nan_to_num(tw_vals[i], nan=0.0)
            target_dollar = w * total
            with np.errstate(divide="ignore", invalid="ignore"):
                target_shares = np.where(px > 0, target_dollar / px, 0.0)
            target_shares = np.nan_to_num(target_shares)

            trade_shares = target_shares - shares
            trade_val = np.nansum(np.abs(trade_shares) * px)
            cost = trade_val * cost_rate

            cash -= np.nansum(trade_shares * px)       # buy: cash down; sell: up
            cash -= cost
            shares = target_shares

            total_after = cash + np.nansum(shares * px)
            equity[i] = total_after
            turnover[i] = trade_val / prev_val if prev_val > 0 else 0.0
            if total_after > 0:
                weights_rec[i] = np.nan_to_num(shares * px) / total_after
            prev_val = total_after

        idx = prices.index
        self._equity = pd.Series(equity, index=idx, name="equity")
        self._turnover = pd.Series(turnover, index=idx, name="turnover")
        self._weights = pd.DataFrame(weights_rec, index=idx, columns=cols)
        return self

    # --- accessors -----------------------------------------------------
    @property
    def equity(self) -> pd.Series:
        if self._equity is None:
            raise RuntimeError("Call run() first.")
        return self._equity

    @property
    def weights(self) -> pd.DataFrame:
        if self._weights is None:
            raise RuntimeError("Call run() first.")
        return self._weights

    def drawdown(self) -> pd.Series:
        eq = self.equity
        return eq / eq.cummax() - 1.0

    def metrics(self, rf: float = 0.0) -> dict:
        eq = self.equity
        rets = eq.pct_change(fill_method=None).dropna()
        n = len(eq)
        years = n / self.ppy if self.ppy else np.nan
        total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0 if years > 0 else np.nan
        ann_vol = rets.std(ddof=1) * np.sqrt(self.ppy)
        ann_ret = rets.mean() * self.ppy
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
        downside = rets[rets < 0].std(ddof=1) * np.sqrt(self.ppy)
        sortino = (ann_ret - rf) / downside if downside and downside > 0 else np.nan
        dd = self.drawdown()
        max_dd = dd.min()
        calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
        win_rate = (rets > 0).mean()
        avg_turnover = self._turnover.mean()
        return {
            "Total Return": total_return,
            "CAGR": cagr,
            "Ann Vol": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": max_dd,
            "Calmar": calmar,
            "Daily Win Rate": win_rate,
            "Avg Turnover": avg_turnover,
            "Years": years,
        }


# ----------------------------------------------------------------------
# Strategies  (each returns a dates x tickers weight matrix)
# ----------------------------------------------------------------------
def buy_and_hold(prices: pd.DataFrame, weights=None) -> pd.DataFrame:
    """Equal-weight (or custom) basket, bought once and held (with drift)."""
    cols = prices.columns
    if weights is None:
        w = pd.Series(1.0 / len(cols), index=cols)
    else:
        w = pd.Series(weights).reindex(cols).fillna(0.0)
        s = w.sum()
        w = w / s if s else w
    tw = pd.DataFrame(np.nan, index=prices.index, columns=cols)
    first = prices.dropna(how="all").index[0]
    tw.loc[first] = w.values
    return tw


def sma_crossover(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Trend following: hold a name only while its fast SMA > slow SMA.

    Capital is split equally across whichever names are currently 'in trend';
    fully in cash when none are. Weights are emitted only on days the signal
    set changes, so there is no spurious daily turnover.
    """
    sma_f = prices.rolling(fast).mean()
    sma_s = prices.rolling(slow).mean()
    in_mkt = (sma_f > sma_s).astype(float)
    n_in = in_mkt.sum(axis=1)
    w = in_mkt.div(n_in.replace(0, np.nan), axis=0).fillna(0.0)
    w = w.where(sma_s.notna().any(axis=1))             # mask warmup

    prev = w.shift()
    cell_changed = (w != prev) & ~(w.isna() & prev.isna())
    row_changed = cell_changed.any(axis=1)
    first_valid = w.dropna(how="all").index[0]
    row_changed.loc[first_valid] = True

    tw = pd.DataFrame(np.nan, index=w.index, columns=w.columns)
    tw.loc[row_changed] = w.loc[row_changed].values
    return tw


def cross_sectional_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    top_n: int = 5,
    rebalance: str = "M",
) -> pd.DataFrame:
    """Classic 12-1 momentum: each period, hold the top_n names ranked by
    their past return from (t - lookback) to (t - skip), equal weighted.

    The 'skip' (default ~1 month) avoids the short-term reversal effect, which
    is why this is '12 minus 1' rather than plain trailing 12-month return.
    """
    mom = prices.shift(skip) / prices.shift(lookback) - 1.0
    idx = prices.index
    period = idx.to_period(rebalance)
    last_days = pd.Series(idx, index=idx).groupby(period).max()

    tw = pd.DataFrame(np.nan, index=idx, columns=prices.columns)
    for d in last_days:
        m = mom.loc[d].dropna()
        if m.empty:
            continue
        winners = m.sort_values(ascending=False).head(top_n).index
        if len(winners) == 0:
            continue
        w = pd.Series(0.0, index=prices.columns)
        w[winners] = 1.0 / len(winners)
        tw.loc[d] = w.values
    return tw


# ----------------------------------------------------------------------
# Comparison helper
# ----------------------------------------------------------------------
def compare(
    prices: pd.DataFrame,
    strategies: dict[str, pd.DataFrame],
    initial_capital: float = 100_000.0,
    rf: float = 0.0,
    **cost_kwargs,
):
    """Run several strategies on the same prices; return (metrics_table, equity_curves)."""
    rows, curves = {}, {}
    for name, tw in strategies.items():
        bt = Backtest(prices, initial_capital=initial_capital, **cost_kwargs).run(tw)
        rows[name] = bt.metrics(rf=rf)
        curves[name] = bt.equity
    table = pd.DataFrame(rows).T
    return table, pd.DataFrame(curves)


def format_table(table: pd.DataFrame) -> str:
    pct = ["Total Return", "CAGR", "Ann Vol", "Max Drawdown", "Daily Win Rate", "Avg Turnover"]
    ratio = ["Sharpe", "Sortino", "Calmar", "Years"]
    out = table.copy()
    for c in pct:
        if c in out:
            out[c] = (out[c] * 100).map(lambda x: f"{x:6.1f}%")
    for c in ratio:
        if c in out:
            out[c] = out[c].map(lambda x: f"{x:6.2f}")
    return out.to_string()


# ----------------------------------------------------------------------
# Screen: rank a whole universe by who is rising
# ----------------------------------------------------------------------
def screen_universe(
    prices: pd.DataFrame,
    sort_by: str = "Ret_3M",
    windows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """One row per ticker, ranked best-to-worst by trailing return.

    This is the 'show me every stock and who's increasing' view. Feed it any
    universe (15 names or the whole S&P 500) and it returns a table of trailing
    returns over several windows. A positive number means the price ROSE over
    that window; negative means it FELL. Sorted so the fastest risers are on top.

    sort_by : which column to rank on (e.g. 'Ret_1M', 'Ret_3M', 'Mom_12_1').
    windows : {label: trading_days}. Defaults to 1M/3M/6M/12M.

    Note: this is a momentum snapshot, NOT a buy list. A name 'increasing' fast
    can be overextended; a faller can be a value setup. It's a starting point
    for the due-diligence step, not the end of it.
    """
    if windows is None:
        windows = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
    px = prices.ffill()
    latest = px.iloc[-1]

    out = pd.DataFrame(index=px.columns)
    out["Price"] = latest.round(2)
    for label, n in windows.items():
        if len(px) > n:
            out[f"Ret_{label}"] = (px.iloc[-1] / px.iloc[-1 - n] - 1).round(4)
    if len(px) > 252:  # 12-1 momentum: last 12 months, skipping the last ~month
        out["Mom_12_1"] = (px.iloc[-1 - 21] / px.iloc[-1 - 252] - 1).round(4)
    if len(px) > 200:  # simple trend flag
        sma200 = px.rolling(200).mean().iloc[-1]
        out["Above_200dma"] = latest > sma200

    sort_col = sort_by if sort_by in out.columns else out.columns[1]
    return out.sort_values(sort_col, ascending=False)


def breakout_scan(
    prices: pd.DataFrame,
    near_high_pct: float = 0.03,
    min_mom_12_1: float = 0.20,
    lookback: int = 252,
) -> pd.DataFrame:
    """Flag stocks breaking out: at/near a 52-week high WITH strong momentum.

    This is the 'catch a trend early' scan you'd run daily and alert on. Read
    the columns honestly:
      - Pct_From_High:  0 means sitting at a new 52-week high; -0.05 means 5%
                        below it. Near 0 = breaking out.
      - Mom_12_1:       12-1 month momentum. Strong + breaking out = the setup
                        that caught NVDA/MU/SNDK *mid-trend*.

    What this CANNOT do: tell you a stock will 40x, or get you in at the bottom.
    A new high means the move already started. Most breakouts also FAIL and
    reverse -- this is a starting filter for due diligence, never a buy trigger.
    Pair it with position sizing and a stop, or the failed ones will hurt.

    near_high_pct  : how close to the 52w high counts (0.03 = within 3%).
    min_mom_12_1   : minimum 12-1 momentum to qualify (0.20 = up 20%+).
    """
    px = prices.ffill()
    latest = px.iloc[-1]
    window = px.iloc[-lookback:] if len(px) >= lookback else px
    high_52w = window.max()
    pct_from_high = latest / high_52w - 1.0

    out = pd.DataFrame(index=px.columns)
    out["Price"] = latest.round(2)
    out["52w_High"] = high_52w.round(2)
    out["Pct_From_High"] = pct_from_high.round(4)
    if len(px) > 21:
        out["Ret_1M"] = (px.iloc[-1] / px.iloc[-1 - 21] - 1).round(4)
    if len(px) > 252:
        out["Mom_12_1"] = (px.iloc[-1 - 21] / px.iloc[-1 - 252] - 1).round(4)
    else:
        out["Mom_12_1"] = np.nan

    hits = out[
        (out["Pct_From_High"] >= -near_high_pct)
        & (out["Mom_12_1"].fillna(-1.0) >= min_mom_12_1)
    ]
    return hits.sort_values("Mom_12_1", ascending=False)


# ----------------------------------------------------------------------
# Live data (network required -> run this on your own machine)
# ----------------------------------------------------------------------
def load_prices(tickers, start="2010-01-01", end=None) -> pd.DataFrame:
    """Download adjusted daily closes via yfinance. Returns dates x tickers."""
    import yfinance as yf  # lazy import so the rest of the module is portable

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = [tickers] if isinstance(tickers, str) else list(tickers)
    return prices.dropna(how="all")


def sp500_tickers() -> list[str]:
    """Current S&P 500 constituents, scraped from Wikipedia.

    NOTE: Wikipedia rejects requests without a real browser User-Agent (this is
    why a bare pd.read_html(url) works locally but 403s on a cloud host). We
    fetch with requests + a UA, then parse. For reliability prefer fmp_sp500()
    if you have an FMP key -- it's a structured API, not a scrape.
    """
    import io
    import requests

    html = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (compatible; stock-scanner/1.0)"},
        timeout=30,
    ).text
    tables = pd.read_html(io.StringIO(html))
    # yfinance uses '-' for share classes (BRK-B); Wikipedia uses '.' (BRK.B)
    return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


# ----------------------------------------------------------------------
# FMP-backed universes  (reliable; needs your FMP api_key)
# ----------------------------------------------------------------------
def fmp_sp500(api_key: str) -> list[str]:
    """S&P 500 constituents straight from FMP's API (one call, no scraping)."""
    import requests

    url = "https://financialmodelingprep.com/api/v3/sp500_constituent"
    data = requests.get(url, params={"apikey": api_key}, timeout=30).json()
    if not isinstance(data, list):
        raise RuntimeError(f"FMP sp500 returned: {str(data)[:200]}")
    return [d["symbol"].replace(".", "-") for d in data if d.get("symbol")]


def fmp_universe(api_key: str, min_market_cap: float = 2_000_000_000,
                 exchanges: str = "NASDAQ,NYSE,AMEX", limit: int = 3000) -> list[str]:
    """A broad US equity universe from FMP's screener, filtered to stay sane.

    This is how you scan BEYOND the S&P 500 -- into the mid/small caps where the
    big breakouts (SanDisk, etc.) usually start before they're index-large.

    min_market_cap : dollar floor. $2B (default) is a manageable few hundred-ish
                     names; drop it to reach smaller caps, but every name you add
                     is another price download -> slower, and heavy on free infra.
    """
    import requests

    url = "https://financialmodelingprep.com/api/v3/stock-screener"
    params = {
        "marketCapMoreThan": int(min_market_cap),
        "exchange": exchanges,
        "isActivelyTrading": "true",
        "limit": int(limit),
        "apikey": api_key,
    }
    data = requests.get(url, params=params, timeout=60).json()
    if not isinstance(data, list):
        raise RuntimeError(f"FMP screener returned: {str(data)[:200]}")
    return [d["symbol"].replace(".", "-") for d in data if d.get("symbol")]


def plot_equity(curves: pd.DataFrame, log: bool = True, title: str = "Equity curves"):
    import matplotlib.pyplot as plt

    ax = curves.plot(figsize=(11, 6), logy=log, title=title)
    ax.set_ylabel("Portfolio value ($)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


# ----------------------------------------------------------------------
# Offline self-test (synthetic data -> proves the engine math, no network)
# ----------------------------------------------------------------------
def _synthetic_prices(n_tickers=8, start="2010-01-01", end="2024-12-31", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    # Give each name its own drift/vol so momentum has signal to find.
    mu = rng.uniform(0.02, 0.18, n_tickers) / 252
    sig = rng.uniform(0.15, 0.40, n_tickers) / np.sqrt(252)
    shocks = rng.standard_normal((T, n_tickers)) * sig + mu
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    cols = [f"SIM{i+1}" for i in range(n_tickers)]
    return pd.DataFrame(prices, index=dates, columns=cols)


if __name__ == "__main__":
    print("Running OFFLINE self-test on synthetic data "
          "(swap in load_prices([...]) for real tickers).\n")
    px = _synthetic_prices()

    strategies = {
        "Buy & Hold (EW)": buy_and_hold(px),
        "SMA 50/200 trend": sma_crossover(px, 50, 200),
        "Momentum top-3 (12-1)": cross_sectional_momentum(px, top_n=3),
    }
    table, curves = compare(px, strategies, commission_bps=2, slippage_bps=3)

    print(format_table(table))
    print("\nFinal portfolio values:")
    print((curves.iloc[-1].map(lambda x: f"  ${x:,.0f}")).to_string())

    print("\n--- Universe screen (who's increasing) ---")
    print(screen_universe(px, sort_by="Ret_3M").to_string())

    print("\n--- Breakout scan (near 52w high + strong momentum) ---")
    bo = breakout_scan(px, near_high_pct=0.10, min_mom_12_1=0.10)
    print(bo.to_string() if not bo.empty else "  (no breakouts in this synthetic set)")

    print("\nEngine self-test complete. Math checks: equity starts at "
          f"${100_000:,.0f}, costs applied on turnover, signals lagged 1 bar.")
