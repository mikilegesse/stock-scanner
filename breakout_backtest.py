"""
breakout_backtest.py - prove (or disprove) the breakout edge with evidence.

Backtests a Donchian-style breakout system so you can see its REAL trade win
rate, expectancy, and drawdowns before trusting the live alerts.

Strategy:
  ENTRY: price makes a new `entry_n`-day high (a new 52-week high by default).
         This is the same event the live breakout_scan flags.
  EXIT:  price makes a new `exit_n`-day low (a trailing channel stop). This is
         the risk management that 'cuts the losers' -- without it, a breakout
         backtest is meaningless.
Capital is split equally across whatever names are currently in an uptrend.

The point of the trade-level stats: breakout systems typically have a LOW win
rate (most breakouts fail and you stop out for a small loss) but positive
expectancy, because the few winners run very far. If the numbers don't show
that shape on your universe, the edge isn't there and you shouldn't trade it.

Run on real data:
    from equity_backtester import load_prices
    px = load_prices(['NVDA','MU','SNDK',...], start='2015-01-01')
    main(px)
"""

import numpy as np
import pandas as pd

from equity_backtester import Backtest, buy_and_hold, format_table


def donchian_breakout(prices, entry_n=252, exit_n=100):
    """Return (target_weights, position_matrix).

    position_matrix is 0/1 per ticker (in an uptrend or not); target_weights is
    the equal-weight allocation the backtester consumes, emitted only on days
    the holding set changes.
    """
    upper = prices.rolling(entry_n).max().shift(1)   # prior N-day high
    lower = prices.rolling(exit_n).min().shift(1)     # prior M-day low
    up = prices > upper
    dn = prices < lower

    pos = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    pos[up] = 1.0
    pos[dn] = 0.0
    pos = pos.ffill().fillna(0.0)

    n = pos.sum(axis=1)
    w = pos.div(n.replace(0, np.nan), axis=0).fillna(0.0)

    prev = w.shift()
    changed = ((w != prev) & ~(w.isna() & prev.isna())).any(axis=1)
    changed.loc[w.dropna(how="all").index[0]] = True
    tw = pd.DataFrame(np.nan, index=w.index, columns=w.columns)
    tw.loc[changed] = w.loc[changed].values
    return tw, pos


def trade_stats(prices, pos):
    """Per-trade win/loss analysis from the 0/1 position matrix."""
    trades = []
    for tk in prices.columns:
        p = pos[tk].values
        px = prices[tk].values
        in_pos, entry_px, entry_i = False, None, None
        for i in range(len(p)):
            if not in_pos and p[i] == 1.0:
                in_pos, entry_px, entry_i = True, px[i], i
            elif in_pos and p[i] == 0.0:
                trades.append({"ticker": tk, "ret": px[i] / entry_px - 1.0,
                               "days": i - entry_i, "open": False})
                in_pos = False
        if in_pos:  # still open at end -> mark to last price
            trades.append({"ticker": tk, "ret": px[-1] / entry_px - 1.0,
                           "days": len(p) - 1 - entry_i, "open": True})

    td = pd.DataFrame(trades)
    if td.empty:
        return td, {}

    wins = td.loc[td["ret"] > 0, "ret"]
    losses = td.loc[td["ret"] <= 0, "ret"]
    summary = {
        "Trades": len(td),
        "Win rate": (td["ret"] > 0).mean(),
        "Avg win": wins.mean() if len(wins) else 0.0,
        "Avg loss": losses.mean() if len(losses) else 0.0,
        "Profit factor": (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
        "Expectancy/trade": td["ret"].mean(),
        "Avg hold (days)": td["days"].mean(),
        "Best trade": td["ret"].max(),
        "Worst trade": td["ret"].min(),
    }
    return td, summary


def _print_trade_stats(ts):
    pct_keys = {"Win rate", "Avg win", "Avg loss", "Expectancy/trade", "Best trade", "Worst trade"}
    for k, v in ts.items():
        if k in pct_keys:
            print(f"  {k:18s}: {v:8.2%}")
        elif isinstance(v, float):
            print(f"  {k:18s}: {v:8.2f}")
        else:
            print(f"  {k:18s}: {v:>8}")


def main(prices, entry_n=252, exit_n=100, initial_capital=100_000,
         save_chart="breakout_backtest.png"):
    tw, pos = donchian_breakout(prices, entry_n=entry_n, exit_n=exit_n)
    costs = dict(commission_bps=2, slippage_bps=3)
    bt = Backtest(prices, initial_capital=initial_capital, **costs).run(tw)
    bh = Backtest(prices, initial_capital=initial_capital, **costs).run(buy_and_hold(prices))

    comp = pd.DataFrame({"Breakout": bt.metrics(), "Buy & Hold": bh.metrics()}).T
    print("=== Breakout vs Buy & Hold (portfolio level) ===")
    print(format_table(comp))

    td, ts = trade_stats(prices, pos)
    print("\n=== Trade-level reality check ===")
    _print_trade_stats(ts)
    if not td.empty:
        win = (td["ret"] > 0).mean()
        print(f"\n  Read this: ~{win:.0%} of breakouts won. The system makes money "
              f"\n  only if the winners (avg {ts['Avg win']:.0%}) outrun the many "
              f"\n  losers (avg {ts['Avg loss']:.0%}). That asymmetry IS the edge.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dd = bt.drawdown()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                       height_ratios=[2, 1], sharex=True)
        ax1.plot(bt.equity, label="Breakout", lw=1.4)
        ax1.plot(bh.equity, label="Buy & Hold", lw=1.2, alpha=0.7)
        ax1.set_yscale("log")
        ax1.set_title("Equity curve (log scale)")
        ax1.legend(); ax1.grid(alpha=0.3)
        ax2.fill_between(dd.index, dd.values * 100, 0, color="crimson", alpha=0.4)
        ax2.set_title("Breakout drawdown (%)")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_chart, dpi=120)
        print(f"\nSaved chart -> {save_chart}")
    except Exception as e:
        print("chart skipped:", e)

    return bt, bh, td, ts


if __name__ == "__main__":
    # Offline demo on synthetic trending data (no network needed).
    # NOTE: smooth synthetic trends FLATTER the strategy -- real markets whipsaw
    # more, so expect a lower live win rate. Swap in load_prices() for the truth.
    from equity_backtester import _synthetic_prices
    px = _synthetic_prices(n_tickers=12, end="2024-12-31", seed=7)
    main(px)
