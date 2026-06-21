"""
pipeline.py - live deployment layer for the equity toolkit.

What it does:
  fetch_news(ticker)   -> recent headlines from your data vendor (normalized)
  news_brief(ticker)   -> an LLM-written, skeptical due-diligence digest
  send_alert(message)  -> email or Telegram notification
  run_daily_scan(univ) -> the job GitHub Actions runs every day

Secrets come from environment variables (set them as GitHub Actions secrets,
never hardcode). Heavy imports (requests, openai, smtplib) are lazy so this
module imports cleanly even without those packages installed.

Why it's built this way (the whole point is INFORMED decisions):
  - news_brief only summarizes the articles it is handed, and is told to ground
    strictly in them. LLMs can still misread or overstate -- treat the brief as
    a fast first pass, then click through to the primary sources (the actual
    articles, the 10-Q / 8-K) before you act.
  - a breakout alert is a candidate to RESEARCH, never a buy trigger.

Env vars used:
  NEWS_API_KEY     - your data/news vendor token (falls back to DATA_API_KEY)
  OPENAI_API_KEY   - your GPT key for news_brief
  SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / ALERT_TO   - email alerts
  TELEGRAM_TOKEN / TELEGRAM_CHAT_ID                          - telegram alerts
  ALERT_CHANNEL    - "email" (default) or "telegram"
  DRY_RUN          - "1" to print instead of sending (for local testing)
"""

import os
import datetime as dt


# ----------------------------------------------------------------------
# 1. News fetch  (normalized across vendors)
# ----------------------------------------------------------------------
def fetch_news(ticker, provider="fmp", api_key=None, limit=10):
    """Return recent articles as [{title, summary, url, date, source}, ...].

    provider: 'fmp' | 'tiingo' | 'finnhub' | 'alphavantage'
    Endpoint params occasionally change -- check the vendor's current docs if a
    call returns nothing. All keys are read from env if not passed explicitly.
    """
    import requests

    def _ensure_list(data, provider):
        """Return data if it's a list of articles; else raise a readable error.

        Vendors signal problems (paid-only endpoint, bad key, rate limit) by
        returning a dict/string instead of a list -- which would otherwise blow
        up as a cryptic slice error downstream.
        """
        if isinstance(data, list):
            return data
        msg = data
        if isinstance(data, dict):
            msg = (data.get("Error Message") or data.get("Information")
                   or data.get("error") or data.get("message") or data)
        raise RuntimeError(
            f"{provider} returned no article list. API said: {str(msg)[:300]} "
            f"| Note: FMP news requires a PAID plan. For free news switch the "
            f"provider to 'finnhub' or 'alphavantage'."
        )

    api_key = api_key or os.environ.get("NEWS_API_KEY") or os.environ.get("DATA_API_KEY")
    if not api_key:
        raise RuntimeError("No news API key. Set NEWS_API_KEY (or pass api_key).")

    ticker = ticker.upper()
    out = []

    if provider == "fmp":
        url = "https://financialmodelingprep.com/api/v3/stock_news"
        params = {"tickers": ticker, "limit": limit, "apikey": api_key}
        data = requests.get(url, params=params, timeout=30).json()
        for a in _ensure_list(data, "fmp")[:limit]:
            out.append({
                "title": a.get("title", ""),
                "summary": a.get("text", "")[:600],
                "url": a.get("url", ""),
                "date": a.get("publishedDate", ""),
                "source": a.get("site", ""),
            })

    elif provider == "tiingo":
        url = "https://api.tiingo.com/tiingo/news"
        params = {"tickers": ticker.lower(), "limit": limit, "token": api_key}
        data = requests.get(url, params=params, timeout=30).json()
        for a in _ensure_list(data, "tiingo")[:limit]:
            out.append({
                "title": a.get("title", ""),
                "summary": a.get("description", "")[:600],
                "url": a.get("url", ""),
                "date": a.get("publishedDate", ""),
                "source": a.get("source", ""),
            })

    elif provider == "finnhub":
        today = dt.date.today()
        frm = (today - dt.timedelta(days=14)).isoformat()
        url = "https://finnhub.io/api/v1/company-news"
        params = {"symbol": ticker, "from": frm, "to": today.isoformat(), "token": api_key}
        data = requests.get(url, params=params, timeout=30).json()
        for a in _ensure_list(data, "finnhub")[:limit]:
            out.append({
                "title": a.get("headline", ""),
                "summary": a.get("summary", "")[:600],
                "url": a.get("url", ""),
                "date": dt.datetime.fromtimestamp(a.get("datetime", 0)).isoformat(),
                "source": a.get("source", ""),
            })

    elif provider == "alphavantage":
        url = "https://www.alphavantage.co/query"
        params = {"function": "NEWS_SENTIMENT", "tickers": ticker, "apikey": api_key,
                  "limit": limit}
        data = requests.get(url, params=params, timeout=30).json()
        feed = data.get("feed") if isinstance(data, dict) else None
        if feed is None:
            _ensure_list(data, "alphavantage")  # raises with the API message
        for a in (feed or [])[:limit]:
            out.append({
                "title": a.get("title", ""),
                "summary": a.get("summary", "")[:600],
                "url": a.get("url", ""),
                "date": a.get("time_published", ""),
                "source": a.get("source", ""),
            })

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return out


# ----------------------------------------------------------------------
# 2. LLM news brief  (the "scan the news when I click" piece)
# ----------------------------------------------------------------------
_BRIEF_SYSTEM = """You are a skeptical equity research assistant. You are given \
RECENT NEWS ARTICLES about one stock. Using ONLY the information in those \
articles (do not invent facts, prices, events, or numbers not present in them), \
write a concise due-diligence briefing for an investor who wants to make an \
informed decision.

Use exactly these sections:
SUMMARY: 2-3 sentences on what is driving the news.
BULL: the strongest points supporting the stock, drawn from the articles.
BEAR: the strongest concerns or risks, drawn from the articles.
WHAT TO VERIFY: 3-5 specific, checkable questions the investor should answer \
from PRIMARY sources (filings, earnings releases, official statements) before \
investing.

If the articles are thin, stale, or one-sided, say so plainly. Do NOT give a \
buy/sell call or a price target. End with exactly:
'This is a digest of headlines, not advice - verify against primary sources.'"""


def news_brief(ticker, provider="fmp", news_api_key=None,
               model="gpt-4o-mini", openai_api_key=None, limit=10):
    """Fetch recent news for `ticker` and return an LLM due-diligence digest."""
    articles = fetch_news(ticker, provider=provider, api_key=news_api_key, limit=limit)
    if not articles:
        return f"No recent news found for {ticker.upper()}."

    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"))

    context = "\n\n".join(
        f"[{a['date']}] {a['source']}: {a['title']}\n{a['summary']}\nURL: {a['url']}"
        for a in articles
    )
    user = f"Ticker: {ticker.upper()}\n\nRecent articles:\n{context}\n\nWrite the briefing."

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _BRIEF_SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------------
# 3. Alerts
# ----------------------------------------------------------------------
def send_alert(message, subject="Breakout scan", channel="email", dry_run=False):
    """Send `message` by email (SMTP) or Telegram. Creds come from env vars."""
    if dry_run:
        print(f"[DRY RUN] {subject}\n{'-' * 40}\n{message}")
        return True

    if channel == "email":
        import smtplib
        from email.mime.text import MIMEText

        user = os.environ["SMTP_USER"]
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = os.environ.get("ALERT_TO", user)
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", 587))) as s:
            s.starttls()
            s.login(user, os.environ["SMTP_PASS"])
            s.send_message(msg)
        return True

    if channel == "telegram":
        import requests
        token = os.environ["TELEGRAM_TOKEN"]
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": message},
            timeout=30,
        )
        return True

    raise ValueError(f"Unknown channel: {channel}")


# ----------------------------------------------------------------------
# 4. Daily scan orchestration  (what GitHub Actions runs)
# ----------------------------------------------------------------------
def run_daily_scan(universe, load_fn=None, alert_fn=None, start="2024-01-01",
                   near_high_pct=0.03, min_mom=0.30, channel="email", dry_run=False):
    """Load prices, run the breakout scan, and alert on the hits.

    load_fn / alert_fn are injectable so this is testable without network.
    By default it uses yfinance (fine for a few hundred names); swap load_fn to
    your vendor loader for the full universe.
    """
    from equity_backtester import load_prices, breakout_scan

    load_fn = load_fn or (lambda u: load_prices(u, start=start))
    alert_fn = alert_fn or send_alert

    prices = load_fn(universe)
    hits = breakout_scan(prices, near_high_pct=near_high_pct, min_mom_12_1=min_mom)

    today = dt.date.today().isoformat()
    if hits.empty:
        msg = (f"{today}: no breakouts matched "
               f"(within {near_high_pct:.0%} of 52w high AND 12-1 momentum >= {min_mom:.0%}).")
    else:
        msg = (f"{today}: {len(hits)} breakout candidate(s) to RESEARCH "
               f"(not a buy signal):\n\n{hits.to_string()}\n\n"
               f"Run news_brief() on these and check primary sources before acting.")

    alert_fn(msg, subject=f"Breakout scan {today}", channel=channel, dry_run=dry_run)
    return hits


if __name__ == "__main__":
    import sys

    # CLI: `python pipeline.py news NVDA`  -> print an LLM news brief for a ticker
    if len(sys.argv) >= 3 and sys.argv[1] == "news":
        print(news_brief(sys.argv[2]))
    else:
        # Default: run the daily scan. EDIT this universe (or swap in a vendor
        # loader returning your full list). DRY_RUN=1 prints instead of sending.
        UNIVERSE = ["NVDA", "MU", "SNDK", "AVGO", "SMCI", "MRVL", "AMD", "TSM", "ASML", "ANET"]
        run_daily_scan(
            UNIVERSE,
            channel=os.environ.get("ALERT_CHANNEL", "email"),
            dry_run=os.environ.get("DRY_RUN", "0") == "1",
        )
