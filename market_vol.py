"""Implied volatility surface solved from market option quotes.

fetch_option_chain pulls quotes from yfinance. surface_from_chain is pure
(no network) so it can be unit-tested with a synthetic chain.
"""

import numpy as np
import pandas as pd

from quant_core import implied_vol

CHAIN_COLUMNS = ["strike", "expiry", "bid", "ask", "option_type"]
TARGET_TENOR_DAYS = (14, 30, 60, 90, 180, 365, 540, 730)


def pick_expiries(expiries, today, max_expiries=6, min_days=7, targets=TARGET_TENOR_DAYS):
    """Choose listed expiries closest to a ladder of tenors.

    Tickers like SPY list an expiry for almost every trading day, so taking the
    first few would cover one week. Picking by target tenor gives a term
    structure instead. Expiries inside min_days are skipped as too noisy.
    """
    dated = [(exp, (pd.Timestamp(exp) - today).days) for exp in expiries]
    dated = [(exp, d) for exp, d in dated if d >= min_days]
    if not dated:
        return []
    chosen = []
    for target in targets:
        best = min(dated, key=lambda item: abs(item[1] - target))[0]
        if best not in chosen:
            chosen.append(best)
        if len(chosen) == max_expiries:
            break
    return chosen


def fetch_option_chain(ticker, max_expiries=6):
    """Return (spot, chain) for a ticker using yfinance.

    The chain has one row per listed call and put with columns strike, expiry
    (years), expiry_date, bid, ask, volume and option_type.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    closes = tk.history(period="5d")["Close"].dropna()   # today's bar can be NaN after hours
    if closes.empty:
        raise ValueError(f"no price history for {ticker}")
    spot = float(closes.iloc[-1])

    today = pd.Timestamp.today().normalize()
    expiries = pick_expiries(list(tk.options), today, max_expiries)
    if not expiries:
        raise ValueError(f"no listed options for {ticker}")

    frames = []
    for exp in expiries:
        chain = tk.option_chain(exp)
        years = (pd.Timestamp(exp) - today).days / 365.0
        for option_type, quotes in (("call", chain.calls), ("put", chain.puts)):
            frames.append(pd.DataFrame({
                "strike": quotes["strike"].to_numpy(dtype=float),
                "expiry": years,
                "expiry_date": exp,
                "bid": quotes["bid"].fillna(0).to_numpy(dtype=float),
                "ask": quotes["ask"].fillna(0).to_numpy(dtype=float),
                "volume": quotes["volume"].fillna(0).to_numpy(dtype=float)
                if "volume" in quotes else 0.0,
                "option_type": option_type,
            }))
    return spot, pd.concat(frames, ignore_index=True)


def surface_from_chain(chain, S, r, moneyness_range=0.15, min_bid=0.05, otm_only=True):
    """Solve implied vol for every usable quote in a chain.

    Quotes are kept when the bid is live, the market is not crossed, and the
    strike sits within moneyness_range of spot. With otm_only the surface uses
    out-of-the-money calls above spot and puts below spot, which is the liquid
    side of the market and the usual convention for a vol surface.
    """
    missing = [c for c in CHAIN_COLUMNS if c not in chain.columns]
    if missing:
        raise ValueError(f"chain is missing columns: {missing}")

    df = chain.copy()
    df["price"] = (df["bid"] + df["ask"]) / 2
    df["moneyness"] = (df["strike"] - S) / S

    keep = (df["bid"] >= min_bid) & (df["ask"] >= df["bid"]) & (df["moneyness"].abs() <= moneyness_range)
    if otm_only:
        keep &= ((df["option_type"] == "call") & (df["strike"] >= S)) | \
                ((df["option_type"] == "put") & (df["strike"] < S))
    df = df[keep].copy()

    df["implied_vol"] = [
        implied_vol(p, S, k, t, r, o)
        for p, k, t, o in zip(df["price"], df["strike"], df["expiry"], df["option_type"])
    ]
    df = df.dropna(subset=["implied_vol"])
    df = df[(df["implied_vol"] > 0.01) & (df["implied_vol"] < 3.0)]

    cols = ["strike", "expiry", "implied_vol", "price", "moneyness", "option_type"]
    if "expiry_date" in df.columns:
        cols.append("expiry_date")
    return df[cols].sort_values(["expiry", "strike"]).reset_index(drop=True)


def atm_vol_by_expiry(surface):
    """Implied vol of the quote nearest to spot, one value per expiry."""
    if surface.empty:
        return pd.Series(dtype=float)
    nearest = surface.loc[surface.groupby("expiry")["moneyness"].apply(lambda m: m.abs().idxmin())]
    return nearest.set_index("expiry")["implied_vol"].sort_index()


def atm_vol(surface):
    """Implied vol of the quote nearest to spot on the shortest expiry."""
    series = atm_vol_by_expiry(surface)
    return float(series.iloc[0]) if len(series) else float("nan")
