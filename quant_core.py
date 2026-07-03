"""
Core quantitative functions for the Quantitative Analytics Engine.

Pure, UI-free functions so they can be imported and unit-tested
without pulling in Streamlit.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option pricing model."""
    if T <= 0 or sigma <= 0:
        return max(S - K if option_type == "call" else K - S, 0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calc_greeks(S, K, T, r, sigma):
    """Calculate option Greeks."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    return {
        "delta": norm.cdf(d1),
        "gamma": nd1 / (S * sigma * np.sqrt(T)),
        "theta": -(S * nd1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2),
        "vega": S * nd1 * np.sqrt(T) / 100,
        "rho": K * T * np.exp(-r * T) * norm.cdf(d2) / 100,
    }


def generate_vol_surface(S, r, base_sigma, strikes_range=0.15, n_strikes=20, expiries=None):
    """Generate implied volatility surface with smile effect."""
    if expiries is None:
        expiries = [0.08, 0.17, 0.25, 0.5, 0.75, 1.0]
    strikes = np.linspace(S * (1 - strikes_range), S * (1 + strikes_range), n_strikes)

    data = []
    for T in expiries:
        for K in strikes:
            moneyness = (K - S) / S
            smile_adj = 1 + 0.15 * moneyness**2 + 0.05 * abs(moneyness)
            term_adj = 1 + 0.1 * (1 / np.sqrt(T) - 1) if T > 0 else 1
            iv = base_sigma * smile_adj * term_adj
            price = black_scholes(S, K, T, r, iv, "call")
            data.append({
                "strike": K, "expiry": T, "implied_vol": iv,
                "price": price, "moneyness": moneyness
            })
    return pd.DataFrame(data)


def monte_carlo_var(portfolio_value, daily_vol, n_sims=10000, n_days=10, confidence=0.95):
    """Monte Carlo VaR simulation."""
    np.random.seed(42)
    returns = np.random.normal(0, daily_vol, (n_sims, n_days))
    cumulative_returns = np.sum(returns, axis=1)
    portfolio_changes = portfolio_value * cumulative_returns
    var = np.percentile(portfolio_changes, (1 - confidence) * 100)
    cvar = np.mean(portfolio_changes[portfolio_changes <= var])
    return -var, -cvar, portfolio_changes
