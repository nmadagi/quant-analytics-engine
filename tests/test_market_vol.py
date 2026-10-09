import numpy as np
import pandas as pd
import pytest

from market_vol import atm_vol, atm_vol_by_expiry, pick_expiries, surface_from_chain
from quant_core import black_scholes, implied_vol


def synthetic_chain(S=100.0, r=0.04, half_spread=0.02):
    """Quotes priced from a known smile so the solver has a right answer to hit."""
    rows = []
    for T in (0.1, 0.5):
        for K in np.linspace(88, 112, 13):
            m = (K - S) / S
            true_iv = 0.20 * (1 + 2.0 * m**2 - 0.5 * m)
            for opt in ("call", "put"):
                fair = black_scholes(S, K, T, r, true_iv, opt)
                rows.append({"strike": K, "expiry": T, "bid": fair - half_spread,
                             "ask": fair + half_spread, "option_type": opt, "true_iv": true_iv})
    return pd.DataFrame(rows)


class TestImpliedVol:
    def test_roundtrip_recovers_sigma(self):
        for opt in ("call", "put"):
            price = black_scholes(100, 95, 0.5, 0.03, 0.27, opt)
            assert implied_vol(price, 100, 95, 0.5, 0.03, opt) == pytest.approx(0.27, abs=1e-6)

    def test_prices_outside_arbitrage_bounds_return_nan(self):
        assert np.isnan(implied_vol(0.5, 100, 80, 0.5, 0.03, "call"))   # below intrinsic
        assert np.isnan(implied_vol(150, 100, 80, 0.5, 0.03, "call"))   # above spot
        assert np.isnan(implied_vol(5, 100, 100, 0, 0.03, "call"))      # expired


class TestSurfaceFromChain:
    def test_recovers_true_smile_from_quotes(self):
        chain = synthetic_chain()
        surf = surface_from_chain(chain, 100.0, 0.04, moneyness_range=0.12)
        assert not surf.empty
        merged = surf.merge(chain[["strike", "expiry", "option_type", "true_iv"]],
                            on=["strike", "expiry", "option_type"])
        # mid of a symmetric spread is the fair price, so IV should match to solver tolerance
        assert np.allclose(merged["implied_vol"], merged["true_iv"], atol=2e-3)

    def test_keeps_only_otm_side(self):
        surf = surface_from_chain(synthetic_chain(), 100.0, 0.04)
        assert (surf.loc[surf.option_type == "call", "strike"] >= 100).all()
        assert (surf.loc[surf.option_type == "put", "strike"] < 100).all()

    def test_drops_dead_and_crossed_quotes(self):
        chain = synthetic_chain()
        chain.loc[0, "bid"] = 0.0                      # no bid
        chain.loc[1, ["bid", "ask"]] = [3.0, 2.0]      # crossed market
        surf = surface_from_chain(chain, 100.0, 0.04)
        bad = chain.loc[[0, 1], ["strike", "expiry", "option_type"]]
        hits = surf.merge(bad, on=["strike", "expiry", "option_type"])
        assert hits.empty

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError):
            surface_from_chain(pd.DataFrame({"strike": [100]}), 100.0, 0.04)

    def test_atm_vol_uses_nearest_strike_on_front_expiry(self):
        surf = surface_from_chain(synthetic_chain(), 100.0, 0.04)
        assert atm_vol(surf) == pytest.approx(0.20, abs=5e-3)
        curve = atm_vol_by_expiry(surf)
        assert list(curve.index) == [0.1, 0.5]
        assert np.allclose(curve.to_numpy(), 0.20, atol=5e-3)


class TestPickExpiries:
    def test_builds_a_term_structure_from_daily_listings(self):
        today = pd.Timestamp("2026-10-08")
        daily = [str((today + pd.Timedelta(days=d)).date()) for d in range(1, 400)]
        chosen = pick_expiries(daily, today, max_expiries=4)
        days = [(pd.Timestamp(e) - today).days for e in chosen]
        assert days == [14, 30, 60, 90]

    def test_skips_expiries_inside_min_days_and_dedupes(self):
        today = pd.Timestamp("2026-10-08")
        listed = ["2026-10-09", "2026-10-30", "2027-01-15"]
        chosen = pick_expiries(listed, today, max_expiries=6)
        assert chosen == ["2026-10-30", "2027-01-15"]
