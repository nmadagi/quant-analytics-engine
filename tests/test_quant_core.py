import numpy as np
import pytest

from quant_core import black_scholes, calc_greeks, generate_vol_surface, monte_carlo_var


class TestBlackScholes:
    def test_call_matches_known_value(self):
        # Classic textbook case: S=100, K=100, T=1, r=5%, sigma=20% -> ~10.4506
        price = black_scholes(100, 100, 1.0, 0.05, 0.20, "call")
        assert price == pytest.approx(10.4506, abs=1e-3)

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100, 105, 0.5, 0.03, 0.25
        call = black_scholes(S, K, T, r, sigma, "call")
        put = black_scholes(S, K, T, r, sigma, "put")
        assert call - put == pytest.approx(S - K * np.exp(-r * T), abs=1e-9)

    def test_expired_option_returns_intrinsic_value(self):
        assert black_scholes(110, 100, 0, 0.05, 0.2, "call") == 10
        assert black_scholes(90, 100, 0, 0.05, 0.2, "put") == 10
        assert black_scholes(90, 100, 0, 0.05, 0.2, "call") == 0


class TestGreeks:
    def test_delta_bounds_and_gamma_positive(self):
        g = calc_greeks(100, 100, 1.0, 0.05, 0.20)
        assert 0 < g["delta"] < 1
        assert g["gamma"] > 0
        assert g["vega"] > 0

    def test_deep_itm_call_delta_near_one(self):
        g = calc_greeks(200, 100, 0.5, 0.05, 0.20)
        assert g["delta"] == pytest.approx(1.0, abs=1e-3)

    def test_degenerate_inputs_return_zeros(self):
        g = calc_greeks(100, 100, 0, 0.05, 0.20)
        assert all(v == 0 for v in g.values())


class TestVolSurface:
    def test_shape_and_smile(self):
        df = generate_vol_surface(100, 0.05, 0.20, n_strikes=10, expiries=[0.25, 1.0])
        assert len(df) == 20
        # smile: wings have higher IV than ATM for the same expiry
        near = df[df.expiry == 1.0]
        atm_iv = near.iloc[(near.moneyness.abs()).argmin()]["implied_vol"]
        wing_iv = near.iloc[near.moneyness.argmax()]["implied_vol"]
        assert wing_iv > atm_iv


class TestMonteCarloVar:
    def test_var_and_cvar_ordering(self):
        var, cvar, changes = monte_carlo_var(1_000_000, 0.02, n_sims=5000)
        assert var > 0
        assert cvar >= var  # expected shortfall is at least as severe as VaR
        assert len(changes) == 5000

    def test_var_scales_with_volatility(self):
        var_lo, _, _ = monte_carlo_var(1_000_000, 0.01)
        var_hi, _, _ = monte_carlo_var(1_000_000, 0.03)
        assert var_hi > var_lo
