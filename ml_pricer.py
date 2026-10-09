"""Random Forest and neural network option pricers benchmarked against Black-Scholes.

The benchmark asks one question: given only what a flat-vol Black-Scholes
pricer sees (spot, strike, expiry, rate, one at-the-money vol), can a learned
model price closer to the market? The gap is the volatility smile the formula
cannot see. Each learned pricer is a hybrid: it starts from the Black-Scholes
price and learns the correction (market minus Black-Scholes, scaled by spot).
Models are trained on earlier quote days and scored on later ones.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant_core import black_scholes

FEATURES = ["log_moneyness", "expiry", "rate", "atm_vol", "is_call"]
BASELINE = "Black-Scholes (flat vol)"
EXPIRIES = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)


def smile_vol(atm_vol, log_moneyness, expiry):
    """Stylised market smile: skewed to the downside and steeper near expiry."""
    return atm_vol * (1 + 0.6 * log_moneyness**2 / np.sqrt(expiry) - 0.25 * log_moneyness)


def make_synthetic_dataset(n_days=40, quotes_per_day=150, seed=42):
    """Daily snapshots of OTM quotes priced off a smile, with bid-ask noise.

    Each day has its own spot and at-the-money vol so a time-based split is a
    genuine out-of-sample test rather than a shuffle of the same surface.
    """
    rng = np.random.default_rng(seed)
    rows = []
    spot = 100.0
    for day in range(n_days):
        spot *= float(np.exp(rng.normal(0, 0.012)))
        atm = float(np.clip(0.22 + 0.04 * np.sin(day / 6) + rng.normal(0, 0.01), 0.10, 0.60))
        for _ in range(quotes_per_day):
            lm = float(rng.uniform(-0.2, 0.2))
            expiry = float(rng.choice(EXPIRIES))
            is_call = int(lm >= 0)
            strike = spot * np.exp(lm)
            sigma = smile_vol(atm, lm, expiry)
            fair = black_scholes(spot, strike, expiry, 0.045, sigma, "call" if is_call else "put")
            price = max(fair * (1 + rng.normal(0, 0.01)), 0.01)
            rows.append({
                "quote_day": day, "spot": spot, "strike": strike, "expiry": expiry,
                "rate": 0.045, "atm_vol": atm, "is_call": is_call,
                "log_moneyness": lm, "market_price": price,
            })
    return pd.DataFrame(rows)


def dataset_from_surface(surface, spot, rate, atm):
    """Turn a market_vol surface into benchmark rows (one quote day, so the split is random).

    atm may be a single vol or a Series indexed by expiry, so the Black-Scholes
    baseline can respect the term structure and only miss the smile.
    """
    expiry = surface["expiry"].to_numpy()
    atm_col = pd.Series(expiry).map(atm).to_numpy() if isinstance(atm, pd.Series) else atm
    df = pd.DataFrame({
        "quote_day": 0,
        "spot": spot,
        "strike": surface["strike"].to_numpy(),
        "expiry": expiry,
        "rate": rate,
        "atm_vol": atm_col,
        "is_call": (surface["option_type"] == "call").astype(int).to_numpy(),
        "market_price": surface["price"].to_numpy(),
    })
    df["log_moneyness"] = np.log(df["strike"] / df["spot"])
    return df


def black_scholes_prices(df):
    """Flat-vol baseline: every quote priced with the day's single ATM vol."""
    return np.array([
        black_scholes(s, k, t, r, v, "call" if c else "put")
        for s, k, t, r, v, c in zip(df["spot"], df["strike"], df["expiry"],
                                    df["rate"], df["atm_vol"], df["is_call"])
    ])


def split_dataset(df, test_frac=0.25, seed=42):
    """Time-based split when the data spans several quote days, random otherwise."""
    days = np.sort(df["quote_day"].unique())
    if len(days) > 1:
        n_test = max(int(round(len(days) * test_frac)), 1)
        cutoff = days[-n_test]
        return df[df["quote_day"] < cutoff], df[df["quote_day"] >= cutoff], "time"
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_frac
    return df[~mask], df[mask], "random"


def build_models(n_estimators=300, nn_iter=2000, seed=42):
    """Two learners for the Black-Scholes correction. The network scales both
    inputs and target, since the correction is a few cents on a $100 spot."""
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=2,
                               random_state=seed, n_jobs=-1)
    nn = TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32, 32), activation="relu", max_iter=nn_iter,
                         n_iter_no_change=50, random_state=seed),
        ),
        transformer=StandardScaler(),
    )
    return {"Random Forest": rf, "Neural Network": nn}


def moneyness_bucket(log_moneyness):
    if log_moneyness < -0.05:
        return "OTM put"
    if log_moneyness > 0.05:
        return "OTM call"
    return "Near ATM"


@dataclass(frozen=True)
class BenchmarkResult:
    metrics: pd.DataFrame
    by_bucket: pd.DataFrame
    predictions: pd.DataFrame
    split: str


def run_benchmark(df, n_estimators=300, nn_iter=2000, seed=42, test_frac=0.25):
    """Fit the models on the training slice and score every pricer on the test slice.

    Models learn (market price - Black-Scholes price) / spot, so one model serves
    every spot level; errors are reported back in dollars per contract.
    """
    train, test, split = split_dataset(df, test_frac, seed)
    if train.empty or test.empty:
        raise ValueError("dataset too small to split")

    X_train, X_test = train[FEATURES], test[FEATURES]
    bs_train, bs_test = black_scholes_prices(train), black_scholes_prices(test)
    y_train = (train["market_price"].to_numpy() - bs_train) / train["spot"].to_numpy()

    preds = {BASELINE: bs_test}
    for name, model in build_models(n_estimators, nn_iter, seed).items():
        model.fit(X_train, y_train)
        preds[name] = bs_test + model.predict(X_test) * test["spot"].to_numpy()

    actual = test["market_price"].to_numpy()
    metrics = pd.DataFrame([{
        "model": name,
        "MSE": mean_squared_error(actual, p),
        "RMSE": float(np.sqrt(mean_squared_error(actual, p))),
        "MAE": mean_absolute_error(actual, p),
    } for name, p in preds.items()]).set_index("model")

    predictions = test.assign(bucket=test["log_moneyness"].map(moneyness_bucket))
    for name, p in preds.items():
        predictions = predictions.assign(**{name: p})

    by_bucket = pd.DataFrame({
        name: predictions.groupby("bucket").apply(
            lambda g, n=name: mean_absolute_error(g["market_price"], g[n]), include_groups=False)
        for name in preds
    }).rename_axis("MAE by bucket")

    return BenchmarkResult(metrics=metrics, by_bucket=by_bucket,
                           predictions=predictions, split=split)
