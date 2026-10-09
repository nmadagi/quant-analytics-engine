import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from ml_pricer import (BASELINE, FEATURES, dataset_from_surface, make_synthetic_dataset,
                       moneyness_bucket, run_benchmark, split_dataset)
from market_vol import atm_vol_by_expiry, surface_from_chain
from tests.test_market_vol import synthetic_chain


@pytest.fixture(scope="module")
def small_result():
    df = make_synthetic_dataset(n_days=16, quotes_per_day=80, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return run_benchmark(df, n_estimators=40, nn_iter=300, seed=1)


class TestDataset:
    def test_synthetic_shape_and_columns(self):
        df = make_synthetic_dataset(n_days=3, quotes_per_day=10)
        assert len(df) == 30
        assert set(FEATURES + ["market_price", "spot", "quote_day"]) <= set(df.columns)
        assert (df["market_price"] > 0).all()

    def test_time_split_keeps_test_days_after_train_days(self):
        df = make_synthetic_dataset(n_days=20, quotes_per_day=5)
        train, test, split = split_dataset(df, test_frac=0.25)
        assert split == "time"
        assert train["quote_day"].max() < test["quote_day"].min()
        assert test["quote_day"].nunique() == 5

    def test_single_day_falls_back_to_random_split(self):
        surf = surface_from_chain(synthetic_chain(), 100.0, 0.04)
        df = dataset_from_surface(surf, 100.0, 0.04, atm=atm_vol_by_expiry(surf))
        assert df["atm_vol"].notna().all()
        train, test, split = split_dataset(df, test_frac=0.3)
        assert split == "random"
        assert len(train) + len(test) == len(df)

    def test_buckets(self):
        assert moneyness_bucket(-0.1) == "OTM put"
        assert moneyness_bucket(0.0) == "Near ATM"
        assert moneyness_bucket(0.1) == "OTM call"


class TestBenchmark:
    def test_reports_every_pricer(self, small_result):
        assert set(small_result.metrics.index) == {BASELINE, "Random Forest", "Neural Network"}
        assert np.isfinite(small_result.metrics.to_numpy()).all()
        assert small_result.split == "time"

    def test_learned_models_beat_flat_vol_black_scholes(self, small_result):
        m = small_result.metrics["MSE"]
        assert m["Random Forest"] < m[BASELINE]
        assert m["Neural Network"] < m[BASELINE]

    def test_bucket_table_and_predictions_align(self, small_result):
        assert set(small_result.by_bucket.index) <= {"OTM put", "Near ATM", "OTM call"}
        assert set(small_result.by_bucket.columns) == set(small_result.metrics.index)
        assert len(small_result.predictions) == 4 * 80

    def test_too_small_dataset_raises(self):
        df = make_synthetic_dataset(n_days=1, quotes_per_day=2)
        with pytest.raises(ValueError):
            run_benchmark(df, n_estimators=5, nn_iter=50, test_frac=0.0)
