"""
Unit tests for ClientDataSimulator.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.data_simulator import ClientDataSimulator, build_all_simulators, CLIENT_SPECS, FEATURE_DIM


class TestClientDataSimulator:
    """Tests for synthetic data generation."""

    def _make_sim(self, client_id: int = 1, num_windows: int = 50) -> ClientDataSimulator:
        nw, hr, st = CLIENT_SPECS[client_id]
        return ClientDataSimulator(
            client_id=client_id,
            num_windows=num_windows,
            heart_rate=hr,
            signal_type=st,
            seed=0,
        )

    def test_output_shapes(self):
        sim = self._make_sim(num_windows=100)
        X, y = sim.generate()
        assert X.shape == (100, FEATURE_DIM), f"Expected (100, {FEATURE_DIM}), got {X.shape}"
        assert y.shape == (100,), f"Expected (100,), got {y.shape}"

    def test_output_dtype(self):
        sim = self._make_sim()
        X, y = sim.generate()
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_heart_rate_range(self):
        sim = self._make_sim(num_windows=200)
        _, y = sim.generate()
        assert y.min() >= 100.0, "HR label below physiological floor"
        assert y.max() <= 200.0, "HR label above physiological ceiling"

    def test_heart_rate_cluster(self):
        """Labels should be centred near the nominal heart rate."""
        sim = self._make_sim(client_id=1, num_windows=500)  # HR=140
        _, y = sim.generate()
        assert abs(np.mean(y) - 140.0) < 5.0, "Mean HR deviates too far from nominal"

    def test_reproducibility(self):
        sim1 = self._make_sim(num_windows=30)
        sim2 = self._make_sim(num_windows=30)
        X1, y1 = sim1.generate()
        X2, y2 = sim2.generate()
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        nw, hr, st = CLIENT_SPECS[1]
        sim1 = ClientDataSimulator(1, 50, hr, st, seed=1)
        sim2 = ClientDataSimulator(1, 50, hr, st, seed=2)
        X1, _ = sim1.generate()
        X2, _ = sim2.generate()
        assert not np.allclose(X1, X2), "Different seeds should produce different data"

    def test_non_iid(self):
        """Different clients should produce statistically different features."""
        sim1 = ClientDataSimulator(1, 200, 140, "band-pass-filtered", seed=10)
        sim2 = ClientDataSimulator(3, 200, 148, "emd-denoised", seed=10)
        X1, _ = sim1.generate()
        X2, _ = sim2.generate()
        # Mean features should differ
        assert not np.allclose(np.mean(X1, axis=0), np.mean(X2, axis=0))

    def test_train_val_test_split_sizes(self):
        sim = self._make_sim(num_windows=1000)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = sim.train_val_test_split(
            train_frac=0.70, val_frac=0.15
        )
        total = len(y_tr) + len(y_va) + len(y_te)
        assert total == 1000
        assert len(y_tr) == pytest.approx(700, abs=2)
        assert len(y_va) == pytest.approx(150, abs=2)

    def test_build_all_simulators_count(self):
        sims = build_all_simulators(seed=0)
        assert len(sims) == 10

    def test_total_windows(self):
        """The 10 default clients total 409,852 windows."""
        total = sum(nw for nw, _, _ in CLIENT_SPECS.values())
        assert total == 409_852

    @pytest.mark.parametrize("client_id", list(CLIENT_SPECS.keys()))
    def test_all_signal_types_run(self, client_id):
        nw, hr, st = CLIENT_SPECS[client_id]
        sim = ClientDataSimulator(client_id, num_windows=20, heart_rate=hr, signal_type=st, seed=0)
        X, y = sim.generate()
        assert X.shape[1] == FEATURE_DIM
        assert not np.any(np.isnan(X)), f"NaN in features for client {client_id}"
        assert not np.any(np.isnan(y)), f"NaN in labels for client {client_id}"
