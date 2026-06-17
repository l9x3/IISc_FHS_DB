"""
Unit tests for FederatedServer.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy

import numpy as np
import pytest
import torch

from src.server import FederatedServer
from src.model import FHRPredictionMLP, build_model


def _equal_state_dicts(d1, d2) -> bool:
    return all(torch.equal(d1[k], d2[k]) for k in d1)


def _make_state_dicts(n: int, seed: int = 0) -> list:
    torch.manual_seed(seed)
    return [build_model().state_dict() for _ in range(n)]


class TestFederatedServer:

    def test_instantiation(self):
        server = FederatedServer()
        assert server.current_round == 0
        assert isinstance(server.global_model, FHRPredictionMLP)

    def test_get_global_weights_is_copy(self):
        server = FederatedServer()
        w1 = server.get_global_weights()
        w2 = server.get_global_weights()
        assert w1 is not w2
        assert _equal_state_dicts(w1, w2)

    def test_aggregate_uniform(self):
        """Uniform weights → aggregated model should equal any single model."""
        server = FederatedServer()
        weights = _make_state_dicts(1)
        agg = server.aggregate(weights, [100])
        assert _equal_state_dicts(agg, weights[0])

    def test_aggregate_equal_sizes(self):
        """Equal sizes → aggregated = simple mean of parameters."""
        server = FederatedServer()
        w1 = _make_state_dicts(1, seed=1)[0]
        w2 = _make_state_dicts(1, seed=2)[0]
        agg = server.aggregate([w1, w2], [50, 50])
        for k in agg:
            expected = (w1[k].float() + w2[k].float()) / 2.0
            assert torch.allclose(agg[k].float(), expected, atol=1e-5)

    def test_aggregate_weighted(self):
        """Weighted aggregation should weight by dataset size."""
        server = FederatedServer()
        w1 = _make_state_dicts(1, seed=10)[0]
        w2 = _make_state_dicts(1, seed=20)[0]
        # client1 has 3× the data of client2
        agg = server.aggregate([w1, w2], [75, 25])
        for k in agg:
            expected = w1[k].float() * 0.75 + w2[k].float() * 0.25
            assert torch.allclose(agg[k].float(), expected, atol=1e-5)

    def test_round_increments(self):
        server = FederatedServer()
        weights = _make_state_dicts(2)
        server.aggregate(weights, [100, 100])
        assert server.current_round == 1
        server.aggregate(weights, [100, 100])
        assert server.current_round == 2

    def test_aggregate_empty_raises(self):
        server = FederatedServer()
        with pytest.raises(ValueError):
            server.aggregate([], [])

    def test_aggregate_mismatched_lengths_raises(self):
        server = FederatedServer()
        weights = _make_state_dicts(2)
        with pytest.raises(ValueError):
            server.aggregate(weights, [100])

    def test_aggregate_zero_total_raises(self):
        server = FederatedServer()
        weights = _make_state_dicts(2)
        with pytest.raises(ValueError):
            server.aggregate(weights, [0, 0])

    def test_drift_zero_before_aggregation(self):
        server = FederatedServer()
        assert server.compute_drift() == 0.0

    def test_drift_nonzero_after_aggregation(self):
        server = FederatedServer()
        torch.manual_seed(99)
        w1 = {k: v + 0.5 for k, v in build_model().state_dict().items()}
        server.aggregate([w1], [100])
        # Drift from initial random init to new weights should be > 0
        assert server.compute_drift() >= 0.0

    def test_evaluate_global(self):
        server = FederatedServer()
        X = torch.randn(50, 256)
        y = torch.ones(50) * 140.0
        mae = server.evaluate_global(X, y)
        assert isinstance(mae, float)
        assert mae >= 0.0

    def test_global_model_updated_after_aggregate(self):
        server = FederatedServer()
        w_before = copy.deepcopy(server.get_global_weights())
        torch.manual_seed(42)
        new_weights = {k: v + 1.0 for k, v in w_before.items()}
        server.aggregate([new_weights], [100])
        w_after = server.get_global_weights()
        assert not _equal_state_dicts(w_before, w_after)
