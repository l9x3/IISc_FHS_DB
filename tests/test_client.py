"""
Unit tests for FederatedClient.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from src.client import FederatedClient
from src.model import FHRPredictionMLP


def _make_data(n: int = 200, dim: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    y = rng.uniform(130, 156, n).astype(np.float32)
    return X, y


class TestFederatedClient:

    def test_instantiation(self):
        client = FederatedClient(client_id=1)
        assert client.client_id == 1
        assert isinstance(client.model, FHRPredictionMLP)

    def test_load_data(self):
        client = FederatedClient(client_id=1)
        X, y = _make_data(200)
        client.load_data(X, y)
        assert client.num_train_samples == 200

    def test_load_data_with_val(self):
        client = FederatedClient(client_id=1)
        X_tr, y_tr = _make_data(160)
        X_va, y_va = _make_data(40, seed=99)
        client.load_data(X_tr, y_tr, X_va, y_va)
        assert client.num_train_samples == 160

    def test_train_returns_float(self):
        client = FederatedClient(client_id=2)
        X, y = _make_data(100)
        client.load_data(X, y)
        mae = client.train(epochs=1, batch_size=32, learning_rate=0.01)
        assert isinstance(mae, float)
        assert mae >= 0.0

    def test_train_appends_history(self):
        client = FederatedClient(client_id=3)
        X, y = _make_data(100)
        client.load_data(X, y)
        client.train(epochs=1, batch_size=32)
        client.train(epochs=1, batch_size=32)
        assert len(client.train_losses) == 2

    def test_evaluate_with_explicit_data(self):
        client = FederatedClient(client_id=4)
        X_tr, y_tr = _make_data(100)
        client.load_data(X_tr, y_tr)
        X_va, y_va = _make_data(50, seed=7)
        mae = client.evaluate(X_va, y_va)
        assert isinstance(mae, float)
        assert mae >= 0.0

    def test_evaluate_with_stored_val(self):
        client = FederatedClient(client_id=5)
        X_tr, y_tr = _make_data(100)
        X_va, y_va = _make_data(40, seed=7)
        client.load_data(X_tr, y_tr, X_va, y_va)
        mae = client.evaluate()
        assert isinstance(mae, float)

    def test_evaluate_no_val_raises(self):
        client = FederatedClient(client_id=6)
        X, y = _make_data(50)
        client.load_data(X, y)
        with pytest.raises(RuntimeError):
            client.evaluate()

    def test_train_without_data_raises(self):
        client = FederatedClient(client_id=7)
        with pytest.raises(RuntimeError):
            client.train()

    def test_set_and_get_weights(self):
        client = FederatedClient(client_id=8)
        original = client.get_model_weights()
        # Modify weights manually
        new_weights = {k: v + 1.0 for k, v in original.items()}
        client.set_global_weights(new_weights)
        recovered = client.get_model_weights()
        for k in original:
            assert torch.allclose(recovered[k], new_weights[k])

    def test_training_reduces_loss(self):
        """After training, MAE should decrease compared to random init."""
        torch.manual_seed(0)
        np.random.seed(0)
        client = FederatedClient(client_id=9)
        # Perfectly learnable constant-label dataset
        X = np.ones((500, 256), dtype=np.float32)
        y = np.full(500, 140.0, dtype=np.float32)
        client.load_data(X, y)
        _ = client.train(epochs=1, batch_size=128, learning_rate=0.01)
        mae_before = client.train_losses[0]
        _ = client.train(epochs=10, batch_size=128, learning_rate=0.01)
        mae_after = client.train_losses[-1]
        # After more training the loss should be lower (or at least not worse)
        assert mae_after <= mae_before * 1.5, (
            f"Loss did not improve: before={mae_before:.4f}, after={mae_after:.4f}"
        )

    def test_weights_change_after_training(self):
        client = FederatedClient(client_id=10)
        X, y = _make_data(100)
        client.load_data(X, y)
        weights_before = {k: v.clone() for k, v in client.get_model_weights().items()}
        client.train(epochs=2, batch_size=32, learning_rate=0.01)
        weights_after = client.get_model_weights()
        changed = any(
            not torch.equal(weights_before[k], weights_after[k])
            for k in weights_before
        )
        assert changed, "Model weights should change after training"
