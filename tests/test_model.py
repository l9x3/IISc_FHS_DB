"""
Unit tests for FHRPredictionMLP and MAELoss.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from src.model import FHRPredictionMLP, MAELoss, build_model


class TestFHRPredictionMLP:
    """Tests for model architecture."""

    def test_default_architecture(self):
        model = FHRPredictionMLP()
        # Verify layer sequence by checking Linear layers
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        expected = [(256, 128), (128, 64), (64, 32), (32, 1)]
        actual = [(l.in_features, l.out_features) for l in linears]
        assert actual == expected, f"Layer dims mismatch: {actual}"

    def test_custom_input_dim(self):
        model = FHRPredictionMLP(input_dim=128)
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert linears[0].in_features == 128

    def test_dropout_present(self):
        model = FHRPredictionMLP(dropout_rate=0.3)
        dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        assert len(dropouts) == 3, "Expected 3 Dropout layers (one per hidden layer)"
        for d in dropouts:
            assert d.p == pytest.approx(0.3)

    def test_forward_shape_batch(self):
        model = FHRPredictionMLP()
        x = torch.randn(16, 256)
        out = model(x)
        assert out.shape == (16, 1), f"Expected (16, 1), got {out.shape}"

    def test_forward_shape_single(self):
        model = FHRPredictionMLP()
        x = torch.randn(1, 256)
        out = model(x)
        assert out.shape == (1, 1)

    def test_output_is_continuous(self):
        """Output layer has no activation (linear), so output can be any float."""
        model = FHRPredictionMLP()
        model.eval()
        x = torch.zeros(1, 256)
        out = model(x)
        # Should not raise and should be finite
        assert torch.isfinite(out).all()

    def test_predict_method(self):
        model = FHRPredictionMLP()
        x = torch.randn(8, 256)
        preds = model.predict(x)
        assert preds.shape == (8,), f"Expected (8,), got {preds.shape}"

    def test_get_set_weights(self):
        model1 = FHRPredictionMLP()
        model2 = FHRPredictionMLP()
        # Set model2 weights to model1's
        weights = model1.get_weights()
        model2.set_weights(weights)
        for k in model1.state_dict():
            assert torch.equal(model1.state_dict()[k], model2.state_dict()[k])

    def test_count_parameters(self):
        model = FHRPredictionMLP()
        # 256*128+128 + 128*64+64 + 64*32+32 + 32*1+1
        expected = (256 * 128 + 128) + (128 * 64 + 64) + (64 * 32 + 32) + (32 * 1 + 1)
        assert model.count_parameters() == expected

    def test_training_mode_vs_eval_mode(self):
        """Dropout should be active in train mode and disabled in eval mode."""
        model = FHRPredictionMLP(dropout_rate=0.9)
        x = torch.ones(100, 256)
        model.train()
        out_train = model(x)
        model.eval()
        with torch.no_grad():
            out_eval = model(x)
        # In eval mode outputs should be identical on repeated calls
        with torch.no_grad():
            out_eval2 = model(x)
        assert torch.equal(out_eval, out_eval2), "Eval mode outputs should be deterministic"

    def test_build_model_factory(self):
        model = build_model(input_dim=256, dropout_rate=0.3)
        assert isinstance(model, FHRPredictionMLP)

    def test_no_activation_on_output(self):
        """Verify output layer is directly Linear (no ReLU on top)."""
        model = FHRPredictionMLP()
        # Inspect the last sub-module
        children = list(model.network.children())
        assert isinstance(children[-1], nn.Linear), "Last layer must be nn.Linear"


class TestMAELoss:
    """Tests for the MAE loss function."""

    def test_zero_loss(self):
        loss_fn = MAELoss()
        y = torch.tensor([130.0, 140.0, 150.0])
        loss = loss_fn(y, y)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_known_loss(self):
        loss_fn = MAELoss()
        preds = torch.tensor([131.0, 142.0, 148.0])
        targets = torch.tensor([130.0, 140.0, 150.0])
        # MAE = (1 + 2 + 2) / 3 = 5/3 ≈ 1.6667
        expected = (1.0 + 2.0 + 2.0) / 3.0
        assert loss_fn(preds, targets).item() == pytest.approx(expected, rel=1e-5)

    def test_squeezed_input(self):
        """Accepts (N, 1) predictions and (N,) targets."""
        loss_fn = MAELoss()
        preds = torch.ones(8, 1) * 140.0
        targets = torch.ones(8) * 145.0
        loss = loss_fn(preds, targets)
        assert loss.item() == pytest.approx(5.0, abs=1e-5)

    def test_differentiable(self):
        loss_fn = MAELoss()
        preds = torch.tensor([140.0], requires_grad=True)
        targets = torch.tensor([135.0])
        loss = loss_fn(preds, targets)
        loss.backward()
        assert preds.grad is not None
