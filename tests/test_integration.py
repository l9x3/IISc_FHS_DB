"""
Integration tests: end-to-end federated learning workflow.

Uses minimal data (small num_windows, few rounds) so the suite runs fast.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from src.client import FederatedClient
from src.data_simulator import ClientDataSimulator, FEATURE_DIM
from src.evaluation import FederatedEvaluator
from src.federated_trainer import FederatedTrainer
from src.model import build_model
from src.server import FederatedServer
from src.utils import compute_mae


# ── minimal configuration for fast testing ────────────────────────────────────

MINI_CONFIG = {
    "model": {"input_dim": FEATURE_DIM, "dropout_rate": 0.3},
    "training": {
        "batch_size": 32,
        "learning_rate": 0.01,
        "local_epochs": 1,
        "optimizer": "sgd",
        "loss_function": "mae",
    },
    "federated": {
        "num_communication_rounds": 3,
        "client_participation_rate": 1.0,
        "aggregation_method": "fedavg_weighted",
        "seed": 42,
    },
    "clients": [
        {"client_id": 1, "num_windows": 60, "heart_rate": 140, "signal_type": "band-pass-filtered"},
        {"client_id": 2, "num_windows": 50, "heart_rate": 144, "signal_type": "wavelet-filtered"},
        {"client_id": 3, "num_windows": 40, "heart_rate": 148, "signal_type": "emd-denoised"},
    ],
    "evaluation": {
        "kfold": 3,
        "train_split": 0.70,
        "val_split": 0.15,
        "test_split": 0.15,
    },
}


class TestEndToEnd:

    @pytest.fixture(scope="class")
    def tmp_results(self, tmp_path_factory):
        return str(tmp_path_factory.mktemp("results"))

    @pytest.fixture(scope="class")
    def trainer(self, tmp_results):
        import copy
        t = FederatedTrainer(copy.deepcopy(MINI_CONFIG), results_dir=tmp_results)
        t.setup()
        return t

    def test_setup_creates_clients(self, trainer):
        assert len(trainer.clients) == 3

    def test_setup_loads_data(self, trainer):
        for client in trainer.clients:
            assert client.num_train_samples > 0

    def test_run_returns_metrics(self, trainer):
        metrics = trainer.run()
        assert len(metrics) == 3  # 3 rounds
        assert "global_mae" in metrics[0]
        assert "client_test_maes" in metrics[0]

    def test_global_mae_is_positive(self, trainer):
        assert all(m["global_mae"] > 0 for m in trainer.round_metrics)

    def test_round_counter_correct(self, trainer):
        rounds = [m["round"] for m in trainer.round_metrics]
        assert rounds == [1, 2, 3]

    def test_server_round_incremented(self, trainer):
        assert trainer.server.current_round == 3

    def test_convergence_plot_created(self, trainer, tmp_results):
        plot_path = os.path.join(tmp_results, "convergence_plots", "federated_convergence.png")
        assert os.path.exists(plot_path)

    def test_round_metrics_json_created(self, trainer, tmp_results):
        json_path = os.path.join(tmp_results, "metrics", "round_metrics.json")
        assert os.path.exists(json_path)

    def test_checkpoint_created(self, trainer, tmp_results):
        ckpt = os.path.join(tmp_results, "model_checkpoints", "global_model_final.pt")
        assert os.path.exists(ckpt)

    def test_all_clients_have_test_maes(self, trainer):
        last = trainer.round_metrics[-1]
        assert set(last["client_test_maes"].keys()) == {1, 2, 3}


class TestEvaluatorIntegration:

    def test_export_round_metrics(self, tmp_path):
        evaluator = FederatedEvaluator(results_dir=str(tmp_path))
        fake_metrics = [
            {
                "round": 1,
                "global_mae": 5.0,
                "client_test_maes": {1: 5.2, 2: 4.8},
                "client_train_maes": {1: 5.0, 2: 5.0},
                "model_drift": 0.1,
                "wall_time_s": 1.0,
            }
        ]
        csv_path = evaluator.export_round_metrics(fake_metrics, "metrics/test_rounds.csv")
        assert os.path.exists(csv_path)
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert df.shape[0] == 1
        assert "global_mae" in df.columns

    def test_convergence_rate(self):
        ev = FederatedEvaluator()
        history = [10.0, 8.0, 6.0, 5.0, 4.5]
        rate = ev.compute_convergence_rate(history, window=3)
        assert rate > 0, "Decreasing MAE should yield positive convergence rate"

    def test_communication_bytes_positive(self):
        ev = FederatedEvaluator()
        model = build_model()
        total = ev.estimate_communication_bytes(model, num_rounds=10, num_clients=3)
        assert total > 0

    def test_kfold_crossval_runs(self):
        ev = FederatedEvaluator(kfold=3)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 256)).astype(np.float32)
        y = rng.uniform(130, 156, 120).astype(np.float32)
        results = ev.kfold_crossval(X, y, epochs=2, batch_size=32)
        assert "summary" in results
        assert "test_mae" in results["summary"]

    def test_centralised_baseline_runs(self):
        ev = FederatedEvaluator()
        rng = np.random.default_rng(1)
        all_X = [rng.standard_normal((50, 256)).astype(np.float32) for _ in range(3)]
        all_y = [rng.uniform(130, 156, 50).astype(np.float32) for _ in range(3)]
        metrics = ev.centralised_baseline(all_X, all_y, epochs=2, batch_size=32)
        assert "mae" in metrics
        assert metrics["mae"] > 0


class TestDataFlowPrivacy:
    """Verify that raw data never needs to leave the client object."""

    def test_only_weights_transferred(self):
        """Clients expose get_model_weights() returning a state_dict, not raw X/y."""
        client = FederatedClient(client_id=1)
        X = np.random.randn(50, 256).astype(np.float32)
        y = np.random.uniform(130, 156, 50).astype(np.float32)
        client.load_data(X, y)
        client.train(epochs=1, batch_size=32)
        weights = client.get_model_weights()
        # Weights should be a dict of tensors, NOT the raw data
        assert isinstance(weights, dict)
        for v in weights.values():
            assert isinstance(v, torch.Tensor)
            assert v.shape != torch.Size([50, 256])

    def test_server_aggregates_weights_not_data(self):
        server = FederatedServer()
        from src.model import build_model
        w = [build_model().state_dict()]
        result = server.aggregate(w, [100])
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, torch.Tensor)
