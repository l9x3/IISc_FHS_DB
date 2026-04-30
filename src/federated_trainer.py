"""
Main federated learning orchestrator.

Ties together data simulation, client training, server aggregation, and
evaluation into a single :class:`FederatedTrainer` class that drives the
full communication-round loop.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .client import FederatedClient
from .data_simulator import ClientDataSimulator, build_all_simulators
from .evaluation import FederatedEvaluator
from .logger import get_logger
from .model import FHRPredictionMLP
from .server import FederatedServer
from .utils import compute_mae, load_config, plot_convergence, save_json

log = get_logger(__name__, log_dir="")


class FederatedTrainer:
    """
    Orchestrates federated learning across all clients and the central server.

    Parameters
    ----------
    config : dict or str
        Either the parsed YAML configuration dictionary or a path to the
        YAML file (``config/federated_config.yaml``).
    results_dir : str
        Root directory for artefacts (plots, metrics, checkpoints).
    device : torch.device, optional
        Compute device (defaults to CPU).
    """

    def __init__(
        self,
        config: Any,
        results_dir: str = "results",
        device: Optional[torch.device] = None,
    ) -> None:
        if isinstance(config, str):
            config = load_config(config)
        self.cfg = config
        self.results_dir = results_dir
        self.device = device or torch.device("cpu")

        # ── reproducibility ──────────────────────────────────────────────────
        seed: int = self.cfg.get("federated", {}).get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ── hyper-parameters ─────────────────────────────────────────────────
        self._input_dim: int = self.cfg.get("model", {}).get("input_dim", 256)
        self._dropout: float = self.cfg.get("model", {}).get("dropout_rate", 0.3)
        self._num_rounds: int = self.cfg["federated"]["num_communication_rounds"]
        self._participation: float = self.cfg["federated"].get(
            "client_participation_rate", 1.0
        )
        train_cfg = self.cfg.get("training", {})
        self._local_epochs: int = train_cfg.get("local_epochs", 5)
        self._batch_size: int = train_cfg.get("batch_size", 512)
        self._lr: float = train_cfg.get("learning_rate", 0.01)

        # ── evaluation config ─────────────────────────────────────────────────
        eval_cfg = self.cfg.get("evaluation", {})
        self._train_split: float = eval_cfg.get("train_split", 0.70)
        self._val_split: float = eval_cfg.get("val_split", 0.15)

        # ── build components ──────────────────────────────────────────────────
        self.server = FederatedServer(
            input_dim=self._input_dim,
            dropout_rate=self._dropout,
            device=self.device,
        )
        self.clients: List[FederatedClient] = []
        self.simulators: List[ClientDataSimulator] = []
        self._test_data: List[Tuple[np.ndarray, np.ndarray]] = []

        # ── metrics ───────────────────────────────────────────────────────────
        self.round_metrics: List[Dict[str, Any]] = []

        # Create result directories
        for sub in ["convergence_plots", "model_checkpoints", "metrics"]:
            os.makedirs(os.path.join(results_dir, sub), exist_ok=True)

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        Instantiate simulators, clients, and load each client's local data.

        Should be called once before :meth:`run`.
        """
        client_specs = self.cfg.get("clients", None)
        seed: int = self.cfg.get("federated", {}).get("seed", 42)

        self.simulators = build_all_simulators(client_specs=client_specs, seed=seed)

        for sim in self.simulators:
            log.info(
                "Generating data for client %d (%d windows, %.0f BPM, %s) …",
                sim.client_id, sim.num_windows, sim.heart_rate, sim.signal_type,
            )
            (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = sim.train_val_test_split(
                train_frac=self._train_split,
                val_frac=self._val_split,
            )

            client = FederatedClient(
                client_id=sim.client_id,
                input_dim=self._input_dim,
                device=self.device,
            )
            client.load_data(X_tr, y_tr, X_va, y_va)
            self.clients.append(client)
            self._test_data.append((X_te, y_te))

        log.info(
            "Setup complete: %d clients, %d communication rounds.",
            len(self.clients), self._num_rounds,
        )

    # ── training loop ─────────────────────────────────────────────────────────

    def run(self) -> List[Dict[str, Any]]:
        """
        Execute the federated training loop.

        Returns
        -------
        round_metrics : list of dict
            Per-round metrics including global MAE, per-client MAE, drift,
            and wall-clock time.
        """
        if not self.clients:
            raise RuntimeError("Call setup() before run().")

        log.info("Starting federated training for %d rounds …", self._num_rounds)
        global_mae_history: List[float] = []

        for rnd in range(1, self._num_rounds + 1):
            t_start = time.perf_counter()

            # ── 1. Sample participating clients ─────────────────────────────
            selected = self._sample_clients()

            # ── 2. Broadcast global model → clients ──────────────────────────
            global_weights = self.server.get_global_weights()
            for client in selected:
                client.set_global_weights(global_weights)

            # ── 3. Local training ─────────────────────────────────────────────
            client_weights = []
            client_sizes = []
            client_train_maes: Dict[int, float] = {}
            for client in selected:
                train_mae = client.train(
                    epochs=self._local_epochs,
                    batch_size=self._batch_size,
                    learning_rate=self._lr,
                )
                client_weights.append(client.get_model_weights())
                client_sizes.append(client.num_train_samples)
                client_train_maes[client.client_id] = train_mae

            # ── 4. Server aggregation (weighted FedAvg) ───────────────────────
            self.server.aggregate(client_weights, client_sizes)

            # ── 5. Evaluate global model on each client's test set ────────────
            client_test_maes: Dict[int, float] = {}
            all_preds = []
            all_labels = []
            for client, (X_te, y_te) in zip(self.clients, self._test_data):
                X_t = torch.tensor(X_te, dtype=torch.float32).to(self.device)
                y_t = torch.tensor(y_te, dtype=torch.float32).to(self.device)
                # Evaluate using the freshly aggregated global model
                self.server.global_model.set_weights(self.server.get_global_weights())
                mae = self.server.evaluate_global(X_t, y_t)
                client_test_maes[client.client_id] = mae
                all_preds.append(
                    self.server.global_model.predict(X_t).cpu().numpy()
                )
                all_labels.append(y_te)

            global_mae = float(
                np.mean(
                    np.abs(
                        np.concatenate(all_preds) - np.concatenate(all_labels)
                    )
                )
            )
            global_mae_history.append(global_mae)

            drift = self.server.compute_drift()
            elapsed = time.perf_counter() - t_start

            metrics: Dict[str, Any] = {
                "round": rnd,
                "global_mae": global_mae,
                "client_test_maes": client_test_maes,
                "client_train_maes": client_train_maes,
                "model_drift": drift,
                "wall_time_s": elapsed,
                "num_clients": len(selected),
            }
            self.round_metrics.append(metrics)

            if rnd == 1 or rnd % 10 == 0 or rnd == self._num_rounds:
                log.info(
                    "Round %3d/%d | global MAE=%.4f | drift=%.4f | %.1fs",
                    rnd, self._num_rounds, global_mae, drift, elapsed,
                )

        # ── post-training: save artefacts ─────────────────────────────────────
        self._save_artefacts(global_mae_history)
        return self.round_metrics

    # ── internal helpers ──────────────────────────────────────────────────────

    def _sample_clients(self) -> List[FederatedClient]:
        """Return the subset of clients participating this round."""
        if self._participation >= 1.0:
            return self.clients
        k = max(1, int(len(self.clients) * self._participation))
        indices = np.random.choice(len(self.clients), k, replace=False)
        return [self.clients[i] for i in indices]

    def _save_artefacts(self, global_mae_history: List[float]) -> None:
        """Save convergence plot, metrics JSON, and final checkpoint."""
        rounds = list(range(1, len(global_mae_history) + 1))
        client_maes: Dict[int, List[float]] = {}
        for m in self.round_metrics:
            for cid, mae in m["client_test_maes"].items():
                client_maes.setdefault(cid, []).append(mae)

        plot_convergence(
            rounds,
            global_mae_history,
            client_maes=client_maes,
            save_path=os.path.join(
                self.results_dir, "convergence_plots", "federated_convergence.png"
            ),
        )

        save_json(
            self.round_metrics,
            os.path.join(self.results_dir, "metrics", "round_metrics.json"),
        )

        # Save global model checkpoint
        import torch

        ckpt_path = os.path.join(
            self.results_dir, "model_checkpoints", "global_model_final.pt"
        )
        torch.save(self.server.global_model.state_dict(), ckpt_path)
        log.info("Artefacts saved to %s", self.results_dir)
