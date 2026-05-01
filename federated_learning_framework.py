"""
federated_learning_framework.py
================================
Orchestrates the full Federated Learning experiment:

  1. Initialise global model on the server.
  2. For each communication round:
     a. Broadcast global weights to (sampled) clients.
     b. Each client performs local_epochs of SGD.
     c. Server aggregates updated weights via FedAvg.
     d. Evaluate global model on every client.
  3. Record per-round metrics and return a results dict.
"""

from __future__ import annotations

import os
import random
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from client_model import FederatedClient, build_model
from server_aggregator import ServerAggregator
from data_simulator import (
    generate_all_clients,
    get_client_specs,
    DEFAULT_CLIENT_SPECS,
)

log = logging.getLogger(__name__)


# ── FL Orchestrator ───────────────────────────────────────────────────────────

class FederatedLearning:
    """
    End-to-end Federated Learning simulation.

    Parameters
    ----------
    config : dict loaded from config.yaml (or None to use defaults)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}

        fl_cfg     = cfg.get("federated_learning", {})
        model_cfg  = cfg.get("model", {})
        train_cfg  = cfg.get("training", {})
        paths_cfg  = cfg.get("paths", {})

        # Hyper-parameters
        self.num_clients      = fl_cfg.get("num_clients", 10)
        self.num_rounds       = fl_cfg.get("num_rounds", 50)
        self.clients_per_round = fl_cfg.get("clients_per_round", 10)
        self.local_epochs     = fl_cfg.get("local_epochs", 5)
        self.seed             = fl_cfg.get("seed", 42)

        self.input_dim    = model_cfg.get("input_dim", 20)
        self.hidden_dims  = model_cfg.get("hidden_dims", [256, 128, 64, 32])
        self.dropout_rate = model_cfg.get("dropout_rate", 0.3)

        self.batch_size = train_cfg.get("batch_size", 512)
        self.lr         = train_cfg.get("learning_rate", 0.01)
        self.momentum   = train_cfg.get("momentum", 0.9)

        self.results_dir = paths_cfg.get("results_dir", "results/federated")
        os.makedirs(self.results_dir, exist_ok=True)

        # Reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Build clients
        client_specs = get_client_specs(cfg)
        datasets = generate_all_clients(
            client_specs=client_specs,
            n_features=self.input_dim,
            seed=self.seed,
        )

        self.clients: List[FederatedClient] = []
        for spec, (X, y) in zip(client_specs, datasets):
            client = FederatedClient(
                client_id=spec.client_id,
                X=X,
                y=y,
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                dropout_rate=self.dropout_rate,
                batch_size=self.batch_size,
                lr=self.lr,
                momentum=self.momentum,
                local_epochs=self.local_epochs,
            )
            self.clients.append(client)

        # Build global model & server aggregator
        global_model = build_model(self.input_dim, self.hidden_dims, self.dropout_rate)
        initial_weights = [p.detach().cpu().numpy().copy()
                           for p in global_model.parameters()]
        self.aggregator = ServerAggregator(initial_weights, strategy="fedavg")

        # Initialise all clients with the same global weights
        for client in self.clients:
            client.set_weights(initial_weights)

        # Metrics storage
        self.round_server_loss: List[float] = []          # mean MAE across clients per round
        self.round_client_losses: List[List[float]] = []  # per-client MAE per round
        self.client_specs = client_specs

        log.info(
            "FL initialised: %d clients | %d rounds | %d local epochs | "
            "lr=%.4f | batch=%d",
            self.num_clients, self.num_rounds, self.local_epochs, self.lr, self.batch_size,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Execute num_rounds communication rounds and return metrics dict."""
        for rnd in range(1, self.num_rounds + 1):
            # Sample clients for this round
            sampled = self._sample_clients()

            # Broadcast global weights, train locally, collect updates
            global_w = self.aggregator.get_weights()
            client_weights_list: List[List[np.ndarray]] = []
            sample_counts: List[int] = []

            for client in sampled:
                client.set_weights(global_w)
                new_weights, _ = client.train_local()
                client_weights_list.append(new_weights)
                sample_counts.append(client.n_samples)

            # Aggregate
            new_global_w = self.aggregator.aggregate(client_weights_list, sample_counts)

            # Evaluate all clients with new global model
            client_losses: List[float] = []
            for client in self.clients:
                client.set_weights(new_global_w)
                loss = client.evaluate()
                client_losses.append(loss)

            mean_loss = float(np.mean(client_losses))
            self.round_server_loss.append(mean_loss)
            self.round_client_losses.append(client_losses)

            if rnd % 10 == 0 or rnd == 1:
                log.info("Round %3d/%d | Global MAE: %.4f", rnd, self.num_rounds, mean_loss)

        return self._build_results()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sample_clients(self) -> List[FederatedClient]:
        k = min(self.clients_per_round, len(self.clients))
        return random.sample(self.clients, k)

    def _build_results(self) -> Dict:
        return {
            "num_rounds": self.num_rounds,
            "num_clients": self.num_clients,
            "server_loss": self.round_server_loss,
            "client_losses": self.round_client_losses,   # list[list[float]]
            "client_specs": [
                {
                    "client_id":     spec.client_id,
                    "n_samples":     spec.n_samples,
                    "dominant_hr":   spec.dominant_hr,
                    "signal_quality": spec.signal_quality,
                }
                for spec in self.client_specs
            ],
            "config": {
                "batch_size":    self.batch_size,
                "learning_rate": self.lr,
                "local_epochs":  self.local_epochs,
                "hidden_dims":   self.hidden_dims,
                "dropout_rate":  self.dropout_rate,
            },
        }


# ── Hyperparameter sweep helper ───────────────────────────────────────────────

def run_hyperparam_sweep(
    base_config: Optional[dict] = None,
    batch_sizes: List[int] | None = None,
    learning_rates: List[float] | None = None,
    num_rounds: int = 20,
) -> Dict:
    """
    Run a grid sweep over batch sizes × learning rates.

    Returns a dict mapping (batch_size, lr) → final server MAE.
    """
    if batch_sizes is None:
        batch_sizes = [128, 256, 512]
    if learning_rates is None:
        learning_rates = [1e-4, 1e-3, 1e-2]

    sweep_results: Dict[Tuple[int, float], float] = {}

    for bs in batch_sizes:
        for lr in learning_rates:
            cfg = dict(base_config or {})
            cfg.setdefault("federated_learning", {})["num_rounds"] = num_rounds
            cfg.setdefault("training", {})["batch_size"] = bs
            cfg["training"]["learning_rate"] = lr

            fl = FederatedLearning(cfg)
            results = fl.run()
            final_loss = results["server_loss"][-1]
            sweep_results[(bs, lr)] = final_loss
            log.info("Sweep bs=%d lr=%.4f → final MAE=%.4f", bs, lr, final_loss)

    return sweep_results
