"""
Evaluation framework for the federated learning system.

Provides:
  - Group k-fold cross-validation per client
  - Per-round metrics tracking (MAE, RMSE, R², MAPE)
  - Convergence-rate estimation
  - Communication-volume estimation
  - Comparison: centralised vs. federated learning
  - CSV / JSON export
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from .logger import get_logger
from .model import FHRPredictionMLP, build_model
from .utils import (
    compute_all_metrics,
    compute_mae,
    compute_r2,
    compute_rmse,
    save_json,
)

log = get_logger(__name__, log_dir="")


class FederatedEvaluator:
    """
    Comprehensive evaluator for a federated learning experiment.

    Parameters
    ----------
    results_dir : str
        Root directory where all evaluation artefacts are written.
    kfold : int
        Number of folds for cross-validation (default: 5).
    device : torch.device, optional
        Torch device for inference.
    """

    def __init__(
        self,
        results_dir: str = "results",
        kfold: int = 5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.results_dir = results_dir
        self.kfold = kfold
        self.device = device or torch.device("cpu")
        os.makedirs(results_dir, exist_ok=True)

    # ── per-round tracking ────────────────────────────────────────────────────

    def summarise_rounds(
        self, round_metrics: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Flatten a list of per-round metric dicts into a tidy DataFrame.

        Parameters
        ----------
        round_metrics : list of dict
            As returned by :meth:`FederatedTrainer.run`.

        Returns
        -------
        df : DataFrame  – one row per round
        """
        rows = []
        for m in round_metrics:
            row: Dict[str, Any] = {
                "round": m["round"],
                "global_mae": m["global_mae"],
                "model_drift": m.get("model_drift", float("nan")),
                "wall_time_s": m.get("wall_time_s", float("nan")),
            }
            # Per-client test MAE
            for cid, mae in m.get("client_test_maes", {}).items():
                row[f"client_{cid}_test_mae"] = mae
            # Per-client train MAE
            for cid, mae in m.get("client_train_maes", {}).items():
                row[f"client_{cid}_train_mae"] = mae
            rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def compute_convergence_rate(
        self, global_mae_history: List[float], window: int = 5
    ) -> float:
        """
        Estimate convergence rate as the average relative decrease in MAE
        over the last *window* rounds.

        Returns
        -------
        rate : float  (positive = improving)
        """
        if len(global_mae_history) < 2:
            return 0.0
        tail = global_mae_history[-window:]
        if len(tail) < 2 or tail[0] == 0:
            return 0.0
        return float((tail[0] - tail[-1]) / (abs(tail[0]) + 1e-8))

    def estimate_communication_bytes(
        self,
        model: FHRPredictionMLP,
        num_rounds: int,
        num_clients: int,
    ) -> int:
        """
        Estimate total bytes transferred (upload + download) over training.

        Assumes 32-bit (4-byte) parameters with no compression.

        Returns
        -------
        total_bytes : int
        """
        param_bytes = sum(p.numel() * 4 for p in model.parameters())
        # Each round: server → all clients (broadcast) + all clients → server
        return param_bytes * num_rounds * num_clients * 2

    # ── group k-fold CV per client ────────────────────────────────────────────

    def kfold_crossval(
        self,
        X: np.ndarray,
        y: np.ndarray,
        input_dim: int = 256,
        dropout_rate: float = 0.3,
        epochs: int = 20,
        batch_size: int = 512,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run group k-fold cross-validation on a single client's dataset.

        Parameters
        ----------
        X : ndarray, shape (n, input_dim)
        y : ndarray, shape (n,)
        input_dim, dropout_rate, epochs, batch_size, learning_rate, seed :
            Model and training hyperparameters.

        Returns
        -------
        results : dict with keys ``fold_metrics`` and ``summary``
        """
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        from .model import MAELoss

        kf = KFold(n_splits=self.kfold, shuffle=True, random_state=seed)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            model = build_model(input_dim=input_dim, dropout_rate=dropout_rate)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            loss_fn = MAELoss()

            X_tr = torch.tensor(X[train_idx], dtype=torch.float32)
            y_tr = torch.tensor(y[train_idx], dtype=torch.float32)
            X_te = torch.tensor(X[test_idx], dtype=torch.float32)
            y_te = y[test_idx]

            loader = DataLoader(
                TensorDataset(X_tr, y_tr),
                batch_size=batch_size, shuffle=True,
            )
            model.train()
            for _ in range(epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss_fn(model(xb).squeeze(1), yb).backward()
                    optimizer.step()

            preds = model.predict(X_te).numpy()
            metrics = compute_all_metrics(y_te, preds, prefix="test")
            metrics["fold"] = fold_idx
            fold_metrics.append(metrics)

        df = pd.DataFrame(fold_metrics).set_index("fold")
        summary = {
            col: {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=1))}
            for col in df.columns
        }
        return {"fold_metrics": df.to_dict(orient="index"), "summary": summary}

    # ── centralised baseline ──────────────────────────────────────────────────

    def centralised_baseline(
        self,
        all_X: List[np.ndarray],
        all_y: List[np.ndarray],
        input_dim: int = 256,
        dropout_rate: float = 0.3,
        epochs: int = 20,
        batch_size: int = 512,
        learning_rate: float = 0.01,
        test_fraction: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Train a single centralised model on all pooled data (simulating
        full data sharing) and return test metrics.

        Parameters
        ----------
        all_X, all_y : lists of per-client arrays
        Returns
        -------
        metrics : dict with keys ``mae``, ``rmse``, ``r2``, ``mape``
        """
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        from .model import MAELoss

        X_pool = np.concatenate(all_X, axis=0)
        y_pool = np.concatenate(all_y, axis=0)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y_pool))
        n_test = max(1, int(len(y_pool) * test_fraction))
        train_idx, test_idx = idx[n_test:], idx[:n_test]

        model = build_model(input_dim=input_dim, dropout_rate=dropout_rate)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = MAELoss()

        X_tr = torch.tensor(X_pool[train_idx], dtype=torch.float32)
        y_tr = torch.tensor(y_pool[train_idx], dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
        )
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(model(xb).squeeze(1), yb).backward()
                optimizer.step()

        X_te = torch.tensor(X_pool[test_idx], dtype=torch.float32)
        preds = model.predict(X_te).numpy()
        return compute_all_metrics(y_pool[test_idx], preds)

    # ── export helpers ────────────────────────────────────────────────────────

    def export_round_metrics(
        self,
        round_metrics: List[Dict[str, Any]],
        filename: str = "metrics/round_metrics.csv",
    ) -> str:
        """Save per-round metrics to CSV; return the file path."""
        df = self.summarise_rounds(round_metrics)
        path = os.path.join(self.results_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        log.info("Round metrics saved → %s", path)
        return path

    def export_summary(
        self,
        summary: Dict[str, Any],
        filename: str = "metrics/summary.json",
    ) -> str:
        """Save a summary dict to JSON; return the file path."""
        path = os.path.join(self.results_dir, filename)
        save_json(summary, path)
        log.info("Summary saved → %s", path)
        return path

    # ── distribution statistics ───────────────────────────────────────────────

    @staticmethod
    def client_data_stats(
        client_id: int,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Return descriptive statistics for a client's label distribution.

        Parameters
        ----------
        client_id : int
        y : ndarray of HR labels

        Returns
        -------
        stats : dict
        """
        return {
            "client_id": client_id,
            "n": len(y),
            "mean_hr": float(np.mean(y)),
            "std_hr": float(np.std(y)),
            "min_hr": float(np.min(y)),
            "max_hr": float(np.max(y)),
            "p25_hr": float(np.percentile(y, 25)),
            "p50_hr": float(np.percentile(y, 50)),
            "p75_hr": float(np.percentile(y, 75)),
        }
