"""
Central federated server implementing weighted FedAvg aggregation.

The server holds the global model, collects local updates from all
participating clients, aggregates them using weighted averaging (weighted
by each client's dataset size), and broadcasts the result.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch

from .model import FHRPredictionMLP, build_model
from .logger import get_logger
from .utils import compute_model_drift

log = get_logger(__name__, log_dir="")


class FederatedServer:
    """
    Central aggregation server for federated learning.

    Parameters
    ----------
    input_dim : int
        Input feature dimension of the global model.
    dropout_rate : float
        Dropout rate used in the global model.
    device : torch.device, optional
        Device for the global model (defaults to CPU).
    """

    def __init__(
        self,
        input_dim: int = 256,
        dropout_rate: float = 0.3,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.global_model: FHRPredictionMLP = build_model(
            input_dim=input_dim,
            dropout_rate=dropout_rate,
            device=self.device,
        )
        self._round: int = 0
        self._prev_weights: Optional[Dict[str, torch.Tensor]] = None

    # ── round counter ─────────────────────────────────────────────────────────

    @property
    def current_round(self) -> int:
        return self._round

    # ── weight broadcasting ───────────────────────────────────────────────────

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Return a deep copy of the current global model state-dict."""
        return copy.deepcopy(self.global_model.state_dict())

    # ── aggregation ───────────────────────────────────────────────────────────

    def aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Weighted FedAvg aggregation.

        .. math::
            \\theta_{\\text{global}} = \\sum_{k=1}^{K}
            \\frac{n_k}{n_{\\text{total}}} \\theta_k

        Parameters
        ----------
        client_weights : list of state_dicts
            One state-dict per participating client.
        client_sizes : list of int
            Number of training samples for each client (must be same length
            as ``client_weights``).

        Returns
        -------
        aggregated : state_dict
            New global model weights.

        Raises
        ------
        ValueError
            If ``client_weights`` and ``client_sizes`` have different lengths
            or ``client_weights`` is empty.
        """
        if not client_weights:
            raise ValueError("client_weights must not be empty.")
        if len(client_weights) != len(client_sizes):
            raise ValueError(
                "client_weights and client_sizes must have the same length."
            )

        total = sum(client_sizes)
        if total <= 0:
            raise ValueError("Total client sample count must be positive.")

        # Store previous weights for drift computation
        self._prev_weights = self.get_global_weights()

        # Compute weighted sum of state-dicts
        aggregated: Dict[str, torch.Tensor] = {}
        for key in client_weights[0]:
            weighted_sum = torch.zeros_like(client_weights[0][key], dtype=torch.float32)
            for weights, size in zip(client_weights, client_sizes):
                weighted_sum += weights[key].float() * (size / total)
            aggregated[key] = weighted_sum

        # Update global model
        self.global_model.set_weights(aggregated)
        self._round += 1

        log.debug(
            "Round %d: aggregated %d clients (total=%d samples)",
            self._round, len(client_weights), total,
        )
        return copy.deepcopy(aggregated)

    # ── metrics ───────────────────────────────────────────────────────────────

    def compute_drift(self) -> float:
        """
        Return the model drift (Frobenius-norm distance) between the
        current and previous global model weights.  Returns 0.0 before
        the first aggregation.
        """
        if self._prev_weights is None:
            return 0.0
        return compute_model_drift(self._prev_weights, self.get_global_weights())

    def evaluate_global(
        self,
        X: "torch.Tensor",
        y: "torch.Tensor",
    ) -> float:
        """
        Evaluate the global model on a held-out test set.

        Parameters
        ----------
        X : Tensor, shape (n, input_dim)
        y : Tensor, shape (n,)

        Returns
        -------
        mae : float
        """
        X = X.to(self.device)
        y = y.to(self.device)
        preds = self.global_model.predict(X)
        return float(torch.mean(torch.abs(preds - y)).item())

    # ── convenience ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"FederatedServer(round={self._round}, "
            f"model={self.global_model})"
        )
