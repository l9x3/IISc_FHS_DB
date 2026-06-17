"""
Federated client: local data management and training.

Each client holds a private local dataset and trains an MLP using
mini-batch SGD with MAE loss.  Only model weights are shared with the
central server – raw data never leaves the client.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import FHRPredictionMLP, MAELoss, build_model
from .logger import get_logger

log = get_logger(__name__, log_dir="")


class FederatedClient:
    """
    Represents a single federated learning participant.

    Parameters
    ----------
    client_id : int
        Unique client identifier.
    input_dim : int
        Feature dimension of each data sample (must match global model).
    device : torch.device, optional
        Compute device (CPU/CUDA).  Defaults to CPU.
    """

    def __init__(
        self,
        client_id: int,
        input_dim: int = 256,
        device: Optional[torch.device] = None,
    ) -> None:
        self.client_id = client_id
        self.input_dim = input_dim
        self.device = device or torch.device("cpu")

        self.model: FHRPredictionMLP = build_model(input_dim=input_dim, device=self.device)
        self.loss_fn = MAELoss()

        # Training data (set via load_data)
        self._X_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None
        self._X_val: Optional[torch.Tensor] = None
        self._y_val: Optional[torch.Tensor] = None

        # Metrics history
        self.train_losses: List[float] = []
        self.val_maes: List[float] = []

    # ── data loading ──────────────────────────────────────────────────────────

    def load_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        Store local training (and optional validation) data.

        Parameters
        ----------
        X_train : ndarray, shape (n_train, input_dim)
        y_train : ndarray, shape (n_train,)
        X_val : ndarray, shape (n_val, input_dim), optional
        y_val : ndarray, shape (n_val,), optional
        """
        self._X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self._y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        if X_val is not None and y_val is not None:
            self._X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            self._y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        log.debug(
            "Client %d: loaded %d train / %d val samples",
            self.client_id,
            len(X_train),
            len(X_val) if X_val is not None else 0,
        )

    @property
    def num_train_samples(self) -> int:
        """Number of local training samples."""
        if self._X_train is None:
            return 0
        return self._X_train.shape[0]

    # ── weight management ─────────────────────────────────────────────────────

    def set_global_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Overwrite local model weights with the global model's state-dict."""
        self.model.set_weights(state_dict)

    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Return a copy of the local model's state-dict."""
        return self.model.get_weights()

    # ── training ──────────────────────────────────────────────────────────────

    def train(
        self,
        epochs: int = 5,
        batch_size: int = 512,
        learning_rate: float = 0.01,
    ) -> float:
        """
        Train the local model for *epochs* epochs using mini-batch SGD.

        Parameters
        ----------
        epochs : int
            Number of local epochs to run before returning weights to server.
        batch_size : int
            Mini-batch size (128, 256, or 512 per specification).
        learning_rate : float
            SGD learning rate.

        Returns
        -------
        avg_train_loss : float
            Average MAE over the last training epoch.

        Raises
        ------
        RuntimeError
            If :meth:`load_data` has not been called first.
        """
        if self._X_train is None:
            raise RuntimeError(
                f"Client {self.client_id}: call load_data() before train()."
            )

        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        dataset = TensorDataset(self._X_train, self._y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        last_epoch_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.model(X_batch).squeeze(1)
                loss = self.loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            last_epoch_loss = epoch_loss / max(num_batches, 1)

        self.train_losses.append(last_epoch_loss)
        log.debug(
            "Client %d: trained %d epoch(s), train MAE=%.4f",
            self.client_id, epochs, last_epoch_loss,
        )
        return last_epoch_loss

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> float:
        """
        Evaluate the local model on *X* / *y* or the stored validation set.

        Parameters
        ----------
        X : ndarray, shape (n, input_dim), optional
            If ``None``, uses the stored validation data.
        y : ndarray, shape (n,), optional

        Returns
        -------
        mae : float
        """
        if X is not None and y is not None:
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        elif self._X_val is not None and self._y_val is not None:
            X_t, y_t = self._X_val, self._y_val
        else:
            raise RuntimeError(
                f"Client {self.client_id}: no validation data available. "
                "Pass X/y or call load_data() with val arrays."
            )

        preds = self.model.predict(X_t)
        mae = float(torch.mean(torch.abs(preds - y_t)).item())
        self.val_maes.append(mae)
        return mae

    # ── convenience ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"FederatedClient(id={self.client_id}, "
            f"samples={self.num_train_samples}, device={self.device})"
        )
