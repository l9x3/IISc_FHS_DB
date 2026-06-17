"""
client_model.py
===============
Individual federated client: holds local data, builds the baseline MLP,
and performs local SGD training returning updated weights.

Model architecture (from the paper):
  Input → Dense(256, ReLU) → Dropout(0.3)
        → Dense(128, ReLU) → Dropout(0.3)
        → Dense(64,  ReLU) → Dropout(0.3)
        → Dense(32,  ReLU)
        → Dense(1,   linear)

Loss : MAE
Optimiser : SGD (momentum 0.9)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── MLP definition ────────────────────────────────────────────────────────────

class FetalHRNet(nn.Module):
    """Baseline 4-layer MLP for fetal heart-rate regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] | None = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        layers: List[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            in_dim = h_dim
        # Output layer – linear activation
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_model(
    input_dim: int,
    hidden_dims: List[int] | None = None,
    dropout_rate: float = 0.3,
) -> FetalHRNet:
    """Factory for FetalHRNet."""
    return FetalHRNet(input_dim, hidden_dims, dropout_rate)


# ── Client ────────────────────────────────────────────────────────────────────

class FederatedClient:
    """
    Encapsulates one federated participant.

    Parameters
    ----------
    client_id    : integer identifier
    X            : feature matrix (n_samples, n_features), float32 numpy array
    y            : target vector  (n_samples,),             float32 numpy array
    input_dim    : number of input features
    hidden_dims  : layer widths for the MLP
    dropout_rate : dropout probability applied after each hidden layer
    batch_size   : mini-batch size for local SGD
    lr           : SGD learning rate
    momentum     : SGD momentum
    local_epochs : number of local training epochs per FL round
    device       : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
        input_dim: int = 20,
        hidden_dims: List[int] | None = None,
        dropout_rate: float = 0.3,
        batch_size: int = 512,
        lr: float = 0.01,
        momentum: float = 0.9,
        local_epochs: int = 5,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.local_epochs = local_epochs

        # Build model
        self.model = build_model(input_dim, hidden_dims, dropout_rate).to(self.device)

        # DataLoader
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.n_samples = len(X)

    # ── Weight access ─────────────────────────────────────────────────────────

    def get_weights(self) -> List[np.ndarray]:
        """Return current model parameters as a list of numpy arrays."""
        return [p.detach().cpu().numpy().copy() for p in self.model.parameters()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Load new model parameters from a list of numpy arrays."""
        with torch.no_grad():
            for param, w in zip(self.model.parameters(), weights):
                param.copy_(torch.tensor(w, dtype=torch.float32))

    # ── Local training ────────────────────────────────────────────────────────

    def train_local(self) -> Tuple[List[np.ndarray], float]:
        """
        Run local_epochs epochs of mini-batch SGD.

        Returns
        -------
        weights   : updated model parameters
        avg_loss  : mean MAE over the last local epoch
        """
        self.model.train()
        optimiser = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )
        criterion = nn.L1Loss()  # MAE

        epoch_losses: List[float] = []
        for _ in range(self.local_epochs):
            batch_losses: List[float] = []
            for X_batch, y_batch in self.loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimiser.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimiser.step()
                batch_losses.append(loss.item())
            epoch_losses.append(float(np.mean(batch_losses)))

        avg_loss = float(np.mean(epoch_losses[-1:]))  # last-epoch loss
        return self.get_weights(), avg_loss

    def evaluate(self) -> float:
        """Evaluate MAE on all local data (no gradient)."""
        self.model.eval()
        criterion = nn.L1Loss()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in self.loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = self.model(X_batch)
                total_loss += criterion(preds, y_batch).item()
                n_batches += 1
        return total_loss / max(n_batches, 1)
