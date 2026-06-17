"""
MLP model for fetal heart rate (FHR) prediction.

Architecture
------------
  Input (256) → FC(256→128) + ReLU + Dropout(0.3)
              → FC(128→ 64) + ReLU + Dropout(0.3)
              → FC( 64→ 32) + ReLU + Dropout(0.3)
              → FC( 32→  1) [linear output]

Loss: Mean Absolute Error (MAE)
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn


class FHRPredictionMLP(nn.Module):
    """
    Feedforward MLP for fetal heart rate regression.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input feature vector (default: 256).
    dropout_rate : float
        Dropout probability applied after each hidden layer (default: 0.3).
    """

    def __init__(self, input_dim: int = 256, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        self.network = nn.Sequential(
            # Hidden layer 1: 256 → 128
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Hidden layer 2: 128 → 64
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Hidden layer 3: 64 → 32
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Output layer: 32 → 1  (linear activation for regression)
            nn.Linear(32, 1),
        )

        self._init_weights()

    # ── weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Kaiming-normal initialisation for Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)

        Returns
        -------
        out : Tensor, shape (batch, 1)
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference (eval mode, no gradient computation).

        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)

        Returns
        -------
        out : Tensor, shape (batch,)  – squeezed scalar per sample
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x).squeeze(1)

    # ── serialisation ─────────────────────────────────────────────────────────

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return a deep copy of the model's ``state_dict``."""
        return copy.deepcopy(self.state_dict())

    def set_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load weights from a ``state_dict``."""
        self.load_state_dict(copy.deepcopy(state_dict))

    # ── convenience ───────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"FHRPredictionMLP(input_dim={self.input_dim}, "
            f"dropout={self.dropout_rate}, params={params:,})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

class MAELoss(nn.Module):
    """
    Mean Absolute Error loss.

    .. math::
        \\mathcal{L}_{\\text{MAE}} = \\frac{1}{N} \\sum_{i=1}^{N}
        \\left| f(\\mathbf{x}_i; \\boldsymbol{\\theta}) - y_i \\right|
    """

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : Tensor, shape (N,) or (N, 1)
        targets : Tensor, shape (N,)

        Returns
        -------
        loss : scalar Tensor
        """
        return torch.mean(torch.abs(predictions.squeeze() - targets))


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    input_dim: int = 256,
    dropout_rate: float = 0.3,
    device: Optional[torch.device] = None,
) -> FHRPredictionMLP:
    """
    Build and return an :class:`FHRPredictionMLP` on the specified device.

    Parameters
    ----------
    input_dim : int
    dropout_rate : float
    device : torch.device, optional – defaults to CPU

    Returns
    -------
    FHRPredictionMLP
    """
    if device is None:
        device = torch.device("cpu")
    model = FHRPredictionMLP(input_dim=input_dim, dropout_rate=dropout_rate)
    return model.to(device)
