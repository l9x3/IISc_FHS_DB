"""
Federated learning package for fetal heart rate prediction.
"""

from .model import FHRPredictionMLP
from .data_simulator import ClientDataSimulator
from .client import FederatedClient
from .server import FederatedServer
from .federated_trainer import FederatedTrainer
from .evaluation import FederatedEvaluator
from .utils import (
    compute_mae,
    compute_rmse,
    compute_r2,
    compute_mape,
    load_config,
)

__all__ = [
    "FHRPredictionMLP",
    "ClientDataSimulator",
    "FederatedClient",
    "FederatedServer",
    "FederatedTrainer",
    "FederatedEvaluator",
    "compute_mae",
    "compute_rmse",
    "compute_r2",
    "compute_mape",
    "load_config",
]
