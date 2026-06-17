"""
server_aggregator.py
====================
Server-side parameter aggregation for Federated Learning.

Implements:
  - FedAvg  : weighted average of client model parameters
  - FedMedian (optional robust aggregation)

Reference: McMahan et al., "Communication-Efficient Learning of Deep
Networks from Decentralized Data", AISTATS 2017.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


def fedavg(
    client_weights: List[List[np.ndarray]],
    client_sample_counts: List[int],
) -> List[np.ndarray]:
    """
    Federated Averaging (FedAvg).

    Computes a weighted average of model parameters, where each client's
    contribution is proportional to its local dataset size.

    Parameters
    ----------
    client_weights      : list of weight lists, one per client.
                          Each element is a list of numpy arrays (one per
                          model parameter tensor).
    client_sample_counts: number of training samples for each client,
                          used to weight the contribution.

    Returns
    -------
    aggregated_weights : list of numpy arrays representing the averaged
                         global model parameters.
    """
    if not client_weights:
        raise ValueError("client_weights must not be empty")

    total_samples = sum(client_sample_counts)
    if total_samples == 0:
        raise ValueError("Total sample count must be positive")

    # Initialise accumulated weights to zero (same shape as first client)
    aggregated = [np.zeros_like(w) for w in client_weights[0]]

    for weights, n_samples in zip(client_weights, client_sample_counts):
        weight_factor = n_samples / total_samples
        for agg_param, client_param in zip(aggregated, weights):
            agg_param += weight_factor * client_param

    return aggregated


def fedmedian(
    client_weights: List[List[np.ndarray]],
) -> List[np.ndarray]:
    """
    Coordinate-wise median aggregation (robust to outliers / Byzantine clients).

    Parameters
    ----------
    client_weights : list of weight lists, one per participating client.

    Returns
    -------
    median_weights : list of numpy arrays.
    """
    if not client_weights:
        raise ValueError("client_weights must not be empty")

    n_layers = len(client_weights[0])
    aggregated = []
    for layer_idx in range(n_layers):
        stacked = np.stack([w[layer_idx] for w in client_weights], axis=0)
        aggregated.append(np.median(stacked, axis=0))
    return aggregated


class ServerAggregator:
    """
    Maintains the global model state and applies FedAvg aggregation each round.

    Parameters
    ----------
    initial_weights : initial global model parameter list
    strategy        : 'fedavg' (default) or 'fedmedian'
    """

    def __init__(
        self,
        initial_weights: List[np.ndarray],
        strategy: str = "fedavg",
    ):
        self.global_weights = [w.copy() for w in initial_weights]
        self.strategy = strategy
        self.round_num = 0

    def aggregate(
        self,
        client_weights: List[List[np.ndarray]],
        client_sample_counts: List[int],
    ) -> List[np.ndarray]:
        """
        Perform one aggregation step.

        Parameters
        ----------
        client_weights       : updated weights from each participating client
        client_sample_counts : dataset sizes used for weighting (FedAvg only)

        Returns
        -------
        new global weights (also stored in self.global_weights)
        """
        if self.strategy == "fedmedian":
            new_weights = fedmedian(client_weights)
        else:
            new_weights = fedavg(client_weights, client_sample_counts)

        self.global_weights = new_weights
        self.round_num += 1
        return self.global_weights

    def get_weights(self) -> List[np.ndarray]:
        return [w.copy() for w in self.global_weights]
