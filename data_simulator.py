"""
data_simulator.py
=================
Generates realistic, non-IID simulated data for 10 federated clients.

Each client has:
  - A specific number of feature windows (samples)
  - A dominant fetal heart rate (BPM)
  - A signal-quality profile (Band-pass, Wavelet, EMD, ICA)

The non-IID nature is introduced via:
  - Different mean / std of the target heart rate per client
  - Different SNR / noise levels in the feature vectors
  - Different feature distributions modelling real sensor heterogeneity
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# ── Client specification table ────────────────────────────────────────────────

@dataclass
class ClientSpec:
    client_id: int
    n_samples: int           # feature windows
    dominant_hr: float       # mean fetal heart rate (BPM)
    signal_quality: str      # band_pass | wavelet | emd | ica
    hr_std: float = 5.0      # intra-client heart-rate std (BPM)
    noise_scale: float = 0.1 # additive feature noise scale

    # Signal-quality → noise multiplier mapping
    _quality_noise: Dict[str, float] = field(default_factory=lambda: {
        "band_pass": 1.0,
        "wavelet":   0.8,
        "emd":       1.2,
        "ica":       0.9,
    }, init=False, repr=False)

    @property
    def effective_noise(self) -> float:
        return self.noise_scale * self._quality_noise.get(self.signal_quality, 1.0)


# Default 10-client configuration from the research table
DEFAULT_CLIENT_SPECS: List[ClientSpec] = [
    ClientSpec(1,  70000, 140, "band_pass"),
    ClientSpec(2,  55000, 144, "wavelet"),
    ClientSpec(3,  48000, 135, "emd"),
    ClientSpec(4,  40000, 130, "ica"),
    ClientSpec(5,  30000, 146, "band_pass"),
    ClientSpec(6,  25000, 142, "wavelet"),
    ClientSpec(7,  20000, 148, "emd"),
    ClientSpec(8,  15000, 156, "ica"),
    ClientSpec(9,  10000, 150, "band_pass"),
    ClientSpec(10,  5800, 138, "wavelet"),
]


# ── Data generation ──────────────────────────────────────────────────────────

def _generate_features(
    n_samples: int,
    n_features: int,
    dominant_hr: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate n_features-dimensional feature vectors correlated with heart rate.

    Features are constructed from:
      - A set of 'informative' bases correlated with heart rate
      - Additive Gaussian noise whose scale depends on signal quality
    """
    # Normalised heart-rate signal for this client (used as latent variable)
    hr_latent = rng.normal(dominant_hr, 5.0, size=n_samples)          # (N,)

    # Base feature matrix: first half informative, second half noise
    n_info = n_features // 2
    # Informative features: linear + sinusoidal functions of hr_latent
    X_info = np.column_stack([
        np.sin(hr_latent / (10 * (k + 1)) * np.pi) + rng.normal(0, noise_scale, n_samples)
        for k in range(n_info)
    ])
    # Uninformative (noise) features
    X_noise = rng.normal(0, noise_scale * 2, size=(n_samples, n_features - n_info))

    X = np.hstack([X_info, X_noise]).astype(np.float32)
    return X, hr_latent.astype(np.float32)


def generate_client_data(
    spec: ClientSpec,
    n_features: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (X, y) for a single client.

    Parameters
    ----------
    spec       : ClientSpec describing the client
    n_features : number of input features (must be consistent across clients)
    seed       : base random seed (offset by client_id for reproducibility)

    Returns
    -------
    X : float32 array of shape (n_samples, n_features)
    y : float32 array of shape (n_samples,)  – fetal heart rate in BPM
    """
    rng = np.random.default_rng(seed + spec.client_id)
    X, y = _generate_features(
        n_samples=spec.n_samples,
        n_features=n_features,
        dominant_hr=spec.dominant_hr,
        noise_scale=spec.effective_noise,
        rng=rng,
    )
    # Clip to physiological range 100–200 BPM
    y = np.clip(y, 100.0, 200.0)
    return X, y


def generate_all_clients(
    client_specs: List[ClientSpec] | None = None,
    n_features: int = 20,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return a list of (X, y) tuples, one per client.

    Parameters
    ----------
    client_specs : list of ClientSpec; defaults to DEFAULT_CLIENT_SPECS
    n_features   : feature dimensionality
    seed         : base random seed

    Returns
    -------
    list of (X, y) tuples indexed 0–9 (client 1 is index 0)
    """
    if client_specs is None:
        client_specs = DEFAULT_CLIENT_SPECS
    return [
        generate_client_data(spec, n_features=n_features, seed=seed)
        for spec in client_specs
    ]


def get_client_specs(config: dict | None = None) -> List[ClientSpec]:
    """
    Build ClientSpec list from a config dict (loaded from config.yaml)
    or return the default list.
    """
    if config is None:
        return DEFAULT_CLIENT_SPECS

    specs = []
    for c in config.get("clients", []):
        specs.append(ClientSpec(
            client_id=c["id"],
            n_samples=c["feature_windows"],
            dominant_hr=c["dominant_hr"],
            signal_quality=c["signal_quality"],
        ))
    return specs or DEFAULT_CLIENT_SPECS


if __name__ == "__main__":
    datasets = generate_all_clients()
    for i, (X, y) in enumerate(datasets, start=1):
        print(f"Client {i:2d}: X={X.shape}  y range=[{y.min():.1f}, {y.max():.1f}]  "
              f"y mean={y.mean():.1f} BPM")
