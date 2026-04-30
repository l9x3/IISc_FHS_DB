"""
Utility helpers for the federated learning system.

Includes:
  - Configuration loading
  - Metric computation (MAE, RMSE, R², MAPE)
  - Signal preprocessing stubs (band-pass, wavelet, EMD, ICA)
  - Model checkpoint management
  - Visualisation helpers
"""

from __future__ import annotations

import json
import os
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def save_json(data: Any, path: str) -> None:
    """Serialise *data* to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=_json_serialiser)


def _json_serialiser(obj: Any) -> Any:
    """Fallback JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ─────────────────────────────────────────────────────────────────────────────
# Regression metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (in %)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
) -> Dict[str, float]:
    """Return a dict with MAE, RMSE, R², and MAPE."""
    sep = "_" if prefix else ""
    return {
        f"{prefix}{sep}mae": compute_mae(y_true, y_pred),
        f"{prefix}{sep}rmse": compute_rmse(y_true, y_pred),
        f"{prefix}{sep}r2": compute_r2(y_true, y_pred),
        f"{prefix}{sep}mape": compute_mape(y_true, y_pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Signal preprocessing (synthetic implementations for simulation)
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 20.0,
    highcut: float = 500.0,
    fs: float = 4000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Butterworth band-pass filter.

    Parameters
    ----------
    signal : ndarray, shape (n,)
    lowcut, highcut : float – passband edges in Hz
    fs : float – sampling rate in Hz
    order : int – filter order

    Returns
    -------
    filtered : ndarray, shape (n,)
    """
    from scipy.signal import butter, sosfilt

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = np.clip(low, 1e-6, 1.0 - 1e-6)
    high = np.clip(high, 1e-6, 1.0 - 1e-6)
    if low >= high:
        high = min(low + 0.01, 1.0 - 1e-6)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, signal).astype(np.float32)


def wavelet_filter(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
    threshold_mode: str = "soft",
) -> np.ndarray:
    """
    Discrete Wavelet Transform denoising using universal threshold.

    Parameters
    ----------
    signal : ndarray, shape (n,)
    wavelet : str – PyWavelets wavelet name
    level : int – decomposition level
    threshold_mode : str – ``"soft"`` or ``"hard"``

    Returns
    -------
    denoised : ndarray, shape (n,)
    """
    try:
        import pywt
    except ImportError:
        return signal.astype(np.float32)

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=threshold_mode) for c in coeffs[1:]
    ]
    return pywt.waverec(denoised_coeffs, wavelet)[: len(signal)].astype(np.float32)


def emd_filter(signal: np.ndarray, n_imfs_to_remove: int = 1) -> np.ndarray:
    """
    Empirical Mode Decomposition (EMD) denoising.

    Decomposes signal into IMFs and reconstructs after discarding the
    highest-frequency IMFs that capture noise.

    Parameters
    ----------
    signal : ndarray, shape (n,)
    n_imfs_to_remove : int – number of high-frequency IMFs to drop

    Returns
    -------
    denoised : ndarray, shape (n,)
    """
    try:
        from PyEMD import EMD  # type: ignore
        emd = EMD()
        imfs = emd(signal)
        if len(imfs) > n_imfs_to_remove:
            return np.sum(imfs[n_imfs_to_remove:], axis=0).astype(np.float32)
        return signal.astype(np.float32)
    except ImportError:
        # Fallback: lightweight moving-average denoising
        window = max(3, len(signal) // 50)
        if window % 2 == 0:
            window += 1
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(signal, size=window).astype(np.float32)


def ica_filter(signal: np.ndarray, n_components: int = 4) -> np.ndarray:
    """
    ICA-based denoising via FastICA on sliding windows.

    Parameters
    ----------
    signal : ndarray, shape (n,)
    n_components : int – number of independent components

    Returns
    -------
    denoised : ndarray, shape (n,)
    """
    try:
        from sklearn.decomposition import FastICA

        n = len(signal)
        step = max(n_components, n // n_components)
        # Build a simple multichannel view via time-delayed copies
        channels = np.stack(
            [signal[i : n - (n_components - 1) + i] for i in range(n_components)],
            axis=1,
        )
        ica = FastICA(n_components=n_components, random_state=0, max_iter=200)
        sources = ica.fit_transform(channels)
        # Reconstruct using only the component with highest variance
        best = np.argmax(np.var(sources, axis=0))
        mask = np.zeros(n_components)
        mask[best] = 1.0
        reconstructed = (sources * mask) @ ica.mixing_.T + ica.mean_
        denoised = reconstructed[:, 0]
        # Pad / trim to original length
        result = np.zeros(n, dtype=np.float32)
        result[: len(denoised)] = denoised
        return result
    except Exception:
        return signal.astype(np.float32)


def apply_preprocessing(signal: np.ndarray, signal_type: str, fs: float = 4000.0) -> np.ndarray:
    """
    Dispatch to the appropriate preprocessing function.

    Parameters
    ----------
    signal : ndarray, shape (n,)
    signal_type : str – one of ``band-pass-filtered``, ``wavelet-filtered``,
        ``emd-denoised``, ``ica-denoised``
    fs : float – sampling rate in Hz

    Returns
    -------
    preprocessed : ndarray, shape (n,)
    """
    st = signal_type.lower()
    if "band" in st:
        return bandpass_filter(signal, fs=fs)
    if "wavelet" in st:
        return wavelet_filter(signal)
    if "emd" in st:
        return emd_filter(signal)
    if "ica" in st:
        return ica_filter(signal)
    return signal.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state_dict: Dict, path: str) -> None:
    """Save a PyTorch state-dict (or plain dict) to *path*."""
    import torch

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state_dict, path)


def load_checkpoint(path: str) -> Dict:
    """Load a checkpoint saved with :func:`save_checkpoint`."""
    import torch

    return torch.load(path, map_location="cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    rounds: List[int],
    global_mae: List[float],
    client_maes: Optional[Dict[int, List[float]]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot global MAE and per-client MAE curves over communication rounds.

    Parameters
    ----------
    rounds : list[int]
    global_mae : list[float]
    client_maes : dict {client_id: list[float]}, optional
    save_path : str, optional – if given, figure is saved to this path
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, global_mae, linewidth=2.5, label="Global model", color="black")
    if client_maes:
        for cid, maes in client_maes.items():
            ax.plot(rounds[: len(maes)], maes, linewidth=1, alpha=0.6, label=f"Client {cid}")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("MAE (BPM)")
    ax.set_title("Federated Learning Convergence")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_model_drift(weights_a: Dict, weights_b: Dict) -> float:
    """
    Compute Frobenius-norm distance between two weight state-dicts.

    Parameters
    ----------
    weights_a, weights_b : dict – PyTorch state-dicts (string → Tensor)

    Returns
    -------
    drift : float
    """
    import torch

    total = 0.0
    count = 0
    for key in weights_a:
        if key in weights_b:
            diff = weights_a[key].float() - weights_b[key].float()
            total += torch.norm(diff).item() ** 2
            count += 1
    return float(np.sqrt(total)) if count else 0.0
