"""
Utility helpers for the federated learning system.

Includes:
  - Configuration loading
  - Metric computation (MAE, RMSE, R², MAPE)
  - Signal preprocessing stubs (band-pass, wavelet, EMD, ICA)
  - Model checkpoint management
  - Visualisation helpers (individual plots + full dashboard)
"""

from __future__ import annotations

import json
import os
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette helpers
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_COLORS: Dict[str, str] = {
    "band-pass-filtered": "#1f77b4",
    "wavelet-filtered":   "#2ca02c",
    "emd-denoised":       "#d62728",
    "ica-denoised":       "#ff7f0e",
}


def _signal_color(signal_type: str) -> str:
    return _SIGNAL_COLORS.get(signal_type.lower(), "#7f7f7f")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Global MAE convergence (enhanced)
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


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Per-client MAE trajectories (separate panel per client)
# ─────────────────────────────────────────────────────────────────────────────

def plot_client_mae_trajectories(
    rounds: List[int],
    client_maes: Dict[int, List[float]],
    signal_types: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    One subplot per client showing MAE trajectory over rounds.

    Parameters
    ----------
    rounds : list[int]
    client_maes : dict {client_id: list[float]}
    signal_types : dict {client_id: str}, optional – used for colouring
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    client_ids = sorted(client_maes.keys())
    n = len(client_ids)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for idx, cid in enumerate(client_ids):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_visible(True)
        color = _signal_color(signal_types.get(cid, "") if signal_types else "")
        maes = client_maes[cid]
        ax.plot(rounds[: len(maes)], maes, color=color, linewidth=1.5)
        ax.set_title(f"Client {cid}", fontsize=9)
        ax.set_xlabel("Round", fontsize=8)
        ax.set_ylabel("MAE (BPM)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Legend for signal types
    if signal_types:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=c, linewidth=2, label=label)
            for label, c in _SIGNAL_COLORS.items()
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4,
                   fontsize=8, title="Signal type", bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Per-Client Test MAE over Communication Rounds", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Model drift over rounds
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_drift(
    rounds: List[int],
    drifts: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Line chart of model drift (Frobenius-norm distance between consecutive
    global models) over communication rounds.

    Parameters
    ----------
    rounds : list[int]
    drifts : list[float]
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds, drifts, color="#9467bd", linewidth=2)
    ax.fill_between(rounds, drifts, alpha=0.15, color="#9467bd")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Model drift (‖Δθ‖₂)")
    ax.set_title("Global Model Drift Between Consecutive Rounds")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Client data distribution (sample counts + heart rates)
# ─────────────────────────────────────────────────────────────────────────────

def plot_client_data_distribution(
    client_specs: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart of per-client window counts, coloured by signal type,
    with the nominal heart rate annotated.

    Parameters
    ----------
    client_specs : list of dicts with keys
        ``client_id``, ``num_windows``, ``heart_rate``, ``signal_type``
    save_path : str, optional
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    specs = sorted(client_specs, key=lambda s: s["client_id"])
    labels = [f"Client {s['client_id']}" for s in specs]
    sizes = [s["num_windows"] for s in specs]
    colors = [_signal_color(s["signal_type"]) for s in specs]
    heart_rates = [s["heart_rate"] for s in specs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel – window counts
    ax = axes[0]
    bars = ax.barh(labels, sizes, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature windows")
    ax.set_title("Dataset Size per Client")
    ax.invert_yaxis()
    for bar, sz in zip(bars, sizes):
        ax.text(bar.get_width() + max(sizes) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{sz:,}", va="center", fontsize=8)
    ax.set_xlim(0, max(sizes) * 1.15)
    ax.grid(axis="x", alpha=0.3)

    # Right panel – heart rate per client
    ax2 = axes[1]
    ax2.barh(labels, heart_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Nominal heart rate (BPM)")
    ax2.set_title("Heart Rate per Client")
    ax2.invert_yaxis()
    for i, hr in enumerate(heart_rates):
        ax2.text(hr + 0.3, i, f"{hr} BPM", va="center", fontsize=8)
    ax2.set_xlim(min(heart_rates) - 5, max(heart_rates) + 10)
    ax2.grid(axis="x", alpha=0.3)

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in _SIGNAL_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=8, title="Signal type", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Client Data Distribution (Non-IID Setup)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Label (HR) distributions per client  (violin / KDE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_label_distributions(
    client_labels: Dict[int, np.ndarray],
    signal_types: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Violin plot of HR label distributions across clients.

    Parameters
    ----------
    client_labels : dict {client_id: 1-D ndarray of HR values}
    signal_types : dict {client_id: str}, optional
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    client_ids = sorted(client_labels.keys())
    data = [client_labels[cid] for cid in client_ids]
    colors = [
        _signal_color(signal_types.get(cid, "") if signal_types else "")
        for cid in client_ids
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    parts = ax.violinplot(data, positions=list(range(1, len(client_ids) + 1)),
                          showmedians=True, showextrema=True)

    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[part].set_color("black")
        parts[part].set_linewidth(1)

    ax.set_xticks(range(1, len(client_ids) + 1))
    ax.set_xticklabels([f"C{cid}" for cid in client_ids])
    ax.set_xlabel("Client")
    ax.set_ylabel("Heart rate (BPM)")
    ax.set_title("HR Label Distributions per Client (Non-IID)")
    ax.grid(axis="y", alpha=0.3)

    if signal_types:
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=c, label=l, alpha=0.6) for l, c in _SIGNAL_COLORS.items()]
        ax.legend(handles=patches, fontsize=8, title="Signal type", loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – Federated vs. centralised MAE comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_federated_vs_centralised(
    federated_mae: float,
    centralised_mae: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side bar chart comparing federated and centralised final MAE.

    Parameters
    ----------
    federated_mae : float
    centralised_mae : float
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    values = [federated_mae, centralised_mae]
    labels = ["Federated\n(FedAvg)", "Centralised\n(pooled)"]
    colors_bar = ["#1f77b4", "#ff7f0e"]
    finite_vals = [v for v in values if np.isfinite(v)]
    ylim_top = (max(finite_vals) * 1.25) if finite_vals else 1.0
    bars = ax.bar(
        labels,
        [v if np.isfinite(v) else 0.0 for v in values],
        color=colors_bar,
        width=0.45,
        edgecolor="white",
    )
    for bar, val in zip(bars, values):
        label_str = f"{val:.3f}" if np.isfinite(val) else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ylim_top * 0.02,
                label_str, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("MAE (BPM)")
    ax.set_title("Federated vs. Centralised Learning\n(Final Test MAE)")
    ax.set_ylim(0, ylim_top)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 – Final per-client test MAE bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_final_client_maes(
    client_maes: Dict[int, float],
    signal_types: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart of each client's final test MAE after federation.

    Parameters
    ----------
    client_maes : dict {client_id: float}
    signal_types : dict {client_id: str}, optional
    save_path : str, optional
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    client_ids = sorted(client_maes.keys())
    maes = [client_maes[cid] for cid in client_ids]
    colors = [
        _signal_color(signal_types.get(cid, "") if signal_types else "")
        for cid in client_ids
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([f"C{cid}" for cid in client_ids], maes,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(np.mean(maes), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(maes):.3f} BPM")
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Client")
    ax.set_ylabel("Test MAE (BPM)")
    ax.set_title("Final Per-Client Test MAE (Global Model)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    if signal_types:
        patches = [mpatches.Patch(color=c, label=l) for l, c in _SIGNAL_COLORS.items()]
        ax.legend(handles=patches + ax.get_legend_handles_labels()[0][:1],
                  fontsize=8, title="Signal type")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 – Cumulative communication cost
# ─────────────────────────────────────────────────────────────────────────────

def plot_communication_cost(
    rounds: List[int],
    bytes_per_round: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Line chart of cumulative bytes transferred vs. communication rounds.

    Parameters
    ----------
    rounds : list[int]
    bytes_per_round : float – bytes transmitted per round (upload + download)
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    cumulative_mb = [r * bytes_per_round / 1_048_576 for r in rounds]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds, cumulative_mb, color="#17becf", linewidth=2)
    ax.fill_between(rounds, cumulative_mb, alpha=0.15, color="#17becf")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative data transferred (MB)")
    ax.set_title("Communication Cost over Training")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 – Wall-clock time per round
# ─────────────────────────────────────────────────────────────────────────────

def plot_round_times(
    rounds: List[int],
    times: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Bar/line chart of wall-clock seconds per communication round.

    Parameters
    ----------
    rounds : list[int]
    times : list[float]  – seconds per round
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(rounds, times, color="#8c564b", alpha=0.7, width=0.8)
    # Rolling mean
    window = max(1, len(times) // 10)
    rolling_mean = np.convolve(times, np.ones(window) / window, mode="valid")
    offset = window // 2
    ax.plot(
        rounds[offset : offset + len(rolling_mean)],
        rolling_mean, color="black", linewidth=1.5, label=f"Rolling mean (w={window})"
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Time (s)")
    ax.set_title("Wall-Clock Time per Communication Round")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 – Summary dashboard (multi-panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(
    rounds: List[int],
    global_mae: List[float],
    drifts: List[float],
    client_maes_final: Dict[int, float],
    times: List[float],
    federated_mae: float,
    centralised_mae: float,
    signal_types: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    6-panel summary dashboard combining the most important figures.

    Panels
    ------
    (A) Global MAE convergence
    (B) Model drift per round
    (C) Final per-client test MAE
    (D) Federated vs. centralised bar
    (E) Wall-clock time per round
    (F) Dataset size distribution (text table rendered as bar)

    Parameters
    ----------
    rounds, global_mae, drifts, client_maes_final, times : as in individual plots
    federated_mae, centralised_mae : float
    signal_types : dict {client_id: str}, optional
    save_path : str, optional
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── (A) Global MAE convergence ─────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(rounds, global_mae, color="black", linewidth=2)
    ax_a.set_xlabel("Round", fontsize=9)
    ax_a.set_ylabel("MAE (BPM)", fontsize=9)
    ax_a.set_title("(A) Global Model Convergence", fontsize=10, fontweight="bold")
    ax_a.grid(True, alpha=0.3)

    # ── (B) Model drift ────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(rounds, drifts, color="#9467bd", linewidth=2)
    ax_b.fill_between(rounds, drifts, alpha=0.15, color="#9467bd")
    ax_b.set_xlabel("Round", fontsize=9)
    ax_b.set_ylabel("‖Δθ‖₂", fontsize=9)
    ax_b.set_title("(B) Model Drift per Round", fontsize=10, fontweight="bold")
    ax_b.grid(True, alpha=0.3)

    # ── (C) Final per-client test MAE ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    client_ids = sorted(client_maes_final.keys())
    maes = [client_maes_final[cid] for cid in client_ids]
    colors = [
        _signal_color(signal_types.get(cid, "") if signal_types else "")
        for cid in client_ids
    ]
    ax_c.bar([f"C{cid}" for cid in client_ids], maes, color=colors, edgecolor="white")
    ax_c.axhline(np.mean(maes), color="black", linestyle="--", linewidth=1.2,
                 label=f"Mean {np.mean(maes):.2f}")
    ax_c.set_xlabel("Client", fontsize=9)
    ax_c.set_ylabel("MAE (BPM)", fontsize=9)
    ax_c.set_title("(C) Final Client Test MAE", fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=8)
    ax_c.grid(axis="y", alpha=0.3)
    ax_c.tick_params(axis="x", labelsize=7)

    # ── (D) Federated vs. centralised ─────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    fed_vals = [federated_mae, centralised_mae]
    finite_fed = [v for v in fed_vals if np.isfinite(v)]
    ylim_d = (max(finite_fed) * 1.3) if finite_fed else 1.0
    bars_d = ax_d.bar(
        ["Federated\n(FedAvg)", "Centralised\n(pooled)"],
        [v if np.isfinite(v) else 0.0 for v in fed_vals],
        color=["#1f77b4", "#ff7f0e"], width=0.45, edgecolor="white",
    )
    for bar, val in zip(bars_d, fed_vals):
        label_str = f"{val:.3f}" if np.isfinite(val) else "N/A"
        ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ylim_d * 0.02,
                  label_str, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("MAE (BPM)", fontsize=9)
    ax_d.set_title("(D) Federated vs. Centralised", fontsize=10, fontweight="bold")
    ax_d.set_ylim(0, ylim_d)
    ax_d.grid(axis="y", alpha=0.3)

    # ── (E) Wall-clock time ────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.bar(rounds, times, color="#8c564b", alpha=0.7, width=0.8)
    window = max(1, len(times) // 10)
    roll = np.convolve(times, np.ones(window) / window, mode="valid")
    offset = window // 2
    ax_e.plot(rounds[offset : offset + len(roll)], roll,
              color="black", linewidth=1.5, label=f"Rolling mean")
    ax_e.set_xlabel("Round", fontsize=9)
    ax_e.set_ylabel("Time (s)", fontsize=9)
    ax_e.set_title("(E) Wall-Clock Time per Round", fontsize=10, fontweight="bold")
    ax_e.legend(fontsize=8)
    ax_e.grid(axis="y", alpha=0.3)

    # ── (F) Signal-type legend / info box ─────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")

    # Summary statistics text
    summary_lines = [
        f"Rounds completed : {len(rounds)}",
        f"Best global MAE  : {min(global_mae):.4f} BPM (round {rounds[int(np.argmin(global_mae))]})",
        f"Final global MAE : {global_mae[-1]:.4f} BPM",
        f"Δ vs centralised : {federated_mae - centralised_mae:+.4f} BPM",
        f"Mean round time  : {np.mean(times):.2f} s",
    ]
    ax_f.text(0.05, 0.95, "\n".join(summary_lines),
              transform=ax_f.transAxes,
              fontsize=10, verticalalignment="top",
              fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))
    ax_f.set_title("(F) Summary Statistics", fontsize=10, fontweight="bold")

    # Global legend for signal types
    if signal_types:
        patches = [mpatches.Patch(color=c, label=l) for l, c in _SIGNAL_COLORS.items()]
        fig.legend(handles=patches, loc="lower center", ncol=4,
                   fontsize=9, title="Signal type", bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Federated Learning for Fetal Heart Rate Prediction – Summary Dashboard",
        fontsize=14, fontweight="bold",
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate all figures at once
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(
    round_metrics: List[Dict[str, Any]],
    client_specs: List[Dict[str, Any]],
    client_labels: Optional[Dict[int, np.ndarray]],
    federated_mae: float,
    centralised_mae: float,
    output_dir: str,
    bytes_per_round: float = 0.0,
) -> List[str]:
    """
    Produce all standard figures and save them to *output_dir*.

    Parameters
    ----------
    round_metrics : list of per-round dicts (as returned by FederatedTrainer.run)
    client_specs : list of dicts with ``client_id``, ``num_windows``,
        ``heart_rate``, ``signal_type``
    client_labels : dict {client_id: ndarray of HR labels}, or ``None``
    federated_mae : float – final global MAE from the federated run
    centralised_mae : float – MAE from the centralised baseline
    output_dir : str – directory where PNGs are written
    bytes_per_round : float – estimated bytes per round (for Fig 8)

    Returns
    -------
    paths : list[str] – file paths of all figures written
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Extract series from round_metrics ────────────────────────────────────
    rounds = [m["round"] for m in round_metrics]
    global_mae = [m["global_mae"] for m in round_metrics]
    drifts = [m.get("model_drift", 0.0) for m in round_metrics]
    times = [m.get("wall_time_s", 0.0) for m in round_metrics]

    client_mae_series: Dict[int, List[float]] = {}
    for m in round_metrics:
        for cid, mae in m.get("client_test_maes", {}).items():
            client_mae_series.setdefault(cid, []).append(mae)

    final_client_maes: Dict[int, float] = round_metrics[-1].get("client_test_maes", {}) \
        if round_metrics else {}

    signal_types: Dict[int, str] = {
        s["client_id"]: s["signal_type"] for s in client_specs
    }

    paths: List[str] = []

    def _save(name: str) -> str:
        return os.path.join(output_dir, name)

    # ── Fig 1: Global MAE convergence ────────────────────────────────────────
    p = _save("fig01_global_convergence.png")
    plot_convergence(rounds, global_mae, client_maes=client_mae_series, save_path=p)
    paths.append(p)

    # ── Fig 2: Per-client MAE trajectories ───────────────────────────────────
    if client_mae_series:
        p = _save("fig02_client_mae_trajectories.png")
        plot_client_mae_trajectories(
            rounds, client_mae_series, signal_types=signal_types, save_path=p
        )
        paths.append(p)

    # ── Fig 3: Model drift ───────────────────────────────────────────────────
    if any(d > 0 for d in drifts):
        p = _save("fig03_model_drift.png")
        plot_model_drift(rounds, drifts, save_path=p)
        paths.append(p)

    # ── Fig 4: Client data distribution ──────────────────────────────────────
    p = _save("fig04_client_data_distribution.png")
    plot_client_data_distribution(client_specs, save_path=p)
    paths.append(p)

    # ── Fig 5: Label (HR) distributions ─────────────────────────────────────
    if client_labels:
        p = _save("fig05_label_distributions.png")
        plot_label_distributions(client_labels, signal_types=signal_types, save_path=p)
        paths.append(p)

    # ── Fig 6: Federated vs. centralised ─────────────────────────────────────
    p = _save("fig06_federated_vs_centralised.png")
    plot_federated_vs_centralised(federated_mae, centralised_mae, save_path=p)
    paths.append(p)

    # ── Fig 7: Final per-client test MAE ─────────────────────────────────────
    if final_client_maes:
        p = _save("fig07_final_client_maes.png")
        plot_final_client_maes(final_client_maes, signal_types=signal_types, save_path=p)
        paths.append(p)

    # ── Fig 8: Communication cost ─────────────────────────────────────────────
    if bytes_per_round > 0:
        p = _save("fig08_communication_cost.png")
        plot_communication_cost(rounds, bytes_per_round, save_path=p)
        paths.append(p)

    # ── Fig 9: Wall-clock time ────────────────────────────────────────────────
    if any(t > 0 for t in times):
        p = _save("fig09_round_times.png")
        plot_round_times(rounds, times, save_path=p)
        paths.append(p)

    # ── Fig 10: Summary dashboard ─────────────────────────────────────────────
    p = _save("fig10_summary_dashboard.png")
    plot_summary_dashboard(
        rounds=rounds,
        global_mae=global_mae,
        drifts=drifts,
        client_maes_final=final_client_maes,
        times=times,
        federated_mae=federated_mae,
        centralised_mae=centralised_mae,
        signal_types=signal_types,
        save_path=p,
    )
    paths.append(p)

    return paths

