"""
Synthetic data simulator for fetal heart sound (FHS) / ECG signals.

Each client receives a distinct simulated dataset that reflects:
  - A client-specific heart rate (130–156 BPM)
  - A client-specific signal preprocessing method
    (band-pass, wavelet, EMD, or ICA)
  - A realistic number of feature windows
  - Non-IID data distribution across clients

Generated features are 256-dimensional to match the MLP input layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import apply_preprocessing
from .logger import get_logger

log = get_logger(__name__, log_dir="")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DIM: int = 256          # MLP input dimension
SAMPLE_RATE: float = 4000.0     # Hz  (typical FHS acquisition rate)
WINDOW_SAMPLES: int = 4096      # Raw samples per analysis window  (≈1 s at 4 kHz)

# Client table: id → (num_windows, heart_rate_bpm, signal_type)
CLIENT_SPECS: Dict[int, Tuple[int, int, str]] = {
    1:  (70_000, 140, "band-pass-filtered"),
    2:  (55_000, 144, "wavelet-filtered"),
    3:  (48_000, 148, "emd-denoised"),
    4:  (42_000, 152, "ica-denoised"),
    5:  (45_000, 136, "wavelet-filtered"),
    6:  (38_000, 132, "band-pass-filtered"),
    7:  (30_000, 156, "emd-denoised"),
    8:  (28_000, 150, "ica-denoised"),
    9:  (48_000, 138, "wavelet-filtered"),
    10: ( 5_852, 130, "band-pass-filtered"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Core simulator class
# ─────────────────────────────────────────────────────────────────────────────

class ClientDataSimulator:
    """
    Generate synthetic fetal heart sound feature data for a single client.

    Parameters
    ----------
    client_id : int
        Client identifier (1–10).
    num_windows : int
        Number of feature windows to generate.
    heart_rate : float
        Nominal heart rate in BPM for this client (130–156).
    signal_type : str
        Preprocessing applied: ``band-pass-filtered``, ``wavelet-filtered``,
        ``emd-denoised``, or ``ica-denoised``.
    feature_dim : int
        Output feature dimensionality (default 256 to match the MLP).
    seed : int, optional
        Random seed for reproducibility.  If ``None``, results vary each run.
    """

    def __init__(
        self,
        client_id: int,
        num_windows: int,
        heart_rate: float,
        signal_type: str,
        feature_dim: int = FEATURE_DIM,
        seed: Optional[int] = None,
    ) -> None:
        self.client_id = client_id
        self.num_windows = num_windows
        self.heart_rate = float(heart_rate)
        self.signal_type = signal_type
        self.feature_dim = feature_dim
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ── public API ────────────────────────────────────────────────────────────

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ``(X, y)`` for this client.

        Returns
        -------
        X : ndarray, shape (num_windows, feature_dim)  – float32
        y : ndarray, shape (num_windows,)              – float32 (BPM)
        """
        log.debug(
            "Simulating client %d: %d windows  HR=%.0f BPM  type=%s",
            self.client_id, self.num_windows, self.heart_rate, self.signal_type,
        )
        X, y = self._generate_features_and_labels()
        return X.astype(np.float32), y.astype(np.float32)

    def train_val_test_split(
        self,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        Generate data and split into train / validation / test sets.

        Returns
        -------
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        X, y = self.generate()
        n = len(y)
        idx = self._rng.permutation(n)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]

        return (
            (X[train_idx], y[train_idx]),
            (X[val_idx], y[val_idx]),
            (X[test_idx], y[test_idx]),
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _generate_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesise feature matrices and corresponding HR labels.

        Strategy
        --------
        1. Draw a small label jitter around the nominal heart rate so that
           each window has a slightly different ground-truth HR, reflecting
           beat-to-beat variability.
        2. Construct a raw cardiac signal segment at the correct frequency.
        3. Apply the client-specific preprocessing.
        4. Extract a 256-D feature vector from each processed window.
        """
        # Per-window HR labels with physiological variability (±2 BPM std)
        hr_jitter = self._rng.normal(0.0, 2.0, size=self.num_windows)
        y = np.clip(self.heart_rate + hr_jitter, 100.0, 200.0)

        # Build feature matrix window-by-window in batches
        batch_size = 500   # generate batches to keep memory reasonable
        X_parts: List[np.ndarray] = []
        for start in range(0, self.num_windows, batch_size):
            end = min(start + batch_size, self.num_windows)
            batch_y = y[start:end]
            X_batch = self._batch_features(batch_y)
            X_parts.append(X_batch)

        X = np.concatenate(X_parts, axis=0)
        return X, y

    def _batch_features(self, hr_values: np.ndarray) -> np.ndarray:
        """
        Generate a batch of feature vectors for the given HR values.

        Parameters
        ----------
        hr_values : ndarray, shape (b,)

        Returns
        -------
        features : ndarray, shape (b, feature_dim)
        """
        b = len(hr_values)
        features = np.zeros((b, self.feature_dim), dtype=np.float32)

        for i, hr in enumerate(hr_values):
            raw = self._synthesise_raw_signal(hr)
            processed = apply_preprocessing(raw, self.signal_type, fs=SAMPLE_RATE)
            features[i] = self._extract_features(processed)

        # Add preprocessing-type-specific distribution shift to ensure Non-IID
        features += self._distribution_shift()
        return features

    def _synthesise_raw_signal(self, hr_bpm: float) -> np.ndarray:
        """
        Synthesise a realistic fetal cardiac signal at *hr_bpm* BPM.

        The signal is a superposition of:
          - Fundamental heartbeat frequency and harmonics (S1, S2 sounds)
          - Murmur-like broadband noise
          - Maternal motion artefact (low-frequency sinusoid)
        """
        t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
        f0 = hr_bpm / 60.0        # fundamental in Hz

        # Cardiac waveform: weighted harmonics
        signal = (
            0.8 * np.sin(2 * np.pi * f0 * t)
            + 0.4 * np.sin(2 * np.pi * 2 * f0 * t + 0.3)
            + 0.2 * np.sin(2 * np.pi * 3 * f0 * t + 0.6)
            + 0.1 * np.sin(2 * np.pi * 4 * f0 * t + 0.9)
        )

        # S1 / S2 impulse bursts
        beat_samples = int(SAMPLE_RATE * 60.0 / hr_bpm)
        for k in range(0, WINDOW_SAMPLES, beat_samples):
            width = int(0.04 * SAMPLE_RATE)   # 40 ms per sound
            if k + width < WINDOW_SAMPLES:
                burst = np.exp(-np.linspace(0, 6, width)) * 0.5
                signal[k : k + width] += burst
            s2_start = k + int(beat_samples * 0.35)
            if s2_start + width < WINDOW_SAMPLES:
                burst2 = np.exp(-np.linspace(0, 8, width)) * 0.3
                signal[s2_start : s2_start + width] += burst2

        # Gaussian noise (murmur / background)
        signal += self._rng.normal(0, 0.15, WINDOW_SAMPLES)

        # Maternal motion artefact (low-frequency)
        signal += 0.1 * np.sin(2 * np.pi * 0.5 * t)

        return signal.astype(np.float32)

    def _extract_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract a 256-D feature vector from a processed signal segment.

        Feature groups
        --------------
        - Time-domain statistics (16 features)
        - Spectral statistics (16 features)
        - Log-magnitude spectrum bins (128 features)
        - Auto-correlation lag features (64 features)
        - Envelope / energy per sub-band (32 features)
        """
        feat = np.zeros(self.feature_dim, dtype=np.float32)
        pos = 0

        # ── Time-domain (16) ──────────────────────────────────────────────────
        pos = self._time_domain_features(signal, feat, pos)

        # ── Spectral statistics (16) ──────────────────────────────────────────
        pos = self._spectral_stat_features(signal, feat, pos)

        # ── Log-magnitude spectrum (128) ──────────────────────────────────────
        pos = self._log_spectrum_features(signal, feat, pos)

        # ── Auto-correlation (64) ─────────────────────────────────────────────
        pos = self._autocorr_features(signal, feat, pos)

        # ── Sub-band energy (32) ──────────────────────────────────────────────
        self._subband_energy_features(signal, feat, pos)

        return feat

    def _time_domain_features(
        self, signal: np.ndarray, feat: np.ndarray, pos: int
    ) -> int:
        tdf = np.array([
            np.mean(signal),                    # mean
            np.std(signal),                     # std
            np.max(signal),                     # max
            np.min(signal),                     # min
            np.sqrt(np.mean(signal ** 2)),      # RMS
            np.max(np.abs(signal)) / (np.sqrt(np.mean(signal ** 2)) + 1e-8),  # crest factor
            float(np.sum(signal[:-1] * signal[1:] < 0)) / len(signal),        # ZCR
            np.percentile(signal, 25),
            np.percentile(signal, 50),
            np.percentile(signal, 75),
            np.percentile(signal, 90),
            np.percentile(signal, 10),
            np.percentile(signal, 75) - np.percentile(signal, 25),  # IQR
            float(np.sum((signal[1:] - signal[:-1]) ** 2)),          # energy of diff
            np.mean(np.abs(signal[1:] - signal[:-1])),               # mean abs delta
            np.sum(signal ** 2) / len(signal),                       # power
        ], dtype=np.float32)
        n = len(tdf)
        feat[pos : pos + n] = tdf
        return pos + n

    def _spectral_stat_features(
        self, signal: np.ndarray, feat: np.ndarray, pos: int
    ) -> int:
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / SAMPLE_RATE)
        power = fft_vals ** 2
        total_power = np.sum(power) + 1e-8

        spectral_centroid = float(np.sum(freqs * power) / total_power)
        spectral_spread = float(
            np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / total_power)
        )
        spectral_entropy = float(
            -np.sum((power / total_power + 1e-12) * np.log2(power / total_power + 1e-12))
        )
        spectral_flatness = float(
            np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-8)
        )
        spectral_rolloff_idx = int(
            np.searchsorted(np.cumsum(power), 0.85 * total_power)
        )
        spectral_rolloff = float(freqs[min(spectral_rolloff_idx, len(freqs) - 1)])
        spectral_skewness = float(
            np.sum(((freqs - spectral_centroid) ** 3) * power)
            / (total_power * spectral_spread ** 3 + 1e-8)
        )
        spectral_kurtosis = float(
            np.sum(((freqs - spectral_centroid) ** 4) * power)
            / (total_power * spectral_spread ** 4 + 1e-8)
        )
        # Peak frequency and its amplitude
        peak_idx = int(np.argmax(power))
        peak_freq = float(freqs[peak_idx])
        peak_amp = float(fft_vals[peak_idx])
        spectral_crest = float(np.max(power) / total_power)

        spf = np.array([
            spectral_centroid,
            spectral_spread,
            spectral_entropy,
            spectral_flatness,
            spectral_rolloff,
            spectral_skewness,
            spectral_kurtosis,
            peak_freq,
            peak_amp,
            spectral_crest,
            float(np.mean(fft_vals)),
            float(np.std(fft_vals)),
            float(np.max(fft_vals)),
            float(np.percentile(fft_vals, 75)),
            float(np.percentile(fft_vals, 25)),
            float(np.sum(power[: len(power) // 2]) / (total_power + 1e-8)),
        ], dtype=np.float32)
        n = len(spf)
        feat[pos : pos + n] = spf
        return pos + n

    def _log_spectrum_features(
        self, signal: np.ndarray, feat: np.ndarray, pos: int
    ) -> int:
        """128 log-magnitude spectral bins."""
        n_bins = 128
        fft_vals = np.abs(np.fft.rfft(signal, n=WINDOW_SAMPLES))
        # Subsample to n_bins evenly
        indices = np.linspace(0, len(fft_vals) - 1, n_bins, dtype=int)
        log_spectrum = np.log1p(fft_vals[indices]).astype(np.float32)
        feat[pos : pos + n_bins] = log_spectrum
        return pos + n_bins

    def _autocorr_features(
        self, signal: np.ndarray, feat: np.ndarray, pos: int
    ) -> int:
        """64 normalised auto-correlation lag values."""
        n_lags = 64
        if len(signal) < n_lags * 2:
            feat[pos : pos + n_lags] = 0.0
            return pos + n_lags
        variance = np.var(signal) + 1e-8
        lags = np.arange(1, n_lags + 1)
        acf = np.array(
            [np.mean(signal[lag:] * signal[: len(signal) - lag]) / variance for lag in lags],
            dtype=np.float32,
        )
        feat[pos : pos + n_lags] = acf
        return pos + n_lags

    def _subband_energy_features(
        self, signal: np.ndarray, feat: np.ndarray, pos: int
    ) -> None:
        """32 energy values across equal-width frequency sub-bands."""
        n_bands = 32
        fft_vals = np.abs(np.fft.rfft(signal, n=WINDOW_SAMPLES)) ** 2
        band_size = max(1, len(fft_vals) // n_bands)
        for b in range(n_bands):
            start = b * band_size
            end = start + band_size
            energy = float(np.sum(fft_vals[start:end]))
            feat[pos + b] = np.log1p(energy)

    def _distribution_shift(self) -> np.ndarray:
        """
        Return a small per-feature bias that creates a Non-IID distribution
        shift specific to this client's preprocessing type.
        """
        # Use a fixed deterministic mapping instead of hash() which can be
        # non-reproducible across Python sessions due to hash randomisation.
        _SIGNAL_TYPE_SEEDS = {
            "band-pass-filtered": 1001,
            "wavelet-filtered": 1002,
            "emd-denoised": 1003,
            "ica-denoised": 1004,
        }
        seed_offset = _SIGNAL_TYPE_SEEDS.get(self.signal_type.lower(), 1000)
        rng = np.random.default_rng(seed_offset)
        shift = rng.normal(
            loc=self.heart_rate / 140.0 - 1.0,    # ~0 for 140 BPM
            scale=0.05,
            size=self.feature_dim,
        ).astype(np.float32)
        return shift


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def build_all_simulators(
    client_specs: Optional[List[Dict]] = None,
    seed: int = 42,
) -> List[ClientDataSimulator]:
    """
    Build a list of :class:`ClientDataSimulator` objects.

    Parameters
    ----------
    client_specs : list of dict, optional
        If ``None``, ``CLIENT_SPECS`` is used.  Each dict must have keys:
        ``client_id``, ``num_windows``, ``heart_rate``, ``signal_type``.
    seed : int
        Base seed; each client gets ``seed + client_id``.

    Returns
    -------
    list[ClientDataSimulator]
    """
    if client_specs is None:
        specs = [
            {
                "client_id": cid,
                "num_windows": nw,
                "heart_rate": hr,
                "signal_type": st,
            }
            for cid, (nw, hr, st) in CLIENT_SPECS.items()
        ]
    else:
        specs = client_specs

    simulators = []
    for spec in specs:
        sim = ClientDataSimulator(
            client_id=spec["client_id"],
            num_windows=spec["num_windows"],
            heart_rate=spec["heart_rate"],
            signal_type=spec["signal_type"],
            seed=seed + spec["client_id"],
        )
        simulators.append(sim)
    return simulators
