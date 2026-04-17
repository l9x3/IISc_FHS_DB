"""
train_fhr_model.py
------------------
Deep learning model for Fetal Heart Rate (FHR) prediction from the
Fetal Heart Sound (FHS) dataset.

Pipeline:
  1. Load clinical data from dataset/Records.csv
  2. Extract audio features (MFCC, spectral) from WAV files in dataset/
  3. Build a dual-branch neural network (audio + tabular)
  4. Run 5-fold cross-validation
  5. Compute Train/Val/Test MAE, RMSE, PPA, MAPE, R²  (+ std-dev across folds)
  6. Save per-fold and summary CSV files to results/
  7. Save trained models to results/
"""

import os
import warnings
import json

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
N_FOLDS = 5
N_MFCC = 20
SAMPLE_RATE = 4000          # FHS recordings are typically at 4 kHz
MAX_DURATION = 30           # seconds to load per file
MIN_WAV_SIZE = 100          # bytes — below this the file is a placeholder
PPA_THRESHOLD = 0.10        # 10 % of actual FHR counts as "accurate"
EPOCHS = 300
BATCH_SIZE = 8
PATIENCE_ES = 30            # early-stopping patience
PATIENCE_LR = 15            # reduce-LR patience

TABULAR_COLS = [
    "Gestational age (weeks)",
    "Age",
    "Weight (kg)",
    "Height (cm)",
    "Gravida",
    "Para",
    "has_condition",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_clinical_data(csv_path: str) -> pd.DataFrame:
    """Load and clean Records.csv.  Returns rows that have a valid FHR target."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Normalise missing-value markers
    df.replace("-", np.nan, inplace=True)
    df.replace("", np.nan, inplace=True)

    numeric_cols = [
        "Gestational age (weeks)", "Age", "Weight (kg)", "Height (cm)",
        "Gravida", "Para", "FHR (bpm) from medical file",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary: does the subject have a documented clinical condition?
    df["has_condition"] = df["Clinical Conditions"].notna().astype(float)

    # Only keep rows where the target is known
    df = df.dropna(subset=["FHR (bpm) from medical file"]).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Audio feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def _get_wav_path_for_subject(subject_id: int) -> str | None:
    """
    Return the path to the WAV file for *subject_id*, or None if not found.
    Handles the case where the file is named 'subject_48.wav\\' (trailing
    backslash introduced by the original upload).
    """
    candidates = [
        os.path.join(DATASET_DIR, f"subject_{subject_id:02d}.wav"),
        os.path.join(DATASET_DIR, f"subject_{subject_id:02d}.wav\\"),
        os.path.join(DATASET_DIR, f"subject_{subject_id}.wav"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def extract_audio_features(wav_path: str) -> np.ndarray | None:
    """
    Extract MFCC, spectral, and chroma features from a WAV file.
    Returns a 1-D float32 array.
    Returns None when the file is a placeholder (< MIN_WAV_SIZE bytes) or
    when loading fails.
    """
    if os.path.getsize(wav_path) < MIN_WAV_SIZE:
        return None

    try:
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        if len(y) == 0:
            return None

        feats: list[float] = []

        # MFCC (mean + std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        feats.extend(np.mean(mfcc, axis=1).tolist())
        feats.extend(np.std(mfcc, axis=1).tolist())

        # MFCC delta (mean + std)
        mfcc_delta = librosa.feature.delta(mfcc)
        feats.extend(np.mean(mfcc_delta, axis=1).tolist())
        feats.extend(np.std(mfcc_delta, axis=1).tolist())

        # Spectral centroid
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats += [float(np.mean(sc)), float(np.std(sc))]

        # Spectral rolloff
        sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr)
        feats += [float(np.mean(sr_)), float(np.std(sr_))]

        # Spectral bandwidth
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        feats += [float(np.mean(sb)), float(np.std(sb))]

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        feats += [float(np.mean(zcr)), float(np.std(zcr))]

        # RMS energy
        rms = librosa.feature.rms(y=y)
        feats += [float(np.mean(rms)), float(np.std(rms))]

        # Chroma (mean + std)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats.extend(np.mean(chroma, axis=1).tolist())
        feats.extend(np.std(chroma, axis=1).tolist())

        return np.array(feats, dtype=np.float32)

    except Exception as exc:
        print(f"  [WARN] Could not extract features from {wav_path}: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Build the dataset array
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(df: pd.DataFrame):
    """
    For each subject in *df*, attempt to load audio features and collect
    tabular features.  Returns arrays aligned on valid subjects.
    """
    audio_list, tabular_list, target_list, subject_ids = [], [], [], []

    # Determine audio-feature dimension from a real file
    audio_dim: int | None = None
    for _, row in df.iterrows():
        sid = int(row["Subject"])
        wp = _get_wav_path_for_subject(sid)
        if wp is not None:
            af = extract_audio_features(wp)
            if af is not None:
                audio_dim = len(af)
                break

    if audio_dim is None:
        # Fallback: 20 MFCC × 4 (mean/std/delta-mean/delta-std)
        #           + 5 spectral × 2 + 12 chroma × 2 = 134
        audio_dim = 134
        print("[WARN] No valid audio files found – using zero-padded audio features "
              f"of dimension {audio_dim}.")
    print(f"Audio feature dimension: {audio_dim}")

    for _, row in df.iterrows():
        sid = int(row["Subject"])
        wp = _get_wav_path_for_subject(sid)

        if wp is None:
            af = np.zeros(audio_dim, dtype=np.float32)
            has_audio = 0.0
        else:
            af = extract_audio_features(wp)
            if af is None:
                af = np.zeros(audio_dim, dtype=np.float32)
                has_audio = 0.0
            else:
                has_audio = 1.0

        tab = list(row[TABULAR_COLS].values.astype(float))
        tab.append(has_audio)  # extra indicator feature

        audio_list.append(af)
        tabular_list.append(tab)
        target_list.append(float(row["FHR (bpm) from medical file"]))
        subject_ids.append(sid)

    audio_arr = np.array(audio_list, dtype=np.float32)
    tabular_arr = np.array(tabular_list, dtype=np.float32)
    targets = np.array(target_list, dtype=np.float32)

    print(f"\nDataset summary:")
    print(f"  Total samples  : {len(subject_ids)}")
    print(f"  Audio features : {audio_arr.shape[1]}")
    print(f"  Tabular feats  : {tabular_arr.shape[1]}")
    print(f"  FHR range      : {targets.min():.1f} – {targets.max():.1f} bpm")
    print(f"  Subjects w/ real audio : {int(tabular_arr[:, -1].sum())}")

    return audio_arr, tabular_arr, targets, subject_ids


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Model definition
# ══════════════════════════════════════════════════════════════════════════════

def build_model(audio_dim: int, tabular_dim: int) -> keras.Model:
    """
    Dual-branch feed-forward network.
    – Audio branch  : Dense → BN → Dropout stack
    – Tabular branch: Dense → BN → Dropout stack
    – Merged        : Dense → Dropout → linear output
    """
    l2 = regularizers.l2(0.01)

    # Audio branch
    audio_in = keras.Input(shape=(audio_dim,), name="audio_input")
    xa = layers.Dense(128, activation="relu", kernel_regularizer=l2)(audio_in)
    xa = layers.BatchNormalization()(xa)
    xa = layers.Dropout(0.4)(xa)
    xa = layers.Dense(64, activation="relu", kernel_regularizer=l2)(xa)
    xa = layers.BatchNormalization()(xa)
    xa = layers.Dropout(0.3)(xa)
    xa = layers.Dense(32, activation="relu")(xa)

    # Tabular branch
    tab_in = keras.Input(shape=(tabular_dim,), name="tabular_input")
    xt = layers.Dense(32, activation="relu", kernel_regularizer=l2)(tab_in)
    xt = layers.BatchNormalization()(xt)
    xt = layers.Dropout(0.3)(xt)
    xt = layers.Dense(16, activation="relu")(xt)

    # Fusion
    merged = layers.Concatenate()([xa, xt])
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2)(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="linear", name="fhr_output")(x)

    model = keras.Model(inputs=[audio_in, tab_in], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE, PPA, R² for a single set of predictions."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    safe_true = np.where(y_true != 0, y_true, 1e-9)
    rel_errors = np.abs((y_true - y_pred) / safe_true)
    mape = float(np.mean(rel_errors) * 100)
    ppa  = float(np.mean(rel_errors <= PPA_THRESHOLD) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "PPA": ppa}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  5-Fold Cross-Validation
# ══════════════════════════════════════════════════════════════════════════════

def run_cross_validation(audio_arr, tabular_arr, targets):
    """
    Runs 5-fold CV.  For each fold the training set is further split 80/20
    into train and validation subsets.

    Returns
    -------
    fold_results  : list of per-fold metric dicts
    all_histories : list of Keras history dicts
    all_preds     : list of prediction dicts (actual vs predicted per split)
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)

    # Impute missing tabular values once (fit on full dataset to get stable stats)
    tab_imputer = SimpleImputer(strategy="median")
    tab_imputed = tab_imputer.fit_transform(tabular_arr)

    fold_results: list[dict] = []
    all_histories: list[dict] = []
    all_preds: list[dict] = []

    for fold_idx, (tv_idx, test_idx) in enumerate(kf.split(targets), start=1):
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold_idx}/{N_FOLDS}")
        print(f"{'─'*60}")

        # Split train_val → train + val (80 / 20)
        tv_shuffled = rng.permutation(tv_idx)
        val_n  = max(1, int(0.20 * len(tv_shuffled)))
        val_idx   = tv_shuffled[:val_n]
        train_idx = tv_shuffled[val_n:]

        print(f"  Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

        # Slice data
        Xa_tr, Xa_va, Xa_te = (
            audio_arr[train_idx], audio_arr[val_idx], audio_arr[test_idx]
        )
        Xt_tr, Xt_va, Xt_te = (
            tab_imputed[train_idx], tab_imputed[val_idx], tab_imputed[test_idx]
        )
        y_tr, y_va, y_te = targets[train_idx], targets[val_idx], targets[test_idx]

        # Scale audio features
        a_scaler = StandardScaler()
        Xa_tr = a_scaler.fit_transform(Xa_tr)
        Xa_va = a_scaler.transform(Xa_va)
        Xa_te = a_scaler.transform(Xa_te)

        # Scale tabular features
        t_scaler = StandardScaler()
        Xt_tr = t_scaler.fit_transform(Xt_tr)
        Xt_va = t_scaler.transform(Xt_va)
        Xt_te = t_scaler.transform(Xt_te)

        # Normalise targets (z-score based on training set)
        y_mu, y_sigma = y_tr.mean(), max(y_tr.std(), 1e-9)
        y_tr_n = (y_tr - y_mu) / y_sigma
        y_va_n = (y_va - y_mu) / y_sigma

        # Build and train model
        model = build_model(Xa_tr.shape[1], Xt_tr.shape[1])
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=PATIENCE_ES,
                restore_best_weights=True, verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=PATIENCE_LR, min_lr=1e-6, verbose=0,
            ),
        ]

        history = model.fit(
            [Xa_tr, Xt_tr], y_tr_n,
            validation_data=([Xa_va, Xt_va], y_va_n),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )
        print(f"  Stopped at epoch {len(history.history['loss'])}")

        # Predictions (de-normalise)
        def predict(Xa, Xt):
            return model.predict([Xa, Xt], verbose=0).flatten() * y_sigma + y_mu

        y_pred_tr = predict(Xa_tr, Xt_tr)
        y_pred_va = predict(Xa_va, Xt_va)
        y_pred_te = predict(Xa_te, Xt_te)

        # Metrics
        m_tr = compute_metrics(y_tr, y_pred_tr)
        m_va = compute_metrics(y_va, y_pred_va)
        m_te = compute_metrics(y_te, y_pred_te)

        print(f"  Train  MAE={m_tr['MAE']:.3f}  RMSE={m_tr['RMSE']:.3f}"
              f"  R²={m_tr['R2']:.3f}  MAPE={m_tr['MAPE']:.2f}%  PPA={m_tr['PPA']:.1f}%")
        print(f"  Val    MAE={m_va['MAE']:.3f}  RMSE={m_va['RMSE']:.3f}"
              f"  R²={m_va['R2']:.3f}  MAPE={m_va['MAPE']:.2f}%  PPA={m_va['PPA']:.1f}%")
        print(f"  Test   MAE={m_te['MAE']:.3f}  RMSE={m_te['RMSE']:.3f}"
              f"  R²={m_te['R2']:.3f}  MAPE={m_te['MAPE']:.2f}%  PPA={m_te['PPA']:.1f}%")

        fold_results.append({
            "fold": fold_idx,
            "train_size": len(train_idx), "val_size": len(val_idx), "test_size": len(test_idx),
            "train_mae": m_tr["MAE"],  "val_mae": m_va["MAE"],  "test_mae": m_te["MAE"],
            "train_rmse": m_tr["RMSE"], "val_rmse": m_va["RMSE"], "test_rmse": m_te["RMSE"],
            "train_r2":  m_tr["R2"],   "val_r2":  m_va["R2"],   "test_r2":  m_te["R2"],
            "train_mape": m_tr["MAPE"], "val_mape": m_va["MAPE"], "test_mape": m_te["MAPE"],
            "train_ppa": m_tr["PPA"],  "val_ppa": m_va["PPA"],  "test_ppa": m_te["PPA"],
        })

        all_histories.append(history.history)

        all_preds.append({
            "fold": fold_idx,
            "y_train_actual": y_tr.tolist(),
            "y_train_pred":   y_pred_tr.tolist(),
            "y_val_actual":   y_va.tolist(),
            "y_val_pred":     y_pred_va.tolist(),
            "y_test_actual":  y_te.tolist(),
            "y_test_pred":    y_pred_te.tolist(),
        })

        # Save per-fold model
        model_path = os.path.join(RESULTS_DIR, f"model_fold_{fold_idx}.h5")
        model.save(model_path)
        print(f"  Model saved → {model_path}")

        keras.backend.clear_session()

    return fold_results, all_histories, all_preds


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Summary statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(fold_results: list[dict]) -> pd.DataFrame:
    """Compute mean ± std across folds for every metric."""
    df = pd.DataFrame(fold_results)
    metric_cols = [c for c in df.columns if c not in ("fold", "train_size", "val_size", "test_size")]

    rows = []
    for col in metric_cols:
        rows.append({
            "metric": col,
            "mean": df[col].mean(),
            "std":  df[col].std(),
            "min":  df[col].min(),
            "max":  df[col].max(),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Persistence helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_results(fold_results, all_histories, all_preds):
    # Per-fold metrics
    df_folds = pd.DataFrame(fold_results)
    df_folds.to_csv(os.path.join(RESULTS_DIR, "fold_metrics.csv"), index=False)
    print(f"\nPer-fold metrics → {os.path.join(RESULTS_DIR, 'fold_metrics.csv')}")

    # Summary
    df_summary = compute_summary(fold_results)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "summary_metrics.csv"), index=False)
    print(f"Summary metrics  → {os.path.join(RESULTS_DIR, 'summary_metrics.csv')}")

    # Training histories
    with open(os.path.join(RESULTS_DIR, "training_histories.json"), "w") as f:
        json.dump(all_histories, f, indent=2)

    # Predictions
    with open(os.path.join(RESULTS_DIR, "predictions.json"), "w") as f:
        json.dump(all_preds, f, indent=2)

    return df_folds, df_summary


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Inline quick-plots (full plots in evaluate_and_visualize.py)
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_histories(all_histories):
    n = len(all_histories)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for i, (hist, ax) in enumerate(zip(all_histories, axes), start=1):
        ax.plot(hist["loss"],     label="Train loss", linewidth=1.5)
        ax.plot(hist["val_loss"], label="Val loss",   linewidth=1.5, linestyle="--")
        ax.set_title(f"Fold {i}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Training Loss History (all folds)", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "training_loss_history.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  FHR Prediction – Deep Learning Pipeline")
    print("=" * 60)

    csv_path = os.path.join(DATASET_DIR, "Records.csv")
    print(f"\nLoading clinical data from: {csv_path}")
    df = load_clinical_data(csv_path)
    print(f"Rows with valid FHR: {len(df)}")

    print("\nBuilding dataset (extracting audio + tabular features)...")
    audio_arr, tabular_arr, targets, subject_ids = build_dataset(df)

    print(f"\nStarting {N_FOLDS}-Fold Cross-Validation...")
    fold_results, all_histories, all_preds = run_cross_validation(
        audio_arr, tabular_arr, targets
    )

    print("\nSaving results...")
    df_folds, df_summary = save_results(fold_results, all_histories, all_preds)

    print("\nPlotting training histories...")
    plot_training_histories(all_histories)

    # Print final summary table
    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION SUMMARY  (mean ± std)")
    print("=" * 60)
    key_metrics = [
        ("test_mae",  "Test  MAE  (bpm)"),
        ("test_rmse", "Test  RMSE (bpm)"),
        ("test_r2",   "Test  R²        "),
        ("test_mape", "Test  MAPE (%)  "),
        ("test_ppa",  "Test  PPA  (%)  "),
        ("val_mae",   "Val   MAE  (bpm)"),
        ("train_mae", "Train MAE  (bpm)"),
    ]
    for col, label in key_metrics:
        row = df_summary[df_summary["metric"] == col].iloc[0]
        print(f"  {label}: {row['mean']:.4f} ± {row['std']:.4f}")
    print("=" * 60)
    print("\nDone.  All results saved to:", RESULTS_DIR)
    print("Run evaluate_and_visualize.py to generate comprehensive plots.\n")


if __name__ == "__main__":
    main()
