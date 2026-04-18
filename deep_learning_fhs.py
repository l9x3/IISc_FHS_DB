"""
Deep Learning Model for Fetal Heart Sound Dataset (IISc_FHS_DB)
5-Fold Cross-Validation with comprehensive metrics and visualizations
"""

import os
import json
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "IISc_features_complete_subset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for sub in ["models", "history", "predictions", "metrics", "plots"]:
    os.makedirs(os.path.join(RESULTS_DIR, sub), exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-3
PATIENCE = 30
VAL_SPLIT = 0.20          # fraction of train+val used as validation
PPA_TOLERANCE = 5.0       # ±5 bpm acceptable range for PPA
# Physiological clipping bounds for fetal heart rate (bpm)
HR_MIN, HR_MAX = 100.0, 200.0


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, prefix=""):
    """Return dict of regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE – guard against zero and near-zero targets
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / safe_denom)) * 100
    r2 = r2_score(y_true, y_pred)
    ppa = np.mean(np.abs(y_true - y_pred) <= PPA_TOLERANCE) * 100
    key = (prefix + "_") if prefix else ""
    return {
        f"{key}MAE": mae,
        f"{key}RMSE": rmse,
        f"{key}MAPE": mape,
        f"{key}R2": r2,
        f"{key}PPA": ppa,
    }


def build_model(input_dim: int) -> keras.Model:
    """Fully-connected regression network with batch-norm and dropout."""
    inp = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, name="heart_rate")(x)

    model = keras.Model(inputs=inp, outputs=out, name="FHS_Net")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Load & preprocess data
# ─────────────────────────────────────────────────────────────────────────────

log.info("Loading dataset …")
df = pd.read_csv(DATA_PATH)
log.info("Dataset shape: %s", df.shape)

feature_cols = [c for c in df.columns if c not in ("Subject", "Heart_Rate")]
X_raw = df[feature_cols].values.astype(np.float32)
y_raw = df["Heart_Rate"].values.astype(np.float32)

log.info("Features: %d  |  Samples: %d", X_raw.shape[1], X_raw.shape[0])
log.info("Target range: %.1f – %.1f  (mean %.1f)", y_raw.min(), y_raw.max(), y_raw.mean())


# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

all_fold_metrics = []
all_histories = []
all_predictions = []

log.info("Starting %d-Fold Cross-Validation …", N_FOLDS)

for fold, (trainval_idx, test_idx) in enumerate(kf.split(X_raw), start=1):
    log.info("─── Fold %d / %d ───────────────────────────────", fold, N_FOLDS)

    # ── Split indices ─────────────────────────────────────────────────────────
    X_trainval, y_trainval = X_raw[trainval_idx], y_raw[trainval_idx]
    X_test, y_test = X_raw[test_idx], y_raw[test_idx]

    n_val = max(1, int(len(trainval_idx) * VAL_SPLIT))
    n_train = len(trainval_idx) - n_val
    rng = np.random.default_rng(SEED + fold)
    perm = rng.permutation(len(trainval_idx))
    train_sub, val_sub = perm[:n_train], perm[n_train:]
    X_train, y_train = X_trainval[train_sub], y_trainval[train_sub]
    X_val, y_val = X_trainval[val_sub], y_trainval[val_sub]

    log.info("  Train=%d  Val=%d  Test=%d", len(X_train), len(X_val), len(X_test))

    # ── Feature normalisation (fit on train only) ─────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ── Build & train ─────────────────────────────────────────────────────────
    model = build_model(X_train_s.shape[1])

    ckpt_path = os.path.join(RESULTS_DIR, "models", f"fold_{fold}_best.keras")
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0
        ),
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=0
        ),
    ]

    hist = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=0,
    )
    log.info("  Stopped at epoch %d", len(hist.history["loss"]))

    # ── Predictions (clipped to physiological range) ──────────────────────────
    y_train_pred = np.clip(model.predict(X_train_s, verbose=0).ravel(), HR_MIN, HR_MAX)
    y_val_pred = np.clip(model.predict(X_val_s, verbose=0).ravel(), HR_MIN, HR_MAX)
    y_test_pred = np.clip(model.predict(X_test_s, verbose=0).ravel(), HR_MIN, HR_MAX)

    # ── Metrics ───────────────────────────────────────────────────────────────
    train_m = compute_metrics(y_train, y_train_pred, "train")
    val_m = compute_metrics(y_val, y_val_pred, "val")
    test_m = compute_metrics(y_test, y_test_pred, "test")

    fold_metrics = {"fold": fold, **train_m, **val_m, **test_m}
    all_fold_metrics.append(fold_metrics)

    log.info(
        "  Train MAE=%.3f | Val MAE=%.3f | Test MAE=%.3f | "
        "RMSE=%.3f | R²=%.3f | PPA=%.1f%% | MAPE=%.2f%%",
        train_m["train_MAE"], val_m["val_MAE"], test_m["test_MAE"],
        test_m["test_RMSE"], test_m["test_R2"],
        test_m["test_PPA"], test_m["test_MAPE"],
    )

    # ── Save history ─────────────────────────────────────────────────────────
    hist_df = pd.DataFrame(hist.history)
    hist_df.index.name = "epoch"
    hist_df.to_csv(os.path.join(RESULTS_DIR, "history", f"fold_{fold}_history.csv"))
    all_histories.append(hist.history)

    # ── Save predictions ──────────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "split": (["train"] * len(y_train) + ["val"] * len(y_val) + ["test"] * len(y_test)),
        "actual": np.concatenate([y_train, y_val, y_test]),
        "predicted": np.concatenate([y_train_pred, y_val_pred, y_test_pred]),
    })
    pred_df.to_csv(
        os.path.join(RESULTS_DIR, "predictions", f"fold_{fold}_predictions.csv"),
        index=False,
    )
    all_predictions.append({"fold": fold, "test": (y_test, y_test_pred)})


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate results
# ─────────────────────────────────────────────────────────────────────────────

metrics_df = pd.DataFrame(all_fold_metrics)
metrics_df.set_index("fold", inplace=True)
metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics", "fold_metrics.csv"))

metric_keys = [
    "train_MAE", "val_MAE", "test_MAE",
    "test_RMSE", "test_PPA", "test_MAPE", "test_R2",
]
summary = {}
for k in metric_keys:
    if k in metrics_df.columns:
        summary[k] = {
            "mean": float(metrics_df[k].mean()),
            "std":  float(metrics_df[k].std(ddof=1)),
            "values": metrics_df[k].tolist(),
        }

with open(os.path.join(RESULTS_DIR, "metrics", "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

log.info("═" * 60)
log.info("SUMMARY  (mean ± std across %d folds)", N_FOLDS)
log.info("═" * 60)
for k, v in summary.items():
    log.info("  %-15s  %.4f ± %.4f", k, v["mean"], v["std"])

# ─────────────────────────────────────────────────────────────────────────────
# Summary report (TXT)
# ─────────────────────────────────────────────────────────────────────────────

report_path = os.path.join(RESULTS_DIR, "summary_report.txt")
with open(report_path, "w") as rpt:
    rpt.write("=" * 70 + "\n")
    rpt.write("  Fetal Heart Sound Dataset – Deep Learning Evaluation Report\n")
    rpt.write("=" * 70 + "\n\n")
    rpt.write(f"Dataset      : {DATA_PATH}\n")
    rpt.write(f"Samples      : {len(X_raw)}\n")
    rpt.write(f"Features     : {X_raw.shape[1]}\n")
    rpt.write(f"Target       : Heart_Rate  [{y_raw.min():.0f} – {y_raw.max():.0f} bpm]\n")
    rpt.write(f"CV folds     : {N_FOLDS}\n")
    rpt.write(f"PPA tolerance: ±{PPA_TOLERANCE} bpm\n\n")
    rpt.write("─" * 70 + "\n")
    rpt.write("  Per-Fold Metrics\n")
    rpt.write("─" * 70 + "\n")
    rpt.write(metrics_df[metric_keys].to_string() + "\n\n")
    rpt.write("─" * 70 + "\n")
    rpt.write("  Summary  (mean ± std)\n")
    rpt.write("─" * 70 + "\n")
    for k, v in summary.items():
        rpt.write(f"  {k:<18}  {v['mean']:.4f}  ±  {v['std']:.4f}\n")

log.info("Report saved → %s", report_path)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = plt.cm.tab10.colors
FOLD_LABELS = [f"Fold {i}" for i in range(1, N_FOLDS + 1)]

# ── 1. Training & Validation Loss/MAE per fold ───────────────────────────────
fig, axes = plt.subplots(N_FOLDS, 2, figsize=(14, 3 * N_FOLDS), sharex=False)
fig.suptitle("Training History per Fold", fontsize=15, fontweight="bold", y=1.01)
for i, h in enumerate(all_histories):
    ep = range(1, len(h["loss"]) + 1)
    axes[i, 0].plot(ep, h["loss"], label="Train Loss", color=COLORS[0])
    axes[i, 0].plot(ep, h["val_loss"], label="Val Loss", color=COLORS[1])
    axes[i, 0].set_title(f"Fold {i+1} – Loss (MSE)", fontsize=10)
    axes[i, 0].set_xlabel("Epoch"); axes[i, 0].set_ylabel("MSE")
    axes[i, 0].legend(fontsize=8)

    axes[i, 1].plot(ep, h["mae"], label="Train MAE", color=COLORS[2])
    axes[i, 1].plot(ep, h["val_mae"], label="Val MAE", color=COLORS[3])
    axes[i, 1].set_title(f"Fold {i+1} – MAE", fontsize=10)
    axes[i, 1].set_xlabel("Epoch"); axes[i, 1].set_ylabel("MAE (bpm)")
    axes[i, 1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "01_training_history.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── 2. Metrics comparison across folds (bar charts) ─────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()
fig.suptitle("Per-Fold Metrics Comparison", fontsize=14, fontweight="bold")
folds_x = np.arange(1, N_FOLDS + 1)
for ax_i, key in enumerate(metric_keys):
    if key not in metrics_df.columns:
        continue
    vals = metrics_df[key].values
    bars = axes[ax_i].bar(folds_x, vals, color=[COLORS[i % 10] for i in range(N_FOLDS)],
                          edgecolor="white", width=0.6)
    mean_val = vals.mean()
    axes[ax_i].axhline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mean_val:.3f}")
    axes[ax_i].set_title(key.replace("_", " ").title(), fontsize=11, fontweight="bold")
    axes[ax_i].set_xlabel("Fold"); axes[ax_i].set_ylabel("Value")
    axes[ax_i].set_xticks(folds_x); axes[ax_i].set_xticklabels(FOLD_LABELS, rotation=30)
    axes[ax_i].legend(fontsize=8)
    for bar, v in zip(bars, vals):
        axes[ax_i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
# hide unused subplot
for ax_i in range(len(metric_keys), len(axes)):
    axes[ax_i].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "02_metrics_per_fold.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── 3. Distribution with error bars ─────────────────────────────────────────
means = [summary[k]["mean"] for k in metric_keys if k in summary]
stds = [summary[k]["std"] for k in metric_keys if k in summary]
labels = [k.replace("_", " ").title() for k in metric_keys if k in summary]
x_pos = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x_pos, means, yerr=stds, align="center", alpha=0.8,
       color=COLORS[:len(labels)], edgecolor="white",
       error_kw=dict(ecolor="black", lw=2, capsize=6, capthick=2))
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_title("Mean ± Std of Metrics Across 5 Folds", fontsize=13, fontweight="bold")
ax.set_ylabel("Value")
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + max(means) * 0.01, f"{m:.3f}\n±{s:.3f}",
            ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "03_mean_std_metrics.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── 4. Actual vs Predicted scatter – all test folds ──────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.ravel()
all_actual_test, all_pred_test = [], []

for i, p in enumerate(all_predictions):
    y_t, y_p = p["test"]
    all_actual_test.extend(y_t)
    all_pred_test.extend(y_p)
    ax = axes_flat[i]
    ax.scatter(y_t, y_p, color=COLORS[i], edgecolor="white", s=60, alpha=0.85)
    lo, hi = min(y_raw) - 2, max(y_raw) + 2
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Perfect")
    mae_v = mean_absolute_error(y_t, y_p)
    r2_v = r2_score(y_t, y_p)
    ax.set_title(f"Fold {i+1}  |  MAE={mae_v:.2f}  R²={r2_v:.3f}", fontsize=10)
    ax.set_xlabel("Actual Heart Rate (bpm)")
    ax.set_ylabel("Predicted Heart Rate (bpm)")
    ax.legend(fontsize=8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

# ── Aggregate scatter ─────────────────────────────────────────────────────────
all_actual_test = np.array(all_actual_test)
all_pred_test = np.array(all_pred_test)
ax = axes_flat[5]
ax.scatter(all_actual_test, all_pred_test, color="steelblue", edgecolor="white", s=50, alpha=0.75)
lo, hi = min(y_raw) - 2, max(y_raw) + 2
ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect")
overall_mae = mean_absolute_error(all_actual_test, all_pred_test)
overall_r2 = r2_score(all_actual_test, all_pred_test)
ax.set_title(f"All Folds  |  MAE={overall_mae:.2f}  R²={overall_r2:.3f}", fontsize=10)
ax.set_xlabel("Actual Heart Rate (bpm)")
ax.set_ylabel("Predicted Heart Rate (bpm)")
ax.legend(fontsize=8)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

fig.suptitle("Actual vs Predicted Heart Rate", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "04_actual_vs_predicted.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── 5. Residuals per fold ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_FOLDS, figsize=(18, 4))
for i, p in enumerate(all_predictions):
    y_t, y_p = p["test"]
    residuals = y_t - y_p
    axes[i].bar(range(len(residuals)), residuals,
                color=[COLORS[0] if r >= 0 else COLORS[1] for r in residuals])
    axes[i].axhline(0, color="black", linewidth=1)
    axes[i].set_title(f"Fold {i+1} Residuals", fontsize=10)
    axes[i].set_xlabel("Sample"); axes[i].set_ylabel("Residual (bpm)")
fig.suptitle("Prediction Residuals per Fold", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "05_residuals.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── 6. Std deviation visualisation ───────────────────────────────────────────
std_keys = [k for k in metric_keys if k in summary]
std_vals = [summary[k]["std"] for k in std_keys]
std_labels = [k.replace("_", " ").title() for k in std_keys]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(std_labels, std_vals, color=COLORS[:len(std_keys)], edgecolor="white")
ax.set_xlabel("Standard Deviation")
ax.set_title("Standard Deviation of Metrics Across 5 Folds", fontsize=13, fontweight="bold")
for bar, v in zip(bars, std_vals):
    ax.text(v + max(std_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "06_std_deviation.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── 7. Box-plots of per-fold metrics ─────────────────────────────────────────
selected = ["test_MAE", "test_RMSE", "test_R2", "test_PPA", "test_MAPE"]
sel_vals = [metrics_df[k].values for k in selected if k in metrics_df.columns]
sel_labels = [k.replace("_", " ").title() for k in selected if k in metrics_df.columns]

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(sel_vals, labels=sel_labels, patch_artist=True,
                medianprops=dict(color="red", linewidth=2))
for patch, color in zip(bp["boxes"], COLORS[:len(sel_vals)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Distribution of Test Metrics Across 5 Folds", fontsize=13, fontweight="bold")
ax.set_ylabel("Value")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "07_boxplot_metrics.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── 8. Heatmap of all fold metrics ───────────────────────────────────────────
heat_df = metrics_df[metric_keys].copy()
heat_df.index = FOLD_LABELS
# normalise each column 0-1 for colour scale
heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(heat_norm.T, annot=heat_df.T.round(3), fmt="g",
            cmap="YlGnBu", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Normalised value"})
ax.set_title("Metrics Heatmap Across 5 Folds (values shown; color = normalised)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Fold"); ax.set_ylabel("Metric")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "08_heatmap.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── 9. MAE comparison: Train vs Val vs Test ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(N_FOLDS)
w = 0.25
for j, (key, label) in enumerate(
    [("train_MAE", "Train"), ("val_MAE", "Val"), ("test_MAE", "Test")]
):
    if key in metrics_df.columns:
        ax.bar(x + j * w, metrics_df[key].values, width=w, label=label, color=COLORS[j])
ax.set_xticks(x + w)
ax.set_xticklabels(FOLD_LABELS)
ax.set_title("MAE Comparison: Train / Val / Test per Fold", fontsize=13, fontweight="bold")
ax.set_ylabel("MAE (bpm)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "09_mae_comparison.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── 10. Combined history: mean ± std over all folds ──────────────────────────
max_epochs = max(len(h["loss"]) for h in all_histories)

def pad_series(series, length):
    """Repeat last value to pad to length."""
    arr = np.array(series)
    if len(arr) < length:
        arr = np.concatenate([arr, np.full(length - len(arr), arr[-1])])
    return arr

loss_mat = np.array([pad_series(h["loss"], max_epochs) for h in all_histories])
vloss_mat = np.array([pad_series(h["val_loss"], max_epochs) for h in all_histories])
mae_mat = np.array([pad_series(h["mae"], max_epochs) for h in all_histories])
vmae_mat = np.array([pad_series(h["val_mae"], max_epochs) for h in all_histories])

ep = np.arange(1, max_epochs + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for mat, label, ax, title in [
    (loss_mat, "Train Loss", axes[0], "Loss (MSE)"),
    (vloss_mat, "Val Loss", axes[0], "Loss (MSE)"),
    (mae_mat, "Train MAE", axes[1], "MAE (bpm)"),
    (vmae_mat, "Val MAE", axes[1], "MAE (bpm)"),
]:
    pass  # handled below

for ax, (m1, l1, m2, l2, title) in zip(axes, [
    (loss_mat, "Train Loss", vloss_mat, "Val Loss", "Loss (MSE) – Mean ± Std"),
    (mae_mat, "Train MAE", vmae_mat, "Val MAE", "MAE (bpm) – Mean ± Std"),
]):
    for mat, label, c in [(m1, l1, COLORS[0]), (m2, l2, COLORS[1])]:
        mean_ = mat.mean(axis=0)
        std_ = mat.std(axis=0)
        ax.plot(ep, mean_, color=c, label=label)
        ax.fill_between(ep, mean_ - std_, mean_ + std_, color=c, alpha=0.2)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.legend()
fig.suptitle("Mean ± Std Training History Across 5 Folds", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "10_mean_history.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── 11. Prediction error distribution ────────────────────────────────────────
residuals_all = all_actual_test - all_pred_test
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(residuals_all, bins=20, color=COLORS[0], edgecolor="white", alpha=0.8)
axes[0].axvline(0, color="red", linestyle="--")
axes[0].set_title("Residual Distribution (All Test Samples)", fontsize=11)
axes[0].set_xlabel("Residual (bpm)"); axes[0].set_ylabel("Count")

axes[1].scatter(all_pred_test, residuals_all, color=COLORS[1], edgecolor="white", s=50, alpha=0.75)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Residuals vs Predicted Values", fontsize=11)
axes[1].set_xlabel("Predicted Heart Rate (bpm)"); axes[1].set_ylabel("Residual (bpm)")
fig.suptitle("Error Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "11_error_analysis.png"), dpi=120, bbox_inches="tight")
plt.close()

log.info("All plots saved to %s", os.path.join(RESULTS_DIR, "plots"))
log.info("Done ✓")
