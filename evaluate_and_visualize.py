"""
evaluate_and_visualize.py
--------------------------
Loads the cross-validation results produced by train_fhr_model.py and
generates comprehensive visualisations saved to the results/ directory.

Graphs produced
---------------
 1. mae_comparison.png          – Train / Val / Test MAE across folds
 2. rmse_comparison.png         – Train / Val / Test RMSE across folds
 3. r2_comparison.png           – Train / Val / Test R² across folds
 4. mape_comparison.png         – Train / Val / Test MAPE across folds
 5. ppa_comparison.png          – Train / Val / Test PPA across folds
 6. pred_vs_actual_fold_N.png   – Prediction vs Actual scatter (N=1..5)
 7. error_distribution.png      – Histogram + KDE of test residuals
 8. training_loss_history.png   – Already saved by train script; re-plotted here
 9. summary_metrics.png         – Summary bar chart (mean ± std for test metrics)
10. all_metrics_heatmap.png     – Heatmap of all fold × metric values
11. boxplot_metrics.png         – Box plots of test metrics across folds
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}

# Small constant to guard against division-by-zero in percentage metrics
EPSILON = 1e-9
sns.set_theme(style="whitegrid", palette="muted")


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_results():
    fold_csv    = os.path.join(RESULTS_DIR, "fold_metrics.csv")
    summary_csv = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    hist_json   = os.path.join(RESULTS_DIR, "training_histories.json")
    pred_json   = os.path.join(RESULTS_DIR, "predictions.json")

    df_folds   = pd.read_csv(fold_csv)
    df_summary = pd.read_csv(summary_csv)

    with open(hist_json) as f:
        histories = json.load(f)
    with open(pred_json) as f:
        preds = json.load(f)

    return df_folds, df_summary, histories, preds


def savefig(name: str):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. MAE comparison across folds
# ══════════════════════════════════════════════════════════════════════════════

def plot_mae_comparison(df: pd.DataFrame):
    folds = df["fold"].tolist()
    x = np.arange(len(folds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, df["train_mae"], width, label="Train",      color=PALETTE["train"], alpha=0.85)
    ax.bar(x,          df["val_mae"],  width, label="Validation",  color=PALETTE["val"],   alpha=0.85)
    ax.bar(x + width,  df["test_mae"], width, label="Test",        color=PALETTE["test"],  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("Mean Absolute Error – Train / Validation / Test (per fold)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    savefig("mae_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. RMSE comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_rmse_comparison(df: pd.DataFrame):
    folds = df["fold"].tolist()
    x = np.arange(len(folds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, df["train_rmse"], width, label="Train",      color=PALETTE["train"], alpha=0.85)
    ax.bar(x,          df["val_rmse"],  width, label="Validation",  color=PALETTE["val"],   alpha=0.85)
    ax.bar(x + width,  df["test_rmse"], width, label="Test",        color=PALETTE["test"],  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE (bpm)")
    ax.set_title("Root Mean Square Error – Train / Validation / Test (per fold)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    savefig("rmse_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. R² comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_r2_comparison(df: pd.DataFrame):
    folds = df["fold"].tolist()
    x = np.arange(len(folds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, df["train_r2"], width, label="Train",      color=PALETTE["train"], alpha=0.85)
    ax.bar(x,          df["val_r2"],  width, label="Validation",  color=PALETTE["val"],   alpha=0.85)
    ax.bar(x + width,  df["test_r2"], width, label="Test",        color=PALETTE["test"],  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score – Train / Validation / Test (per fold)", fontweight="bold")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    savefig("r2_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAPE comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_mape_comparison(df: pd.DataFrame):
    folds = df["fold"].tolist()
    x = np.arange(len(folds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, df["train_mape"], width, label="Train",      color=PALETTE["train"], alpha=0.85)
    ax.bar(x,          df["val_mape"],  width, label="Validation",  color=PALETTE["val"],   alpha=0.85)
    ax.bar(x + width,  df["test_mape"], width, label="Test",        color=PALETTE["test"],  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Mean Absolute Percentage Error – Train / Validation / Test (per fold)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    savefig("mape_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PPA comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_ppa_comparison(df: pd.DataFrame):
    folds = df["fold"].tolist()
    x = np.arange(len(folds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, df["train_ppa"], width, label="Train",      color=PALETTE["train"], alpha=0.85)
    ax.bar(x,          df["val_ppa"],  width, label="Validation",  color=PALETTE["val"],   alpha=0.85)
    ax.bar(x + width,  df["test_ppa"], width, label="Test",        color=PALETTE["test"],  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel("PPA (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Percentage Prediction Accuracy (±10 % threshold) (per fold)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    savefig("ppa_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Prediction vs Actual (per fold)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pred_vs_actual(preds: list[dict]):
    for p in preds:
        fold = p["fold"]
        ya = np.array(p["y_test_actual"])
        yp = np.array(p["y_test_pred"])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(ya, yp, color=PALETTE["test"], edgecolors="white",
                   s=70, alpha=0.85, zorder=3)

        # Perfect-prediction diagonal
        lo = min(ya.min(), yp.min()) - 5
        hi = max(ya.max(), yp.max()) + 5
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect fit")

        mae  = float(np.mean(np.abs(ya - yp)))
        rmse = float(np.sqrt(np.mean((ya - yp) ** 2)))
        ax.set_xlabel("Actual FHR (bpm)")
        ax.set_ylabel("Predicted FHR (bpm)")
        ax.set_title(f"Fold {fold} – Predicted vs Actual FHR\n"
                     f"MAE={mae:.2f} bpm  RMSE={rmse:.2f} bpm",
                     fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        savefig(f"pred_vs_actual_fold_{fold}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Error distribution (all test predictions pooled)
# ══════════════════════════════════════════════════════════════════════════════

def plot_error_distribution(preds: list[dict]):
    all_errors = []
    for p in preds:
        ya = np.array(p["y_test_actual"])
        yp = np.array(p["y_test_pred"])
        all_errors.extend((yp - ya).tolist())

    errors = np.array(all_errors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram + KDE
    axes[0].hist(errors, bins=15, color=PALETTE["test"], edgecolor="white",
                 alpha=0.8, density=True)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(errors)
        xs = np.linspace(errors.min() - 3, errors.max() + 3, 300)
        axes[0].plot(xs, kde(xs), color="darkgreen", linewidth=2, label="KDE")
        axes[0].legend()
    except Exception:
        pass
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Prediction Error (bpm)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of Test Prediction Errors", fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Q-Q plot
    from scipy.stats import probplot
    probplot(errors, plot=axes[1])
    axes[1].set_title("Q-Q Plot of Test Prediction Errors", fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    savefig("error_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Training loss history (all folds)
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_loss(histories: list[dict]):
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for i, (hist, ax) in enumerate(zip(histories, axes), start=1):
        ax.plot(hist["loss"],     label="Train", color=PALETTE["train"], linewidth=1.5)
        ax.plot(hist["val_loss"], label="Val",   color=PALETTE["val"],   linewidth=1.5, linestyle="--")
        ax.set_title(f"Fold {i}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Model Training Loss History (all folds)", fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig("training_loss_history.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Summary metrics bar chart (mean ± std)
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_metrics(df_summary: pd.DataFrame):
    test_rows = df_summary[df_summary["metric"].str.startswith("test_")].copy()
    test_rows["label"] = test_rows["metric"].str.replace("test_", "", regex=False).str.upper()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        test_rows["label"], test_rows["mean"],
        yerr=test_rows["std"], capsize=5,
        color=[PALETTE["test"]] * len(test_rows),
        alpha=0.85, edgecolor="white",
        error_kw={"elinewidth": 1.5, "ecolor": "gray"},
    )
    for bar, val in zip(bars, test_rows["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylabel("Value")
    ax.set_title("Test-Set Metrics: Mean ± Std across 5 Folds", fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    savefig("summary_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. Heatmap of all metrics
# ══════════════════════════════════════════════════════════════════════════════

def plot_metrics_heatmap(df: pd.DataFrame):
    metric_cols = [c for c in df.columns
                   if c not in ("fold", "train_size", "val_size", "test_size")]
    pivot = df.set_index("fold")[metric_cols]

    # Normalise each column to [0,1] for the colour scale
    normalised = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(max(12, len(metric_cols) * 0.9), 5))
    sns.heatmap(
        normalised, annot=pivot.round(3), fmt="g",
        cmap="YlGnBu", linewidths=0.5,
        cbar_kws={"label": "Normalised value"},
        ax=ax,
    )
    ax.set_title("All Metrics Across Folds (annotated values, colour = normalised)",
                 fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Fold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    savefig("all_metrics_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11. Box plots of test metrics
# ══════════════════════════════════════════════════════════════════════════════

def plot_boxplot_metrics(df: pd.DataFrame):
    test_cols = [c for c in df.columns if c.startswith("test_")]
    data_long = df[test_cols].melt(var_name="Metric", value_name="Value")
    data_long["Metric"] = data_long["Metric"].str.replace("test_", "", regex=False).str.upper()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=data_long, x="Metric", y="Value", palette="Set2", ax=ax)
    ax.set_title("Box Plot of Test Metrics across 5 Folds", fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    savefig("boxplot_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12. Comprehensive single-page dashboard
# ══════════════════════════════════════════════════════════════════════════════

def plot_dashboard(df: pd.DataFrame, preds: list[dict], histories: list[dict]):
    """4×3 grid showing all key plots on a single figure."""
    folds = df["fold"].tolist()
    x = np.arange(len(folds))
    w = 0.25

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0 ──────────────────────────────────────────────────────────────
    # MAE
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(x - w, df["train_mae"], w, label="Train", color=PALETTE["train"], alpha=0.85)
    ax0.bar(x,     df["val_mae"],   w, label="Val",   color=PALETTE["val"],   alpha=0.85)
    ax0.bar(x + w, df["test_mae"],  w, label="Test",  color=PALETTE["test"],  alpha=0.85)
    ax0.set_title("MAE", fontweight="bold"); ax0.set_ylabel("bpm")
    ax0.set_xticks(x); ax0.set_xticklabels([f"F{f}" for f in folds])
    ax0.legend(fontsize=7); ax0.grid(axis="y", alpha=0.3)

    # RMSE
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(x - w, df["train_rmse"], w, label="Train", color=PALETTE["train"], alpha=0.85)
    ax1.bar(x,     df["val_rmse"],   w, label="Val",   color=PALETTE["val"],   alpha=0.85)
    ax1.bar(x + w, df["test_rmse"],  w, label="Test",  color=PALETTE["test"],  alpha=0.85)
    ax1.set_title("RMSE", fontweight="bold"); ax1.set_ylabel("bpm")
    ax1.set_xticks(x); ax1.set_xticklabels([f"F{f}" for f in folds])
    ax1.legend(fontsize=7); ax1.grid(axis="y", alpha=0.3)

    # R²
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(x - w, df["train_r2"], w, label="Train", color=PALETTE["train"], alpha=0.85)
    ax2.bar(x,     df["val_r2"],   w, label="Val",   color=PALETTE["val"],   alpha=0.85)
    ax2.bar(x + w, df["test_r2"],  w, label="Test",  color=PALETTE["test"],  alpha=0.85)
    ax2.set_title("R²", fontweight="bold"); ax2.set_ylabel("R²")
    ax2.set_xticks(x); ax2.set_xticklabels([f"F{f}" for f in folds])
    ax2.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax2.legend(fontsize=7); ax2.grid(axis="y", alpha=0.3)

    # MAPE
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.bar(x - w, df["train_mape"], w, label="Train", color=PALETTE["train"], alpha=0.85)
    ax3.bar(x,     df["val_mape"],   w, label="Val",   color=PALETTE["val"],   alpha=0.85)
    ax3.bar(x + w, df["test_mape"],  w, label="Test",  color=PALETTE["test"],  alpha=0.85)
    ax3.set_title("MAPE (%)", fontweight="bold"); ax3.set_ylabel("%")
    ax3.set_xticks(x); ax3.set_xticklabels([f"F{f}" for f in folds])
    ax3.legend(fontsize=7); ax3.grid(axis="y", alpha=0.3)

    # ── Row 1  (pred vs actual – first 4 folds) ──────────────────────────
    for col_i, p in enumerate(preds[:4]):
        ax = fig.add_subplot(gs[1, col_i])
        ya = np.array(p["y_test_actual"])
        yp = np.array(p["y_test_pred"])
        ax.scatter(ya, yp, color=PALETTE["test"], s=40, alpha=0.85, edgecolors="white")
        lo, hi = min(ya.min(), yp.min()) - 3, max(ya.max(), yp.max()) + 3
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        mae = float(np.mean(np.abs(ya - yp)))
        ax.set_title(f"Fold {p['fold']} Pred vs Actual\nMAE={mae:.2f}", fontweight="bold", fontsize=9)
        ax.set_xlabel("Actual (bpm)", fontsize=8)
        ax.set_ylabel("Predicted (bpm)", fontsize=8)
        ax.grid(alpha=0.3)

    # ── Row 2  (training loss + error distribution + summary) ────────────
    # Training loss fold 1
    ax_loss = fig.add_subplot(gs[2, 0])
    ax_loss.plot(histories[0]["loss"],     label="Train", color=PALETTE["train"], linewidth=1.5)
    ax_loss.plot(histories[0]["val_loss"], label="Val",   color=PALETTE["val"],   linewidth=1.5, linestyle="--")
    ax_loss.set_title("Loss History (Fold 1)", fontweight="bold")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("MSE")
    ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3)

    # Error distribution
    ax_err = fig.add_subplot(gs[2, 1])
    all_errors = []
    for p in preds:
        ya = np.array(p["y_test_actual"])
        yp = np.array(p["y_test_pred"])
        all_errors.extend((yp - ya).tolist())
    ax_err.hist(np.array(all_errors), bins=15, color=PALETTE["test"],
                edgecolor="white", alpha=0.8, density=True)
    ax_err.axvline(0, color="red", linestyle="--", linewidth=1)
    ax_err.set_title("Test Error Distribution", fontweight="bold")
    ax_err.set_xlabel("Error (bpm)"); ax_err.set_ylabel("Density")
    ax_err.grid(alpha=0.3)

    # PPA
    ax_ppa = fig.add_subplot(gs[2, 2])
    ax_ppa.bar(x - w, df["train_ppa"], w, label="Train", color=PALETTE["train"], alpha=0.85)
    ax_ppa.bar(x,     df["val_ppa"],   w, label="Val",   color=PALETTE["val"],   alpha=0.85)
    ax_ppa.bar(x + w, df["test_ppa"],  w, label="Test",  color=PALETTE["test"],  alpha=0.85)
    ax_ppa.set_title("PPA (%)", fontweight="bold"); ax_ppa.set_ylabel("%")
    ax_ppa.set_xticks(x); ax_ppa.set_xticklabels([f"F{f}" for f in folds])
    ax_ppa.legend(fontsize=7); ax_ppa.grid(axis="y", alpha=0.3)

    # Summary bar chart
    ax_sum = fig.add_subplot(gs[2, 3])
    metrics_labels = ["MAE", "RMSE", "MAPE", "PPA"]
    means = [
        df["test_mae"].mean(), df["test_rmse"].mean(),
        df["test_mape"].mean(), df["test_ppa"].mean(),
    ]
    stds = [
        df["test_mae"].std(), df["test_rmse"].std(),
        df["test_mape"].std(), df["test_ppa"].std(),
    ]
    bars = ax_sum.bar(metrics_labels, means, yerr=stds, capsize=5,
                      color=[PALETTE["test"]] * 4, alpha=0.85, edgecolor="white",
                      error_kw={"elinewidth": 1.5, "ecolor": "gray"})
    for bar, val in zip(bars, means):
        ax_sum.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax_sum.set_title("Test Metrics (mean±std)", fontweight="bold")
    ax_sum.grid(axis="y", alpha=0.3)

    fig.suptitle("FHR Prediction – Cross-Validation Results Dashboard", fontsize=14,
                 fontweight="bold", y=1.01)
    savefig("dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  FHR Model Evaluation & Visualisation")
    print("=" * 60)

    if not os.path.exists(RESULTS_DIR):
        raise FileNotFoundError(
            f"Results directory '{RESULTS_DIR}' not found. "
            "Please run train_fhr_model.py first."
        )

    required = ["fold_metrics.csv", "summary_metrics.csv",
                "training_histories.json", "predictions.json"]
    for f in required:
        fp = os.path.join(RESULTS_DIR, f)
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"Required file not found: {fp}\n"
                "Please run train_fhr_model.py first."
            )

    print("\nLoading saved results...")
    df_folds, df_summary, histories, preds = load_results()

    print(f"\nGenerating {12} plots...")
    plot_mae_comparison(df_folds)
    plot_rmse_comparison(df_folds)
    plot_r2_comparison(df_folds)
    plot_mape_comparison(df_folds)
    plot_ppa_comparison(df_folds)
    plot_pred_vs_actual(preds)
    plot_error_distribution(preds)
    plot_training_loss(histories)
    plot_summary_metrics(df_summary)
    plot_metrics_heatmap(df_folds)
    plot_boxplot_metrics(df_folds)
    plot_dashboard(df_folds, preds, histories)

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    key = [
        ("test_mae",  "Test  MAE  (bpm)"),
        ("val_mae",   "Val   MAE  (bpm)"),
        ("train_mae", "Train MAE  (bpm)"),
        ("test_rmse", "Test  RMSE (bpm)"),
        ("test_r2",   "Test  R²        "),
        ("test_mape", "Test  MAPE (%)  "),
        ("test_ppa",  "Test  PPA  (%)  "),
    ]
    for col, label in key:
        row = df_summary[df_summary["metric"] == col]
        if len(row):
            r = row.iloc[0]
            print(f"  {label} :  {r['mean']:.4f} ± {r['std']:.4f}")

    print("=" * 60)
    print(f"\nAll plots saved to: {RESULTS_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
