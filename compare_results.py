"""
compare_results.py
------------------
Compares the trained deep-learning FHR model against classical ML baselines
using the same 5-fold cross-validation split produced by train_fhr_model.py.

Baseline models (trained on tabular features unless noted):
  1. Mean baseline          – always predicts the training-set mean FHR
  2. Linear Regression
  3. Ridge Regression
  4. Random Forest
  5. Gradient Boosting
  6. Random Forest (tabular + audio features)

The DL model test predictions are loaded from results/predictions.json.

Outputs (all written to results/):
  - baseline_comparison.csv       – per-fold + mean±std metrics for every model
  - model_comparison_mae.png      – MAE grouped bar chart (per fold + mean)
  - model_comparison_rmse.png     – RMSE grouped bar chart
  - model_comparison_r2.png       – R² grouped bar chart
  - model_comparison_ppa.png      – PPA grouped bar chart
  - model_comparison_summary.png  – Heatmap of mean test metrics across models
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Import shared helpers from the training script
from train_fhr_model import (
    BASE_DIR,
    DATASET_DIR,
    N_FOLDS,
    RESULTS_DIR,
    SEED,
    build_dataset,
    compute_metrics,
    load_clinical_data,
)

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")

# ──────────────────────────────────────────────────────────────────────────────
# Baseline model definitions
# ──────────────────────────────────────────────────────────────────────────────

BASELINE_MODELS = {
    "Mean Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=SEED
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def savefig(name: str) -> None:
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def load_dl_metrics(pred_json_path: str) -> list[dict]:
    """
    Load per-fold test predictions saved by train_fhr_model.py and
    recompute test metrics to ensure consistency.
    """
    with open(pred_json_path) as f:
        preds = json.load(f)

    fold_metrics = []
    for p in preds:
        ya = np.array(p["y_test_actual"])
        yp = np.array(p["y_test_pred"])
        m = compute_metrics(ya, yp)
        fold_metrics.append(
            {
                "fold": p["fold"],
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"],
                "MAPE": m["MAPE"],
                "PPA": m["PPA"],
            }
        )
    return fold_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Core comparison routine
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_comparison(
    audio_arr: np.ndarray,
    tabular_arr: np.ndarray,
    targets: np.ndarray,
    dl_fold_metrics: list[dict],
) -> pd.DataFrame:
    """
    Runs all baseline models with the same 5-fold split used by train_fhr_model.py
    and returns a DataFrame with per-fold metrics for every model.
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)

    # Impute missing tabular values (same strategy as training script)
    tab_imputer = SimpleImputer(strategy="median")
    tab_imputed = tab_imputer.fit_transform(tabular_arr)

    records: list[dict] = []

    for fold_idx, (tv_idx, test_idx) in enumerate(kf.split(targets), start=1):
        # Reproduce the exact train / val / test split from train_fhr_model.py
        tv_shuffled = rng.permutation(tv_idx)
        val_n = max(1, int(0.20 * len(tv_shuffled)))
        val_idx = tv_shuffled[:val_n]
        train_idx = tv_shuffled[val_n:]

        y_tr = targets[train_idx]
        y_te = targets[test_idx]

        # Scale tabular features (fit on train only)
        t_scaler = StandardScaler()
        Xt_tr = t_scaler.fit_transform(tab_imputed[train_idx])
        Xt_te = t_scaler.transform(tab_imputed[test_idx])

        # Scale audio features (fit on train only)
        a_scaler = StandardScaler()
        Xa_tr = a_scaler.fit_transform(audio_arr[train_idx])
        Xa_te = a_scaler.transform(audio_arr[test_idx])

        # Combined feature matrix (tabular + audio)
        Xcomb_tr = np.concatenate([Xt_tr, Xa_tr], axis=1)
        Xcomb_te = np.concatenate([Xt_te, Xa_te], axis=1)

        # ── Tabular-only baselines ──────────────────────────────────────────
        for model_name, clf in BASELINE_MODELS.items():
            clf.fit(Xt_tr, y_tr)
            y_pred = clf.predict(Xt_te)
            m = compute_metrics(y_te, y_pred)
            records.append(
                {
                    "model": model_name,
                    "features": "tabular",
                    "fold": fold_idx,
                    **{k: v for k, v in m.items()},
                }
            )

        # ── Random Forest on tabular + audio ───────────────────────────────
        rf_audio = RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1
        )
        rf_audio.fit(Xcomb_tr, y_tr)
        y_pred_rf_audio = rf_audio.predict(Xcomb_te)
        m = compute_metrics(y_te, y_pred_rf_audio)
        records.append(
            {
                "model": "RF (tabular + audio)",
                "features": "tabular+audio",
                "fold": fold_idx,
                **{k: v for k, v in m.items()},
            }
        )

        # ── Deep Learning (from saved predictions) ─────────────────────────
        dl = dl_fold_metrics[fold_idx - 1]
        records.append(
            {
                "model": "Deep Learning",
                "features": "tabular+audio",
                "fold": fold_idx,
                "MAE": dl["MAE"],
                "RMSE": dl["RMSE"],
                "R2": dl["R2"],
                "MAPE": dl["MAPE"],
                "PPA": dl["PPA"],
            }
        )

        print(
            f"  Fold {fold_idx}: baselines evaluated "
            f"(train={len(train_idx)}, test={len(test_idx)})"
        )

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std across folds for every model × metric."""
    metrics = ["MAE", "RMSE", "R2", "MAPE", "PPA"]
    rows = []
    for (model, feats), grp in df.groupby(["model", "features"], sort=False):
        row = {"model": model, "features": feats}
        for m in metrics:
            row[f"{m}_mean"] = grp[m].mean()
            row[f"{m}_std"] = grp[m].std()
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Visualisations
# ══════════════════════════════════════════════════════════════════════════════

MODEL_ORDER = [
    "Mean Baseline",
    "Linear Regression",
    "Ridge Regression",
    "Random Forest",
    "Gradient Boosting",
    "RF (tabular + audio)",
    "Deep Learning",
]

# Colour for each model
MODEL_COLOURS = [
    "#9E9E9E",  # grey  – Mean
    "#2196F3",  # blue  – LR
    "#03A9F4",  # light blue – Ridge
    "#FF9800",  # orange – RF
    "#F44336",  # red   – GB
    "#9C27B0",  # purple – RF+audio
    "#4CAF50",  # green – DL
]


def _model_palette(df_summary: pd.DataFrame) -> dict:
    models_present = df_summary["model"].tolist()
    return {
        m: c
        for m, c in zip(MODEL_ORDER, MODEL_COLOURS)
        if m in models_present
    }


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    lower_is_better: bool = True,
) -> None:
    """Grouped bar chart: one group per fold, one bar per model."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    folds = sorted(df["fold"].unique())
    n_models = len(models)
    n_folds = len(folds)

    x = np.arange(n_folds)
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    palette = _model_palette(df.assign(**{metric: df[metric]}))

    fig, ax = plt.subplots(figsize=(max(10, n_folds * 2.5), 5))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].sort_values("fold")
        values = [
            sub[sub["fold"] == f][metric].values[0] if len(sub[sub["fold"] == f]) else np.nan
            for f in folds
        ]
        ax.bar(
            x + offsets[i],
            values,
            width,
            label=model,
            color=palette.get(model, "#888888"),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    savefig(filename)


def plot_summary_heatmap(df_summary: pd.DataFrame) -> None:
    """Heatmap of mean test metrics across models."""
    metrics = ["MAE", "RMSE", "R2", "MAPE", "PPA"]
    models = [m for m in MODEL_ORDER if m in df_summary["model"].values]

    data = {}
    for m in metrics:
        col = f"{m}_mean"
        data[m] = [
            df_summary[df_summary["model"] == mdl][col].values[0]
            if len(df_summary[df_summary["model"] == mdl]) > 0
            else np.nan
            for mdl in models
        ]

    heatmap_df = pd.DataFrame(data, index=models)

    # Normalise each column to [0,1] for the colour map
    normed = heatmap_df.copy()
    for col in normed.columns:
        col_min, col_max = normed[col].min(), normed[col].max()
        denom = max(col_max - col_min, 1e-9)
        normed[col] = (normed[col] - col_min) / denom

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.7)))
    sns.heatmap(
        normed,
        annot=heatmap_df.round(3),
        fmt="g",
        cmap="YlOrRd_r",
        linewidths=0.5,
        cbar_kws={"label": "Normalised value (lower = better for error metrics)"},
        ax=ax,
    )
    ax.set_title(
        "Mean Test Metrics Across Models (5-Fold CV)\n"
        "annotated values = raw mean, colour = normalised (lower = better for MAE/RMSE/MAPE)",
        fontweight="bold",
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model")
    plt.tight_layout()
    savefig("model_comparison_summary.png")


def plot_mean_bar_chart(df_summary: pd.DataFrame) -> None:
    """Side-by-side bar chart of mean test MAE, RMSE, R², PPA for each model."""
    metrics = [("MAE", "MAE (bpm)"), ("RMSE", "RMSE (bpm)"), ("R2", "R²"), ("PPA", "PPA (%)")]
    models = [m for m in MODEL_ORDER if m in df_summary["model"].values]
    palette = _model_palette(df_summary)

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))

    for ax, (m, label) in zip(axes, metrics):
        means = [
            df_summary[df_summary["model"] == mdl][f"{m}_mean"].values[0]
            for mdl in models
        ]
        stds = [
            df_summary[df_summary["model"] == mdl][f"{m}_std"].values[0]
            for mdl in models
        ]
        colours = [palette.get(mdl, "#888888") for mdl in models]
        x = np.arange(len(models))
        ax.bar(x, means, yerr=stds, capsize=4, color=colours, alpha=0.85,
               edgecolor="white", error_kw={"elinewidth": 1.5, "ecolor": "gray"})
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="y", alpha=0.4)
        if m == "R2":
            ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.suptitle(
        "Model Comparison – Mean ± Std of Test Metrics (5-Fold CV)",
        fontweight="bold",
        fontsize=12,
    )
    plt.tight_layout()
    savefig("model_comparison_mean.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  FHR Model Comparison – DL vs Classical ML Baselines")
    print("=" * 65)

    # 1. Load dataset
    csv_path = os.path.join(DATASET_DIR, "Records.csv")
    print(f"\nLoading clinical data from: {csv_path}")
    df_clinical = load_clinical_data(csv_path)
    print(f"Rows with valid FHR: {len(df_clinical)}")

    print("\nBuilding feature arrays (audio + tabular)...")
    audio_arr, tabular_arr, targets, _ = build_dataset(df_clinical)

    # 2. Load saved DL predictions
    pred_json = os.path.join(RESULTS_DIR, "predictions.json")
    if not os.path.exists(pred_json):
        raise FileNotFoundError(
            f"DL predictions not found: {pred_json}\n"
            "Please run train_fhr_model.py first."
        )
    print(f"\nLoading DL predictions from: {pred_json}")
    dl_fold_metrics = load_dl_metrics(pred_json)

    # 3. Run baseline comparison
    print(f"\nRunning {N_FOLDS}-fold comparison with baseline models...")
    df_results = run_baseline_comparison(audio_arr, tabular_arr, targets, dl_fold_metrics)

    # 4. Build summary
    df_summary = build_summary(df_results)

    # 5. Save CSV outputs
    results_csv = os.path.join(RESULTS_DIR, "baseline_comparison.csv")
    df_results.to_csv(results_csv, index=False)
    print(f"\nPer-fold comparison → {results_csv}")

    summary_csv = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Summary comparison  → {summary_csv}")

    # 6. Generate plots
    print("\nGenerating comparison plots...")

    plot_metric_comparison(
        df_results, "MAE", "MAE (bpm)",
        "MAE – All Models per Fold", "model_comparison_mae.png",
        lower_is_better=True,
    )
    plot_metric_comparison(
        df_results, "RMSE", "RMSE (bpm)",
        "RMSE – All Models per Fold", "model_comparison_rmse.png",
        lower_is_better=True,
    )
    plot_metric_comparison(
        df_results, "R2", "R² Score",
        "R² – All Models per Fold", "model_comparison_r2.png",
        lower_is_better=False,
    )
    plot_metric_comparison(
        df_results, "PPA", "PPA (%)",
        "PPA (±10 % threshold) – All Models per Fold", "model_comparison_ppa.png",
        lower_is_better=False,
    )
    plot_summary_heatmap(df_summary)
    plot_mean_bar_chart(df_summary)

    # 7. Print summary table
    print("\n" + "=" * 65)
    print("  COMPARISON SUMMARY  (mean ± std across 5 folds)")
    print("=" * 65)
    col_w = max(len(m) for m in df_summary["model"].values) + 2
    header = f"{'Model':<{col_w}} {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  {'PPA (%)':>9}"
    print(header)
    print("-" * len(header))
    for _, row in df_summary.iterrows():
        model_str = row["model"]
        print(
            f"{model_str:<{col_w}} "
            f"{row['MAE_mean']:>6.3f}±{row['MAE_std']:.3f}  "
            f"{row['RMSE_mean']:>6.3f}±{row['RMSE_std']:.3f}  "
            f"{row['R2_mean']:>6.3f}±{row['R2_std']:.3f}  "
            f"{row['PPA_mean']:>7.2f}±{row['PPA_std']:.2f}"
        )
    print("=" * 65)
    print(f"\nDone. All comparison outputs saved to: {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()
