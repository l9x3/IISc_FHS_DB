"""
Inference Script for Fetal Heart Rate Prediction
Usage:
    python inference.py --input new_features.csv
    python inference.py --input new_features.csv --model ensemble
Available model choices: dense_dropout | batchnorm_deep | residual_net | ensemble
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf

MODELS_DIR = "saved_models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_params.json")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_scaler_params(path: str = SCALER_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler parameters not found at '{path}'. "
            "Run train_model.py first."
        )
    with open(path) as f:
        return json.load(f)


def normalise(X: np.ndarray, scaler_params: dict) -> np.ndarray:
    mean = np.array(scaler_params["mean"], dtype=np.float32)
    scale = np.array(scaler_params["scale"], dtype=np.float32)
    return (X - mean) / scale


def load_keras_model(name: str) -> tf.keras.Model:
    path = os.path.join(MODELS_DIR, f"{name}.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{name}' not found at '{path}'. "
            "Run train_model.py first."
        )
    return tf.keras.models.load_model(path)


# ─── Ensemble helper ──────────────────────────────────────────────────────────

class EnsembleModel:
    def __init__(self, models: list, weights: list = None):
        self.models = models
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = np.array(weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack(
            [m.predict(X, verbose=0).ravel() for m in self.models], axis=1
        )
        return (preds * self.weights).sum(axis=1)


def load_report_weights() -> list:
    """Read ensemble weights saved during training."""
    report_path = "model_report.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        ew = report.get("ensemble_weights", {})
        return [
            ew.get("dense_dropout", 1 / 3),
            ew.get("batchnorm_deep", 1 / 3),
            ew.get("residual_net", 1 / 3),
        ]
    return [1 / 3, 1 / 3, 1 / 3]


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict(input_path: str, model_name: str = "ensemble",
            output_path: str = None) -> pd.DataFrame:
    """
    Load features from ``input_path``, normalise them, and return predictions.

    Parameters
    ----------
    input_path  : path to CSV file containing feature columns.
                  Must have the same feature columns as the training data.
                  A 'Subject' column is dropped if present.
    model_name  : one of 'dense_dropout', 'batchnorm_deep',
                  'residual_net', 'ensemble'.
    output_path : if provided, save predictions CSV to this path.

    Returns
    -------
    pd.DataFrame with original rows plus a 'Predicted_Heart_Rate' column.
    """
    # 1. Load input
    df = pd.read_csv(input_path)
    df_orig = df.copy()
    df = df.drop(columns=["Subject", "Heart_Rate"], errors="ignore")

    X = df.values.astype(np.float32)

    # 2. Load scaler and normalise
    scaler_params = load_scaler_params()
    expected_features = scaler_params["feature_names"]

    # Align columns to training feature order
    if list(df.columns) != expected_features:
        try:
            X = df[expected_features].values.astype(np.float32)
        except KeyError as e:
            raise ValueError(
                f"Input CSV is missing expected feature column: {e}"
            )

    X_norm = normalise(X, scaler_params)

    # 3. Load model(s) and predict
    if model_name == "ensemble":
        models = [
            load_keras_model("dense_dropout"),
            load_keras_model("batchnorm_deep"),
            load_keras_model("residual_net"),
        ]
        weights = load_report_weights()
        predictor = EnsembleModel(models, weights)
        y_pred = predictor.predict(X_norm)
    else:
        model = load_keras_model(model_name)
        y_pred = model.predict(X_norm, verbose=0).ravel()

    # 4. Assemble output
    df_orig["Predicted_Heart_Rate"] = np.round(y_pred, 2)

    if output_path:
        df_orig.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    return df_orig


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict fetal heart rate from feature CSV."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input CSV file with feature columns."
    )
    parser.add_argument(
        "--model", "-m", default="ensemble",
        choices=["dense_dropout", "batchnorm_deep", "residual_net", "ensemble"],
        help="Which model to use for inference (default: ensemble)."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to save prediction CSV (optional)."
    )
    args = parser.parse_args()

    results = predict(args.input, model_name=args.model,
                      output_path=args.output)

    print("\nPredictions:")
    cols_to_show = ["Subject"] if "Subject" in results.columns else []
    cols_to_show.append("Predicted_Heart_Rate")
    if "Heart_Rate" in results.columns:
        cols_to_show.append("Heart_Rate")
    print(results[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
