# IISc_FHS_DB – Fetal Heart Sound Dataset

Clinical deep-learning model that predicts **Fetal Heart Rate (FHR)** from
tabular patient records, using data from the
[IISc Fetal Heart Sound Database](https://www.physionet.org/content/fetalheartsounddata/1.0/).

---

## Dataset

`Records.csv` – 60 pregnant women with 8 clinical features:

| Column | Description |
|---|---|
| `gestational_age` | Gestational age (weeks) |
| `maternal_age` | Maternal age (years) |
| `weight` | Maternal weight (kg) |
| `height` | Maternal height (cm) |
| `gravida` | Number of pregnancies |
| `para` | Number of live births |
| `bmi` | Body Mass Index (kg/m²) |
| `clinical_risk_score` | Composite clinical risk score (0–5) |
| `fhr` | **Target** – Fetal Heart Rate (bpm); 44/60 samples have measurements |

---

## Model Architecture

```
Input (8 features)
  │
  ├─ Dense(64) → BatchNorm → ReLU → Dropout(0.3)
  ├─ Dense(32) → BatchNorm → ReLU → Dropout(0.2)
  ├─ Dense(16) → BatchNorm → ReLU → Dropout(0.1)
  └─ Dense(1, sigmoid) → scale to [110, 160] bpm
```

* **Optimizer**: Adam (lr=1e-3)
* **Loss**: Mean Squared Error (MSE)
* **Early stopping**: patience=20, monitored on val_loss
* **LR reduction**: ×0.5 when val_loss plateaus (patience=10)

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train & evaluate
python fhr_prediction_model.py
```

### Example output

```
=======================================================
  Test-Set Evaluation Metrics
=======================================================
  MAE   (Mean Absolute Error)         : X.XXXX bpm
  RMSE  (Root Mean Squared Error)     : X.XXXX bpm
  R²    Score                         : X.XXXX
  MAPE  (Mean Abs. Percentage Error)  : X.XXXX %
=======================================================
```

Visualisations are saved to `outputs/`:

| File | Description |
|---|---|
| `training_history.png` | Loss & MAE curves over epochs |
| `predictions_vs_actual.png` | Predicted vs actual FHR scatter |
| `residual_plot.png` | Residuals vs predicted values |
| `error_distribution.png` | Histogram of prediction errors |
