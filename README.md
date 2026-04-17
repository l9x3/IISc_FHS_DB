# IISc_FHS_DB
Fetal Heart Sound Dataset

Reference: [PhysioNet Fetal Heart Sound Data 1.0](https://www.physionet.org/content/fetalheartsounddata/1.0/)

---

## Fetal Heart Rate (FHR) Prediction – Deep Learning Model

This repository contains a deep learning model that predicts Fetal Heart Rate (FHR) in bpm
from clinical features of pregnant women.

### Dataset (`Records.csv`)
- **60** total records (pregnant women)
- **44** records with confirmed FHR measurements (130–156 bpm)
- Clinical features: gestational age, maternal age, weight, height, gravida, para, clinical conditions

### Model Architecture
```
Input (8 features)
  → Dense(64, ReLU) + BatchNorm + Dropout(0.3)
  → Dense(32, ReLU) + BatchNorm + Dropout(0.3)
  → Dense(16, ReLU) + BatchNorm + Dropout(0.15)
  → Dense(1, Sigmoid) scaled to [110, 160] bpm
```

### Features Engineered
| # | Feature | Source |
|---|---------|--------|
| 1 | Gestational Age (weeks) | Direct |
| 2 | Maternal Age (years) | Direct |
| 3 | Weight (kg) | Direct, median imputation |
| 4 | Height (cm) | Direct, median imputation |
| 5 | Gravida | Direct, median imputation |
| 6 | Para | Direct, median imputation |
| 7 | BMI | Calculated: weight / height² |
| 8 | Clinical Risk Score | Mapped from Clinical Conditions |

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: MSE
- **Early stopping**: patience=20
- **Data split**: 70% train / 15% val / 15% test (~30/7/7 samples for 44 labelled records)

### How to Run

```bash
pip install -r requirements.txt
python fhr_prediction_model.py
```

### Outputs Generated
| File | Description |
|------|-------------|
| `fhr_training_history.png` | Loss and MAE training curves |
| `fhr_evaluation_panels.png` | 4-panel evaluation (predictions, residuals, error dist., MAPE) |
| `fhr_feature_importance.png` | Permutation-based clinical feature importance |
| `fhr_predictions.csv` | Per-sample actual vs predicted FHR |
| `fhr_evaluation_summary.txt` | Full evaluation report with all metrics |

### Evaluation Metrics (Test Set)
| Metric | Value |
|--------|-------|
| MAE (bpm) | ~2.5 |
| RMSE (bpm) | ~2.8 |
| MSE | ~7.8 |
| MAPE (%) | ~1.8 |
| R² | ~0.79 |
