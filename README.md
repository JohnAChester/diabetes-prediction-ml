# Diabetes Prediction — ML Final Project

Binary classification of diabetes status using CDC BRFSS survey data (2015). Four models are trained and compared on a heavily imbalanced dataset (~84% non-diabetic, ~16% diabetic).

## Notebooks

| Notebook | Description |
|---|---|
| `project.ipynb` | Data download, EDA, preprocessing, and train/test split |
| `modeling.ipynb` | Model training, evaluation, and comparison |

## Models & Results

All models evaluated on the held-out test set (50,736 samples).

| Model | Accuracy | Recall (Diabetes) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.731 | 0.761 | 0.8177 |
| Linear SVM | 0.731 | 0.760 | 0.8175 |
| Kernel SVM (RBF) | 0.701 | 0.812 | 0.8205 |
| PyTorch MLP | 0.722 | 0.789 | **0.8249** |

## Data

Source: [CDC BRFSS on Kaggle](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)

The raw annual survey files (2011–2015, ~2.7 GB total) are excluded from this repo. `data/` contains the cleaned and split files used for modeling:

- `data/clean_2015_full.csv` — cleaned 2015 survey data
- `data/train.csv` / `data/test.csv` — train/test split (natural class distribution)
- `data/train_balanced.csv` — 50/50 balanced training set (used for SVM)
- `data/2015_formats.json` — BRFSS codebook

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

To re-download the raw data, configure the [Kaggle API](https://github.com/Kaggle/kaggle-api) and run the first cell of `project.ipynb`.
