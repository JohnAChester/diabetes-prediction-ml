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

## Running the Scripts

The two `.py` files are standalone versions of the notebooks. Run them in order:

```
preprocessing.py >> modeling.py
```

### 1. Install dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

### 2. Configure Kaggle credentials

`preprocessing.py` downloads the raw BRFSS dataset automatically on first run. You need a Kaggle account and API token. See the [Kaggle API docs](https://github.com/Kaggle/kaggle-api) for setup.

### 3. Run preprocessing

```bash
python preprocessing.py
```

This will:
- Download and unzip `data/2015.csv` from Kaggle (skipped if already present)
- Select 21 clinically-relevant features from the 330-column BRFSS survey
- Clean missing/invalid codes, recode binary features, rescale BMI
- Split data 80/20 (stratified) and create a 50/50 undersampled training set
- Write four CSVs to `data/`: `clean_2015_full.csv`, `train.csv`, `test.csv`, `train_balanced.csv`
- Save EDA charts to `outputs/`

### 4. Run modeling

```bash
python modeling.py
```

This will:
- Load the split CSVs from `data/`
- Train and tune four models via 5-fold CV: Logistic Regression, Linear SVM, Kernel SVM (RBF), and a PyTorch MLP
- Run three MLP iterations (Adam → AdamW → AdamW + tuned pos_weight)
- Save model checkpoints: `_best_mlp_v1.pt`, `_best_mlp_v2.pt`, `_best_mlp_v3.pt`
- Save ROC curves, confusion matrices, and a summary table to `outputs/`

> **Note:** The Kernel SVM subsamples 20,000 rows due to memory constraints. The MLP grid search trains 9 configurations per iteration — expect the full modeling run to take 20–40 minutes on CPU.

## Requirements

Python 3.11+. Dependencies are listed in `pyproject.toml` and pinned in `uv.lock`.
