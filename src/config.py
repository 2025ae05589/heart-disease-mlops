"""All the settings of the project, in one place.

Data source and file paths, split and cross-validation settings, metrics,
the selection rule, the quality gate, the grid of each model, MLflow names
and the log format. Only values are kept here; all other modules read
their settings from this file. The feature lists and their valid values come from
src/data/data_dictionary.py.
"""
import os
from pathlib import Path

# not used in this file; re-exported so that other modules can import them
# from here
from src.data.data_dictionary import (  # noqa: F401
    ABOVE_ZERO,
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    VALID_CODES,
    ZERO_OR_ABOVE,
)

# The project folder. The data paths below are relative to this folder; the
# functions that read or write the files put ROOT in front of them, so the
# code runs the same way from the project root and from the notebooks folder.
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- data source

# folder of the data files on the UCI site
BASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
)

# One file per hospital, with its row count from the UCI documentation and
# the SHA-256 hash of the file. Only Cleveland is used for modelling.
HOSPITALS = {
    "cleveland": (
        "processed.cleveland.data", 303,
        "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    ),
    "hungarian": (
        "processed.hungarian.data", 294,
        "d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796",
    ),
    "switzerland": (
        "processed.switzerland.data", 123,
        "834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314",
    ),
    "va": (
        "processed.va.data", 200,
        "e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f",
    ),
}

# The source files have no header row. The names and meanings are from the
# UCI documentation (heart-disease.names).
COLUMNS = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1 to 4)
    "trestbps",  # resting blood pressure in mm Hg
    "chol",      # serum cholesterol in mg/dl
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # resting electrocardiographic results (0 to 2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # slope of the peak exercise ST segment (1 to 3)
    "ca",        # number of major vessels (0 to 3) colored by fluoroscopy
    "thal",      # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num",       # diagnosis: 0 = no presence, 1 to 4 = presence
]

# ----------------------------------------------------------------- data files

RAW_DIR = Path("data/raw")
# file as published by UCI
SOURCE_PATH = RAW_DIR / "processed.cleveland.data"
# same data with column names added
RAW_PATH = RAW_DIR / "heart_cleveland_raw.csv"
# cleaned data, used for training
CLEAN_PATH = Path("data/processed/heart_clean.csv")

# ------------------------------------------------------------------- features

# NUMERIC, CATEGORICAL, FEATURES and the valid values are imported at the top
# from the data dictionary, as the API uses them too.
TARGET = "target"

# ------------------------------------------------- split and cross-validation

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10

# -------------------------------------------------------- metrics and choices

# name used here -> name used by scikit-learn
METRICS = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# decides the best combination of settings of a model
TUNING_METRIC = "roc_auc"

# models within this margin of the best mean ROC-AUC are treated as close
ROC_AUC_GAP = 0.01

# quality gate: lowest ROC-AUC on the test set accepted for the selected model
ROC_AUC_GATE = 0.80

# ---------------------------------------------------------- grids of settings

# "model__C" is the setting C of the pipeline step named "model".
# In each grid the values are written from the simplest to the most complex.

# the same values are tried for Gradient Boosting and XGBoost
BOOSTING_GRID = {
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [1, 2, 3],
    "model__n_estimators": [50, 100, 200],
}

# model name -> grid
GRIDS = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
    "Random Forest": {
        "model__max_depth": [3, 5, None],
        "model__min_samples_leaf": [5, 3, 1],
        "model__n_estimators": [100, 300],
    },
    "Gradient Boosting": BOOSTING_GRID,
    "XGBoost": [
        BOOSTING_GRID,
        # the default setting of XGBoost, which is outside the values above
        {
            "model__learning_rate": [0.3],
            "model__max_depth": [6],
            "model__n_estimators": [100],
        },
    ],
}

# --------------------------------------------------------------------- MLflow

# A local file by default. A shared MLflow server is used instead when its
# address is given in the environment variable MLFLOW_TRACKING_URI.
MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "sqlite:///" + (ROOT / "mlflow.db").as_posix()
)
# where the artifacts of the local store are kept
ARTIFACT_DIR = ROOT / "mlruns"
EXPERIMENT = "Heart Disease Prediction"

# name in the model registry; the aliases "challenger" and "champion" point
# to versions of it
MODEL_NAME = "heart-disease-classifier"
# the champion is copied here for the API and the Docker image
EXPORT_DIR = ROOT / "models" / "champion"

# -------------------------------------------------------------------- logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
