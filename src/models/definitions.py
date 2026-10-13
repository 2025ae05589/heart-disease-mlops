"""The four models that are compared, each with its grid of settings.

MODELS gives, for each model name, the model and its grid. The grids are
kept in src/config.py.

Used by step 3 of the pipeline, the notebook and the tests.
"""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import GRIDS, RANDOM_STATE

# model name -> (model, grid)
MODELS = {
    "Logistic Regression": (
        LogisticRegression(random_state=RANDOM_STATE),
        GRIDS["Logistic Regression"],
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        GRIDS["Random Forest"],
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        GRIDS["Gradient Boosting"],
    ),
    "XGBoost": (
        # n_jobs=1 because the grid search already runs the combinations in parallel
        XGBClassifier(random_state=RANDOM_STATE, n_jobs=1),
        GRIDS["XGBoost"],
    ),
}
