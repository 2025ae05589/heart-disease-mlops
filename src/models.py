"""The models that are compared, and the grid of settings tried for each."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluation import RANDOM_STATE

# "model__C" is the setting C of the pipeline step named "model".
# In each grid the values are written from the simplest to the most complex.

# the same values are tried for Gradient Boosting and XGBoost
BOOSTING_GRID = {
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [1, 2, 3],
    "model__n_estimators": [50, 100, 200],
}

# model name -> (model, grid)
MODELS = {
    "Logistic Regression": (
        LogisticRegression(random_state=RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [5, 3, 1],
            "model__n_estimators": [100, 300],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        BOOSTING_GRID,
    ),
    "XGBoost": (
        # n_jobs=1 because the grid search already runs the combinations in parallel
        XGBClassifier(random_state=RANDOM_STATE, n_jobs=1),
        [
            BOOSTING_GRID,
            # the default setting of XGBoost, which is outside the values above
            {
                "model__learning_rate": [0.3],
                "model__max_depth": [6],
                "model__n_estimators": [100],
            },
        ],
    ),
}
