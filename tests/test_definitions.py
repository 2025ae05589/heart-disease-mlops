"""Tests for src/models/definitions.py: the four models and their grids.

It is checked that every setting named in a grid exists in the pipeline,
that each grid has the expected number of combinations, that the seed of
each model is fixed, and that only the XGBoost model asks for the xgboost
package.
"""
import pytest
from sklearn.model_selection import ParameterGrid

from src.config import RANDOM_STATE
from src.features.preprocessing import make_pipeline
from src.models.definitions import MODELS
from src.tracking.experiments import model_requirements

COMBINATIONS = {
    "Logistic Regression": 4,
    "Random Forest": 18,
    "Gradient Boosting": 18,
    "XGBoost": 19,
}


def test_the_four_models():
    assert list(MODELS) == list(COMBINATIONS)


@pytest.mark.parametrize("model_name", list(MODELS))
def test_grid_settings_exist_in_the_pipeline(model_name):
    model, grid = MODELS[model_name]
    known_settings = make_pipeline(model).get_params()

    for combination in ParameterGrid(grid):
        for setting in combination:
            assert setting in known_settings


@pytest.mark.parametrize("model_name", list(MODELS))
def test_number_of_combinations(model_name):
    _, grid = MODELS[model_name]

    assert len(ParameterGrid(grid)) == COMBINATIONS[model_name]


@pytest.mark.parametrize("model_name", list(MODELS))
def test_random_state_is_fixed(model_name):
    model, _ = MODELS[model_name]

    assert model.get_params()["random_state"] == RANDOM_STATE


@pytest.mark.parametrize("model_name", list(MODELS))
def test_xgboost_is_required_only_by_the_xgboost_model(model_name):
    model, _ = MODELS[model_name]
    packages = []
    for requirement in model_requirements(make_pipeline(model)):
        packages.append(requirement.split("==")[0])

    assert ("xgboost" in packages) == (model_name == "XGBoost")
    assert {"scikit-learn", "pandas", "numpy", "skops"} <= set(packages)
