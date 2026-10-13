"""Tests for src/models/definitions.py: the four models and their grids.

It is checked that every setting named in a grid exists in the pipeline,
that each grid has the expected number of combinations, and that the seed
of each model is fixed.
"""
import pytest
from sklearn.model_selection import ParameterGrid

from src.config import RANDOM_STATE
from src.features.preprocessing import make_pipeline
from src.models.definitions import MODELS

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
