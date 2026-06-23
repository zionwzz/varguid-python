from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from varguid import VarGuidRegressor, generate_cobra2d, lmv, lmv_formula, load_cobra2d


def _small_data() -> tuple[pd.DataFrame, pd.Series]:
    data = load_cobra2d().iloc[:90]
    return data[["x1", "x2", "x3", "x4"]].copy(), data["y"].copy()


def test_dataframe_prediction_reorders_columns_safely() -> None:
    predictors, response = _small_data()
    fit = lmv(predictors, response, M=2)

    prediction = fit.predict(predictors)
    reversed_prediction = fit.predict(predictors[predictors.columns[::-1]])
    assert np.allclose(prediction, reversed_prediction)


def test_dataframe_prediction_rejects_missing_or_extra_columns() -> None:
    predictors, response = _small_data()
    fit = lmv(predictors, response, M=2)

    with pytest.raises(ValueError, match="missing columns"):
        fit.predict(predictors.drop(columns=["x2"]))
    with pytest.raises(ValueError, match="unexpected columns"):
        fit.predict(predictors.assign(extra=1.0))


def test_array_fit_accepts_dataframe_positionally() -> None:
    predictors, response = _small_data()
    matrix = predictors.to_numpy()
    fit = lmv(matrix, response, M=2)
    renamed = pd.DataFrame(matrix, columns=["a", "b", "c", "d"])
    assert np.allclose(fit.predict(matrix), fit.predict(renamed))


def test_wrong_array_feature_count_is_rejected() -> None:
    predictors, response = _small_data()
    fit = lmv(predictors, response, M=2)
    with pytest.raises(ValueError, match="expects 4"):
        fit.predict(np.ones((5, 3)))


def test_regressor_uses_not_fitted_error() -> None:
    model = VarGuidRegressor(max_iter=2)
    with pytest.raises(NotFittedError):
        model.predict(np.ones((3, 2)))
    with pytest.raises(NotFittedError):
        model.summary()


def test_regressor_rejects_feature_mismatch() -> None:
    predictors, response = _small_data()
    model = VarGuidRegressor(max_iter=2).fit(predictors, response)
    with pytest.raises(ValueError, match="missing columns"):
        model.predict(predictors.drop(columns=["x1"]))


def test_lasso_is_invariant_to_predictor_units() -> None:
    generator = np.random.default_rng(2026)
    predictors = generator.normal(size=(100, 4))
    response = (
        2.0 * predictors[:, 0] - 0.6 * predictors[:, 2] + generator.normal(scale=0.2, size=100)
    )
    rescaled = predictors.copy()
    rescaled[:, 0] *= 100.0

    fit_original = lmv(
        predictors,
        response,
        M=2,
        lasso=True,
        cv_folds=5,
        rng=42,
    )
    fit_rescaled = lmv(
        rescaled,
        response,
        M=2,
        lasso=True,
        cv_folds=5,
        rng=42,
    )
    assert np.allclose(
        fit_original.predict(predictors),
        fit_rescaled.predict(rescaled),
        rtol=1e-6,
        atol=1e-8,
    )


def test_generate_cobra2d_rng_and_legacy_seed_agree() -> None:
    preferred = generate_cobra2d(n_samples=25, rng=7)
    with pytest.deprecated_call():
        legacy = generate_cobra2d(n_samples=25, random_state=7)
    pd.testing.assert_frame_equal(preferred, legacy)


def test_generate_cobra2d_expands_small_feature_count() -> None:
    generated = generate_cobra2d(n_samples=12, n_features=3, rng=5)
    assert generated.shape == (12, 11)
    assert list(generated.columns) == [*[f"x{i}" for i in range(1, 11)], "y"]


def test_lmv_rng_generator_is_reproducible() -> None:
    predictors, response = _small_data()
    fit_1 = lmv(
        predictors,
        response,
        M=2,
        lasso=True,
        cv_folds=3,
        rng=np.random.default_rng(11),
    )
    fit_2 = lmv(
        predictors,
        response,
        M=2,
        lasso=True,
        cv_folds=3,
        rng=np.random.default_rng(11),
    )
    assert np.allclose(fit_1.beta, fit_2.beta)


def test_lmv_legacy_random_state_warns_and_conflicts_with_rng() -> None:
    predictors, response = _small_data()
    with pytest.deprecated_call(match="use rng instead"):
        fit = lmv(
            predictors,
            response,
            M=1,
            lasso=True,
            cv_folds=3,
            random_state=7,
        )
    assert np.isfinite(fit.beta).all()

    with pytest.raises(ValueError, match="only one"):
        lmv(
            predictors,
            response,
            M=1,
            lasso=True,
            cv_folds=3,
            random_state=7,
            rng=7,
        )


def test_formula_missing_values_raise_instead_of_dropping_rows() -> None:
    predictors, response = _small_data()
    data = predictors.assign(y=response)
    missing_fit = data.copy()
    missing_fit.loc[missing_fit.index[0], "x1"] = np.nan
    with pytest.raises(ValueError, match="Could not construct formula design matrices"):
        lmv_formula("y ~ x1 + x2", data=missing_fit, M=1)

    fit = lmv_formula("y ~ x1 + x2", data=data, M=1)
    missing_prediction = data.iloc[:5].copy()
    missing_prediction.loc[missing_prediction.index[0], "x2"] = np.nan
    with pytest.raises(ValueError, match="Could not construct the formula design matrix"):
        fit.predict(missing_prediction)


def test_packaged_dataset_shape_and_columns() -> None:
    data = load_cobra2d()
    assert data.shape == (500, 16)
    assert list(data.columns) == [*[f"x{i}" for i in range(1, 16)], "y"]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("M", 0, "at least 1"),
        ("step", 0.0, "positive"),
        ("tol", -1.0, "non-negative"),
        ("cv_folds", 1, "at least 2"),
    ],
)
def test_invalid_parameters_are_rejected(keyword: str, value: object, message: str) -> None:
    predictors, response = _small_data()
    arguments: dict[str, Any] = {"M": 2, "step": 1.0, "tol": 1e-4, "cv_folds": 3}
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        lmv(predictors, response, **arguments)
