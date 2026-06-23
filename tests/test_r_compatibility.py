from __future__ import annotations

from typing import Any, cast

import numpy as np
import statsmodels.api as sm
from numpy.typing import NDArray

from varguid import lmv, load_cobra2d

FloatArray = NDArray[np.float64]


def _add_intercept(predictors: FloatArray) -> FloatArray:
    return np.asarray(sm.add_constant(predictors, has_constant="add"), dtype=np.float64)


def _r_order_reference(
    predictors: FloatArray,
    response: FloatArray,
    *,
    iterations: int,
    step: float,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Independent translation of the update order in R/irls.R."""

    def beta_est(weights: FloatArray, current_step: float) -> Any:
        sample_weights = np.exp(-current_step * weights)
        return sm.WLS(response, _add_intercept(predictors), weights=sample_weights).fit()

    current = beta_est(np.ones(predictors.shape[0]), step)
    beta = np.asarray(current.params, dtype=np.float64)
    final = current
    difference = step
    fitted_variance: FloatArray = cast(FloatArray, np.zeros(predictors.shape[0], dtype=np.float64))

    for _ in range(iterations):
        old_beta = beta.copy()
        variance_model = sm.OLS(
            np.asarray(current.resid, dtype=np.float64) ** 2,
            _add_intercept(predictors**2),
        ).fit()
        fitted_variance = cast(
            FloatArray, np.asarray(variance_model.fittedvalues, dtype=np.float64)
        )
        normalized = fitted_variance / float(np.max(fitted_variance))
        candidate = beta_est(normalized, step)
        current = candidate
        if difference > tolerance:
            beta = np.asarray(candidate.params, dtype=np.float64)
            final = candidate
        else:
            step *= 0.1
            continue
        difference = float(np.nansum((beta - old_beta) ** 2))

    return beta, np.asarray(final.params, dtype=np.float64), fitted_variance


def test_non_lasso_matches_attached_r_update_order() -> None:
    data = load_cobra2d().iloc[:120]
    predictors = data[["x1", "x2", "x3", "x4", "x5"]].to_numpy(dtype=np.float64)
    response = data["y"].to_numpy(dtype=np.float64)

    expected_beta, expected_final_params, expected_variance = _r_order_reference(
        predictors,
        response,
        iterations=5,
        step=1.0,
        tolerance=float(np.exp(-10)),
    )
    fitted = lmv(predictors, response, M=5, step=1.0, lasso=False)

    assert np.allclose(fitted.beta, expected_beta, rtol=1e-11, atol=1e-12)
    assert np.allclose(
        np.asarray(fitted.obj_varGuid.params),
        expected_final_params,
        rtol=1e-11,
        atol=1e-12,
    )
    expected_last_variance = expected_variance
    assert fitted.res is not None
    actual_last_variance = np.asarray(fitted.res.fittedvalues, dtype=np.float64)
    assert np.allclose(actual_last_variance, expected_last_variance, rtol=1e-11, atol=1e-12)


def test_zero_variance_case_falls_back_to_uniform_weights() -> None:
    predictors = np.arange(12, dtype=np.float64).reshape(-1, 1)
    response = 2.0 + 3.0 * predictors[:, 0]
    fitted = lmv(predictors, response, M=2, lasso=False)
    assert np.allclose(fitted.beta, [2.0, 3.0], atol=1e-10)
    assert np.isfinite(fitted.predict(predictors)).all()
