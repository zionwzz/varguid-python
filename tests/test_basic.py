from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from varguid import VarGuidRegressor, lmv, lmv_formula, load_cobra2d, prd, predict


def _cobra_subset(n: int = 120) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    data = load_cobra2d().iloc[:n].copy()
    response = data["y"]
    predictors = data.drop(columns=["y"])
    return data, predictors, response


def test_lmv_ols_smoke_and_summary_frame() -> None:
    _, predictors, response = _cobra_subset()
    fit = lmv(predictors, response, M=3, lasso=False)
    baseline_prediction = fit.predict(predictors, model="baseline")
    varguid_prediction = fit.predict(predictors, model="varGuid")

    assert baseline_prediction.shape == varguid_prediction.shape == (len(predictors),)
    assert len(fit.beta) == predictors.shape[1] + 1
    assert np.isfinite(varguid_prediction).all()
    frame = fit.summary_frame(cov_type="HC3")
    assert list(frame.columns) == [
        "coef",
        "std_err",
        "t",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]
    assert frame.shape[0] == predictors.shape[1] + 1


def test_lmv_lasso_smoke_and_reproducibility() -> None:
    _, predictors, response = _cobra_subset(100)
    fit_1 = lmv(
        predictors,
        response,
        M=2,
        lasso=True,
        cv_folds=3,
        rng=17,
    )
    fit_2 = lmv(
        predictors,
        response,
        M=2,
        lasso=True,
        cv_folds=3,
        rng=17,
    )

    prediction_1 = fit_1.predict(predictors, model="varGuid")
    prediction_2 = fit_2.predict(predictors, model="varGuid")
    assert prediction_1.shape == (len(predictors),)
    assert np.isfinite(prediction_1).all()
    assert np.allclose(fit_1.beta, fit_2.beta)
    assert np.allclose(prediction_1, prediction_2)
    with pytest.raises(NotImplementedError):
        fit_1.summary()


def test_top_level_predict_and_deprecated_prd_agree() -> None:
    _, predictors, response = _cobra_subset()
    fit = lmv(predictors, response, M=3, lasso=False)
    prediction = predict(fit, predictors, model="varGuid")
    with pytest.deprecated_call():
        deprecated_prediction = prd(fit, predictors, model="varGuid")
    assert np.allclose(prediction, deprecated_prediction)


def test_regressor_wrapper_exposes_summary_frame() -> None:
    _, predictors, response = _cobra_subset()
    model = VarGuidRegressor(max_iter=3, use_lasso=False)
    model.fit(predictors, response)

    prediction = model.predict(predictors)
    assert prediction.shape == (len(predictors),)
    assert model.n_features_in_ == predictors.shape[1]
    assert model.coef_.shape == (predictors.shape[1],)
    frame = model.summary_frame(cov_type="HC1")
    assert not frame.empty


def test_formula_interface_matches_matrix_fit() -> None:
    data, predictors, response = _cobra_subset()
    columns = ["x1", "x2", "x3", "x4", "x5"]
    fit_matrix = lmv(predictors[columns], response, M=3, lasso=False)
    fit_formula = lmv_formula(
        "y ~ x1 + x2 + x3 + x4 + x5",
        data=data,
        M=3,
        lasso=False,
    )

    assert np.allclose(fit_matrix.beta, fit_formula.beta)
    assert np.allclose(fit_matrix.predict(predictors[columns]), fit_formula.predict(data))


def test_formula_prediction_reconstructs_transforms_and_categories() -> None:
    data, _, _ = _cobra_subset()
    data["group"] = np.where(data["x1"] >= 0, "nonnegative", "negative")
    fit = lmv_formula(
        "y ~ I(x1 ** 2) + C(group) + x2:C(group)",
        data=data,
        M=3,
        lasso=False,
    )

    newdata = data.iloc[:7].copy()
    prediction = fit.predict(newdata[newdata.columns[::-1]])
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()


def test_no_intercept_formula_matches_matrix_fit() -> None:
    data, predictors, response = _cobra_subset()
    columns = ["x1", "x2", "x3"]
    fit_matrix = lmv(
        predictors[columns],
        response,
        M=3,
        lasso=False,
        fit_intercept=False,
    )
    fit_formula = lmv_formula("y ~ 0 + x1 + x2 + x3", data=data, M=3, lasso=False)

    assert not fit_formula.fit_intercept
    assert len(fit_formula.beta) == len(columns)
    assert "Intercept" not in fit_formula.summary_frame().index
    assert np.allclose(fit_matrix.beta, fit_formula.beta)
    assert np.allclose(fit_matrix.predict(predictors[columns]), fit_formula.predict(data))


def test_baseline_ols_matches_statsmodels_exactly() -> None:
    _, predictors, response = _cobra_subset()
    fit = lmv(predictors, response, M=3, lasso=False)

    design = sm.add_constant(predictors, has_constant="add")
    ordinary_least_squares = sm.OLS(response, design).fit()

    assert fit.obj_OLS is not None
    assert np.allclose(np.asarray(fit.obj_OLS.params), np.asarray(ordinary_least_squares.params))
    assert np.allclose(
        fit.predict(predictors, model="baseline"),
        np.asarray(ordinary_least_squares.predict(design)),
    )


def test_hc3_summary_matches_statsmodels() -> None:
    _, predictors, response = _cobra_subset()
    fit = lmv(predictors, response, M=3, lasso=False)
    ours = fit.summary_frame(cov_type="HC3")
    statsmodels_hc3 = fit.obj_varGuid.get_robustcov_results(cov_type="HC3")
    expected = pd.DataFrame(
        {
            "coef": np.asarray(statsmodels_hc3.params),
            "std_err": np.asarray(statsmodels_hc3.bse),
            "t": np.asarray(statsmodels_hc3.tvalues),
            "p_value": np.asarray(statsmodels_hc3.pvalues),
            "ci_lower": np.asarray(statsmodels_hc3.conf_int()[:, 0]),
            "ci_upper": np.asarray(statsmodels_hc3.conf_int()[:, 1]),
        },
        index=ours.index,
    )
    assert np.allclose(ours.to_numpy(), expected.to_numpy())


def test_regressor_formula_wrapper() -> None:
    data, _, _ = _cobra_subset()
    model = VarGuidRegressor(max_iter=3, use_lasso=False)
    model.fit_formula("y ~ x1 + x2 + x3 + x4 + x5", data=data)
    prediction = model.predict(data.iloc[:8])
    assert prediction.shape == (8,)
