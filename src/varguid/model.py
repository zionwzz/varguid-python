"""Variance-guided regression models for Python.

The implementation follows the update order used by the CRAN ``varGuid``
package for the non-lasso algorithm.  Sparse fits use scikit-learn's lasso
solver with glmnet-like feature standardization and randomized cross-validation
folds.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, cast

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from numpy.typing import ArrayLike as NumPyArrayLike
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y, validate_data

type ArrayLike = NumPyArrayLike | pd.DataFrame | pd.Series
type FloatArray = NDArray[np.float64]
type RNGLike = (
    np.random.Generator
    | np.random.BitGenerator
    | np.random.SeedSequence
    | int
    | np.integer[Any]
    | None
)


@dataclass
class LmvResult:
    """Result returned by :func:`lmv` or :func:`lmv_formula`.

    The ``obj_OLS``, ``obj_lasso``, ``obj_varGuid`` and ``obj_varGuid_coef``
    attribute names intentionally mirror the attached R package where practical.
    """

    beta: FloatArray
    obj_OLS: Any | None
    obj_lasso: Any | None
    obj_varGuid: Any
    res: Any | None
    obj_varGuid_coef: dict[str, Any]
    X: FloatArray
    feature_names: list[str] = field(default_factory=list)
    lasso: bool = False
    fit_intercept: bool = True
    formula: str | None = None
    design_info: Any | None = None
    response_name: str | None = None
    n_iter: int = 0
    converged: bool = False
    final_step: float = 1.0

    def _prepare_newdata(self, newdata: ArrayLike) -> FloatArray:
        if self.design_info is not None:
            if not isinstance(newdata, pd.DataFrame):
                raise TypeError(
                    "Formula-based fits require newdata to be a pandas DataFrame "
                    "containing the raw formula variables."
                )
            try:
                design = patsy.build_design_matrices(
                    [self.design_info],
                    newdata,
                    return_type="dataframe",
                    NA_action="raise",
                )[0]
            except patsy.PatsyError as exc:
                raise ValueError(f"Could not construct the formula design matrix: {exc}") from exc
            if self.fit_intercept:
                design = design.drop(columns=["Intercept"], errors="raise")
            matrix = design.to_numpy(dtype=float)
        elif isinstance(newdata, pd.DataFrame):
            if self.feature_names:
                matrix = _dataframe_to_expected_matrix(newdata, self.feature_names)
            else:
                matrix = newdata.to_numpy(dtype=float)
        else:
            matrix, _ = _as_2d_array(newdata)

        matrix = cast(
            FloatArray,
            check_array(matrix, dtype=np.float64, ensure_2d=True, ensure_min_features=0),
        )
        if matrix.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"newdata has {matrix.shape[1]} features, but the fitted model expects "
                f"{self.X.shape[1]}."
            )
        return matrix

    def predict(self, newdata: ArrayLike, model: str = "varGuid") -> FloatArray:
        """Predict with the final variance-guided or baseline model."""
        return predict(self, newdata, model=model)

    def summary_frame(self, model: str = "varGuid", cov_type: str = "WLS") -> pd.DataFrame:
        """Return a coefficient summary as a :class:`pandas.DataFrame`."""
        if self.lasso:
            raise NotImplementedError(
                "summary_frame() is unavailable for lasso fits because the sklearn "
                "LassoCV object does not define statsmodels-style standard errors."
            )
        result = _resolve_sm_result(self, model=model, cov_type=cov_type)
        return _summary_df(result, names=self._coefficient_names())

    def summary(self, model: str = "varGuid", cov_type: str = "WLS") -> Any:
        """Return a statsmodels-style summary object for a non-lasso fit."""
        if self.lasso:
            raise NotImplementedError(
                "summary() is unavailable for lasso fits because the sklearn LassoCV "
                "object does not define statsmodels-style standard errors."
            )
        result = _resolve_sm_result(self, model=model, cov_type=cov_type)
        return result.summary()

    def _coefficient_names(self) -> list[str]:
        prefix = ["Intercept"] if self.fit_intercept else []
        if self.feature_names:
            return [*prefix, *self.feature_names]
        generated = [f"x{i}" for i in range(1, self.X.shape[1] + 1)]
        return [*prefix, *generated]


class _WeightedLassoModel:
    """LassoCV wrapper that exposes coefficients on the original feature scale."""

    def __init__(
        self,
        estimator: LassoCV,
        feature_mean: FloatArray,
        feature_scale: FloatArray,
        fit_intercept: bool,
    ) -> None:
        self.estimator = estimator
        self.feature_mean_ = feature_mean.copy()
        self.feature_scale_ = feature_scale.copy()
        self.fit_intercept = fit_intercept
        self.alpha_ = float(estimator.alpha_)
        self.coef_ = np.asarray(estimator.coef_, dtype=float) / self.feature_scale_
        if fit_intercept:
            self.intercept_ = float(estimator.intercept_ - self.feature_mean_ @ self.coef_)
        else:
            self.intercept_ = 0.0

    def predict(self, X: ArrayLike) -> FloatArray:
        matrix, _ = _as_2d_array(X)
        if matrix.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {matrix.shape[1]} features, but the lasso model expects "
                f"{self.coef_.shape[0]}."
            )
        return np.asarray(matrix @ self.coef_ + self.intercept_, dtype=float)


class VarGuidRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn-compatible wrapper for stage-1 varGuid regression.

    Parameters
    ----------
    max_iter
        Number of variance-guided reweighting iterations.
    step
        Exponential weight step used by the R algorithm.
    tol
        Squared-coefficient-change threshold used to reduce ``step``.
    use_lasso
        Fit the sparse lasso variant instead of weighted least squares.
    cv_folds
        Requested lasso cross-validation folds. The R-compatible default uses
        three folds for at most 80 observations and ten folds otherwise.
    random_state
        Controls randomized lasso cross-validation folds.
    fit_intercept
        Whether matrix-based fits include an intercept. Formula-based fits take
        the intercept setting from the formula itself.
    """

    def __init__(
        self,
        max_iter: int = 10,
        step: float = 1.0,
        tol: float = float(np.exp(-10)),
        use_lasso: bool = False,
        cv_folds: int = 10,
        random_state: int | np.random.RandomState | None = None,
        fit_intercept: bool = True,
    ) -> None:
        self.max_iter = max_iter
        self.step = step
        self.tol = tol
        self.use_lasso = use_lasso
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.fit_intercept = fit_intercept

    def fit(self, X: ArrayLike, y: ArrayLike) -> VarGuidRegressor:
        """Fit a matrix-based variance-guided regression model."""
        _validate_parameters(
            max_iter=self.max_iter,
            step=self.step,
            tol=self.tol,
            cv_folds=self.cv_folds,
            fit_intercept=self.fit_intercept,
        )
        validated = validate_data(
            self,
            X,
            y,
            reset=True,
            dtype=np.float64,
            ensure_2d=True,
            ensure_min_samples=2,
            y_numeric=True,
        )
        X_valid, y_valid = cast(tuple[FloatArray, FloatArray], validated)

        if hasattr(self, "feature_names_in_"):
            columns = [str(name) for name in self.feature_names_in_]
            fit_X: ArrayLike = pd.DataFrame(X_valid, columns=columns)
        else:
            fit_X = X_valid

        result = lmv(
            X=fit_X,
            Y=y_valid,
            M=self.max_iter,
            step=self.step,
            tol=self.tol,
            lasso=self.use_lasso,
            cv_folds=self.cv_folds,
            fit_intercept=self.fit_intercept,
            rng=_rng_seed_from_sklearn_random_state(self.random_state),
        )
        self._assign_from_result(result)
        return self

    def fit_formula(self, formula: str, data: pd.DataFrame) -> VarGuidRegressor:
        """Fit using a Patsy/statsmodels-style formula."""
        _validate_parameters(
            max_iter=self.max_iter,
            step=self.step,
            tol=self.tol,
            cv_folds=self.cv_folds,
            fit_intercept=self.fit_intercept,
        )
        result = lmv_formula(
            formula=formula,
            data=data,
            M=self.max_iter,
            step=self.step,
            tol=self.tol,
            lasso=self.use_lasso,
            cv_folds=self.cv_folds,
            rng=_rng_seed_from_sklearn_random_state(self.random_state),
        )
        self.n_features_in_ = result.X.shape[1]
        self.feature_names_in_ = np.asarray(result.feature_names, dtype=object)
        self._assign_from_result(result)
        return self

    def _assign_from_result(self, result: LmvResult) -> None:
        self.result_ = result
        if result.fit_intercept:
            self.coef_ = np.asarray(result.beta[1:], dtype=float)
            self.intercept_ = float(result.beta[0])
        else:
            self.coef_ = np.asarray(result.beta, dtype=float)
            self.intercept_ = 0.0
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict with the fitted variance-guided model."""
        check_is_fitted(self, attributes=["result_", "coef_", "intercept_"])
        if self.result_.design_info is not None:
            return self.result_.predict(X, model="varGuid")

        prepared: ArrayLike = X
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            expected = [str(name) for name in self.feature_names_in_]
            prepared = _dataframe_in_expected_order(X, expected)
        X_valid = cast(
            FloatArray,
            validate_data(self, prepared, reset=False, dtype=np.float64, ensure_2d=True),
        )
        return self.result_.predict(X_valid, model="varGuid")

    def summary(self, cov_type: str = "WLS") -> Any:
        """Return the fitted non-lasso statsmodels summary."""
        check_is_fitted(self, attributes=["result_"])
        return self.result_.summary(cov_type=cov_type)

    def summary_frame(self, cov_type: str = "WLS") -> pd.DataFrame:
        """Return the fitted non-lasso coefficient table."""
        check_is_fitted(self, attributes=["result_"])
        return self.result_.summary_frame(cov_type=cov_type)

    def get_inference_results(self) -> dict[str, Any]:
        """Return precomputed WLS and heteroscedasticity-robust tables."""
        check_is_fitted(self, attributes=["result_"])
        return self.result_.obj_varGuid_coef


@dataclass
class _WeightModelResult:
    estimator: Any
    fitted_variance: FloatArray
    w: FloatArray


class _SeedStream:
    """Generate deterministic, distinct seeds for successive CV fits."""

    def __init__(
        self,
        *,
        rng: RNGLike = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if rng is not None and random_state is not None:
            raise ValueError("Pass only one of rng and random_state.")
        if random_state is not None:
            self._generator: np.random.Generator | None = None
            self._random_state: np.random.RandomState | None = check_random_state(random_state)
        else:
            self._generator = np.random.default_rng(rng)
            self._random_state = None

    def next_seed(self) -> int:
        upper = np.iinfo(np.int32).max
        if self._generator is not None:
            return int(self._generator.integers(upper))
        if self._random_state is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("The random-number stream was not initialized.")
        return int(self._random_state.randint(upper))


def _seed_stream(
    *,
    rng: RNGLike,
    random_state: int | np.random.RandomState | None,
) -> _SeedStream:
    if rng is not None and random_state is not None:
        raise ValueError("Pass only one of rng and random_state.")
    if random_state is not None:
        warnings.warn(
            "random_state is deprecated for lmv(); use rng instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return _SeedStream(rng=rng, random_state=random_state)


def _rng_seed_from_sklearn_random_state(
    random_state: int | np.random.RandomState | None,
) -> RNGLike:
    """Bridge sklearn's public random_state convention to the SPEC 7 RNG path."""
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(np.iinfo(np.int32).max))
    return random_state


def _column_names(columns: pd.Index[Any]) -> list[str]:
    names = [str(column) for column in columns]
    if len(set(names)) != len(names):
        raise ValueError("DataFrame column names must be unique after conversion to strings.")
    return names


def _dataframe_in_expected_order(data: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    incoming = _column_names(data.columns)
    missing = [name for name in expected if name not in incoming]
    unexpected = [name for name in incoming if name not in expected]
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        raise ValueError(
            "newdata columns do not match the fitted model (" + "; ".join(details) + ")."
        )
    positions = [incoming.index(name) for name in expected]
    return data.iloc[:, positions]


def _dataframe_to_expected_matrix(data: pd.DataFrame, expected: list[str]) -> FloatArray:
    ordered = _dataframe_in_expected_order(data, expected)
    return ordered.to_numpy(dtype=float)


def _as_2d_array(X: ArrayLike) -> tuple[FloatArray, list[str]]:
    if isinstance(X, pd.DataFrame):
        names = _column_names(X.columns)
        matrix = X.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("X must be a 2D array or a pandas DataFrame.")
        names = []
    return np.asarray(matrix, dtype=float), names


def _as_1d_array(y: ArrayLike) -> FloatArray:
    array = np.asarray(y, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError("Y must be one-dimensional.")
    return np.asarray(array, dtype=float)


def _add_intercept(X: FloatArray, fit_intercept: bool) -> FloatArray:
    if not fit_intercept:
        return X
    return np.asarray(sm.add_constant(X, has_constant="add"), dtype=float)


def _predict_any(model: Any, X: FloatArray, fit_intercept: bool) -> FloatArray:
    if isinstance(model, _WeightedLassoModel):
        return model.predict(X)
    return np.asarray(model.predict(_add_intercept(X, fit_intercept)), dtype=float).reshape(-1)


def _fit_linear(
    X: FloatArray,
    y: FloatArray,
    sample_weights: FloatArray | None = None,
    *,
    fit_intercept: bool,
) -> Any:
    design = _add_intercept(X, fit_intercept)
    if sample_weights is None:
        model = sm.OLS(y, design)
    else:
        model = sm.WLS(y, design, weights=sample_weights)
    return model.fit()


def _choose_cv_folds(n_samples: int, requested: int) -> int:
    if n_samples < 3:
        raise ValueError("Lasso fitting requires at least three observations.")
    folds = 3 if requested == 10 and n_samples <= 80 else requested
    return min(folds, n_samples)


def _standardize_features(
    X: FloatArray,
    sample_weights: FloatArray | None,
    *,
    fit_intercept: bool,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if sample_weights is None:
        weights: FloatArray = np.ones(X.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("sample weights must be finite, non-negative, and have a positive sum.")

    if fit_intercept:
        feature_mean = np.average(X, axis=0, weights=weights)
        centered = X - feature_mean
    else:
        feature_mean = np.zeros(X.shape[1], dtype=float)
        centered = X

    variance = np.average(centered**2, axis=0, weights=weights)
    feature_scale = np.sqrt(np.asarray(variance, dtype=float))
    invalid = ~np.isfinite(feature_scale) | (feature_scale <= np.finfo(float).eps)
    feature_scale[invalid] = 1.0
    standardized = centered / feature_scale
    return (
        np.asarray(standardized, dtype=float),
        np.asarray(feature_mean, dtype=float),
        feature_scale,
    )


def _fit_lasso(
    X: FloatArray,
    y: FloatArray,
    sample_weights: FloatArray | None,
    cv_folds: int,
    cv_seed: int,
    *,
    fit_intercept: bool,
) -> _WeightedLassoModel:
    if X.shape[1] == 0:
        raise ValueError("Lasso fitting requires at least one predictor.")
    folds = _choose_cv_folds(X.shape[0], cv_folds)
    standardized, feature_mean, feature_scale = _standardize_features(
        X, sample_weights, fit_intercept=fit_intercept
    )
    splitter = KFold(n_splits=folds, shuffle=True, random_state=cv_seed)
    estimator = LassoCV(
        cv=splitter,
        fit_intercept=fit_intercept,
        max_iter=100_000,
        tol=1e-7,
    )
    if sample_weights is None:
        estimator.fit(standardized, y)
    else:
        estimator.fit(standardized, y, sample_weight=sample_weights)
    return _WeightedLassoModel(estimator, feature_mean, feature_scale, fit_intercept)


def _stable_exponential_weights(w: FloatArray, step: float) -> FloatArray:
    exponent = -float(step) * np.asarray(w, dtype=float)
    if not np.all(np.isfinite(exponent)):
        raise FloatingPointError(
            "Non-finite values were produced while constructing sample weights."
        )
    exponent -= float(np.max(exponent))
    weights = np.exp(exponent)
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
        raise FloatingPointError("Could not construct finite positive sample weights.")
    return np.asarray(weights, dtype=float)


def _beta_est(
    X: FloatArray,
    y: FloatArray,
    w: FloatArray,
    step: float,
    lasso: bool,
    cv_folds: int,
    cv_seed: int,
    *,
    fit_intercept: bool,
) -> tuple[FloatArray, Any]:
    sample_weights = _stable_exponential_weights(w, step)
    if lasso:
        estimator = _fit_lasso(
            X,
            y,
            sample_weights,
            cv_folds,
            cv_seed,
            fit_intercept=fit_intercept,
        )
        parts = [estimator.coef_]
        if fit_intercept:
            parts.insert(0, np.asarray([estimator.intercept_]))
        beta = np.concatenate(parts)
        return np.asarray(beta, dtype=float), estimator
    result = _fit_linear(X, y, sample_weights, fit_intercept=fit_intercept)
    return np.asarray(result.params, dtype=float), result


def _normalise_variance_fit(fitted: FloatArray) -> FloatArray:
    fitted = np.asarray(fitted, dtype=float).reshape(-1)
    if not np.all(np.isfinite(fitted)):
        raise FloatingPointError("The fitted variance model contains non-finite values.")
    maximum = float(np.max(fitted))
    if abs(maximum) <= np.finfo(float).eps:
        if np.all(np.abs(fitted) <= np.finfo(float).eps):
            return np.zeros_like(fitted)
        raise FloatingPointError("The fitted variance model has a numerically zero maximum.")
    weights = fitted / maximum
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError("The normalized variance weights contain non-finite values.")
    return np.asarray(weights, dtype=float)


def _w_est(
    X: FloatArray,
    residuals: FloatArray,
    lasso: bool,
    cv_folds: int,
    cv_seed: int,
) -> _WeightModelResult:
    X_sq = X**2
    y_var = residuals**2
    if lasso:
        estimator = _fit_lasso(
            X_sq,
            y_var,
            sample_weights=None,
            cv_folds=cv_folds,
            cv_seed=cv_seed,
            fit_intercept=True,
        )
        fitted = estimator.predict(X_sq)
    else:
        estimator = _fit_linear(X_sq, y_var, sample_weights=None, fit_intercept=True)
        fitted = np.asarray(estimator.predict(_add_intercept(X_sq, True)), dtype=float)
    normalized = _normalise_variance_fit(fitted)
    return _WeightModelResult(estimator, np.asarray(fitted, dtype=float), normalized)


def _summary_df(result: Any, names: list[str] | None = None) -> pd.DataFrame:
    confidence_interval = np.asarray(result.conf_int(), dtype=float)
    data = np.column_stack(
        [
            np.asarray(result.params),
            np.asarray(result.bse),
            np.asarray(result.tvalues),
            np.asarray(result.pvalues),
            confidence_interval[:, 0],
            confidence_interval[:, 1],
        ]
    )
    if names is None:
        names = ["Intercept", *[f"x{i}" for i in range(1, data.shape[0])]]
    columns = ["coef", "std_err", "t", "p_value", "ci_lower", "ci_upper"]
    return pd.DataFrame(data, index=names, columns=columns)


def _coef_summaries(result: Any, names: list[str]) -> dict[str, pd.DataFrame]:
    output = {"WLS": _summary_df(result, names=names)}
    for cov_type in ("HC0", "HC1", "HC2", "HC3"):
        robust = result.get_robustcov_results(cov_type=cov_type)
        output[cov_type] = _summary_df(robust, names=names)
    return output


def _resolve_sm_result(
    model_result: LmvResult, model: str = "varGuid", cov_type: str = "WLS"
) -> Any:
    if model_result.lasso:
        raise NotImplementedError("statsmodels-style summaries are unavailable for lasso fits.")
    _validate_model_choice(model)
    estimator = model_result.obj_varGuid if model == "varGuid" else model_result.obj_OLS
    if estimator is None:
        raise ValueError(f"No fitted estimator is stored for model={model!r}.")
    cov_type_upper = cov_type.upper()
    if cov_type_upper == "WLS":
        return estimator
    if cov_type_upper not in {"HC0", "HC1", "HC2", "HC3"}:
        raise ValueError("cov_type must be one of 'WLS', 'HC0', 'HC1', 'HC2', or 'HC3'.")
    return estimator.get_robustcov_results(cov_type=cov_type_upper)


def _validate_parameters(
    *,
    max_iter: int,
    step: float,
    tol: float,
    cv_folds: int,
    fit_intercept: bool,
) -> None:
    if not isinstance(max_iter, Integral) or isinstance(max_iter, bool) or int(max_iter) < 1:
        raise ValueError("M/max_iter must be an integer of at least 1.")
    if not isinstance(step, Real) or isinstance(step, bool) or not np.isfinite(step) or step <= 0:
        raise ValueError("step must be a finite positive number.")
    if not isinstance(tol, Real) or isinstance(tol, bool) or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite non-negative number.")
    if not isinstance(cv_folds, Integral) or isinstance(cv_folds, bool) or int(cv_folds) < 2:
        raise ValueError("cv_folds must be an integer of at least 2.")
    if not isinstance(fit_intercept, bool):
        raise TypeError("fit_intercept must be a bool.")


def _validate_model_choice(model: str) -> None:
    if model not in {"varGuid", "baseline"}:
        raise ValueError("model must be either 'varGuid' or 'baseline'.")


def lmv(
    X: ArrayLike,
    Y: ArrayLike,
    M: int = 10,
    step: float = 1.0,
    tol: float = float(np.exp(-10)),
    lasso: bool = False,
    cv_folds: int = 10,
    random_state: int | np.random.RandomState | None = None,
    fit_intercept: bool = True,
    *,
    rng: RNGLike = None,
) -> LmvResult:
    """Fit the stage-1 global linear mean-variance model.

    For non-lasso fits, the coefficient update order and un-clipped variance
    predictions follow the attached R package. Sparse fits use scikit-learn's
    :class:`~sklearn.linear_model.LassoCV`, so their coefficients are not
    expected to be bit-for-bit identical to R's ``glmnet`` implementation.

    ``rng`` is the preferred pseudo-random generator control for sparse fits.
    ``random_state`` is retained as a deprecated compatibility alias.
    """
    _validate_parameters(
        max_iter=M,
        step=step,
        tol=tol,
        cv_folds=cv_folds,
        fit_intercept=fit_intercept,
    )
    if not isinstance(lasso, bool):
        raise TypeError("lasso must be a bool.")

    X_matrix, feature_names = _as_2d_array(X)
    y_vector = _as_1d_array(Y)
    checked = check_X_y(
        X_matrix,
        y_vector,
        dtype=np.float64,
        ensure_2d=True,
        ensure_min_samples=2,
        ensure_min_features=0,
        y_numeric=True,
    )
    X_matrix, y_vector = cast(tuple[FloatArray, FloatArray], checked)
    if X_matrix.shape[1] == 0 and not fit_intercept:
        raise ValueError("At least one predictor is required when fit_intercept=False.")

    seed_stream = _seed_stream(rng=rng, random_state=random_state) if lasso else None

    def next_cv_seed() -> int:
        return seed_stream.next_seed() if seed_stream is not None else 0

    n_samples = X_matrix.shape[0]
    current_step = float(step)
    difference = current_step

    beta, initial_model = _beta_est(
        X_matrix,
        y_vector,
        w=np.ones(n_samples, dtype=float),
        step=current_step,
        lasso=lasso,
        cv_folds=cv_folds,
        cv_seed=next_cv_seed(),
        fit_intercept=fit_intercept,
    )
    baseline_ols = None if lasso else initial_model
    baseline_lasso = initial_model if lasso else None
    current_model = initial_model
    final_model = initial_model
    last_weight_result: _WeightModelResult | None = None

    for _ in range(int(M)):
        old_beta = beta.copy()
        residuals = y_vector - _predict_any(current_model, X_matrix, fit_intercept)
        weight_result = _w_est(
            X_matrix,
            residuals,
            lasso=lasso,
            cv_folds=cv_folds,
            cv_seed=next_cv_seed(),
        )
        last_weight_result = weight_result
        candidate_beta, candidate_model = _beta_est(
            X_matrix,
            y_vector,
            w=weight_result.w,
            step=current_step,
            lasso=lasso,
            cv_folds=cv_folds,
            cv_seed=next_cv_seed(),
            fit_intercept=fit_intercept,
        )

        # This assignment deliberately occurs before the convergence branch to
        # match the R implementation's ``o <- beta_est(...)`` update order.
        current_model = candidate_model
        if difference > tol:
            beta = candidate_beta
            final_model = candidate_model
            difference = float(np.nansum((beta - old_beta) ** 2))
        else:
            current_step *= 0.1

    coefficient_summaries: dict[str, Any] = {}
    predictor_names = feature_names or [f"x{i}" for i in range(1, X_matrix.shape[1] + 1)]
    coefficient_names = (["Intercept"] if fit_intercept else []) + predictor_names
    if not lasso:
        coefficient_summaries = _coef_summaries(final_model, names=coefficient_names)

    return LmvResult(
        beta=np.asarray(beta, dtype=float),
        obj_OLS=baseline_ols,
        obj_lasso=baseline_lasso,
        obj_varGuid=final_model,
        res=last_weight_result.estimator if last_weight_result is not None else None,
        obj_varGuid_coef=coefficient_summaries,
        X=np.asarray(X_matrix, dtype=float),
        feature_names=feature_names,
        lasso=lasso,
        fit_intercept=fit_intercept,
        n_iter=int(M),
        converged=bool(difference <= tol),
        final_step=current_step,
    )


def lmv_formula(
    formula: str,
    data: pd.DataFrame,
    M: int = 10,
    step: float = 1.0,
    tol: float = float(np.exp(-10)),
    lasso: bool = False,
    cv_folds: int = 10,
    random_state: int | np.random.RandomState | None = None,
    *,
    rng: RNGLike = None,
) -> LmvResult:
    """Fit the stage-1 model using a Patsy/statsmodels-style formula."""
    if not isinstance(formula, str) or not formula.strip():
        raise TypeError("formula must be a non-empty string.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame for formula-based fitting.")

    try:
        y_frame, X_frame = patsy.dmatrices(
            formula,
            data=data,
            return_type="dataframe",
            NA_action="raise",
        )
    except patsy.PatsyError as exc:
        raise ValueError(f"Could not construct formula design matrices: {exc}") from exc
    if y_frame.shape[1] != 1:
        raise ValueError("The formula response must produce exactly one numeric column.")

    response_name = str(y_frame.columns[0])
    design_info = X_frame.design_info
    fit_intercept = "Intercept" in X_frame.columns
    design_predictors = X_frame.drop(columns=["Intercept"], errors="ignore")
    result = lmv(
        X=design_predictors,
        Y=y_frame.iloc[:, 0],
        M=M,
        step=step,
        tol=tol,
        lasso=lasso,
        cv_folds=cv_folds,
        random_state=random_state,
        fit_intercept=fit_intercept,
        rng=rng,
    )
    result.formula = formula
    result.design_info = design_info
    result.response_name = response_name
    result.feature_names = [str(column) for column in design_predictors.columns]
    return result


def predict(object: LmvResult, newdata: ArrayLike, model: str = "varGuid") -> FloatArray:
    """Predict from a fitted stage-1 varGuid model."""
    _validate_model_choice(model)
    if not isinstance(object, LmvResult):
        raise TypeError("object must be an instance returned by lmv() or lmv_formula().")

    X_matrix = object._prepare_newdata(newdata)
    estimator = (
        object.obj_varGuid
        if model == "varGuid"
        else (object.obj_lasso if object.lasso else object.obj_OLS)
    )
    if estimator is None:
        raise ValueError(f"No fitted estimator is stored for model={model!r}.")
    if isinstance(estimator, _WeightedLassoModel):
        return estimator.predict(X_matrix)
    design = _add_intercept(X_matrix, object.fit_intercept)
    return np.asarray(estimator.predict(design), dtype=float).reshape(-1)


def prd(object: LmvResult, newdata: ArrayLike, model: str = "varGuid") -> FloatArray:
    """Deprecated alias for :func:`predict`."""
    warnings.warn(
        "prd() is deprecated; use predict(object, newdata, model=...) or object.predict(...).",
        DeprecationWarning,
        stacklevel=2,
    )
    return predict(object, newdata, model=model)
