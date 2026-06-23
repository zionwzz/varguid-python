"""Example datasets for :mod:`varguid`."""

from __future__ import annotations

import warnings
from importlib import resources
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.stats import norm


def generate_cobra2d(
    n_samples: int = 500,
    n_features: int = 15,
    noise_std: float = 0.1,
    correlation: float = 0.0,
    random_state: int | None = None,
    *,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic ``cobra2d``-style dataset.

    Parameters
    ----------
    n_samples
        Number of observations.
    n_features
        Requested number of predictors. Values below ten are expanded to ten,
        matching the original package and the attached R example generator.
    noise_std
        Standard deviation of the additive Gaussian noise.
    correlation
        Equicorrelation among predictors. Zero generates independent uniforms.
    random_state
        Deprecated seed alias retained for backward compatibility.
    rng
        Preferred pseudo-random generator or seed. When neither seed argument is
        supplied, seed 1 is used to preserve the package's historical output.
    """
    if not isinstance(n_samples, Integral) or isinstance(n_samples, bool) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(n_features, Integral) or isinstance(n_features, bool) or n_features < 1:
        raise ValueError("n_features must be a positive integer.")
    n_features = max(10, int(n_features))
    if not isinstance(noise_std, Real) or noise_std < 0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and non-negative.")
    if not isinstance(correlation, Real) or not np.isfinite(correlation):
        raise ValueError("correlation must be finite.")
    lower_bound = -1.0 / (n_features - 1)
    if not lower_bound < float(correlation) < 1.0:
        raise ValueError(f"correlation must be greater than {lower_bound:.6g} and less than 1.")
    if rng is not None and random_state is not None:
        raise ValueError("Pass only one of rng and random_state.")
    if random_state is not None:
        warnings.warn(
            "random_state is deprecated for generate_cobra2d(); use rng instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        rng = random_state
    generator = np.random.default_rng(1 if rng is None else rng)

    if correlation != 0:
        mean = np.zeros(n_features)
        covariance = np.full((n_features, n_features), float(correlation))
        np.fill_diagonal(covariance, 1.0)
        z = generator.multivariate_normal(mean, covariance, size=int(n_samples))
        predictors = 2 * norm.cdf(z) - 1
    else:
        predictors = generator.uniform(-1, 1, size=(int(n_samples), n_features))

    true_response = (
        predictors[:, 0] * predictors[:, 1]
        + predictors[:, 2] ** 2
        - predictors[:, 3] * predictors[:, 6]
        + predictors[:, 7] * predictors[:, 9]
        - predictors[:, 5] ** 2
    )
    response = true_response + generator.normal(0.0, float(noise_std), size=int(n_samples))

    columns = [f"x{i + 1}" for i in range(n_features)]
    frame = pd.DataFrame(predictors, columns=columns)
    frame["y"] = response
    return frame


def load_cobra2d() -> pd.DataFrame:
    """Load the packaged ``cobra2d`` dataset from the attached R release."""
    resource = resources.files("varguid").joinpath("data").joinpath("cobra2d.csv")
    with resources.as_file(resource) as path:
        return pd.read_csv(path)
