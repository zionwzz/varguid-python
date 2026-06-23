"""Manual benchmark against direct statsmodels calls.

Run with::

    python tests/benchmarks/benchmark_statsmodels.py
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from varguid import lmv, load_cobra2d


def main() -> None:
    data = load_cobra2d().iloc[:300]
    response = data["y"]
    predictors = data.drop(columns=["y"])

    fitted = lmv(predictors, response, M=5, lasso=False)
    direct = sm.OLS(response, sm.add_constant(predictors, has_constant="add")).fit()

    assert fitted.obj_OLS is not None
    coefficient_difference = np.max(
        np.abs(np.asarray(fitted.obj_OLS.params) - np.asarray(direct.params))
    )
    prediction_difference = np.max(
        np.abs(
            fitted.predict(predictors, model="baseline")
            - np.asarray(direct.predict(sm.add_constant(predictors, has_constant="add")))
        )
    )
    print(f"maximum baseline coefficient difference: {coefficient_difference:.3e}")
    print(f"maximum baseline prediction difference:  {prediction_difference:.3e}")


if __name__ == "__main__":
    main()
