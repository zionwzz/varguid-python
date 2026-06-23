from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from varguid import VarGuidRegressor, lmv, lmv_formula, load_cobra2d, predict


def test_readme_python_blocks_execute_verbatim() -> None:
    readme = Path(__file__).parents[1].joinpath("README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    assert blocks, "README.md contains no Python examples"

    namespace: dict[str, object] = {}
    for number, block in enumerate(blocks, start=1):
        code = compile(block, f"README.md:python-block-{number}", "exec")
        exec(code, namespace)


def test_documented_interfaces_return_expected_shapes() -> None:
    data = load_cobra2d()
    train = data.iloc[:-200].copy()
    response = train["y"]
    predictors = train.drop(columns=["y"])

    fit = lmv(predictors, response, M=3, lasso=False)
    baseline_prediction = fit.predict(predictors, model="baseline")
    varguid_prediction = fit.predict(predictors, model="varGuid")
    helper_prediction = predict(fit, predictors, model="varGuid")
    assert np.allclose(varguid_prediction, helper_prediction)
    assert baseline_prediction.shape == varguid_prediction.shape == (300,)
    assert not fit.summary_frame(cov_type="HC3").empty

    formula_fit = lmv_formula(
        "y ~ x1 + x2 + x3 + x4 + x5",
        data=train,
        M=3,
    )
    assert formula_fit.predict(train.iloc[:5]).shape == (5,)

    estimator = VarGuidRegressor(max_iter=3, random_state=1)
    estimator.fit(predictors, response)
    assert estimator.predict(predictors.iloc[:5]).shape == (5,)
