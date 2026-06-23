from __future__ import annotations

from sklearn.utils.estimator_checks import check_estimator

from varguid import VarGuidRegressor


def test_sklearn_estimator_contract() -> None:
    check_estimator(VarGuidRegressor(max_iter=2))
