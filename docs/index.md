# {varguid}: Variance-Guided Regression Improving Upon OLS and ANOVA for Python

`varguid` implements the stage-1 variance-guided global linear mean-variance
model from Liu and Lu (2026), *Statistics in Medicine* 45(13-14):e70632.

## Public API

- `lmv(X, Y, ...)`: matrix or DataFrame fit.
- `lmv_formula(formula, data, ...)`: Patsy formula fit.
- `LmvResult.predict(newdata, model=...)`: baseline or variance-guided prediction.
- `LmvResult.summary(...)` and `summary_frame(...)`: non-lasso inference.
- `VarGuidRegressor`: scikit-learn-compatible estimator.
- `load_cobra2d()` and `generate_cobra2d(...)`: example data helpers.

## Input safety

A model fitted from a DataFrame remembers its feature names. Prediction accepts
a different column order and reorders safely, but rejects missing or additional
columns. Formula fits retain Patsy's design information and therefore require
raw formula variables at prediction time.

## Algorithm compatibility

The non-lasso implementation follows the update order in the attached R package
and does not clip fitted variance values. Lasso fits standardize predictors and
use reproducible shuffled cross-validation, but the underlying solver is
scikit-learn rather than `glmnet`; exact sparse coefficient equality with R is
not promised.

## Scope

Only Section 2 of the paper is included. The grouping-based nonlinear extension
from Section 3 is outside this package release.

See the repository `README.md` for runnable examples and
`FIX_LOG_0.1.8.md` for the 0.1.8 audit details.

## Citation

See `CITATION.cff` and the repository README for the software and method-paper citations.
