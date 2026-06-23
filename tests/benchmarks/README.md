# Benchmarks against statsmodels

These scripts are not part of the automated test suite. They are lightweight
reference checks showing that the baseline OLS fit and the non-lasso summary
output agree with direct `statsmodels` calculations on the packaged `cobra2d`
example data.

Run from the project root with:

```bash
python tests/benchmarks/benchmark_statsmodels.py
```
