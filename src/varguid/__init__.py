"""Variance-guided regression improving upon OLS and ANOVA for Python."""

from .datasets import generate_cobra2d, load_cobra2d
from .model import LmvResult, VarGuidRegressor, lmv, lmv_formula, prd, predict

__all__ = [
    "LmvResult",
    "VarGuidRegressor",
    "generate_cobra2d",
    "lmv",
    "lmv_formula",
    "load_cobra2d",
    "prd",
    "predict",
]

__version__ = "0.1.8"
