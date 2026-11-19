"""Validated descriptive statistics for the mobile-specification assignment."""
from __future__ import annotations

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "PID", "Blue", "Wi_Fi", "Tch_Scr", "Ext_Mem", "Px_h", "Px_w",
    "Scr_h", "Scr_w", "PC", "FC", "Int_Mem", "Bty_Pwr", "RAM",
    "Depth", "Weight", "Price",
]
BINARY_COLUMNS = ["Blue", "Wi_Fi", "Tch_Scr", "Ext_Mem"]
MEASURE_COLUMNS = [column for column in EXPECTED_COLUMNS if column not in {"PID", *BINARY_COLUMNS}]


def validate_phone_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate expected fields and basic domains without changing input."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(f"Expected columns in this order: {EXPECTED_COLUMNS}")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values.")
    if frame["PID"].astype(str).str.strip().eq("").any():
        raise ValueError("Product identifiers must not be blank.")
    if frame["PID"].duplicated().any():
        raise ValueError("Product identifiers must be unique.")

    for column in BINARY_COLUMNS:
        values = set(frame[column].astype(str).str.strip().str.lower())
        if not values <= {"yes", "no"}:
            raise ValueError(f"{column} must contain only yes/no values.")

    numeric = frame[MEASURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Numeric measurements contain non-numeric values.")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Numeric measurements contain non-finite values.")
    if (numeric <= 0).any().any():
        raise ValueError("Numeric measurements must be greater than zero.")
    return frame


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return distribution statistics and coefficient of variation.

    CV is reported only when the sample mean is non-zero. It is a scale-free
    descriptive ratio, not a universal measure of variability.
    """
    validate_phone_data(frame)
    numeric = frame[MEASURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    summary = numeric.agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).T
    summary["q1"] = numeric.quantile(0.25)
    summary["q3"] = numeric.quantile(0.75)
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary["coefficient_of_variation"] = np.where(
        summary["mean"].ne(0),
        summary["std"] / summary["mean"].abs(),
        np.nan,
    )
    # Keep a percentage-form display column for the notebook while exposing
    # the conventional unitless coefficient expected by the reusable API.
    summary["cv_pct"] = summary["coefficient_of_variation"] * 100
    summary.index.name = "feature"
    return summary[
        [
            "count", "mean", "median", "std", "min", "q1", "q3", "max",
            "iqr", "coefficient_of_variation", "cv_pct",
        ]
    ].round(3)


def price_rank_correlations(frame: pd.DataFrame) -> pd.Series:
    """Return Spearman rank correlations of numeric features with listed price."""
    validate_phone_data(frame)
    numeric = frame[MEASURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    return numeric.corr(method="spearman")["Price"].drop("Price").sort_values(
        key=lambda values: values.abs(), ascending=False
    ).rename("spearman_rho_with_listed_price").round(3)
