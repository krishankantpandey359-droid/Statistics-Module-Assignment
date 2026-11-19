from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.statistics_summary import (
    descriptive_statistics,
    price_rank_correlations,
    validate_phone_data,
)

DATA_PATH = ROOT / "Dataset.csv"
EXPECTED_COLUMNS = [
    "PID", "Blue", "Wi_Fi", "Tch_Scr", "Ext_Mem", "Px_h", "Px_w",
    "Scr_h", "Scr_w", "PC", "FC", "Int_Mem", "Bty_Pwr", "RAM",
    "Depth", "Weight", "Price",
]


def load_dataset():
    return pd.read_csv(DATA_PATH)


def test_dataset_shape_and_schema():
    df = load_dataset()
    assert df.shape == (50, 17)
    assert df.columns.tolist() == EXPECTED_COLUMNS


def test_identifiers_are_present_and_unique():
    df = load_dataset()
    assert df["PID"].notna().all()
    assert df["PID"].is_unique


def test_binary_features_have_only_yes_no_values():
    df = load_dataset()
    for column in ["Blue", "Wi_Fi", "Tch_Scr", "Ext_Mem"]:
        assert set(df[column].dropna().unique()) <= {"yes", "no"}


def test_numeric_measurements_are_complete_and_positive():
    df = load_dataset()
    numeric = df.drop(columns=["PID", "Blue", "Wi_Fi", "Tch_Scr", "Ext_Mem"])
    assert numeric.notna().all().all()
    assert (numeric > 0).all().all()
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_dataset_has_no_exact_duplicate_rows():
    assert not load_dataset().duplicated().any()


def test_validator_accepts_repository_data():
    assert validate_phone_data(load_dataset()).shape == (50, 17)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Blue", "unknown", "yes/no"),
        ("Price", 0, "greater than zero"),
        ("RAM", -1, "greater than zero"),
        ("Price", float("inf"), "non-finite"),
        ("Price", "unknown", "non-numeric"),
        ("PID", "", "blank"),
    ],
)
def test_validator_rejects_invalid_data(column, value, message):
    df = load_dataset()
    df[column] = df[column].astype(object)
    df.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_phone_data(df)


def test_validator_rejects_duplicate_ids():
    df = load_dataset()
    df.loc[1, "PID"] = df.loc[0, "PID"]
    with pytest.raises(ValueError, match="unique"):
        validate_phone_data(df)


def test_descriptive_statistics_include_quartiles_and_cv():
    summary = descriptive_statistics(load_dataset())
    assert "PID" not in summary.index
    assert "Price" in summary.index
    assert summary.loc["Price", "count"] == 50
    assert summary.loc["Price", "q1"] < summary.loc["Price", "q3"]
    assert summary.loc["Price", "cv_pct"] > 0


def test_price_rank_correlations_are_bounded_and_sorted_by_strength():
    correlations = price_rank_correlations(load_dataset())
    assert "Price" not in correlations.index
    assert correlations.between(-1, 1).all()
    assert correlations.abs().is_monotonic_decreasing


def test_summary_functions_do_not_mutate_input():
    df = load_dataset()
    original = df.copy(deep=True)
    descriptive_statistics(df)
    price_rank_correlations(df)
    pd.testing.assert_frame_equal(df, original)
