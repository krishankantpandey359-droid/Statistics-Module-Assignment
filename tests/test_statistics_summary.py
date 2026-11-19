from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.statistics_summary import descriptive_statistics, price_rank_correlations, validate_phone_data

DATA = ROOT / "Dataset.csv"


def load_data():
    return pd.read_csv(DATA)


def test_descriptive_statistics_covers_all_numeric_fields():
    frame = load_data()
    summary = descriptive_statistics(frame)
    assert len(summary) == 12
    assert {"mean", "median", "q1", "q3", "iqr", "coefficient_of_variation"} <= set(summary.columns)


def test_price_correlations_are_sorted_by_absolute_strength():
    result = price_rank_correlations(load_data())
    assert len(result) == 11
    assert result.abs().is_monotonic_decreasing


def test_validator_rejects_duplicate_product_ids():
    frame = load_data()
    frame.loc[1, "PID"] = frame.loc[0, "PID"]
    with pytest.raises(ValueError, match="unique"):
        validate_phone_data(frame)


def test_validator_rejects_invalid_binary_flag():
    frame = load_data()
    frame.loc[0, "Wi_Fi"] = "unknown"
    with pytest.raises(ValueError, match="yes/no"):
        validate_phone_data(frame)


def test_summary_does_not_mutate_source():
    frame = load_data()
    original = frame.copy(deep=True)
    descriptive_statistics(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_validator_rejects_ids_that_duplicate_after_whitespace_normalization():
    frame = load_data()
    frame.loc[0, "PID"] = "  " + str(frame.loc[0, "PID"]) + "  "
    frame.loc[1, "PID"] = str(frame.loc[0, "PID"]).strip()
    with pytest.raises(ValueError, match="after whitespace normalization"):
        validate_phone_data(frame)
