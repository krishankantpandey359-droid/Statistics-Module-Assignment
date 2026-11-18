from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "Dataset.csv"
EXPECTED_COLUMNS = [
    "PID", "Blue", "Wi_Fi", "Tch_Scr", "Ext_Mem", "Px_h", "Px_w",
    "Scr_h", "Scr_w", "PC", "FC", "Int_Mem", "Bty_Pwr", "RAM",
    "Depth", "Weight", "Price",
]
BINARY_COLUMNS = ["Blue", "Wi_Fi", "Tch_Scr", "Ext_Mem"]


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
    for column in BINARY_COLUMNS:
        assert set(df[column].dropna().unique()) <= {"yes", "no"}


def test_numeric_measurements_are_complete_and_positive():
    df = load_dataset()
    numeric = df.drop(columns=["PID", *BINARY_COLUMNS])
    assert numeric.notna().all().all()
    assert (numeric > 0).all().all()


def test_dataset_has_no_exact_duplicate_rows():
    df = load_dataset()
    assert not df.duplicated().any()
