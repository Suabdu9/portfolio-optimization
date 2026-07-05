import pandas as pd
from pathlib import Path


def test_processed_directory_exists():
    assert Path("data/processed").exists()


def test_tsla_dataset_exists():
    assert Path("data/processed/TSLA_processed.csv").exists()


def test_dataset_can_be_loaded():
    df = pd.read_csv("data/processed/TSLA_processed.csv")
    assert not df.empty


def test_required_columns_exist():
    df = pd.read_csv("data/processed/TSLA_processed.csv")

    required = [
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]

    for column in required:
        assert column in df.columns
