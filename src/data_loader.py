from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_processed_data(filename):
    """
    Load a processed CSV file from the data/processed directory.

    Parameters
    ----------
    filename : str
        CSV filename.

    Returns
    -------
    pandas.DataFrame
    """

    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} was not found."
        )

    df = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    required_columns = [
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df
