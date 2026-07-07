from pathlib import Path
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def load_lstm_model(filename="lstm_model.keras"):
    """
    Load a trained LSTM model.

    Parameters
    ----------
    filename : str

    Returns
    -------
    keras.Model
    """

    filepath = MODEL_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Model not found: {filepath}"
        )

    return load_model(filepath)
