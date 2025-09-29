import pandas as pd
from pathlib import Path
from datetime import datetime


CHECKPOINTS_DIR = Path("data/checkpoints")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def save_checkpoint(df: pd.DataFrame, name: str, with_timestamp: bool = False) -> Path:
    """
    Save a DataFrame as a checkpoint in data/checkpoints.

    Args:
        df (pd.DataFrame): DataFrame to save
        name (str): Base name of the checkpoint file (without extension)
        with_timestamp (bool): Whether to append a timestamp to the filename

    Returns:
        Path: Path to the saved file
    """
    filename = f"{name}.csv"
    if with_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.csv"

    file_path = CHECKPOINTS_DIR / filename
    df.to_csv(file_path, index=False)
    return file_path


def load_checkpoint(filename: str) -> pd.DataFrame:
    """
    Load a checkpoint DataFrame from data/checkpoints.

    Args:
        filename (str): Filename of the checkpoint (must exist in data/checkpoints)

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    file_path = CHECKPOINTS_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {file_path}")
    return pd.read_csv(file_path)
