import pandas as pd
import os

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file from a local path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)

