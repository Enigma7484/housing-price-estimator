from pathlib import Path
from typing import Any

import joblib


def load_joblib(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python training/train_housing_model.py` first."
        )
    return joblib.load(path)
