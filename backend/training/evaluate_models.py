from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "app" / "models" / "housing_model.joblib"


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    print(
        {
            "model_name": bundle["model_name"],
            "metrics": bundle["metrics"],
            "trained_at": bundle["trained_at"],
            "dataset_rows": bundle["dataset_rows"],
        }
    )


if __name__ == "__main__":
    main()
