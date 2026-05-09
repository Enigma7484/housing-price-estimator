from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "car"
TRAIN_PATH = DATA_DIR / "train.csv"
MODEL_PATH = ROOT / "app" / "models" / "car_model.joblib"
METRICS_PATH = ROOT / "app" / "models" / "car_metrics.json"
TARGET = "price"

CATEGORICAL_COLUMNS = ["make", "model", "body_type", "fuel_type", "transmission"]
NUMERIC_COLUMNS = [
    "year",
    "mileage",
    "engine_size_l",
    "horsepower",
    "owners",
    "accident_history",
    "condition_score",
]
FEATURE_COLUMNS = [*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]


def create_bootstrap_dataset(path: Path, rows: int = 4200) -> str:
    rng = np.random.default_rng(51)
    path.parent.mkdir(parents=True, exist_ok=True)

    makes = np.array(["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Tesla", "Hyundai", "Chevrolet"])
    make_models = {
        "Toyota": ["Camry", "Corolla", "RAV4"],
        "Honda": ["Civic", "Accord", "CR-V"],
        "Ford": ["F-150", "Escape", "Mustang"],
        "BMW": ["3 Series", "5 Series", "X5"],
        "Mercedes": ["C-Class", "E-Class", "GLC"],
        "Tesla": ["Model 3", "Model Y", "Model S"],
        "Hyundai": ["Elantra", "Tucson", "Sonata"],
        "Chevrolet": ["Malibu", "Equinox", "Silverado"],
    }
    body_by_model = {
        "F-150": "truck",
        "Silverado": "truck",
        "Mustang": "coupe",
        "RAV4": "suv",
        "CR-V": "suv",
        "Escape": "suv",
        "X5": "suv",
        "GLC": "suv",
        "Model Y": "suv",
        "Tucson": "suv",
        "Equinox": "suv",
    }
    make = rng.choice(makes, rows, p=[0.18, 0.17, 0.15, 0.11, 0.1, 0.08, 0.12, 0.09])
    model = np.array([rng.choice(make_models[item]) for item in make])
    body_type = np.array([body_by_model.get(item, "sedan") for item in model])
    fuel_type = np.array([
        "electric" if m == "Tesla" else rng.choice(["gasoline", "hybrid", "diesel"], p=[0.76, 0.18, 0.06])
        for m in make
    ])

    year = rng.integers(2000, 2027, rows)
    age = 2026 - year
    mileage = np.maximum((age * rng.normal(11200, 3600, rows)).astype(int), rng.integers(2000, 22000, rows))
    condition_score = np.clip(5 - (age / 8) - (mileage / 160000) + rng.normal(0.7, 0.65, rows), 1, 5).round().astype(int)
    horsepower = np.array([
        rng.integers(260, 520) if m in {"BMW", "Mercedes", "Tesla"} else rng.integers(120, 330)
        for m in make
    ])
    engine_size_l = np.where(fuel_type == "electric", 0.0, np.round(rng.uniform(1.4, 5.7, rows), 1))
    owners = np.clip((age / 5 + rng.normal(1.1, 0.8, rows)).round().astype(int), 1, 7)
    accident_history = rng.binomial(1, np.clip(0.05 + age * 0.015 + owners * 0.02, 0.04, 0.42), rows)
    transmission = np.where(rng.random(rows) < 0.88, "automatic", "manual")

    make_premium = pd.Series(make).map({
        "Toyota": 3500,
        "Honda": 3200,
        "Ford": 2500,
        "BMW": 18500,
        "Mercedes": 21000,
        "Tesla": 24000,
        "Hyundai": 1300,
        "Chevrolet": 1800,
    }).to_numpy()
    body_premium = pd.Series(body_type).map({"sedan": 0, "suv": 4200, "truck": 6200, "hatchback": -1200, "coupe": 2600, "wagon": 500}).to_numpy()
    fuel_premium = pd.Series(fuel_type).map({"gasoline": 0, "diesel": 1700, "hybrid": 3200, "electric": 7600}).to_numpy()

    price = (
        8500
        + make_premium
        + body_premium
        + fuel_premium
        + (year - 2000) * 1150
        - mileage * 0.105
        + horsepower * 42
        + engine_size_l * 1150
        + condition_score * 2800
        - owners * 950
        - accident_history * 5200
        + rng.normal(0, 3200, rows)
    )
    price = np.maximum(price, 2500)

    df = pd.DataFrame({
        "make": make,
        "model": model,
        "body_type": body_type,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "year": year,
        "mileage": mileage,
        "engine_size_l": engine_size_l,
        "horsepower": horsepower,
        "owners": owners,
        "accident_history": accident_history,
        "condition_score": condition_score,
        TARGET: price.round(2),
    })
    df.to_csv(path, index=False)
    return "synthetic_bootstrap"


def load_dataset() -> tuple[pd.DataFrame, str]:
    source = create_bootstrap_dataset(TRAIN_PATH) if not TRAIN_PATH.exists() else "local_csv"
    df = pd.read_csv(TRAIN_PATH)
    missing_columns = [column for column in [*FEATURE_COLUMNS, TARGET] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Car dataset is missing required columns: {missing_columns}")
    return df[[*FEATURE_COLUMNS, TARGET]].copy(), source


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    predictions = model.predict(x_test)
    return {
        "model_name": name,
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse": round(float(root_mean_squared_error(y_test, predictions)), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }


def make_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train() -> None:
    df, dataset_source = load_dataset()
    x = df[FEATURE_COLUMNS]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    candidates = {
        "RandomForestRegressor": make_pipeline(
            RandomForestRegressor(n_estimators=240, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)
        ),
        "GradientBoostingRegressor": make_pipeline(
            GradientBoostingRegressor(n_estimators=240, learning_rate=0.05, max_depth=4, random_state=42)
        ),
    }

    reports = []
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        reports.append(evaluate_model(name, model, x_test, y_test))

    best_report = max(reports, key=lambda report: float(report["r2"]))
    bundle = {
        "model": candidates[str(best_report["model_name"])],
        "model_name": best_report["model_name"],
        "feature_columns": FEATURE_COLUMNS,
        "metrics": {key: value for key, value in best_report.items() if key != "model_name"},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "target": TARGET,
        "dataset_source": dataset_source,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"candidates": reports, "best": best_report}, indent=2), encoding="utf-8")

    print("Car model training complete.")
    print(json.dumps({"best": best_report, "dataset_source": dataset_source, "artifact": str(MODEL_PATH)}, indent=2))


if __name__ == "__main__":
    train()
