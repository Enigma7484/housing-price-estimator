from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing" / "kc_house_data.csv"
MODEL_PATH = ROOT / "app" / "models" / "housing_model.joblib"
METRICS_PATH = ROOT / "app" / "models" / "housing_metrics.json"


def clean_housing_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]
    cleaned = cleaned.drop(columns=[column for column in ["id", "date"] if column in cleaned.columns])
    cleaned = cleaned.dropna(subset=["price"])

    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        if column != "price":
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    return cleaned


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    predictions = model.predict(x_test)
    return {
        "model_name": name,
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse": round(float(root_mean_squared_error(y_test, predictions)), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }


def train() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Housing dataset not found: {DATA_PATH}")

    df = clean_housing_dataframe(pd.read_csv(DATA_PATH))
    target = "price"
    feature_columns = [column for column in df.columns if column != target]
    x = df[feature_columns]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    candidates: dict[str, Pipeline] = {
        "Linear Regression Baseline": Pipeline(
            steps=[("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "RandomForestRegressor": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=220,
                        max_depth=22,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "GradientBoostingRegressor": Pipeline(
            steps=[
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=280,
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=42,
                    ),
                )
            ]
        ),
    }

    reports = []
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        reports.append(evaluate_model(name, model, x_test, y_test))

    best_report = max(reports, key=lambda report: float(report["r2"]))
    best_model = candidates[str(best_report["model_name"])]
    bundle = {
        "model": best_model,
        "model_name": best_report["model_name"],
        "feature_columns": feature_columns,
        "defaults": x.median(numeric_only=True).to_dict(),
        "metrics": {key: value for key, value in best_report.items() if key != "model_name"},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "target": target,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"candidates": reports, "best": best_report}, indent=2), encoding="utf-8")

    print("Housing model training complete.")
    print(json.dumps({"best": best_report, "artifact": str(MODEL_PATH)}, indent=2))


if __name__ == "__main__":
    train()
