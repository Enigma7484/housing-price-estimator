from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "mobile"
TRAIN_PATH = DATA_DIR / "train.csv"
MODEL_PATH = ROOT / "app" / "models" / "mobile_model.joblib"
METRICS_PATH = ROOT / "app" / "models" / "mobile_metrics.json"

FEATURE_COLUMNS = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
]
TARGET = "price_range"


def create_bootstrap_dataset(path: Path, rows: int = 2500) -> str:
    rng = np.random.default_rng(42)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "battery_power": rng.integers(500, 2201, rows),
            "blue": rng.integers(0, 2, rows),
            "clock_speed": rng.uniform(0.5, 3.2, rows).round(2),
            "dual_sim": rng.integers(0, 2, rows),
            "fc": rng.integers(0, 33, rows),
            "four_g": rng.integers(0, 2, rows),
            "int_memory": rng.integers(2, 257, rows),
            "m_dep": rng.uniform(0.1, 1.2, rows).round(2),
            "mobile_wt": rng.integers(85, 241, rows),
            "n_cores": rng.integers(1, 9, rows),
            "pc": rng.integers(2, 65, rows),
            "px_height": rng.integers(240, 2601, rows),
            "px_width": rng.integers(320, 3001, rows),
            "ram": rng.integers(512, 8193, rows),
            "sc_h": rng.integers(6, 23, rows),
            "sc_w": rng.integers(3, 14, rows),
            "talk_time": rng.integers(3, 31, rows),
            "three_g": rng.integers(0, 2, rows),
            "touch_screen": rng.integers(0, 2, rows),
            "wifi": rng.integers(0, 2, rows),
        }
    )
    df["four_g"] = np.where(df["three_g"] == 0, 0, df["four_g"])

    score = (
        df["ram"] * 0.55
        + df["battery_power"] * 0.35
        + df["int_memory"] * 8
        + df["pc"] * 12
        + (df["px_height"] * df["px_width"]) / 5000
        + df["four_g"] * 180
        + df["wifi"] * 80
        + df["n_cores"] * 35
        - df["mobile_wt"] * 2.5
        + rng.normal(0, 260, rows)
    )
    df[TARGET] = pd.qcut(score, 4, labels=[0, 1, 2, 3]).astype(int)
    df.to_csv(path, index=False)
    return "synthetic_bootstrap"


def load_dataset() -> tuple[pd.DataFrame, str]:
    if not TRAIN_PATH.exists():
        source = create_bootstrap_dataset(TRAIN_PATH)
    else:
        source = "local_csv"

    df = pd.read_csv(TRAIN_PATH)
    missing_columns = [column for column in [*FEATURE_COLUMNS, TARGET] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Mobile dataset is missing required columns: {missing_columns}")

    df = df[[*FEATURE_COLUMNS, TARGET]].copy()
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return df, source


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    predictions = model.predict(x_test)
    return {
        "model_name": name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_macro": round(float(f1_score(y_test, predictions, average="macro")), 4),
    }


def train() -> None:
    df, dataset_source = load_dataset()
    x = df[FEATURE_COLUMNS]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates: dict[str, Pipeline] = {
        "RandomForestClassifier": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=260,
                        max_depth=18,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "GradientBoostingClassifier": Pipeline(
            steps=[
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=180,
                        learning_rate=0.06,
                        max_depth=3,
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

    best_report = max(reports, key=lambda report: float(report["f1_macro"]))
    best_model = candidates[str(best_report["model_name"])]
    bundle = {
        "model": best_model,
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

    print("Mobile model training complete.")
    print(json.dumps({"best": best_report, "dataset_source": dataset_source, "artifact": str(MODEL_PATH)}, indent=2))


if __name__ == "__main__":
    train()
