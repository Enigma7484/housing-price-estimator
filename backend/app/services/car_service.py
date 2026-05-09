from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import get_settings
from app.schemas.car_schema import CarPredictionRequest, CarPredictionResponse
from app.utils.model_loader import load_joblib


@dataclass
class CarModelBundle:
    model: Any
    model_name: str
    feature_columns: list[str]
    metrics: dict[str, float]
    trained_at: str
    dataset_rows: int
    target: str
    dataset_source: str


class CarService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.bundle: CarModelBundle | None = None

    @property
    def model_path(self) -> Path:
        return self.settings.models_dir / "car_model.joblib"

    @property
    def is_loaded(self) -> bool:
        return self.bundle is not None

    def load(self) -> None:
        raw_bundle = load_joblib(self.model_path)
        self.bundle = CarModelBundle(**raw_bundle)

    def metadata(self) -> dict[str, Any]:
        if self.bundle is None:
            self.load()
        assert self.bundle is not None
        return {
            "estimator": "Car Price Estimator",
            "model_name": self.bundle.model_name,
            "metrics": self.bundle.metrics,
            "trained_at": self.bundle.trained_at,
            "dataset_rows": self.bundle.dataset_rows,
            "target": self.bundle.target,
            "dataset_source": self.bundle.dataset_source,
            "status": "ready",
        }

    def predict(self, payload: CarPredictionRequest) -> CarPredictionResponse:
        if self.bundle is None:
            self.load()
        assert self.bundle is not None

        row = self._build_feature_row(payload)
        frame = pd.DataFrame([row], columns=self.bundle.feature_columns)
        predicted_price = max(float(self.bundle.model.predict(frame)[0]), 0.0)
        rmse = float(self.bundle.metrics.get("rmse", predicted_price * 0.16))
        lower = max(predicted_price - rmse, 0.0)
        upper = predicted_price + rmse

        return CarPredictionResponse(
            predicted_price=round(predicted_price, 2),
            formatted_price=self._format_price(predicted_price),
            model_name=self.bundle.model_name,
            price_range={
                "low": round(lower, 2),
                "high": round(upper, 2),
                "formatted_low": self._format_price(lower),
                "formatted_high": self._format_price(upper),
            },
            confidence=self._confidence_label(predicted_price, rmse),
            value_badge=self._value_badge(predicted_price),
            explanation=self._explain(payload, predicted_price),
            input_summary={
                "vehicle": f"{payload.year} {payload.make} {payload.model}",
                "mileage": payload.mileage,
                "condition_score": payload.condition_score,
                "fuel_type": payload.fuel_type,
                "accident_history": payload.accident_history,
            },
        )

    @staticmethod
    def _build_feature_row(payload: CarPredictionRequest) -> dict[str, float | str | int]:
        return {
            "make": payload.make,
            "model": payload.model,
            "body_type": payload.body_type,
            "fuel_type": payload.fuel_type,
            "transmission": payload.transmission,
            "year": payload.year,
            "mileage": payload.mileage,
            "engine_size_l": payload.engine_size_l,
            "horsepower": payload.horsepower,
            "owners": payload.owners,
            "accident_history": int(payload.accident_history),
            "condition_score": payload.condition_score,
        }

    @staticmethod
    def _format_price(value: float) -> str:
        return f"${value:,.0f}"

    @staticmethod
    def _confidence_label(predicted_price: float, rmse: float) -> str:
        if predicted_price <= 0:
            return "Low"
        relative_error = rmse / predicted_price
        if relative_error <= 0.14:
            return "High"
        if relative_error <= 0.25:
            return "Moderate"
        return "Directional"

    @staticmethod
    def _value_badge(value: float) -> str:
        if value >= 60000:
            return "Premium resale vehicle"
        if value >= 32000:
            return "Upper-market vehicle"
        if value >= 16000:
            return "Mainstream resale vehicle"
        return "Budget resale segment"

    @staticmethod
    def _explain(payload: CarPredictionRequest, predicted_price: float) -> list[str]:
        factors: list[str] = []
        vehicle_age = 2026 - payload.year
        if vehicle_age <= 3:
            factors.append("Recent model year strongly supports resale value.")
        elif vehicle_age >= 12:
            factors.append("Older model year pulls the estimate downward.")
        else:
            factors.append("Model year places the vehicle in a common used-market range.")

        if payload.mileage <= 30000:
            factors.append("Low mileage improves the valuation profile.")
        elif payload.mileage >= 120000:
            factors.append("High mileage is a major depreciation signal.")

        if payload.condition_score >= 5:
            factors.append("Excellent condition supports a stronger price estimate.")
        elif payload.condition_score <= 2:
            factors.append("Lower condition score limits the resale estimate.")
        if payload.accident_history:
            factors.append("Accident history reduces the expected resale range.")
        if payload.fuel_type in {"hybrid", "electric"}:
            factors.append("Efficient powertrain adds a positive market signal.")
        factors.append(f"Predicted value falls in the {CarService._value_badge(predicted_price).lower()} band.")
        return factors[:5]


car_service = CarService()
