from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import get_settings
from app.schemas.housing_schema import HousingPredictionRequest, HousingPredictionResponse
from app.utils.model_loader import load_joblib


@dataclass
class HousingModelBundle:
    model: Any
    model_name: str
    feature_columns: list[str]
    defaults: dict[str, float]
    metrics: dict[str, float]
    trained_at: str
    dataset_rows: int
    target: str


class HousingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.bundle: HousingModelBundle | None = None

    @property
    def model_path(self) -> Path:
        return self.settings.models_dir / "housing_model.joblib"

    @property
    def is_loaded(self) -> bool:
        return self.bundle is not None

    def load(self) -> None:
        raw_bundle = load_joblib(self.model_path)
        self.bundle = HousingModelBundle(**raw_bundle)

    def metadata(self) -> dict[str, Any]:
        if self.bundle is None:
            self.load()
        assert self.bundle is not None
        return {
            "estimator": "Housing Price Estimator",
            "model_name": self.bundle.model_name,
            "metrics": self.bundle.metrics,
            "trained_at": self.bundle.trained_at,
            "dataset_rows": self.bundle.dataset_rows,
            "target": self.bundle.target,
            "status": "ready",
        }

    def predict(self, payload: HousingPredictionRequest) -> HousingPredictionResponse:
        if self.bundle is None:
            self.load()
        assert self.bundle is not None

        row = self._build_feature_row(payload)
        frame = pd.DataFrame([row], columns=self.bundle.feature_columns)
        predicted_price = float(self.bundle.model.predict(frame)[0])
        predicted_price = max(predicted_price, 0.0)

        rmse = float(self.bundle.metrics.get("rmse", predicted_price * 0.12))
        lower = max(predicted_price - rmse, 0.0)
        upper = predicted_price + rmse

        return HousingPredictionResponse(
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
                "square_footage": payload.square_footage,
                "bedrooms": payload.bedrooms,
                "bathrooms": payload.bathrooms,
                "floors": payload.floors,
                "zipcode": payload.zipcode,
                "waterfront": payload.waterfront,
                "grade": payload.grade,
            },
        )

    def _build_feature_row(self, payload: HousingPredictionRequest) -> dict[str, float]:
        assert self.bundle is not None
        defaults = self.bundle.defaults.copy()
        sqft_living = float(payload.square_footage)
        basement_sqft = max(sqft_living * 0.22, 250.0) if payload.basement else 0.0
        sqft_above = max(sqft_living - basement_sqft, 0.0)
        lat = payload.latitude if payload.latitude is not None else defaults.get("lat", 47.6)
        long = payload.longitude if payload.longitude is not None else defaults.get("long", -122.2)

        updates = {
            "bedrooms": float(payload.bedrooms),
            "bathrooms": float(payload.bathrooms),
            "sqft_living": sqft_living,
            "sqft_lot": float(payload.lot_size),
            "floors": float(payload.floors),
            "waterfront": 1.0 if payload.waterfront else 0.0,
            "view": float(payload.view),
            "condition": float(payload.condition),
            "grade": float(payload.grade),
            "sqft_above": sqft_above,
            "sqft_basement": basement_sqft,
            "yr_built": float(payload.year_built),
            "yr_renovated": float(payload.year_renovated),
            "zipcode": float(payload.zipcode),
            "lat": float(lat),
            "long": float(long),
            "sqft_living15": sqft_living,
            "sqft_lot15": float(payload.lot_size),
        }

        defaults.update(updates)
        return {column: float(defaults[column]) for column in self.bundle.feature_columns}

    @staticmethod
    def _format_price(value: float) -> str:
        return f"${value:,.0f}"

    @staticmethod
    def _value_badge(value: float) -> str:
        if value >= 1_000_000:
            return "Premium market property"
        if value >= 650_000:
            return "Upper-tier residential asset"
        if value >= 400_000:
            return "Mid-market home"
        return "Affordable market segment"

    @staticmethod
    def _confidence_label(predicted_price: float, rmse: float) -> str:
        if predicted_price <= 0:
            return "Low"
        relative_error = rmse / predicted_price
        if relative_error <= 0.16:
            return "High"
        if relative_error <= 0.28:
            return "Moderate"
        return "Directional"

    @staticmethod
    def _explain(payload: HousingPredictionRequest, predicted_price: float) -> list[str]:
        factors: list[str] = []
        if payload.square_footage >= 3000:
            factors.append("Large living area is a primary positive driver.")
        elif payload.square_footage <= 1200:
            factors.append("Compact living area pulls the estimate toward the lower market range.")
        else:
            factors.append("Living area sits in a common residential range for this market.")

        if payload.grade >= 9:
            factors.append("High construction grade increases the valuation signal.")
        elif payload.grade <= 6:
            factors.append("Lower construction grade limits the predicted valuation.")

        if payload.waterfront:
            factors.append("Waterfront access is treated as a premium location feature.")
        if payload.basement:
            factors.append("Basement space contributes additional usable square footage.")
        if payload.air_conditioning or payload.furnishing_status == "premium":
            factors.append("Premium comfort inputs support a stronger qualitative profile.")
        if payload.year_renovated and payload.year_renovated >= 2000:
            factors.append("Recent renovation status supports a higher estimate.")
        if not factors:
            factors.append("Estimate is mainly driven by size, grade, location, and age.")
        factors.append(f"Predicted value falls in the {HousingService._value_badge(predicted_price).lower()} band.")
        return factors[:5]


housing_service = HousingService()
