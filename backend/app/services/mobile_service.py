from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import get_settings
from app.schemas.mobile_schema import MobilePredictionRequest, MobilePredictionResponse
from app.utils.model_loader import load_joblib


PRICE_RANGE_LABELS = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost",
}


@dataclass
class MobileModelBundle:
    model: Any
    model_name: str
    feature_columns: list[str]
    metrics: dict[str, float]
    trained_at: str
    dataset_rows: int
    target: str
    dataset_source: str


class MobileService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.bundle: MobileModelBundle | None = None

    @property
    def model_path(self) -> Path:
        return self.settings.models_dir / "mobile_model.joblib"

    @property
    def is_loaded(self) -> bool:
        return self.bundle is not None

    def load(self) -> None:
        raw_bundle = load_joblib(self.model_path)
        self.bundle = MobileModelBundle(**raw_bundle)

    def metadata(self) -> dict[str, Any]:
        if self.bundle is None:
            self.load()
        assert self.bundle is not None
        return {
            "estimator": "Mobile Price Range Estimator",
            "model_name": self.bundle.model_name,
            "metrics": self.bundle.metrics,
            "trained_at": self.bundle.trained_at,
            "dataset_rows": self.bundle.dataset_rows,
            "target": self.bundle.target,
            "dataset_source": self.bundle.dataset_source,
            "status": "ready",
        }

    def predict(self, payload: MobilePredictionRequest) -> MobilePredictionResponse:
        if self.bundle is None:
            self.load()
        assert self.bundle is not None

        row = self._build_feature_row(payload)
        frame = pd.DataFrame([row], columns=self.bundle.feature_columns)
        predicted_class = int(self.bundle.model.predict(frame)[0])
        probabilities = self._probabilities(frame)
        confidence = probabilities.get(str(predicted_class), 0.0)

        return MobilePredictionResponse(
            predicted_price_range=predicted_class,
            label=PRICE_RANGE_LABELS[predicted_class],
            confidence=round(confidence, 4),
            probabilities={PRICE_RANGE_LABELS[int(key)]: round(value, 4) for key, value in probabilities.items()},
            model_name=self.bundle.model_name,
            explanation=self._explain(payload, predicted_class),
            input_summary={
                "ram": payload.ram,
                "battery_power": payload.battery_power,
                "internal_memory": payload.internal_memory,
                "camera_profile": f"{payload.primary_camera_mp}MP / {payload.front_camera_mp}MP",
                "connectivity": "4G" if payload.four_g else "3G" if payload.three_g else "Basic",
            },
        )

    def _probabilities(self, frame: pd.DataFrame) -> dict[str, float]:
        assert self.bundle is not None
        if not hasattr(self.bundle.model, "predict_proba"):
            return {}
        values = self.bundle.model.predict_proba(frame)[0]
        classes = [str(int(value)) for value in self.bundle.model.classes_]
        return dict(zip(classes, [float(value) for value in values]))

    @staticmethod
    def _build_feature_row(payload: MobilePredictionRequest) -> dict[str, float]:
        return {
            "battery_power": float(payload.battery_power),
            "blue": 1.0 if payload.bluetooth else 0.0,
            "clock_speed": float(payload.clock_speed),
            "dual_sim": 1.0 if payload.dual_sim else 0.0,
            "fc": float(payload.front_camera_mp),
            "four_g": 1.0 if payload.four_g else 0.0,
            "int_memory": float(payload.internal_memory),
            "m_dep": float(payload.mobile_depth_cm),
            "mobile_wt": float(payload.mobile_weight),
            "n_cores": float(payload.n_cores),
            "pc": float(payload.primary_camera_mp),
            "px_height": float(payload.pixel_height),
            "px_width": float(payload.pixel_width),
            "ram": float(payload.ram),
            "sc_h": float(payload.screen_height_cm),
            "sc_w": float(payload.screen_width_cm),
            "talk_time": float(payload.talk_time),
            "three_g": 1.0 if payload.three_g else 0.0,
            "touch_screen": 1.0 if payload.touch_screen else 0.0,
            "wifi": 1.0 if payload.wifi else 0.0,
        }

    @staticmethod
    def _explain(payload: MobilePredictionRequest, predicted_class: int) -> list[str]:
        factors: list[str] = []
        if payload.ram >= 4096:
            factors.append("High RAM is the strongest positive signal for the price range.")
        elif payload.ram <= 1500:
            factors.append("Limited RAM pulls the device toward a lower market tier.")
        else:
            factors.append("RAM places the device in a competitive mid-range performance band.")

        if payload.battery_power >= 1700:
            factors.append("Large battery capacity supports a higher utility profile.")
        if payload.internal_memory >= 128:
            factors.append("High internal storage improves the device value profile.")
        if payload.primary_camera_mp >= 48:
            factors.append("Camera specification strengthens the premium signal.")
        if payload.four_g:
            factors.append("4G connectivity keeps the device aligned with modern buyer expectations.")
        factors.append(f"The classifier maps this profile to {PRICE_RANGE_LABELS[predicted_class]}.")
        return factors[:5]


mobile_service = MobileService()
