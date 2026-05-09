from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.services.car_service import car_service
from app.services.housing_service import housing_service
from app.services.mobile_service import mobile_service


MetadataProvider = Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class EstimatorDefinition:
    key: str
    name: str
    category: str
    problem_type: str
    route: str | None
    frontend_path: str | None
    phase: str
    description: str
    metadata_provider: MetadataProvider | None = None


LIVE_ESTIMATORS = [
    EstimatorDefinition(
        key="housing",
        name="Housing Price Estimator",
        category="Real Estate",
        problem_type="Regression",
        route="/api/housing/predict",
        frontend_path="/estimators/housing",
        phase="Phase 1",
        description="Predict residential property prices from housing, location, and quality signals.",
        metadata_provider=housing_service.metadata,
    ),
    EstimatorDefinition(
        key="mobile",
        name="Mobile Price Range Estimator",
        category="Consumer Devices",
        problem_type="Classification",
        route="/api/mobile/predict",
        frontend_path="/estimators/mobile",
        phase="Phase 2",
        description="Classify mobile devices into market tiers from hardware and connectivity specs.",
        metadata_provider=mobile_service.metadata,
    ),
    EstimatorDefinition(
        key="car",
        name="Car Price Estimator",
        category="Automotive",
        problem_type="Regression",
        route="/api/car/predict",
        frontend_path="/estimators/car",
        phase="Phase 4",
        description="Predict vehicle resale value using mileage, make, year, condition, and trim.",
        metadata_provider=car_service.metadata,
    ),
]

PLANNED_ESTIMATORS = [
    EstimatorDefinition(
        key="laptop",
        name="Laptop Price Estimator",
        category="Consumer Devices",
        problem_type="Regression",
        route=None,
        frontend_path=None,
        phase="Roadmap",
        description="Estimate laptop value from CPU, RAM, storage, display, GPU, and brand profile.",
    ),
    EstimatorDefinition(
        key="salary",
        name="Salary Estimator",
        category="Labor Market",
        problem_type="Regression",
        route=None,
        frontend_path=None,
        phase="Roadmap",
        description="Model compensation bands from role, location, seniority, skills, and industry.",
    ),
    EstimatorDefinition(
        key="rent",
        name="Rent Price Estimator",
        category="Real Estate",
        problem_type="Regression",
        route=None,
        frontend_path=None,
        phase="Roadmap",
        description="Forecast rental prices from property features, geography, and market signals.",
    ),
    EstimatorDefinition(
        key="insurance",
        name="Insurance Cost Estimator",
        category="Risk",
        problem_type="Regression",
        route=None,
        frontend_path=None,
        phase="Roadmap",
        description="Estimate insurance costs from profile, asset, risk, and coverage signals.",
    ),
    EstimatorDefinition(
        key="used-phone-resale",
        name="Used Phone Resale Value Estimator",
        category="Consumer Devices",
        problem_type="Regression",
        route=None,
        frontend_path=None,
        phase="Roadmap",
        description="Predict resale value from device model, age, condition, storage, and market demand.",
    ),
]


class EstimatorRegistry:
    def __init__(self) -> None:
        self.estimators = [*LIVE_ESTIMATORS, *PLANNED_ESTIMATORS]

    def load_live_models(self) -> None:
        housing_service.load()
        try:
            mobile_service.load()
        except FileNotFoundError:
            pass
        try:
            car_service.load()
        except FileNotFoundError:
            pass

    def health(self) -> dict[str, str]:
        return {
            "housing": "ready" if housing_service.is_loaded else "model_not_loaded",
            "mobile": "ready" if mobile_service.is_loaded else "model_not_loaded",
            "car": "ready" if car_service.is_loaded else "model_not_loaded",
        }

    def catalog(self) -> list[dict[str, Any]]:
        return [self._serialize(definition) for definition in self.estimators]

    def _serialize(self, definition: EstimatorDefinition) -> dict[str, Any]:
        metadata: dict[str, Any] | None = None
        status = "planned"
        if definition.metadata_provider:
            try:
                metadata = definition.metadata_provider()
                status = metadata.get("status", "ready")
            except FileNotFoundError:
                status = "model_not_loaded"

        return {
            "key": definition.key,
            "name": definition.name,
            "category": definition.category,
            "problem_type": definition.problem_type,
            "route": definition.route,
            "frontend_path": definition.frontend_path,
            "phase": definition.phase,
            "description": definition.description,
            "status": status,
            "metadata": metadata,
        }


estimator_registry = EstimatorRegistry()
