from pydantic import BaseModel, Field


class CarPredictionRequest(BaseModel):
    make: str = Field("Toyota", min_length=2, max_length=40)
    model: str = Field("Camry", min_length=1, max_length=40)
    body_type: str = Field("sedan", pattern="^(sedan|suv|truck|hatchback|coupe|wagon)$")
    fuel_type: str = Field("gasoline", pattern="^(gasoline|diesel|hybrid|electric)$")
    transmission: str = Field("automatic", pattern="^(automatic|manual)$")
    year: int = Field(..., ge=1995, le=2026)
    mileage: int = Field(..., ge=0, le=350000)
    engine_size_l: float = Field(..., ge=0.0, le=8.5)
    horsepower: int = Field(..., ge=60, le=1000)
    owners: int = Field(1, ge=1, le=8)
    accident_history: bool = False
    condition_score: int = Field(4, ge=1, le=5)


class CarPredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str
    model_name: str
    price_range: dict[str, float | str]
    confidence: str
    value_badge: str
    explanation: list[str]
    input_summary: dict[str, str | int | float | bool]
