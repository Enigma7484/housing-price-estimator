from pydantic import BaseModel, Field


class HousingPredictionRequest(BaseModel):
    square_footage: int = Field(..., ge=250, le=20000, description="Interior living area in square feet.")
    lot_size: int = Field(5000, ge=500, le=200000, description="Lot size in square feet.")
    bedrooms: int = Field(..., ge=0, le=15)
    bathrooms: float = Field(..., ge=0, le=10)
    floors: float = Field(1, ge=1, le=4)
    waterfront: bool = False
    view: int = Field(0, ge=0, le=4)
    condition: int = Field(3, ge=1, le=5)
    grade: int = Field(7, ge=1, le=13)
    year_built: int = Field(1995, ge=1900, le=2026)
    year_renovated: int = Field(0, ge=0, le=2026)
    zipcode: str = Field("98103", min_length=5, max_length=5)
    latitude: float | None = Field(None, ge=47.0, le=48.0)
    longitude: float | None = Field(None, ge=-123.0, le=-121.0)
    parking: int = Field(1, ge=0, le=8, description="Presentation input; used as a qualitative factor.")
    furnishing_status: str = Field("standard", pattern="^(basic|standard|premium)$")
    main_road_access: bool = True
    basement: bool = False
    air_conditioning: bool = False


class HousingPredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str
    model_name: str
    price_range: dict[str, float | str]
    confidence: str
    value_badge: str
    explanation: list[str]
    input_summary: dict[str, str | int | float | bool]
