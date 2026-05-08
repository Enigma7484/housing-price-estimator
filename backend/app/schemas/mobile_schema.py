from pydantic import BaseModel, Field


class MobilePredictionRequest(BaseModel):
    battery_power: int = Field(..., ge=500, le=2200)
    clock_speed: float = Field(..., ge=0.5, le=3.5)
    ram: int = Field(..., ge=256, le=8192)
    internal_memory: int = Field(..., ge=2, le=256)
    mobile_weight: int = Field(..., ge=80, le=260)
    n_cores: int = Field(..., ge=1, le=12)
    primary_camera_mp: int = Field(..., ge=0, le=108)
    front_camera_mp: int = Field(..., ge=0, le=64)
    pixel_height: int = Field(..., ge=240, le=3200)
    pixel_width: int = Field(..., ge=240, le=3200)
    screen_height_cm: int = Field(..., ge=5, le=25)
    screen_width_cm: int = Field(..., ge=3, le=15)
    talk_time: int = Field(..., ge=2, le=32)
    mobile_depth_cm: float = Field(..., ge=0.1, le=1.5)
    bluetooth: bool = True
    dual_sim: bool = True
    four_g: bool = True
    three_g: bool = True
    touch_screen: bool = True
    wifi: bool = True


class MobilePredictionResponse(BaseModel):
    predicted_price_range: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str
    explanation: list[str]
    input_summary: dict[str, int | float | bool | str]
