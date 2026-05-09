from fastapi import APIRouter, HTTPException

from app.schemas.car_schema import CarPredictionRequest, CarPredictionResponse
from app.services.car_service import car_service

router = APIRouter()


@router.post("/predict", response_model=CarPredictionResponse)
def predict_car_price(payload: CarPredictionRequest):
    try:
        return car_service.predict(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Car prediction failed.") from exc


@router.get("/metadata")
def car_metadata():
    return car_service.metadata()
