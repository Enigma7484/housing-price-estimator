from fastapi import APIRouter, HTTPException

from app.schemas.mobile_schema import MobilePredictionRequest, MobilePredictionResponse
from app.services.mobile_service import mobile_service

router = APIRouter()


@router.post("/predict", response_model=MobilePredictionResponse)
def predict_mobile_price_range(payload: MobilePredictionRequest):
    try:
        return mobile_service.predict(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Mobile prediction failed.") from exc


@router.get("/metadata")
def mobile_metadata():
    return mobile_service.metadata()
