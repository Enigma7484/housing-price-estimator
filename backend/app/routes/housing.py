from fastapi import APIRouter, HTTPException

from app.schemas.housing_schema import HousingPredictionRequest, HousingPredictionResponse
from app.services.housing_service import housing_service

router = APIRouter()


@router.post("/predict", response_model=HousingPredictionResponse)
def predict_housing_price(payload: HousingPredictionRequest):
    try:
        return housing_service.predict(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Housing prediction failed.") from exc


@router.get("/metadata")
def housing_metadata():
    return housing_service.metadata()
