from fastapi import APIRouter

from app.services.estimator_registry import estimator_registry

router = APIRouter()


@router.get("/estimators")
def list_estimators():
    return {
        "platform": "AI Estimator Platform",
        "live_count": sum(1 for item in estimator_registry.catalog() if item["status"] == "ready"),
        "estimators": estimator_registry.catalog(),
    }
