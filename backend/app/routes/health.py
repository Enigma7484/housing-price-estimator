from fastapi import APIRouter

from app.services.estimator_registry import estimator_registry

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "services": estimator_registry.health(),
    }
