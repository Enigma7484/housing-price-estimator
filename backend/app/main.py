import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import estimators, health, housing, mobile
from app.services.estimator_registry import estimator_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s", settings.app_name)
    estimator_registry.load_live_models()
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Production-oriented API for modular machine learning estimators.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[*settings.frontend_origins, "http://localhost:5173", "http://localhost:3000"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_prefix, tags=["Health"])
app.include_router(estimators.router, prefix=settings.api_prefix, tags=["Estimators"])
app.include_router(housing.router, prefix=f"{settings.api_prefix}/housing", tags=["Housing"])
app.include_router(mobile.router, prefix=f"{settings.api_prefix}/mobile", tags=["Mobile"])


@app.get("/")
def root():
    return {"name": settings.app_name, "status": "online", "docs": "/docs"}
