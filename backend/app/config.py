import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "ResaleIQ API"
    api_prefix: str = "/api"
    frontend_origin: str = os.getenv("ESTIMATOR_FRONTEND_ORIGIN", "http://localhost:5173")

    @property
    def frontend_origins(self) -> list[str]:
        origins = os.getenv("ESTIMATOR_FRONTEND_ORIGINS", self.frontend_origin)
        return [origin.strip() for origin in origins.split(",") if origin.strip()]

    @property
    def backend_dir(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def models_dir(self) -> Path:
        return self.backend_dir / "app" / "models"

    @property
    def data_dir(self) -> Path:
        return self.backend_dir / "data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
