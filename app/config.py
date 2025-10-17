"""Configuration settings using Pydantic"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Keys
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    google_places_api_key: str = Field(..., alias="GOOGLE_PLACES_API_KEY")
    xotelo_api_key: str = Field(..., alias="XOTELO_API_KEY")

    # Application Settings
    env: str = Field(default="development", alias="ENV")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    # Model Settings
    model_name: str = "gemini-2.0-flash-lite"
    model_temperature: float = 0.7

    # Budget Validation
    min_budget: float = 100.0
    max_budget: float = 100000.0
    max_trip_days: int = 365

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
