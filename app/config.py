"""Configuration settings using Pydantic"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Keys
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    google_places_api_key: str = Field(..., alias="GOOGLE_PLACES_API_KEY")
    xotelo_api_key: str = Field(..., alias="XOTELO_API_KEY")

    # Supabase Configuration
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_key: str = Field(..., alias="SUPABASE_KEY")

    # JWT Configuration
    jwt_secret_key: str = Field(..., alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=1440, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")  # 24 hours

    # Application Settings
    env: str = Field(default="development", alias="ENV")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    # Model Settings
    # Note: gemini-2.5-flash-lite may have limited structured output support
    # If experiencing issues, try: gemini-1.5-flash or gemini-1.5-pro
    model_name: str = Field(default="gemini-2.5-flash-lite", alias="MODEL_NAME")
    model_temperature: float = 0.2

    # Budget Validation
    min_budget: float = 100.0
    max_budget: float = 100000.0
    max_trip_days: int = 7  # Max 7 days for day-by-day itineraries

    # Security Settings
    request_timeout_seconds: int = Field(default=60, alias="REQUEST_TIMEOUT_SECONDS")
    max_preferences_length: int = Field(default=500, alias="MAX_PREFERENCES_LENGTH")
    max_city_length: int = Field(default=100, alias="MAX_CITY_LENGTH")

    # CORS Settings
    allowed_origins: str = Field(
        default="https://voyaiger.vercel.app",
        alias="ALLOWED_ORIGINS",
        description="Comma-separated list of allowed origins for CORS"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
