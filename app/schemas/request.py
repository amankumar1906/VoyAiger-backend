"""Request schemas for API endpoints"""
from datetime import date, timedelta
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DateRange(BaseModel):
    """Date range for the trip"""
    start: date = Field(..., description="Trip start date")
    end: date = Field(..., description="Trip end date")

    @field_validator("start")
    @classmethod
    def start_is_future(cls, v):
        """Validate start date is in the future"""
        if v < date.today():
            raise ValueError("Start date must be in the future")
        return v

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v, info):
        """Validate end date is after start date"""
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("End date must be after start date")
        return v

    @field_validator("end")
    @classmethod
    def max_duration(cls, v, info):
        """Validate trip duration is reasonable (max 1 year)"""
        if "start" in info.data:
            duration = (v - info.data["start"]).days
            if duration > 365:
                raise ValueError("Trip duration cannot exceed 365 days")
        return v


class CityLocation(BaseModel):
    """City location with coordinates from Google Autocomplete"""
    name: str = Field(..., description="City name (e.g., 'Miami, FL')")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")


class GenerateItineraryRequest(BaseModel):
    """
    Request body for /generate endpoint

    City includes coordinates from Google Autocomplete on frontend
    Budget is extracted from preferences if mentioned
    """
    city: CityLocation = Field(
        ...,
        description="City with coordinates from Google Autocomplete"
    )
    dates: DateRange = Field(..., description="Trip date range")
    preferences: Optional[str] = Field(
        None,
        max_length=500,
        description="Unstructured preferences (e.g., 'I love nightlife and have $1500 budget')"
    )

    @field_validator("preferences")
    @classmethod
    def validate_preferences(cls, v):
        """Validate preferences for prompt injection attempts"""
        if v is None:
            return v

        # Check for prompt injection patterns
        injection_patterns = [
            'ignore previous instructions',
            'ignore all previous',
            'disregard',
            'system:',
            'assistant:',
            'user:',
            '<|im_start|>',
            '<|im_end|>',
            '###',
            'SYSTEM',
            'ASSISTANT',
        ]

        v_lower = v.lower()
        for pattern in injection_patterns:
            if pattern in v_lower:
                raise ValueError("Invalid preferences format - suspicious content detected")

        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "city": {
                    "name": "Miami, FL",
                    "latitude": 25.7617,
                    "longitude": -80.1918
                },
                "dates": {
                    "start": "2025-11-01",
                    "end": "2025-11-05"
                },
                "preferences": "I love nightlife and have $1500 budget for hotels"
            }
        }
