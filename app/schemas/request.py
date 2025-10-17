"""Request schemas for API endpoints"""
from datetime import date
from pydantic import BaseModel, Field, field_validator


class DateRange(BaseModel):
    """Date range for the trip"""
    start: date = Field(..., description="Trip start date")
    end: date = Field(..., description="Trip end date")

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v, info):
        """Validate end date is after start date"""
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("End date must be after start date")
        return v


class GenerateItineraryRequest(BaseModel):
    """Request body for /generate endpoint"""
    city: str = Field(..., min_length=2, max_length=100, description="Destination city")
    budget: float = Field(..., gt=0, description="Total budget in USD")
    dates: DateRange = Field(..., description="Trip date range")

    class Config:
        json_schema_extra = {
            "example": {
                "city": "Paris",
                "budget": 2000,
                "dates": {
                    "start": "2025-06-01",
                    "end": "2025-06-07"
                }
            }
        }
