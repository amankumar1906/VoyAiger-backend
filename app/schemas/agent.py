"""Agent-specific schemas for internal use and LLM output validation"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from .response import DayActivity, OptionalActivity


class DaySchedule(BaseModel):
    """Daily schedule from LLM - with weather integration"""
    model_config = {"extra": "forbid"}

    day_number: int = Field(..., ge=1, description="Day number (1, 2, 3, etc.)")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    weather: Optional[str] = Field(None, description="Weather for this day (e.g., 'Sunny, 75°F')")
    activities: List[DayActivity] = Field(default_factory=list, description="Scheduled activities for this day")

    @field_validator("activities")
    @classmethod
    def validate_activities_not_empty(cls, v):
        """Ensure at least one activity per day"""
        # Allow empty for flexibility days or travel days
        # if not v or len(v) == 0:
        #     raise ValueError("Each day must have at least one activity")
        return v


class ItineraryPlanLLM(BaseModel):
    """
    Schema for final planning LLM output - SINGLE itinerary with optional activities

    This schema validates the JSON output from the second LLM call (planning step)
    """
    model_config = {"extra": "forbid"}

    hotel_index: Optional[int] = Field(
        None,
        ge=0,
        description="Index of selected hotel (null if no budget provided)"
    )
    attraction_indices: List[int] = Field(
        ...,
        min_length=1,
        description="Indices of main attractions to include"
    )
    restaurant_indices: List[int] = Field(
        default_factory=list,
        description="Indices of main restaurants to include (empty if no restaurants searched)"
    )
    daily_schedule: List[DaySchedule] = Field(
        ...,
        min_length=1,
        description="Day-by-day schedule with adaptive time granularity"
    )
    optional_activities: List[OptionalActivity] = Field(
        default_factory=list,
        description="Alternative activities user can swap in"
    )
    estimated_total: Optional[str] = Field(
        None,
        description="Estimated total cost (null if no budget)"
    )
    reasoning: str = Field(
        ...,
        min_length=50,
        max_length=400,
        description="Concise 2-3 sentence summary explaining the itinerary's key themes and why these choices work for the user (max 400 characters)"
    )

    @field_validator("daily_schedule")
    @classmethod
    def validate_schedule_order(cls, v):
        """Ensure days are in sequential order"""
        if not v:
            raise ValueError("Daily schedule cannot be empty")

        for i, day in enumerate(v):
            expected_day = i + 1
            if day.day_number != expected_day:
                raise ValueError(f"Day numbers must be sequential. Expected {expected_day}, got {day.day_number}")

        return v
