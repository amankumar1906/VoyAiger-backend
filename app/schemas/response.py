"""Response schemas for API endpoints"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Hotel(BaseModel):
    """Hotel option"""
    name: str = Field(..., description="Hotel name")
    address: str = Field(..., description="Hotel address")
    price_per_night: float = Field(..., description="Price per night in USD")
    total_price: float = Field(..., description="Total price for the stay")
    rating: Optional[float] = Field(None, description="Hotel rating (1-5)")
    amenities: List[str] = Field(default_factory=list, description="Hotel amenities")


class Attraction(BaseModel):
    """Attraction option"""
    name: str = Field(..., description="Attraction name")
    address: str = Field(..., description="Attraction address")
    price_level: Optional[int] = Field(None, description="Price level 0-4 (0=Free, 1=$, 2=$$, 3=$$$, 4=$$$$)")
    price_display: Optional[str] = Field(None, description="Price display (Free, $, $$, $$$, $$$$)")
    rating: Optional[float] = Field(None, description="Attraction rating (1-5)")
    category: Optional[str] = Field(None, description="Category (museum, landmark, etc.)")


class Restaurant(BaseModel):
    """Restaurant option"""
    name: str = Field(..., description="Restaurant name")
    address: str = Field(..., description="Restaurant address")
    price_level: Optional[int] = Field(None, description="Price level 0-4 (0=Free, 1=$5-10, 2=$15-20, 3=$25-30, 4=$35+)")
    price_display: Optional[str] = Field(None, description="Price display (Free, $, $$, $$$, $$$$)")
    cuisine: Optional[str] = Field(None, description="Cuisine type")
    rating: Optional[float] = Field(None, description="Restaurant rating (1-5)")


class DayActivity(BaseModel):
    """Activity for a specific time slot"""
    model_config = {"extra": "forbid"}

    time: str = Field(..., description="Time of activity (e.g., '9:00 AM', 'Lunch', 'Evening')")
    type: str = Field(..., description="Type of activity: 'attraction', 'restaurant', 'hotel'")
    venue: str = Field(..., description="Name of the venue")
    address: str = Field(..., description="Address of the venue")
    price_display: Optional[str] = Field(None, description="Price display (Free, $, $$, $$$, $$$$)")
    notes: Optional[str] = Field(None, description="Additional notes about the activity")


class DayPlan(BaseModel):
    """Plan for a single day"""
    model_config = {"extra": "forbid"}

    day_number: int = Field(..., description="Day number (1, 2, 3, etc.)")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    weather: Optional[str] = Field(None, description="Weather forecast for this day (e.g., 'Sunny, 75°F')")
    activities: List[DayActivity] = Field(..., description="Activities scheduled for this day")


class OptionalActivity(BaseModel):
    """Optional activity that user can choose to add"""
    type: str = Field(..., description="Type: 'attraction' or 'restaurant'")
    venue: str = Field(..., description="Name of the venue")
    address: str = Field(..., description="Address")
    price_display: Optional[str] = Field(None, description="Price display")
    notes: str = Field(..., description="Why this is a good alternative option")


class Itinerary(BaseModel):
    """
    Complete single itinerary (not 3 tiers anymore)

    CHANGED: Now returns ONE itinerary with optional alternatives
    instead of 3 separate budget/balanced/premium options
    """
    hotel: Optional[Hotel] = Field(None, description="Selected hotel (null if no budget provided)")
    daily_plans: List[DayPlan] = Field(..., description="Day-by-day schedule with adaptive granularity")
    optional_activities: List[OptionalActivity] = Field(
        default_factory=list,
        description="Alternative activities user can swap in"
    )
    estimated_total: Optional[str] = Field(
        None,
        description="Estimated total cost range (e.g., '$1200-$1500'). Null if no budget."
    )


class GenerateItineraryResponse(BaseModel):
    """Response for /generate endpoint"""
    city: str = Field(..., description="Destination city")
    itinerary: Itinerary = Field(..., description="Single personalized itinerary")
    message: Optional[str] = Field(None, description="Additional message or notes")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")
