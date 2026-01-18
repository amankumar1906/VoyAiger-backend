"""Request schemas for API endpoints"""
from datetime import date, timedelta
from typing import Optional, Dict, Any, List
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
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="City name (e.g., 'Miami, FL')"
    )
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
        min_length=1,
        max_length=500,
        description="Additional notes like budget, dietary restrictions, etc."
    )
    user_preferences: Optional[List[str]] = Field(
        None,
        description="User's saved profile preferences (e.g., ['outdoor', 'beaches', 'nightlife'])"
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
                "preferences": "Budget is $1500 for hotels, prefer vegetarian food",
                "user_preferences": ["beaches", "nightlife", "outdoor", "relaxation"]
            }
        }


class SaveItineraryRequest(BaseModel):
    """Request body for /itineraries/save endpoint"""
    city: str = Field(..., min_length=1, max_length=100, description="City name")
    start_date: str = Field(..., description="Trip start date (ISO format)")
    end_date: str = Field(..., description="Trip end date (ISO format)")
    preferences: Optional[str] = Field(None, description="User preferences")
    itinerary_data: Dict[str, Any] = Field(..., description="Complete itinerary JSON data")

    class Config:
        json_schema_extra = {
            "example": {
                "city": "Miami, FL",
                "start_date": "2025-11-01",
                "end_date": "2025-11-05",
                "preferences": "I love beaches and nightlife",
                "itinerary_data": {
                    "hotel": {
                        "name": "Beachfront Hotel",
                        "address": "123 Ocean Drive",
                        "price_per_night": 200,
                        "total_price": 800,
                        "rating": 4.5,
                        "amenities": ["Pool", "WiFi"]
                    },
                    "daily_plans": [],
                    "optional_activities": [],
                    "estimated_total": "$1200-$1500"
                }
            }
        }


class UpdateItineraryItemRequest(BaseModel):
    """Request body for updating a specific item within a day"""
    time: Optional[str] = Field(None, description="Time of activity (e.g., '9:00 AM', 'Lunch', 'Evening')")
    venue: Optional[str] = Field(None, description="Name of the venue")
    address: Optional[str] = Field(None, description="Address of the venue")
    price_display: Optional[str] = Field(None, description="Price display (Free, $, $$, $$$, $$$$)")
    notes: Optional[str] = Field(None, description="Additional notes about the activity")
    expected_version: Optional[int] = Field(None, description="Expected version number for optimistic locking")

    class Config:
        json_schema_extra = {
            "example": {
                "time": "10:00 AM",
                "venue": "South Beach",
                "address": "1001 Ocean Dr, Miami Beach, FL 33139",
                "price_display": "Free",
                "notes": "Enjoy the beautiful beach and sunshine"
            }
        }


class AddActivityRequest(BaseModel):
    """Request body for adding a new activity to a day"""
    time: str = Field(..., description="Time of activity (e.g., '9:00 AM', 'Lunch', 'Evening')")
    type: str = Field(..., description="Type of activity (e.g., 'Restaurant', 'Attraction', 'Nightlife', 'Sightseeing')")
    venue: str = Field(..., description="Name of the venue")
    address: str = Field(..., description="Address of the venue")
    price_display: str = Field(..., description="Price display (Free, $, $$, $$$, $$$$)")
    notes: Optional[str] = Field(None, description="Additional notes about the activity")
    expected_version: Optional[int] = Field(None, description="Expected version number for optimistic locking")

    class Config:
        json_schema_extra = {
            "example": {
                "time": "3:00 PM",
                "type": "Sightseeing",
                "venue": "Art Deco Historic District",
                "address": "Ocean Dr, Miami Beach, FL 33139",
                "price_display": "Free",
                "notes": "Walking tour of the historic Art Deco buildings"
            }
        }


class SendInviteRequest(BaseModel):
    """Request body for sending an itinerary invite"""
    invitee_email: str = Field(
        ...,
        min_length=3,
        max_length=255,
        description="Email address of the person to invite"
    )

    @field_validator("invitee_email")
    @classmethod
    def validate_email(cls, v):
        """Basic email validation"""
        if "@" not in v or "." not in v:
            raise ValueError("Invalid email format")
        return v.lower().strip()

    class Config:
        json_schema_extra = {
            "example": {
                "invitee_email": "friend@example.com"
            }
        }


class RespondToInviteRequest(BaseModel):
    """Request body for accepting or rejecting an invite"""
    status: str = Field(..., description="Response status: 'accepted' or 'rejected'")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        """Validate status is either accepted or rejected"""
        if v not in ['accepted', 'rejected']:
            raise ValueError("Status must be 'accepted' or 'rejected'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "status": "accepted"
            }
        }
