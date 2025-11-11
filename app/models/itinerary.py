"""Itinerary database model"""
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class Itinerary(BaseModel):
    """Itinerary model matching Supabase itineraries table schema"""
    id: Optional[str] = None
    user_id: str = Field(..., description="Foreign key to users table")
    city: str = Field(..., min_length=1, max_length=100)
    start_date: str
    end_date: str
    preferences: Optional[str] = None
    itinerary_data: dict = Field(..., description="JSON data containing the full itinerary")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ItineraryInDB(BaseModel):
    """Itinerary model as stored in database"""
    id: str
    user_id: str
    city: str
    start_date: str
    end_date: str
    preferences: Optional[str]
    itinerary_data: dict
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
