"""Itinerary feedback database model"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator


class ItineraryFeedbackCreate(BaseModel):
    """Request model for creating/updating feedback"""
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5 stars")
    feedback_text: Optional[str] = Field(None, max_length=1000, description="Optional text feedback")

    @validator('rating')
    def validate_rating(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Rating must be between 1 and 5')
        return v


class ItineraryFeedback(BaseModel):
    """Itinerary feedback model matching Supabase schema"""
    id: Optional[str] = None
    itinerary_id: str = Field(..., description="Foreign key to itineraries table")
    user_id: str = Field(..., description="Foreign key to users table")
    rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ItineraryFeedbackInDB(BaseModel):
    """Feedback model as stored in database"""
    id: str
    itinerary_id: str
    user_id: str
    rating: int
    feedback_text: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ItineraryFeedbackResponse(BaseModel):
    """Response model for feedback"""
    id: str
    itinerary_id: str
    rating: int
    feedback_text: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
