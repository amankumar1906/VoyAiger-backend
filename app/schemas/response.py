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
    price: float = Field(..., description="Estimated cost in USD")
    rating: Optional[float] = Field(None, description="Attraction rating (1-5)")
    category: Optional[str] = Field(None, description="Category (museum, landmark, etc.)")


class Restaurant(BaseModel):
    """Restaurant option"""
    name: str = Field(..., description="Restaurant name")
    address: str = Field(..., description="Restaurant address")
    estimated_cost_per_meal: float = Field(..., description="Estimated cost per meal in USD")
    cuisine: Optional[str] = Field(None, description="Cuisine type")
    rating: Optional[float] = Field(None, description="Restaurant rating (1-5)")


class Itinerary(BaseModel):
    """Complete itinerary option"""
    hotel: Hotel = Field(..., description="Selected hotel")
    attractions: List[Attraction] = Field(..., description="List of attractions")
    restaurants: List[Restaurant] = Field(..., description="List of restaurants")
    total_cost: float = Field(..., description="Total estimated cost")
    remaining_budget: float = Field(..., description="Remaining budget after expenses")


class GenerateItineraryResponse(BaseModel):
    """Response for /generate endpoint"""
    city: str = Field(..., description="Destination city")
    options: List[Itinerary] = Field(..., max_length=3, description="Up to 3 itinerary options")
    message: Optional[str] = Field(None, description="Additional message or notes")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")
