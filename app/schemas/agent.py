"""Agent-specific schemas for internal use"""
from typing import List
from pydantic import BaseModel, Field
from .response import Hotel, Attraction, Restaurant


class HotelAgentOutput(BaseModel):
    """Output schema for Hotel Agent"""
    options: List[Hotel] = Field(..., max_length=3, description="Up to 3 hotel options")
    total_allocated_budget: float = Field(..., description="Budget allocated for hotels")


class AttractionsAgentOutput(BaseModel):
    """Output schema for Attractions Agent"""
    options: List[Attraction] = Field(..., max_length=3, description="Up to 3 attraction options")
    total_allocated_budget: float = Field(..., description="Budget allocated for attractions")


class RestaurantAgentOutput(BaseModel):
    """Output schema for Restaurant Agent"""
    options: List[Restaurant] = Field(..., max_length=3, description="Up to 3 restaurant options")
    total_allocated_budget: float = Field(..., description="Budget allocated for restaurants")


class BudgetAllocation(BaseModel):
    """Budget allocation across categories"""
    hotel_budget: float = Field(..., description="Budget for hotels")
    attractions_budget: float = Field(..., description="Budget for attractions")
    restaurants_budget: float = Field(..., description="Budget for restaurants")
    contingency: float = Field(..., description="Remaining contingency budget")
