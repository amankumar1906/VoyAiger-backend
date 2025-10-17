"""Input validation for API requests"""
from datetime import date, timedelta
from typing import Tuple
from ..config import settings
from ..schemas.request import GenerateItineraryRequest


class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


def validate_budget(budget: float) -> None:
    """
    Validate budget is within reasonable limits

    Args:
        budget: Budget amount in USD

    Raises:
        ValidationError: If budget is invalid
    """
    if budget < settings.min_budget:
        raise ValidationError(
            f"Budget too low. Minimum budget is ${settings.min_budget}",
            {"min_budget": settings.min_budget, "provided_budget": budget}
        )

    if budget > settings.max_budget:
        raise ValidationError(
            f"Budget too high. Maximum budget is ${settings.max_budget}",
            {"max_budget": settings.max_budget, "provided_budget": budget}
        )


def validate_dates(start_date: date, end_date: date) -> Tuple[int, bool]:
    """
    Validate trip dates are reasonable

    Args:
        start_date: Trip start date
        end_date: Trip end date

    Returns:
        Tuple of (number_of_days, is_valid)

    Raises:
        ValidationError: If dates are invalid
    """
    today = date.today()

    # Check if start date is in the past
    if start_date < today:
        raise ValidationError(
            "Start date cannot be in the past",
            {"start_date": str(start_date), "today": str(today)}
        )

    # Calculate trip duration
    duration = (end_date - start_date).days

    # Check if trip is too long
    if duration > settings.max_trip_days:
        raise ValidationError(
            f"Trip duration too long. Maximum is {settings.max_trip_days} days",
            {"duration_days": duration, "max_days": settings.max_trip_days}
        )

    # Check if trip is at least 1 day
    if duration < 1:
        raise ValidationError(
            "Trip must be at least 1 day long",
            {"duration_days": duration}
        )

    return duration, True


def validate_city(city: str) -> None:
    """
    Validate city name

    Args:
        city: City name

    Raises:
        ValidationError: If city name is invalid
    """
    # Basic validation (more can be added)
    if not city or city.strip() == "":
        raise ValidationError("City name cannot be empty")

    # Check for suspicious characters
    invalid_chars = ['<', '>', '{', '}', '[', ']', '|', '\\']
    if any(char in city for char in invalid_chars):
        raise ValidationError(
            "City name contains invalid characters",
            {"invalid_characters": invalid_chars}
        )


def validate_request(request: GenerateItineraryRequest) -> dict:
    """
    Validate complete request

    Args:
        request: The itinerary generation request

    Returns:
        dict: Validation metadata (duration, etc.)

    Raises:
        ValidationError: If request is invalid
    """
    # Validate city
    validate_city(request.city)

    # Validate budget
    validate_budget(request.budget)

    # Validate dates
    duration, _ = validate_dates(request.dates.start, request.dates.end)

    return {
        "trip_duration_days": duration,
        "budget_per_day": request.budget / duration
    }
