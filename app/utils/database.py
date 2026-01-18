"""Supabase database utility functions"""
from supabase import create_client, Client
from app.config import settings
from typing import Optional, Dict, Any, List


class SupabaseClient:
    """Singleton Supabase client wrapper"""
    _instance: Optional[Client] = None

    @classmethod
    def get_client(cls) -> Client:
        """Get or create Supabase client instance"""
        if cls._instance is None:
            cls._instance = create_client(
                supabase_url=settings.supabase_url,
                supabase_key=settings.supabase_key
            )
        return cls._instance


# Database operations
async def create_user(name: str, email: str, password_hash: str) -> Dict[str, Any]:
    """
    Create a new user in the users table

    Args:
        name: User's full name
        email: User's email address
        password_hash: Hashed password

    Returns:
        Created user data

    Raises:
        Exception: If user creation fails or email already exists
    """
    client = SupabaseClient.get_client()

    # Check if user already exists
    existing = client.table('users').select('*').eq('email', email).execute()
    if existing.data:
        raise ValueError("User with this email already exists")

    # Insert new user
    result = client.table('users').insert({
        'name': name,
        'email': email,
        'password_hash': password_hash
    }).execute()

    if not result.data:
        raise Exception("Failed to create user")

    return result.data[0]


async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Get user by email address

    Args:
        email: User's email address

    Returns:
        User data if found, None otherwise
    """
    client = SupabaseClient.get_client()
    result = client.table('users').select('*').eq('email', email).execute()

    if result.data:
        return result.data[0]
    return None


async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user by ID

    Args:
        user_id: User's UUID

    Returns:
        User data if found, None otherwise
    """
    client = SupabaseClient.get_client()
    result = client.table('users').select('*').eq('id', user_id).execute()

    if result.data:
        return result.data[0]
    return None


async def create_itinerary(
    user_id: str,
    city: str,
    start_date: str,
    end_date: str,
    preferences: Optional[str],
    itinerary_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new itinerary in the itineraries table

    Args:
        user_id: User's UUID (foreign key)
        city: City name
        start_date: Trip start date (ISO format)
        end_date: Trip end date (ISO format)
        preferences: Optional user preferences
        itinerary_data: Complete itinerary JSON data

    Returns:
        Created itinerary data

    Raises:
        Exception: If itinerary creation fails
    """
    client = SupabaseClient.get_client()

    result = client.table('itineraries').insert({
        'user_id': user_id,
        'city': city,
        'start_date': start_date,
        'end_date': end_date,
        'preferences': preferences,
        'itinerary_data': itinerary_data
    }).execute()

    if not result.data:
        raise Exception("Failed to create itinerary")

    return result.data[0]


async def get_user_itineraries(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get all itineraries for a user

    Args:
        user_id: User's UUID
        limit: Maximum number of itineraries to return

    Returns:
        List of itinerary data
    """
    client = SupabaseClient.get_client()
    result = client.table('itineraries')\
        .select('*')\
        .eq('user_id', user_id)\
        .order('created_at', desc=True)\
        .limit(limit)\
        .execute()

    return result.data if result.data else []


async def get_itinerary_by_id(itinerary_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific itinerary by ID (must belong to the user)

    Args:
        itinerary_id: Itinerary UUID
        user_id: User's UUID (for ownership verification)

    Returns:
        Itinerary data if found and owned by user, None otherwise
    """
    client = SupabaseClient.get_client()
    result = client.table('itineraries')\
        .select('*')\
        .eq('id', itinerary_id)\
        .eq('user_id', user_id)\
        .execute()

    if result.data:
        return result.data[0]
    return None


async def delete_itinerary(itinerary_id: str, user_id: str) -> bool:
    """
    Delete an itinerary (must belong to the user)

    Args:
        itinerary_id: Itinerary UUID
        user_id: User's UUID (for ownership verification)

    Returns:
        True if deleted successfully, False if not found or unauthorized

    Raises:
        Exception if deletion fails
    """
    client = SupabaseClient.get_client()

    # First verify the itinerary belongs to the user
    itinerary = await get_itinerary_by_id(itinerary_id, user_id)
    if not itinerary:
        return False

    # Delete the itinerary (feedback will be cascade deleted due to ON DELETE CASCADE)
    result = client.table('itineraries')\
        .delete()\
        .eq('id', itinerary_id)\
        .eq('user_id', user_id)\
        .execute()

    return True


async def update_user_preferences(user_id: str, preferences: list) -> Dict[str, Any]:
    """
    Update user preferences

    Args:
        user_id: User's UUID
        preferences: List of preference strings

    Returns:
        Updated user data

    Raises:
        Exception if update fails
    """
    client = SupabaseClient.get_client()
    result = client.table('users')\
        .update({'preferences': preferences})\
        .eq('id', user_id)\
        .execute()

    if result.data:
        return result.data[0]
    raise Exception("Failed to update preferences")


# Feedback operations
async def create_or_update_feedback(
    itinerary_id: str,
    user_id: str,
    rating: int,
    feedback_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create or update feedback for an itinerary

    Args:
        itinerary_id: UUID of the itinerary
        user_id: UUID of the user
        rating: Rating from 1-5
        feedback_text: Optional text feedback

    Returns:
        Created/updated feedback data

    Raises:
        Exception: If operation fails
    """
    client = SupabaseClient.get_client()

    # Check if feedback already exists
    existing = client.table('itinerary_feedback')\
        .select('*')\
        .eq('itinerary_id', itinerary_id)\
        .eq('user_id', user_id)\
        .execute()

    feedback_data = {
        'rating': rating,
        'feedback_text': feedback_text
    }

    if existing.data:
        # Update existing feedback
        result = client.table('itinerary_feedback')\
            .update(feedback_data)\
            .eq('itinerary_id', itinerary_id)\
            .eq('user_id', user_id)\
            .execute()
    else:
        # Create new feedback
        feedback_data['itinerary_id'] = itinerary_id
        feedback_data['user_id'] = user_id
        result = client.table('itinerary_feedback')\
            .insert(feedback_data)\
            .execute()

    if result.data:
        return result.data[0]
    raise Exception("Failed to create/update feedback")


async def get_feedback_by_itinerary(itinerary_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get feedback for an itinerary by the user

    Args:
        itinerary_id: UUID of the itinerary
        user_id: UUID of the user

    Returns:
        Feedback data or None if not found
    """
    client = SupabaseClient.get_client()
    result = client.table('itinerary_feedback')\
        .select('*')\
        .eq('itinerary_id', itinerary_id)\
        .eq('user_id', user_id)\
        .execute()

    if result.data:
        return result.data[0]
    return None


async def delete_feedback(itinerary_id: str, user_id: str) -> bool:
    """
    Delete feedback for an itinerary

    Args:
        itinerary_id: UUID of the itinerary
        user_id: UUID of the user

    Returns:
        True if deleted successfully

    Raises:
        Exception: If deletion fails
    """
    client = SupabaseClient.get_client()
    result = client.table('itinerary_feedback')\
        .delete()\
        .eq('itinerary_id', itinerary_id)\
        .eq('user_id', user_id)\
        .execute()

    return True


async def update_itinerary_item(
    itinerary_id: str,
    user_id: str,
    day_number: int,
    activity_index: int,
    updated_item: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update a specific item within a day's activities

    Args:
        itinerary_id: UUID of the itinerary
        user_id: UUID of the user (for ownership verification)
        day_number: Day number (1-indexed)
        activity_index: Index of the activity within the day (0-indexed)
        updated_item: Updated activity data (time, venue, address, etc.)

    Returns:
        Updated itinerary data

    Raises:
        ValueError: If itinerary not found or day/activity doesn't exist
        Exception: If update fails
    """
    client = SupabaseClient.get_client()

    # Get the existing itinerary
    itinerary = await get_itinerary_by_id(itinerary_id, user_id)
    if not itinerary:
        raise ValueError("Itinerary not found or you don't have access to it")

    # Get the itinerary data
    itinerary_data = itinerary.get('itinerary_data', {})
    daily_plans = itinerary_data.get('daily_plans', [])

    # Find the day
    day_plan = None
    for day in daily_plans:
        if day.get('day_number') == day_number:
            day_plan = day
            break

    if not day_plan:
        raise ValueError(f"Day {day_number} not found in itinerary")

    # Validate activity index
    activities = day_plan.get('activities', [])
    if activity_index < 0 or activity_index >= len(activities):
        raise ValueError(f"Activity index {activity_index} out of range for day {day_number}")

    # Update the specific activity
    activities[activity_index].update(updated_item)

    # Update the itinerary in the database
    result = client.table('itineraries')\
        .update({'itinerary_data': itinerary_data})\
        .eq('id', itinerary_id)\
        .eq('user_id', user_id)\
        .execute()

    if result.data:
        return result.data[0]
    raise Exception("Failed to update itinerary item")


async def add_activity_to_day(
    itinerary_id: str,
    user_id: str,
    day_number: int,
    new_activity: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Add a new activity to a specific day

    Args:
        itinerary_id: UUID of the itinerary
        user_id: UUID of the user (for ownership verification)
        day_number: Day number (1-indexed)
        new_activity: New activity data (time, venue, address, etc.)

    Returns:
        Updated itinerary data

    Raises:
        ValueError: If itinerary not found or day doesn't exist
        Exception: If update fails
    """
    client = SupabaseClient.get_client()

    # Get the existing itinerary
    itinerary = await get_itinerary_by_id(itinerary_id, user_id)
    if not itinerary:
        raise ValueError("Itinerary not found or you don't have access to it")

    # Get the itinerary data
    itinerary_data = itinerary.get('itinerary_data', {})
    daily_plans = itinerary_data.get('daily_plans', [])

    # Find the day
    day_plan = None
    for day in daily_plans:
        if day.get('day_number') == day_number:
            day_plan = day
            break

    if not day_plan:
        raise ValueError(f"Day {day_number} not found in itinerary")

    # Add the new activity to the day's activities
    activities = day_plan.get('activities', [])
    activities.append(new_activity)

    # Update the itinerary in the database
    result = client.table('itineraries')\
        .update({'itinerary_data': itinerary_data})\
        .eq('id', itinerary_id)\
        .eq('user_id', user_id)\
        .execute()

    if result.data:
        return result.data[0]
    raise Exception("Failed to add activity to day")


async def delete_activity_from_day(
    itinerary_id: str,
    user_id: str,
    day_number: int,
    activity_index: int
) -> Dict[str, Any]:
    """
    Delete a specific activity from a day

    Args:
        itinerary_id: UUID of the itinerary
        user_id: UUID of the user (for ownership verification)
        day_number: Day number (1-indexed)
        activity_index: Index of the activity within the day (0-indexed)

    Returns:
        Updated itinerary data

    Raises:
        ValueError: If itinerary not found or day/activity doesn't exist
        Exception: If deletion fails
    """
    client = SupabaseClient.get_client()

    # Get the existing itinerary
    itinerary = await get_itinerary_by_id(itinerary_id, user_id)
    if not itinerary:
        raise ValueError("Itinerary not found or you don't have access to it")

    # Get the itinerary data
    itinerary_data = itinerary.get('itinerary_data', {})
    daily_plans = itinerary_data.get('daily_plans', [])

    # Find the day
    day_plan = None
    for day in daily_plans:
        if day.get('day_number') == day_number:
            day_plan = day
            break

    if not day_plan:
        raise ValueError(f"Day {day_number} not found in itinerary")

    # Validate activity index
    activities = day_plan.get('activities', [])
    if activity_index < 0 or activity_index >= len(activities):
        raise ValueError(f"Activity index {activity_index} out of range for day {day_number}")

    # Delete the activity
    activities.pop(activity_index)

    # Update the itinerary in the database
    result = client.table('itineraries')\
        .update({'itinerary_data': itinerary_data})\
        .eq('id', itinerary_id)\
        .eq('user_id', user_id)\
        .execute()

    if result.data:
        return result.data[0]
    raise Exception("Failed to delete activity from day")
