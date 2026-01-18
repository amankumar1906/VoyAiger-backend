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


async def get_itinerary_by_id_with_access(itinerary_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific itinerary by ID if user has access (owner or accepted invitee)

    Args:
        itinerary_id: Itinerary UUID
        user_id: User's UUID (for access verification)

    Returns:
        Itinerary data if found and user has access, None otherwise
    """
    # Check if user has access
    if not await has_itinerary_access(itinerary_id, user_id):
        return None

    # Get the itinerary
    client = SupabaseClient.get_client()
    result = client.table('itineraries')\
        .select('*')\
        .eq('id', itinerary_id)\
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


async def update_profile_image(user_id: str, image_url: str) -> Dict[str, Any]:
    """
    Update user's profile image URL

    Args:
        user_id: UUID of the user
        image_url: URL of the uploaded image

    Returns:
        Updated user data

    Raises:
        Exception if update fails
    """
    client = SupabaseClient.get_client()
    result = client.table('users')\
        .update({'profile_image_url': image_url})\
        .eq('id', user_id)\
        .execute()

    if result.data:
        return result.data[0]
    raise Exception("Failed to update profile image")


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
    updated_item: Dict[str, Any],
    expected_version: Optional[int] = None
) -> Dict[str, Any]:
    """
    Update a specific item within a day's activities

    Args:
        itinerary_id: UUID of the itinerary
        user_id: UUID of the user (for access verification - owner or invitee)
        day_number: Day number (1-indexed)
        activity_index: Index of the activity within the day (0-indexed)
        updated_item: Updated activity data (time, venue, address, etc.)
        expected_version: Expected version number for optimistic locking (optional)

    Returns:
        Updated itinerary data

    Raises:
        ValueError: If itinerary not found, day/activity doesn't exist, or version mismatch
        Exception: If update fails
    """
    client = SupabaseClient.get_client()

    # Get the existing itinerary (with access check)
    itinerary = await get_itinerary_by_id_with_access(itinerary_id, user_id)
    if not itinerary:
        raise ValueError("Itinerary not found or you don't have access to it")

    # Check version for concurrent edit detection
    if expected_version is not None and itinerary.get('version') != expected_version:
        raise ValueError(f"Conflict detected: Itinerary was modified by another user. Expected version {expected_version}, but current version is {itinerary.get('version')}")

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

    # Update the itinerary in the database with version check
    update_data = {
        'itinerary_data': itinerary_data,
        'last_modified_by': user_id
    }

    # If version check enabled, add it to the WHERE clause
    query = client.table('itineraries').update(update_data).eq('id', itinerary_id)

    if expected_version is not None:
        query = query.eq('version', expected_version)

    result = query.execute()

    if not result.data:
        # Could be version mismatch or other error
        raise Exception("Failed to update itinerary item - possible concurrent modification")

    return result.data[0]


async def add_activity_to_day(
    itinerary_id: str,
    user_id: str,
    day_number: int,
    new_activity: Dict[str, Any],
    expected_version: Optional[int] = None
) -> Dict[str, Any]:
    """
    Add a new activity to a specific day

    Args:
        itinerary_id: UUID of the itinerary
        user_id: UUID of the user (for access verification - owner or invitee)
        day_number: Day number (1-indexed)
        new_activity: New activity data (time, venue, address, etc.)
        expected_version: Expected version number for optimistic locking (optional)

    Returns:
        Updated itinerary data

    Raises:
        ValueError: If itinerary not found, day doesn't exist, or version mismatch
        Exception: If update fails
    """
    client = SupabaseClient.get_client()

    # Get the existing itinerary (with access check)
    itinerary = await get_itinerary_by_id_with_access(itinerary_id, user_id)
    if not itinerary:
        raise ValueError("Itinerary not found or you don't have access to it")

    # Check version for concurrent edit detection
    if expected_version is not None and itinerary.get('version') != expected_version:
        raise ValueError(f"Conflict detected: Itinerary was modified by another user. Expected version {expected_version}, but current version is {itinerary.get('version')}")

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

    # Update the itinerary in the database with version check
    update_data = {
        'itinerary_data': itinerary_data,
        'last_modified_by': user_id
    }

    query = client.table('itineraries').update(update_data).eq('id', itinerary_id)

    if expected_version is not None:
        query = query.eq('version', expected_version)

    result = query.execute()

    if not result.data:
        raise Exception("Failed to add activity to day - possible concurrent modification")

    return result.data[0]


async def delete_activity_from_day(
    itinerary_id: str,
    user_id: str,
    day_number: int,
    activity_index: int,
    expected_version: Optional[int] = None
) -> Dict[str, Any]:
    """
    Delete a specific activity from a day

    Args:
        itinerary_id: UUID of the itinerary
        user_id: UUID of the user (for access verification - owner or invitee)
        day_number: Day number (1-indexed)
        activity_index: Index of the activity within the day (0-indexed)
        expected_version: Expected version number for optimistic locking (optional)

    Returns:
        Updated itinerary data

    Raises:
        ValueError: If itinerary not found, day/activity doesn't exist, or version mismatch
        Exception: If deletion fails
    """
    client = SupabaseClient.get_client()

    # Get the existing itinerary (with access check)
    itinerary = await get_itinerary_by_id_with_access(itinerary_id, user_id)
    if not itinerary:
        raise ValueError("Itinerary not found or you don't have access to it")

    # Check version for concurrent edit detection
    if expected_version is not None and itinerary.get('version') != expected_version:
        raise ValueError(f"Conflict detected: Itinerary was modified by another user. Expected version {expected_version}, but current version is {itinerary.get('version')}")

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

    # Update the itinerary in the database with version check
    update_data = {
        'itinerary_data': itinerary_data,
        'last_modified_by': user_id
    }

    query = client.table('itineraries').update(update_data).eq('id', itinerary_id)

    if expected_version is not None:
        query = query.eq('version', expected_version)

    result = query.execute()

    if not result.data:
        raise Exception("Failed to delete activity from day - possible concurrent modification")

    return result.data[0]


# Invite operations
async def send_invite(
    itinerary_id: str,
    invited_by_user_id: str,
    invitee_email: str
) -> Dict[str, Any]:
    """
    Send an invite to collaborate on an itinerary

    Args:
        itinerary_id: UUID of the itinerary
        invited_by_user_id: UUID of the user sending the invite
        invitee_email: Email address of the person being invited

    Returns:
        Created invite data

    Raises:
        ValueError: If invite already exists
        Exception: If invite creation fails
    """
    client = SupabaseClient.get_client()

    # Check if invite already exists
    existing = client.table('itinerary_invites')\
        .select('*')\
        .eq('itinerary_id', itinerary_id)\
        .eq('invitee_email', invitee_email)\
        .execute()

    if existing.data:
        raise ValueError("Invite already exists for this email")

    # Look up invitee user_id if they have an account
    invitee_user = await get_user_by_email(invitee_email)
    invitee_user_id = invitee_user['id'] if invitee_user else None

    # Create the invite
    result = client.table('itinerary_invites').insert({
        'itinerary_id': itinerary_id,
        'invited_by_user_id': invited_by_user_id,
        'invitee_email': invitee_email,
        'invitee_user_id': invitee_user_id,
        'status': 'pending'
    }).execute()

    if not result.data:
        raise Exception("Failed to create invite")

    return result.data[0]


async def get_invite(invite_id: str, user_email: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific invite by ID (user must be the invitee)

    Args:
        invite_id: UUID of the invite
        user_email: Email of the user (for verification)

    Returns:
        Invite data if found and belongs to user, None otherwise
    """
    client = SupabaseClient.get_client()
    result = client.table('itinerary_invites')\
        .select('*')\
        .eq('id', invite_id)\
        .eq('invitee_email', user_email)\
        .execute()

    if result.data:
        return result.data[0]
    return None


async def get_itinerary_invites(itinerary_id: str, owner_user_id: str) -> List[Dict[str, Any]]:
    """
    Get all invites for an itinerary (owner only)

    Args:
        itinerary_id: UUID of the itinerary
        owner_user_id: UUID of the itinerary owner (for verification)

    Returns:
        List of invite data

    Raises:
        ValueError: If user is not the owner
    """
    client = SupabaseClient.get_client()

    # Verify ownership
    itinerary = await get_itinerary_by_id(itinerary_id, owner_user_id)
    if not itinerary:
        raise ValueError("Itinerary not found or you are not the owner")

    # Get all invites
    result = client.table('itinerary_invites')\
        .select('*')\
        .eq('itinerary_id', itinerary_id)\
        .order('created_at', desc=True)\
        .execute()

    return result.data if result.data else []


async def get_user_pending_invites(user_email: str) -> List[Dict[str, Any]]:
    """
    Get all pending invites for a user by email

    Args:
        user_email: Email address of the user

    Returns:
        List of pending invite data with itinerary details
    """
    client = SupabaseClient.get_client()

    # Get pending invites with itinerary details
    result = client.table('itinerary_invites')\
        .select('*, itineraries(id, city, start_date, end_date, user_id, users(name, email))')\
        .eq('invitee_email', user_email)\
        .eq('status', 'pending')\
        .order('created_at', desc=True)\
        .execute()

    return result.data if result.data else []


async def respond_to_invite(
    invite_id: str,
    user_id: str,
    user_email: str,
    status: str
) -> Dict[str, Any]:
    """
    Accept or reject an invite

    Args:
        invite_id: UUID of the invite
        user_id: UUID of the user responding
        user_email: Email of the user (for verification)
        status: 'accepted' or 'rejected'

    Returns:
        Updated invite data

    Raises:
        ValueError: If invite not found, already responded, or invalid status
        Exception: If update fails
    """
    if status not in ['accepted', 'rejected']:
        raise ValueError("Status must be 'accepted' or 'rejected'")

    client = SupabaseClient.get_client()

    # Get the invite
    invite = await get_invite(invite_id, user_email)
    if not invite:
        raise ValueError("Invite not found or you are not the invitee")

    if invite['status'] != 'pending':
        raise ValueError(f"Invite has already been {invite['status']}")

    # Update the invite status and link to user account
    result = client.table('itinerary_invites')\
        .update({
            'status': status,
            'invitee_user_id': user_id
        })\
        .eq('id', invite_id)\
        .eq('invitee_email', user_email)\
        .execute()

    if result.data:
        return result.data[0]
    raise Exception("Failed to respond to invite")


async def has_itinerary_access(itinerary_id: str, user_id: str) -> bool:
    """
    Check if a user has access to an itinerary (owner or accepted invitee)

    Args:
        itinerary_id: UUID of the itinerary
        user_id: UUID of the user

    Returns:
        True if user has access, False otherwise
    """
    client = SupabaseClient.get_client()

    # Check if user is the owner
    owner_check = client.table('itineraries')\
        .select('id')\
        .eq('id', itinerary_id)\
        .eq('user_id', user_id)\
        .execute()

    if owner_check.data:
        return True

    # Check if user has accepted invite
    invite_check = client.table('itinerary_invites')\
        .select('id')\
        .eq('itinerary_id', itinerary_id)\
        .eq('invitee_user_id', user_id)\
        .eq('status', 'accepted')\
        .execute()

    return bool(invite_check.data)


async def is_itinerary_owner(itinerary_id: str, user_id: str) -> bool:
    """
    Check if a user is the owner of an itinerary

    Args:
        itinerary_id: UUID of the itinerary
        user_id: UUID of the user

    Returns:
        True if user is the owner, False otherwise
    """
    client = SupabaseClient.get_client()

    result = client.table('itineraries')\
        .select('id')\
        .eq('id', itinerary_id)\
        .eq('user_id', user_id)\
        .execute()

    return bool(result.data)


async def get_all_accessible_itineraries(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get all itineraries accessible to a user (owned + accepted invites)

    Args:
        user_id: User's UUID
        limit: Maximum number of itineraries to return

    Returns:
        List of itinerary data with access information
    """
    client = SupabaseClient.get_client()

    # Get owned itineraries
    owned = client.table('itineraries')\
        .select('*, is_owner:user_id, role')\
        .eq('user_id', user_id)\
        .order('created_at', desc=True)\
        .limit(limit)\
        .execute()

    owned_itineraries = owned.data if owned.data else []

    # Add metadata for owned itineraries
    for itin in owned_itineraries:
        itin['is_owner'] = True
        itin['role'] = 'owner'

    # Get accepted invite itineraries
    invites = client.table('itinerary_invites')\
        .select('*, itineraries(*)')\
        .eq('invitee_user_id', user_id)\
        .eq('status', 'accepted')\
        .order('created_at', desc=True)\
        .execute()

    invited_itineraries = []
    if invites.data:
        for invite in invites.data:
            if invite.get('itineraries'):
                itin = invite['itineraries']
                itin['is_owner'] = False
                itin['role'] = 'collaborator'
                invited_itineraries.append(itin)

    # Combine and sort by created_at
    all_itineraries = owned_itineraries + invited_itineraries
    all_itineraries.sort(key=lambda x: x.get('created_at', ''), reverse=True)

    return all_itineraries[:limit]
