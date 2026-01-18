"""
VoyAIger Backend - AI-powered travel itinerary generator

REFACTORED ARCHITECTURE:
- Budget optional (extracted from preferences)
- Weather-aware suggestions (Open-Meteo API)
- Single itinerary with optional activities
- Strict security (prompt injection prevention)
- Pydantic validation on all LLM outputs
- In-memory rate limiting (2 requests/minute per IP)
- JWT-based authentication with Supabase (HttpOnly cookies)
"""
from fastapi import FastAPI, HTTPException, Request, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError as PydanticValidationError
from typing import Dict, Any, Optional
import logging
from uuid import UUID
from .schemas.request import GenerateItineraryRequest, SaveItineraryRequest, UpdateItineraryItemRequest, AddActivityRequest
from .schemas.response import GenerateItineraryResponse, ErrorResponse, Itinerary
from .schemas.auth import UserRegisterRequest, UserLoginRequest, AuthResponse, UserResponse
from .models.feedback import ItineraryFeedbackCreate, ItineraryFeedbackResponse
from .agents.travel_agent import TravelAgent
from .utils.content_safety import ContentSafetyError
from .utils.rate_limiter import InMemoryRateLimiter
from .utils.auth import hash_password, verify_password, create_access_token, get_token_expiry_seconds
from .utils.database import create_user, get_user_by_email, get_user_by_id, create_itinerary, get_user_itineraries, get_itinerary_by_id, delete_itinerary, update_user_preferences, create_or_update_feedback, get_feedback_by_itinerary, delete_feedback, update_itinerary_item, add_activity_to_day, delete_activity_from_day
from .middleware.security_headers import SecurityHeadersMiddleware
from .middleware.timeout import CustomTimeoutMiddleware
from .middleware.auth import require_auth
from .config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VoyAIger API",
    description="AI-powered travel itinerary generator",
    version="1.0.0"
)

# Initialize rate limiter (10 requests per hour per IP, 10 requests per minute globally)
rate_limiter = InMemoryRateLimiter(requests_per_hour=10, global_requests_per_minute=10)

# Add security middleware in correct order (bottom to top execution)
# 1. Security headers (outermost - applied last)
app.add_middleware(SecurityHeadersMiddleware)

# 2. Request timeout
app.add_middleware(CustomTimeoutMiddleware, timeout_seconds=settings.request_timeout_seconds)

# 3. CORS middleware for frontend communication
# Restricts to allowed origins from environment (default: https://voyaiger.vercel.app)
allowed_origins_list = [origin.strip() for origin in settings.allowed_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# Helper function for RAG indexing
def extract_itinerary_summary(itinerary_data: Dict[str, Any]) -> str:
    """
    Extract summary for RAG - just return empty string.

    We only embed: city + preferences + optional feedback.
    No need for itinerary summary since it doesn't match user queries.

    Args:
        itinerary_data: Complete itinerary JSON (unused)

    Returns:
        Empty string
    """
    return ""


@app.get("/")
@app.head("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "VoyAIger API is running"}

@app.get("/health")
@app.head("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post(
    "/auth/register",
    response_model=AuthResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def register(request: UserRegisterRequest, response: Response):
    """
    Register a new user and set JWT in HttpOnly cookie

    Args:
        request: User registration data (name, email, password)
        response: FastAPI response object to set cookies

    Returns:
        UserResponse with user data (token is in cookie)

    Raises:
        HTTPException: If registration fails or email already exists
    """
    try:
        # Hash password
        password_hash = hash_password(request.password)

        # Create user in database
        user_data = await create_user(
            name=request.name,
            email=request.email,
            password_hash=password_hash
        )

        # Create JWT token
        access_token = create_access_token(
            data={"sub": user_data["id"], "email": user_data["email"]}
        )

        # Set token in HttpOnly cookie
        # Always use secure=True and samesite="none" for cross-origin requests
        # This is required for mobile browsers to accept cookies
        logger.info(f"Setting authentication cookie for user {user_data['email']} (secure=True, samesite=none, path=/, max_age={get_token_expiry_seconds()})")
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=True,  # Required for SameSite=None
            samesite="none",  # Allow cross-site cookies
            path="/",  # Explicitly set path
            max_age=get_token_expiry_seconds()
        )
        logger.info(f"Cookie set in response headers for {user_data['email']}")

        # Return token in response body as well (for mobile browsers that block cookies)
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=get_token_expiry_seconds(),
            user=UserResponse(
                id=user_data["id"],
                name=user_data["name"],
                email=user_data["email"],
                created_at=user_data["created_at"],
                preferences=user_data.get("preferences", []),
                profile_image_url=user_data.get("profile_image_url")
            )
        )

    except ValueError as e:
        # User already exists
        raise HTTPException(
            status_code=400,
            detail={
                "error": "RegistrationError",
                "message": str(e),
                "details": {}
            }
        )

    except Exception as e:
        # Generic error
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to register user",
                "details": {"original_error": str(e)}
            }
        )


@app.post(
    "/auth/login",
    response_model=AuthResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def login(request: UserLoginRequest, response: Response):
    """
    Login with email and password and set JWT in HttpOnly cookie

    Args:
        request: User login credentials (email, password)
        response: FastAPI response object to set cookies

    Returns:
        UserResponse with user data (token is in cookie)

    Raises:
        HTTPException: If credentials are invalid
    """
    try:
        # Get user by email
        user_data = await get_user_by_email(request.email)

        if not user_data:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "AuthenticationError",
                    "message": "Invalid email or password",
                    "details": {}
                }
            )

        # Verify password
        if not verify_password(request.password, user_data["password_hash"]):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "AuthenticationError",
                    "message": "Invalid email or password",
                    "details": {}
                }
            )

        # Create JWT token
        access_token = create_access_token(
            data={"sub": user_data["id"], "email": user_data["email"]}
        )

        # Set token in HttpOnly cookie
        # Always use secure=True and samesite="none" for cross-origin requests
        # This is required for mobile browsers to accept cookies
        logger.info(f"Setting authentication cookie for user {user_data['email']} (secure=True, samesite=none, path=/, max_age={get_token_expiry_seconds()})")
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=True,  # Required for SameSite=None
            samesite="none",  # Allow cross-site cookies
            path="/",  # Explicitly set path
            max_age=get_token_expiry_seconds()
        )
        logger.info(f"Cookie set in response headers for {user_data['email']}")

        # Return token in response body as well (for mobile browsers that block cookies)
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=get_token_expiry_seconds(),
            user=UserResponse(
                id=user_data["id"],
                name=user_data["name"],
                email=user_data["email"],
                created_at=user_data["created_at"],
                preferences=user_data.get("preferences", []),
                profile_image_url=user_data.get("profile_image_url")
            )
        )

    except HTTPException:
        raise

    except Exception as e:
        # Generic error
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to authenticate user",
                "details": {"original_error": str(e)}
            }
        )


@app.post("/auth/logout")
async def logout(response: Response):
    """
    Logout by clearing the authentication cookie

    Args:
        response: FastAPI response object to clear cookies

    Returns:
        Success message
    """
    response.delete_cookie(key="access_token")
    return {"message": "Successfully logged out"}


@app.get("/user/profile")
async def get_profile(
    user: Dict[str, Any] = Depends(require_auth)
):
    """
    Get current user's profile data

    Args:
        user: Current authenticated user

    Returns:
        User profile data including preferences

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Fetch fresh user data from database
        user_data = await get_user_by_id(user["id"])

        if not user_data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": "User not found",
                    "details": {}
                }
            )

        return UserResponse(
            id=user_data["id"],
            name=user_data["name"],
            email=user_data["email"],
            created_at=user_data["created_at"],
            preferences=user_data.get("preferences", []),
            profile_image_url=user_data.get("profile_image_url")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to fetch profile",
                "details": {"original_error": str(e)}
            }
        )


@app.put("/user/preferences")
async def update_preferences(
    request: Request,
    user: Dict[str, Any] = Depends(require_auth)
):
    """
    Update user travel preferences

    Args:
        request: FastAPI request object
        user: Current authenticated user

    Returns:
        Success message with updated user data

    Raises:
        HTTPException: If update fails
    """
    try:
        body = await request.json()
        preferences = body.get("preferences", [])

        # Validate preferences is a list
        if not isinstance(preferences, list):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ValidationError",
                    "message": "Preferences must be a list",
                    "details": {}
                }
            )

        # Update preferences in database
        updated_user = await update_user_preferences(user["id"], preferences)

        return {
            "message": "Preferences updated successfully",
            "user": UserResponse(
                id=updated_user["id"],
                name=updated_user["name"],
                email=updated_user["email"],
                created_at=updated_user["created_at"]
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to update preferences",
                "details": {"original_error": str(e)}
            }
        )


@app.post(
    "/generate",
    response_model=GenerateItineraryResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def generate_itinerary(
    request: GenerateItineraryRequest,
    req: Request,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Generate personalized travel itinerary (PROTECTED - requires authentication)

    NEW: Requires JWT authentication
    NEW: Saves itinerary to database
    NEW: Budget is optional (extracted from preferences)
    NEW: Returns single itinerary with optional activities
    NEW: Rate limited to 10 requests per hour

    Args:
        request: Request with city, dates, and optional preferences
        req: FastAPI Request object for client IP extraction
        current_user: Authenticated user data from JWT token

    Returns:
        GenerateItineraryResponse with single personalized itinerary

    Raises:
        HTTPException: Various errors for validation, API failures, safety, or rate limiting
    """
    # Check rate limit (per-IP and global)
    client_ip = req.client.host
    if not rate_limiter.is_allowed(client_ip):
        remaining = rate_limiter.get_remaining(client_ip)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "RateLimitExceeded",
                "message": "Too many requests. You can make up to 10 requests per hour. Please try again in an hour.",
                "details": {"remaining_requests": remaining, "retry_after_seconds": 3600}
            }
        )

    agent = None

    try:
        # Calculate trip duration
        trip_days = (request.dates.end - request.dates.start).days

        # Create travel agent
        agent = TravelAgent()

        # Generate itinerary (single, not 3 options)
        itinerary = await agent.generate_itinerary(
            city_name=request.city.name,
            latitude=request.city.latitude,
            longitude=request.city.longitude,
            start_date=request.dates.start,
            end_date=request.dates.end,
            preferences=request.preferences,
            user_preferences=request.user_preferences,
            user_id=current_user["id"]  # Pass user_id for RAG personalization
        )

        # NOTE: Itinerary is NOT auto-saved to database
        # User must manually save it using the /itineraries/save endpoint

        return GenerateItineraryResponse(
            city=request.city.name,
            itinerary=itinerary,
            message=f"Generated personalized {trip_days}-day itinerary for {request.city.name}"
        )

    except PydanticValidationError as e:
        # Pydantic validation errors (from request or LLM output)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ValidationError",
                "message": "Invalid input or LLM output format",
                "details": {"errors": str(e)}
            }
        )

    except ContentSafetyError as e:
        # Content safety violations
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ContentSafetyError",
                "message": e.message,
                "details": {"safety_ratings": e.safety_ratings}
            }
        )

    except ValueError as e:
        # Business logic errors (invalid city, budget too low, etc.)
        error_msg = str(e)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "InvalidRequest",
                "message": error_msg,
                "details": {}
            }
        )

    except Exception as e:
        error_message = str(e)

        # API failures
        if any(keyword in error_message.lower() for keyword in ['api', 'timeout', 'connection', 'unavailable']):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "ServiceUnavailable",
                    "message": "External API temporarily unavailable. Please try again.",
                    "details": {"original_error": error_message}
                }
            )

        # Generic error
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "An unexpected error occurred. Please try again.",
                "details": {"original_error": error_message}
            }
        )

    finally:
        # Clean up agent
        if agent:
            await agent.close()


@app.post(
    "/itineraries/save",
    responses={
        201: {"description": "Itinerary saved successfully"},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    status_code=201
)
async def save_itinerary(
    request: SaveItineraryRequest,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Save an itinerary to user's favorites (PROTECTED - requires authentication)

    Args:
        request: Request with city, dates, preferences, and itinerary data
        current_user: Authenticated user data from JWT token

    Returns:
        Success message with itinerary ID

    Raises:
        HTTPException: If save fails
    """
    try:
        # Save itinerary to database
        saved_itinerary = await create_itinerary(
            user_id=current_user["id"],
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            preferences=request.preferences,
            itinerary_data=request.itinerary_data
        )

        return {
            "message": "Itinerary saved successfully",
            "itinerary_id": saved_itinerary["id"],
            "created_at": saved_itinerary["created_at"]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to save itinerary",
                "details": {"original_error": str(e)}
            }
        )


@app.get(
    "/itineraries",
    responses={
        200: {"description": "List of user's saved itineraries"},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def list_itineraries(
    current_user: Dict[str, Any] = Depends(require_auth),
    limit: int = 50
):
    """
    Get all saved itineraries for the authenticated user

    Args:
        current_user: Authenticated user data from JWT token
        limit: Maximum number of itineraries to return (default: 50)

    Returns:
        List of saved itineraries

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        itineraries = await get_user_itineraries(current_user["id"], limit)

        return {
            "itineraries": itineraries,
            "count": len(itineraries)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve itineraries",
                "details": {"original_error": str(e)}
            }
        )


@app.get(
    "/itineraries/{itinerary_id}",
    responses={
        200: {"description": "Itinerary details"},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_itinerary(
    itinerary_id: str,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Get a specific itinerary by ID (must belong to authenticated user)

    Args:
        itinerary_id: UUID of the itinerary
        current_user: Authenticated user data from JWT token

    Returns:
        Itinerary details

    Raises:
        HTTPException: If not found or unauthorized
    """
    try:
        itinerary = await get_itinerary_by_id(itinerary_id, current_user["id"])

        if not itinerary:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": "Itinerary not found or you don't have access to it",
                    "details": {}
                }
            )

        return itinerary

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve itinerary",
                "details": {"original_error": str(e)}
            }
        )


@app.delete(
    "/itineraries/{itinerary_id}",
    responses={
        200: {"description": "Itinerary deleted successfully"},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def delete_user_itinerary(
    itinerary_id: str,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Delete a specific itinerary by ID (must belong to authenticated user)

    Args:
        itinerary_id: UUID of the itinerary
        current_user: Authenticated user data from JWT token

    Returns:
        Success message

    Raises:
        HTTPException: If not found, unauthorized, or deletion fails
    """
    try:
        deleted = await delete_itinerary(itinerary_id, current_user["id"])

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": "Itinerary not found or you don't have access to it",
                    "details": {}
                }
            )

        return {
            "message": "Itinerary deleted successfully",
            "itinerary_id": itinerary_id
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to delete itinerary",
                "details": {"original_error": str(e)}
            }
        )


@app.patch(
    "/itineraries/{itinerary_id}/days/{day_number}/items/{activity_index}",
    responses={
        200: {"description": "Item updated successfully"},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def update_day_item(
    itinerary_id: str,
    day_number: int,
    activity_index: int,
    request: UpdateItineraryItemRequest,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Update a specific item within a day's activities (must belong to authenticated user)

    Args:
        itinerary_id: UUID of the itinerary
        day_number: Day number (1-indexed)
        activity_index: Index of the activity within the day (0-indexed)
        request: Updated item data (time, venue, address, etc.)
        current_user: Authenticated user data from JWT token

    Returns:
        Updated itinerary data

    Raises:
        HTTPException: If not found, unauthorized, or update fails
    """
    try:
        # Convert request to dict, excluding None values
        updated_data = request.model_dump(exclude_none=True)

        if not updated_data:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "BadRequest",
                    "message": "At least one field must be provided to update",
                    "details": {}
                }
            )

        updated_itinerary = await update_itinerary_item(
            itinerary_id=itinerary_id,
            user_id=current_user["id"],
            day_number=day_number,
            activity_index=activity_index,
            updated_item=updated_data
        )

        return {
            "message": "Item updated successfully",
            "itinerary": updated_itinerary
        }

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NotFound",
                "message": str(e),
                "details": {}
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to update item",
                "details": {"original_error": str(e)}
            }
        )


@app.post(
    "/itineraries/{itinerary_id}/days/{day_number}/activities",
    responses={
        201: {"description": "Activity added successfully"},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    status_code=201
)
async def add_activity(
    itinerary_id: str,
    day_number: int,
    request: AddActivityRequest,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Add a new activity to a specific day (must belong to authenticated user)

    Args:
        itinerary_id: UUID of the itinerary
        day_number: Day number (1-indexed)
        request: New activity data (time, venue, address, etc.)
        current_user: Authenticated user data from JWT token

    Returns:
        Updated itinerary data

    Raises:
        HTTPException: If not found, unauthorized, or addition fails
    """
    try:
        # Convert request to dict
        new_activity = request.model_dump()

        updated_itinerary = await add_activity_to_day(
            itinerary_id=itinerary_id,
            user_id=current_user["id"],
            day_number=day_number,
            new_activity=new_activity
        )

        return {
            "message": "Activity added successfully",
            "itinerary": updated_itinerary
        }

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NotFound",
                "message": str(e),
                "details": {}
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to add activity",
                "details": {"original_error": str(e)}
            }
        )


@app.delete(
    "/itineraries/{itinerary_id}/days/{day_number}/activities/{activity_index}",
    responses={
        200: {"description": "Activity deleted successfully"},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def delete_activity(
    itinerary_id: str,
    day_number: int,
    activity_index: int,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Delete a specific activity from a day (must belong to authenticated user)

    Args:
        itinerary_id: UUID of the itinerary
        day_number: Day number (1-indexed)
        activity_index: Index of the activity within the day (0-indexed)
        current_user: Authenticated user data from JWT token

    Returns:
        Updated itinerary data

    Raises:
        HTTPException: If not found, unauthorized, or deletion fails
    """
    try:
        updated_itinerary = await delete_activity_from_day(
            itinerary_id=itinerary_id,
            user_id=current_user["id"],
            day_number=day_number,
            activity_index=activity_index
        )

        return {
            "message": "Activity deleted successfully",
            "itinerary": updated_itinerary
        }

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NotFound",
                "message": str(e),
                "details": {}
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to delete activity",
                "details": {"original_error": str(e)}
            }
        )


# ============================================================================
# FEEDBACK ENDPOINTS
# ============================================================================

@app.post(
    "/itineraries/{itinerary_id}/feedback",
    response_model=ItineraryFeedbackResponse,
    responses={
        200: {"description": "Feedback created/updated successfully"},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def submit_feedback(
    itinerary_id: str,
    feedback: ItineraryFeedbackCreate,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Submit or update feedback for an itinerary

    Args:
        itinerary_id: UUID of the itinerary
        feedback: Feedback data (rating and optional text)
        current_user: Authenticated user data from JWT token

    Returns:
        Created/updated feedback

    Raises:
        HTTPException: If itinerary not found or operation fails
    """
    try:
        # Verify itinerary exists and belongs to user
        itinerary = await get_itinerary_by_id(itinerary_id, current_user["id"])
        if not itinerary:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": "Itinerary not found or you don't have access to it",
                    "details": {}
                }
            )

        # Create or update feedback
        result = await create_or_update_feedback(
            itinerary_id=itinerary_id,
            user_id=current_user["id"],
            rating=feedback.rating,
            feedback_text=feedback.feedback_text
        )

        # RAG Integration: Index high-rated itineraries (≥4 stars)
        try:
            from .rag import get_retriever
            retriever = get_retriever()

            if feedback.rating >= 4:
                # Content safety check before indexing to RAG
                from .utils.content_safety import check_content_safety, ContentSafetyError

                # Get user preferences from users table
                user_preferences_array = current_user.get("preferences", [])

                # Format as natural language sentence for better semantic matching
                if user_preferences_array:
                    if len(user_preferences_array) == 1:
                        user_preferences_text = f"I enjoy {user_preferences_array[0]}"
                    elif len(user_preferences_array) == 2:
                        user_preferences_text = f"I enjoy {user_preferences_array[0]} and {user_preferences_array[1]}"
                    else:
                        user_preferences_text = f"I enjoy {', '.join(user_preferences_array[:-1])}, and {user_preferences_array[-1]}"
                else:
                    user_preferences_text = ""

                # Combine all text fields for embedding
                itinerary_prefs = itinerary.get('preferences', '').strip()
                combined_preferences = f"{itinerary_prefs}. {user_preferences_text}".strip() if user_preferences_text else itinerary_prefs

                text_to_check = f"{combined_preferences} {feedback.feedback_text or ''}"

                try:
                    check_content_safety(text_to_check)

                    # Extract summary from itinerary data
                    itinerary_summary = extract_itinerary_summary(itinerary["itinerary_data"])

                    # Index the itinerary for RAG (only if content is safe)
                    await retriever.index_itinerary_feedback(
                        user_id=UUID(current_user["id"]),
                        itinerary_id=UUID(itinerary_id),
                        feedback_id=UUID(result["id"]),
                        city=itinerary["city"],
                        start_date=str(itinerary["start_date"]),
                        end_date=str(itinerary["end_date"]),
                        preferences=combined_preferences,
                        itinerary_summary=itinerary_summary,
                        rating=feedback.rating,
                        feedback_text=feedback.feedback_text
                    )
                    logger.info(f"Indexed itinerary {itinerary_id} for RAG (rating={feedback.rating})")

                except ContentSafetyError as safety_error:
                    logger.warning(
                        f"Skipped RAG indexing for itinerary {itinerary_id} due to unsafe content: {safety_error}"
                    )
                    # Don't index, but don't fail the request either

            else:
                # If rating dropped below 4, remove from RAG index
                await retriever.remove_itinerary_feedback(UUID(result["id"]))
                logger.info(f"Removed itinerary {itinerary_id} from RAG index (rating={feedback.rating})")

        except Exception as rag_error:
            # Don't fail the request if RAG indexing fails - just log it
            logger.error(f"RAG indexing failed for itinerary {itinerary_id}: {rag_error}")

        return ItineraryFeedbackResponse(**result)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to submit feedback",
                "details": {"original_error": str(e)}
            }
        )


@app.get(
    "/itineraries/{itinerary_id}/feedback",
    response_model=Optional[ItineraryFeedbackResponse],
    responses={
        200: {"description": "Feedback retrieved successfully (null if no feedback given yet)"},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_feedback(
    itinerary_id: str,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Get feedback for an itinerary

    Args:
        itinerary_id: UUID of the itinerary
        current_user: Authenticated user data from JWT token

    Returns:
        Feedback data if exists, None if user hasn't provided feedback yet

    Raises:
        HTTPException: If there's an error retrieving feedback
    """
    try:
        feedback = await get_feedback_by_itinerary(itinerary_id, current_user["id"])

        if not feedback:
            # Return null/None to indicate no feedback has been given yet
            # This is not an error condition - user simply hasn't provided feedback
            return None

        return ItineraryFeedbackResponse(**feedback)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to retrieve feedback",
                "details": {"original_error": str(e)}
            }
        )


@app.delete(
    "/itineraries/{itinerary_id}/feedback",
    responses={
        200: {"description": "Feedback deleted successfully"},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def remove_feedback(
    itinerary_id: str,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Delete feedback for an itinerary

    Args:
        itinerary_id: UUID of the itinerary
        current_user: Authenticated user data from JWT token

    Returns:
        Success message

    Raises:
        HTTPException: If operation fails
    """
    try:
        # Check if feedback exists
        feedback = await get_feedback_by_itinerary(itinerary_id, current_user["id"])
        if not feedback:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": "No feedback found for this itinerary",
                    "details": {}
                }
            )

        # RAG Integration: Remove from index before deleting feedback
        try:
            from .rag import get_retriever
            retriever = get_retriever()
            await retriever.remove_itinerary_feedback(UUID(feedback["id"]))
            logger.info(f"Removed itinerary {itinerary_id} from RAG index on feedback deletion")
        except Exception as rag_error:
            # Don't fail the request if RAG removal fails - just log it
            logger.error(f"RAG removal failed for feedback {feedback['id']}: {rag_error}")

        await delete_feedback(itinerary_id, current_user["id"])

        return {
            "message": "Feedback deleted successfully"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to delete feedback",
                "details": {"original_error": str(e)}
            }
        )
