"""
VoyAIger Backend - AI-powered travel itinerary generator

REFACTORED ARCHITECTURE:
- Budget optional (extracted from preferences)
- Weather-aware suggestions (Open-Meteo API)
- Single itinerary with optional activities
- Strict security (prompt injection prevention)
- Pydantic validation on all LLM outputs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError as PydanticValidationError
from .schemas.request import GenerateItineraryRequest
from .schemas.response import GenerateItineraryResponse, ErrorResponse
from .agents.travel_agent import TravelAgent
from .utils.content_safety import ContentSafetyError

app = FastAPI(
    title="VoyAIger API",
    description="AI-powered travel itinerary generator",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "VoyAIger API is running"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post(
    "/generate",
    response_model=GenerateItineraryResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def generate_itinerary(request: GenerateItineraryRequest):
    """
    Generate personalized travel itinerary

    NEW: Budget is optional (extracted from preferences)
    NEW: Returns single itinerary with optional activities

    Args:
        request: Request with city, dates, and optional preferences

    Returns:
        GenerateItineraryResponse with single personalized itinerary

    Raises:
        HTTPException: Various errors for validation, API failures, or safety
    """
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
            preferences=request.preferences
        )

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
