"""
VoyAIger Backend - AI-powered travel itinerary generator
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas.request import GenerateItineraryRequest
from .schemas.response import GenerateItineraryResponse, ErrorResponse
from .validators.input_validator import validate_request, ValidationError
from .agents.orchestrator import OrchestratorAgent
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
    Generate travel itineraries based on city, budget, and dates

    Args:
        request: Itinerary generation request with city, budget, and dates

    Returns:
        GenerateItineraryResponse with up to 3 itinerary options

    Raises:
        HTTPException: Various errors for validation, budget, or API failures
    """
    orchestrator = None

    try:
        # Validate request
        validation_meta = validate_request(request)

        # Create orchestrator
        orchestrator = OrchestratorAgent()

        # Generate itineraries
        itineraries = await orchestrator.generate_itineraries(
            city=request.city,
            budget=request.budget,
            start_date=request.dates.start,
            end_date=request.dates.end
        )

        return GenerateItineraryResponse(
            city=request.city,
            options=itineraries,
            message=f"Generated {len(itineraries)} itinerary options for your {validation_meta['trip_duration_days']}-day trip"
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ValidationError",
                "message": e.message,
                "details": e.details
            }
        )

    except ContentSafetyError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ContentSafetyError",
                "message": e.message,
                "details": {"safety_ratings": e.safety_ratings}
            }
        )

    except Exception as e:
        error_message = str(e)

        # Check if it's a budget error
        if "budget" in error_message.lower() or "increase" in error_message.lower():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "BudgetError",
                    "message": error_message,
                    "details": {}
                }
            )

        # API failures
        if "api" in error_message.lower() or "search failed" in error_message.lower():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "ServiceUnavailable",
                    "message": error_message,
                    "details": {}
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
        # Clean up orchestrator
        if orchestrator:
            await orchestrator.close()
