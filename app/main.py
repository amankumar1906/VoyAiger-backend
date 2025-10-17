"""
VoyAIger Backend - AI-powered travel itinerary generator
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
