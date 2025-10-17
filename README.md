# VoyAIger Backend

AI-powered travel itinerary generator that creates personalized trip plans based on your budget, dates, and destination.

## What It Does

Give us a city, budget, and travel dates - we'll generate complete itineraries with hotels, attractions, and restaurants that fit your budget.

## How It Works

The system uses multiple AI agents working together:

- **Orchestrator Agent**: Splits your budget intelligently and coordinates the planning
- **Hotel Agent**: Finds accommodation options using Xotelo API
- **Attractions Agent**: Discovers places to visit using Google Places API
- **Restaurant Agent**: Recommends dining spots using Google Places API

Each agent provides 3 options, and the orchestrator combines them into 3 different complete itineraries for you to choose from.

## Key Features

- Budget-aware planning (won't exceed your limits)
- Smart validation (catches unrealistic budgets or date ranges)
- Content safety checks at every step
- Structured, validated responses

## Tech Stack

- **FastAPI**: REST API framework
- **LangChain**: Agent orchestration
- **Gemini 2.0 Flash Lite**: AI model powering the agents
- **Pydantic**: Data validation
- **Google Places API**: Attractions and restaurants data
- **Xotelo API**: Hotel information

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (create `.env` file):
```
GEMINI_API_KEY=your_gemini_key
GOOGLE_PLACES_API_KEY=your_google_places_key
XOTELO_API_KEY=your_xotelo_key
```

4. Run the server:
```bash
uvicorn main:app --reload
```

## API Usage

**Endpoint**: `POST /generate`

**Request**:
```json
{
  "city": "Paris",
  "budget": 2000,
  "dates": {
    "start": "2025-06-01",
    "end": "2025-06-07"
  }
}
```

**Response**: 3 complete itinerary options with hotels, attractions, and restaurants within your budget.

## Project Structure

```
backend/
├── main.py                 # FastAPI application
├── agents/                 # AI agents
│   ├── orchestrator.py
│   ├── hotel_agent.py
│   ├── attractions_agent.py
│   └── restaurant_agent.py
├── tools/                  # External API wrappers
├── schemas/                # Pydantic models
├── validators/             # Input validation
└── utils/                  # Helper functions
```

## Development

See `TODO.md` for the complete development roadmap.
