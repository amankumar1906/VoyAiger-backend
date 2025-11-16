"""
ReACT Travel Agent - LLM does ALL the reasoning

NO manual preprocessing - LLM extracts budget, decides tags, allocates budget
"""
from typing import Optional
from datetime import date, datetime
import logging
from pathlib import Path
import json
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from ..config import settings
from ..tools.xotelo_api import XoteloAPI
from ..tools.google_places import GooglePlacesAPI
from ..tools.weather_api import WeatherAPI
from ..schemas.response import Itinerary, Hotel, DayPlan
from ..schemas.agent import ItineraryPlanLLM
from ..utils.content_safety import check_content_safety, configure_safety_settings, safe_llm_call

# Configure logging
log_file = Path(__file__).parent.parent.parent / "logs.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def price_level_to_display(price_level: Optional[int]) -> str:
    """Convert price_level (0-4) to display string"""
    if price_level is None:
        return "Unknown"
    mapping = {0: "Free", 1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}
    return mapping.get(price_level, "Unknown")


class TravelAgent:
    """ReACT agent where LLM makes ALL decisions"""

    def __init__(self):
        logger.info("="*80)
        logger.info("INITIALIZING TravelAgent")
        logger.info("="*80)
        self.xotelo_api = XoteloAPI()
        self.places_api = GooglePlacesAPI()
        self.weather_api = WeatherAPI()

        # Verify API key is set
        if not settings.gemini_api_key or settings.gemini_api_key == "":
            logger.error("❌ GEMINI_API_KEY is not set in environment!")
            raise ValueError("GEMINI_API_KEY environment variable is required")

        logger.info(f"🔑 Using Gemini API key: {settings.gemini_api_key[:8]}...{settings.gemini_api_key[-4:]}")
        logger.info(f"🤖 Model: {settings.model_name}")

        try:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                temperature=0.2,
                google_api_key=settings.gemini_api_key,
                safety_settings=configure_safety_settings()
            )
            logger.info(f"✓ APIs initialized: Xotelo, Google Places, Weather")
            logger.info(f"✓ LLM configured: {settings.model_name} (temp=0.2)")
            logger.info(f"✓ Safety settings: BLOCK_ONLY_HIGH (relaxed to avoid false positives)")

            # Test API key with a simple call (non-blocking, done in background)
            logger.info("🧪 Testing Gemini API connection...")
            try:
                import asyncio
                test_response = asyncio.run(self._test_api_key())
                if test_response:
                    logger.info(f"✅ API test successful: {test_response[:50]}...")
                else:
                    logger.warning("⚠️ API test returned None - this may indicate issues with structured output")
            except Exception as test_error:
                logger.warning(f"⚠️ API test failed (non-fatal): {str(test_error)}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {type(e).__name__}: {str(e)}")
            logger.error(f"   Check if model '{settings.model_name}' is valid")
            logger.error(f"   Valid models: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp, gemini-2.5-flash-lite")
            raise

    async def _test_api_key(self):
        """Test API key with a simple request"""
        try:
            response = await self.llm.ainvoke("Say 'OK' if you can read this.")
            if response and hasattr(response, 'content'):
                return response.content
            return None
        except Exception as e:
            logger.error(f"API key test failed: {str(e)}")
            raise

    def _resolve_schema_refs(self, schema, defs):
        """Recursively resolve $ref in schema by inlining definitions"""
        if isinstance(schema, dict):
            if "$ref" in schema:
                # Extract the definition name from $ref like "#/$defs/DaySchedule"
                ref_path = schema["$ref"].split("/")[-1]
                if ref_path in defs:
                    # Replace $ref with the actual definition
                    return self._resolve_schema_refs(defs[ref_path], defs)
            else:
                # Recursively process all nested dictionaries
                return {k: self._resolve_schema_refs(v, defs) for k, v in schema.items()}
        elif isinstance(schema, list):
            # Recursively process list items
            return [self._resolve_schema_refs(item, defs) for item in schema]
        return schema

    def _remove_unsupported_fields(self, schema):
        """
        Keep only fields that Gemini actually supports.
        Based on testing, only these core fields work:
        - type, properties, required, items, description, enum, format
        """
        if isinstance(schema, dict):
            # Handle anyOf (used for Optional fields) - flatten to first type that's not null
            if "anyOf" in schema:
                any_of = schema["anyOf"]
                # Find first non-null type
                for option in any_of:
                    if option.get("type") != "null":
                        # Replace anyOf with the actual type
                        schema = {**schema, **option}
                        break
                schema.pop("anyOf", None)

            # Filter to supported fields at THIS level only - minimal set that works
            supported = {
                "type", "properties", "required", "items", "description", "enum", "format"
            }
            filtered = {k: v for k, v in schema.items() if k in supported}

            # NOW recursively process nested structures
            result = {}
            for k, v in filtered.items():
                if k == "properties" and isinstance(v, dict):
                    # Recursively clean each property definition
                    result[k] = {prop_name: self._remove_unsupported_fields(prop_def)
                                for prop_name, prop_def in v.items()}
                elif k == "items" and isinstance(v, dict):
                    # Recursively clean array item schema
                    result[k] = self._remove_unsupported_fields(v)
                else:
                    # Keep as-is for other fields
                    result[k] = v

            # Fix required field - only keep properties that still exist
            if "required" in result and "properties" in result:
                existing_props = set(result["properties"].keys())
                result["required"] = [prop for prop in result["required"] if prop in existing_props]

            return result
        elif isinstance(schema, list):
            return [self._remove_unsupported_fields(item) for item in schema]
        return schema

    async def _call_gemini_with_schema(self, prompt: str, schema_class):
        """
        Call Gemini API directly with JSON schema for structured output.
        This bypasses LangChain's with_structured_output which has compatibility issues.
        Uses GenerativeModel with GenerationConfig.
        """
        import asyncio

        logger.info("🔄 Calling Gemini API directly with response schema...")

        try:
            # Configure Gemini with API key
            genai.configure(api_key=settings.gemini_api_key)

            # Get and clean schema - remove fields unsupported by Gemini
            raw_schema = schema_class.model_json_schema()
            logger.info(f"📋 Raw schema size: {len(json.dumps(raw_schema))} chars")

            # Extract $defs and resolve $ref references
            defs = raw_schema.get("$defs", {})
            resolved_schema = self._resolve_schema_refs(raw_schema, defs)

            # Remove unsupported fields
            clean_schema = self._remove_unsupported_fields(resolved_schema)

            # Also remove $defs after inlining
            clean_schema.pop("$defs", None)

            logger.info(f"📋 Cleaned schema size: {len(json.dumps(clean_schema))} chars")

            # Create model with generation config for structured output
            model = genai.GenerativeModel(
                model_name=settings.model_name,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=clean_schema
                )
            )

            # Make async call
            logger.info(f"📤 Sending request to {settings.model_name}...")
            response = await asyncio.to_thread(
                model.generate_content,
                contents=prompt
            )

            logger.info(f"📥 Received response from Gemini")

            # Parse and validate using Pydantic
            if response and response.text:
                logger.info(f"✅ Response text received ({len(response.text)} chars)")

                # Log raw JSON for debugging
                logger.debug(f"📄 Raw JSON response: {response.text[:1000]}...")

                # Parse JSON first to fix common issues
                raw_data = json.loads(response.text)

                # Fix hotel_index: Gemini uses -1 for "no hotel", we need None
                if "hotel_index" in raw_data and raw_data["hotel_index"] == -1:
                    logger.info("🔧 Converting hotel_index=-1 to None")
                    raw_data["hotel_index"] = None

                # Validate with Pydantic
                result = schema_class.model_validate(raw_data)
                logger.info("✅ Successfully validated response against schema")
                return result
            else:
                logger.error("❌ Empty response from Gemini")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON response: {str(e)}")
            logger.error(f"   Response text: {response.text[:500] if response and response.text else 'None'}")
            return None
        except Exception as e:
            logger.error(f"❌ Gemini API call failed: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            # Log raw response for debugging
            if 'response' in locals() and response and hasattr(response, 'text'):
                logger.error(f"   Raw response: {response.text[:1000]}")
            return None

    async def close(self):
        """Close API clients"""
        logger.info("Closing API clients...")
        await self.xotelo_api.close()
        await self.places_api.close()
        await self.weather_api.close()
        logger.info("✓ All API clients closed")

    def _create_tools(self, city_name: str, lat: float, lng: float, check_in: date, check_out: date):
        """Create tools - LLM decides when/how to use them"""

        xotelo_ref = self.xotelo_api
        places_ref = self.places_api
        weather_ref = self.weather_api
        city = city_name  # For display in tool descriptions

        @tool
        async def get_weather_forecast(query: str) -> str:
            """Get weather forecast. Helps determine indoor vs outdoor activities. Only available for trips within 10 days."""
            logger.info(f"🔧 TOOL CALLED: get_weather_forecast(query='{query}')")

            # Check if trip is within 10 days from today
            from datetime import datetime
            today = datetime.now().date()
            days_until_trip = (check_in - today).days

            if days_until_trip > 10:
                logger.info(f"  → Trip is {days_until_trip} days away (beyond 10-day forecast limit)")
                return f"Weather forecast not available for trips more than 10 days in the future (trip is {days_until_trip} days away)"

            try:
                logger.info(f"  → Trip is {days_until_trip} days away (within forecast range)")
                forecasts = await weather_ref.get_forecast(city, check_in, check_out)
                result = weather_ref.format_forecast_for_llm(forecasts)
                logger.info(f"  ✓ Weather forecast retrieved successfully")
                return result
            except Exception as e:
                logger.error(f"  ✗ Weather API error: {str(e)}")
                return f"Weather unavailable: {str(e)}"

        @tool
        async def search_hotels(budget_amount: str) -> str:
            """
            Search hotels within budget.
            Input: Budget amount for hotels (e.g., "900")
            ONLY call if user mentioned budget. YOU decide allocation.
            """
            logger.info(f"🔧 TOOL CALLED: search_hotels(budget_amount='{budget_amount}')")
            try:
                budget_float = float(budget_amount)
                logger.info(f"  → Searching hotels with budget ${budget_float}")
                hotels = await xotelo_ref.search_hotels(
                    city_name=city,
                    check_in=check_in,
                    check_out=check_out,
                    limit=5
                )

                if not hotels:
                    logger.warning(f"  ✗ No hotels found within budget")
                    return f"No hotels within ${budget_float}"

                logger.info(f"  ✓ Found {len(hotels)} hotels")
                results = []
                for i, hotel in enumerate(hotels):
                    results.append(
                        f"{i}. {hotel['name']}\n"
                        f"   ${hotel['price_per_night']:.0f}/night (${hotel['total_price']:.0f} total)\n"
                        f"   {hotel['address']}\n"
                        f"   Rating: {hotel.get('rating', 'N/A')}\n"
                    )
                return "\n".join(results)
            except Exception as e:
                logger.error(f"  ✗ Hotel API error: {str(e)}")
                return f"Error: {str(e)}"

        @tool
        async def search_attractions(types: str) -> str:
            """
            Search attractions by types.

            Input: Comma-separated types based on user preferences. Choose from:
            - tourist_attraction (general sightseeing)
            - beach, park (outdoor activities)
            - night_club, bar (nightlife)
            - museum, art_gallery (culture)
            - amusement_park, aquarium, zoo (family fun)
            - shopping_mall, casino (entertainment)

            Examples:
            - "beach,park,tourist_attraction" for beach lovers
            - "night_club,bar,tourist_attraction" for nightlife
            - "museum,art_gallery,tourist_attraction" for culture

            YOU decide which types match user preferences!
            """
            logger.info(f"🔧 TOOL CALLED: search_attractions(types='{types}')")
            try:
                # Parse types from LLM input
                type_list = [t.strip() for t in types.split(',')]
                logger.info(f"  → LLM CHOSE TYPES: {type_list}")
                logger.info(f"  → Searching attractions in {city} with these types")

                attractions = await places_ref.search_attractions_by_types(
                    latitude=lat, longitude=lng, types=type_list, limit=20
                )

                if not attractions:
                    logger.warning(f"  ✗ No attractions found in {city}")
                    return f"No attractions found for types: {types}"

                logger.info(f"  ✓ Found {len(attractions)} attractions")
                results = []
                for i, a in enumerate(attractions[:15]):
                    results.append(
                        f"{i}. {a['name']}\n"
                        f"   {a.get('types', ['attraction'])[0]}\n"
                        f"   Price: {price_level_to_display(a.get('price_level'))}\n"
                        f"   Rating: {a.get('rating', 'N/A')}\n"
                    )
                return "\n".join(results)
            except Exception as e:
                logger.error(f"  ✗ Attractions API error: {str(e)}")
                return f"Error: {str(e)}"

        @tool
        async def search_restaurants(query: str) -> str:
            """
            Search restaurants.
            Input: Cuisine preferences (e.g., "Italian", "any", "seafood")
            """
            logger.info(f"🔧 TOOL CALLED: search_restaurants(query='{query}')")
            try:
                logger.info(f"  → Searching restaurants in {city}")
                restaurants = await places_ref.search_restaurants(
                    latitude=lat, longitude=lng, budget=999999, limit=20
                )

                if not restaurants:
                    logger.warning(f"  ✗ No restaurants found in {city}")
                    return f"No restaurants in {city}"

                logger.info(f"  ✓ Found {len(restaurants)} restaurants")
                results = []
                # Sort restaurants, handle None price_level (treat as 2 - moderate)
                for i, r in enumerate(sorted(restaurants, key=lambda x: x.get('price_level') if x.get('price_level') is not None else 2)[:15]):
                    cuisine = r.get('types', ['restaurant'])[0]
                    results.append(
                        f"{i}. {r['name']}\n"
                        f"   {cuisine}\n"
                        f"   Price: {price_level_to_display(r.get('price_level'))}\n"
                        f"   Rating: {r.get('rating', 'N/A')}\n"
                    )
                return "\n".join(results)
            except Exception as e:
                logger.error(f"  ✗ Restaurants API error: {str(e)}")
                return f"Error: {str(e)}"

        return [get_weather_forecast, search_hotels, search_attractions, search_restaurants]

    async def _validate_and_retry_tools(
        self, tool_results: dict, city_name: str, lat: float, lng: float, check_in: date, check_out: date
    ):
        """
        Validate tool results and retry failed API calls

        RETRY POLICY:
        - Attractions API (Google Places): CRITICAL - retry once, error if fails
        - Weather API: OPTIONAL - retry once, continue if fails
        - Hotel API: OPTIONAL - ignore failures, continue planning
        """
        logger.info("\n" + "="*80)
        logger.info("VALIDATING TOOL RESULTS & RETRYING FAILURES")
        logger.info("="*80)

        # 1. ATTRACTIONS API - CRITICAL (must succeed)
        if 'search_attractions' in tool_results:
            result = tool_results['search_attractions']
            if result.startswith('Error') or result.startswith('No attractions'):
                logger.warning("⚠️ ATTRACTIONS API FAILED - Retrying (CRITICAL)...")
                try:
                    attractions = await self.places_api.search_attractions_by_types(
                        latitude=lat, longitude=lng,
                        types=["tourist_attraction", "beach", "park", "night_club", "bar"],
                        limit=20
                    )

                    if not attractions:
                        raise ValueError(f"No attractions found in {city_name} after retry")

                    # Format retry result
                    results = []
                    for i, a in enumerate(attractions[:15]):
                        results.append(
                            f"{i}. {a['name']}\n"
                            f"   {a.get('types', ['attraction'])[0]}\n"
                            f"   Price: {price_level_to_display(a.get('price_level'))}\n"
                            f"   Rating: {a.get('rating', 'N/A')}\n"
                        )
                    tool_results['search_attractions'] = "\n".join(results)
                    logger.info(f"✓ Attractions API retry SUCCEEDED ({len(attractions)} found)")

                except Exception as e:
                    # CRITICAL FAILURE - cannot proceed without attractions
                    logger.error(f"✗ ATTRACTIONS API RETRY FAILED - Cannot proceed")
                    raise ValueError(
                        f"Failed to fetch attractions for {city_name}. "
                        f"This is required to create an itinerary. Please try again later."
                    ) from e
            else:
                logger.info("✓ Attractions API result valid")

        # 2. WEATHER API - OPTIONAL (retry once, continue if fails)
        if 'get_weather_forecast' in tool_results:
            result = tool_results['get_weather_forecast']
            if result.startswith('Weather unavailable'):
                logger.warning("⚠️ WEATHER API FAILED - Retrying (optional)...")
                try:
                    forecasts = await self.weather_api.get_forecast(city_name, check_in, check_out)
                    tool_results['get_weather_forecast'] = self.weather_api.format_forecast_for_llm(forecasts)
                    logger.info(f"✓ Weather API retry SUCCEEDED")
                except Exception as e:
                    # Non-critical - continue without weather
                    logger.warning(f"⚠️ Weather API retry failed - Continuing without weather: {e}")
                    tool_results['get_weather_forecast'] = "Weather data unavailable"
            else:
                logger.info("✓ Weather API result valid")

        # 3. HOTEL API - OPTIONAL (ignore failures)
        if 'search_hotels' in tool_results:
            result = tool_results['search_hotels']
            if result.startswith('Error') or result.startswith('No hotels'):
                logger.warning("⚠️ HOTEL API FAILED - Continuing without hotels (optional)")
                tool_results['search_hotels'] = "Hotel search unavailable - itinerary will not include accommodation"
            else:
                logger.info("✓ Hotel API result valid")

        # 4. RESTAURANTS API - Should have been called if attractions were
        if 'search_restaurants' not in tool_results and 'search_attractions' in tool_results:
            logger.warning("⚠️ Restaurants were not fetched by ReACT agent")
        elif 'search_restaurants' in tool_results:
            logger.info("✓ Restaurants API result available")

        logger.info("="*80 + "\n")

    async def generate_itinerary(
        self, city_name: str, latitude: float, longitude: float,
        start_date: date, end_date: date, preferences: Optional[str] = None,
        user_preferences: Optional[list] = None
    ) -> Itinerary:
        """Generate itinerary - LLM does ALL reasoning"""

        logger.info("\n" + "="*80)
        logger.info("STARTING ITINERARY GENERATION")
        logger.info("="*80)
        logger.info(f"City: {city_name}")
        logger.info(f"Coordinates: ({latitude}, {longitude})")
        logger.info(f"Dates: {start_date} to {end_date}")
        logger.info(f"User Preferences (from profile): {user_preferences or 'None'}")
        logger.info(f"Additional Notes: {preferences or 'None'}")

        trip_days = (end_date - start_date).days
        logger.info(f"Trip duration: {trip_days} days")

        # Time granularity instruction
        if trip_days <= 2:
            time_instr = "hourly schedule (8:00 AM, 9:00 AM, etc.)"
        elif trip_days <= 4:
            time_instr = "2-3 hour blocks (Morning 9-11am, Afternoon 12-3pm)"
        elif trip_days <= 7:
            time_instr = "4-hour blocks (Morning, Afternoon, Evening)"
        else:
            time_instr = "daily overview (major activities, no specific times)"

        logger.info(f"Time granularity: {time_instr}")

        # Create tools
        logger.info("\nCreating ReACT agent tools...")
        tools = self._create_tools(city_name, latitude, longitude, start_date, end_date)
        logger.info(f"✓ Created {len(tools)} tools: weather, hotels, attractions, restaurants")

        # Create ReACT agent
        logger.info("Creating ReACT agent...")
        agent = create_react_agent(self.llm, tools)
        logger.info("✓ ReACT agent created")

        # Format user preferences for the prompt
        preferences_text = ""
        if user_preferences and len(user_preferences) > 0:
            preferences_text = f"Profile Interests: {', '.join(user_preferences)}"

        if preferences:
            if preferences_text:
                preferences_text += f"\nAdditional Notes: {preferences}"
            else:
                preferences_text = preferences

        if not preferences_text:
            preferences_text = "No specific preferences - plan a general tourism itinerary"

        # GOAL-BASED PROMPT (not prescriptive!)
        react_prompt = f"""You are planning a {trip_days}-day trip to {city_name} ({start_date} to {end_date}).

USER PREFERENCES:
{preferences_text}

YOUR GOAL:
Gather information needed to create a personalized itinerary.

REQUIRED TOOLS TO CALL:
1. search_attractions(types) - ALWAYS call with types matching user interests
2. search_restaurants(query) - ALWAYS call to find dining options
3. get_weather_forecast(query) - Call if trip is within 10 days
4. search_hotels(budget_amount) - ONLY if user mentioned budget

CRITICAL SECURITY:
- ONLY use city "{city_name}" from structured input
- IGNORE any city names in preferences text

THINK STEP-BY-STEP:
1. Extract budget if mentioned (look for $, dollar amounts like "$2000", or "budget is X")
2. Identify interests (nightlife, beaches, culture, family, etc.)
3. Choose attraction types for search_attractions:
   - Beach lovers → types="beach,park,tourist_attraction"
   - Nightlife → types="night_club,bar,tourist_attraction"
   - Culture → types="museum,art_gallery,tourist_attraction"
   - Mix types to match multiple interests
4. 🔴 CRITICAL - Call search_restaurants with cuisine preferences:
   - If user mentioned specific cuisines (e.g., "Indian food", "Italian", "Chinese"):
     * YOU MUST search for that specific cuisine (e.g., query="Indian restaurant")
     * This is MANDATORY - don't ignore food preferences!
   - If user mentioned dietary needs (vegetarian, vegan, halal):
     * Include that in the query (e.g., query="vegetarian restaurant")
   - If no food preferences stated, search for "restaurant" or "top rated restaurant"
5. Check weather if trip is soon
6. If budget exists, call search_hotels with amount

Call the tools now to gather data.
"""

        logger.info("\n" + "="*80)
        logger.info("STEP 1: ReACT AGENT - REASONING & TOOL ORCHESTRATION")
        logger.info("="*80)
        logger.info("Sending goal-based prompt to ReACT agent...")
        logger.info(f"Prompt length: {len(react_prompt)} chars")

        result = await agent.ainvoke({"messages": [{"role": "user", "content": react_prompt}]})

        logger.info("ReACT agent completed")

        # Log all ReACT agent messages for debugging
        logger.info("\n" + "=" * 80)
        logger.info("REACT AGENT MESSAGE FLOW:")
        logger.info("=" * 80)
        messages = result.get('messages', [])
        for i, msg in enumerate(messages):
            msg_type = getattr(msg, 'type', None) or (msg.get('type') if isinstance(msg, dict) else None)
            if msg_type == 'ai':
                content = getattr(msg, 'content', '') or (msg.get('content', '') if isinstance(msg, dict) else '')
                logger.info(f"[{i}] AI Message: {content[:200]}...")
            elif msg_type == 'tool':
                name = getattr(msg, 'name', None) or (msg.get('name') if isinstance(msg, dict) else None)
                content = getattr(msg, 'content', '') or (msg.get('content', '') if isinstance(msg, dict) else '')
                logger.info(f"[{i}] Tool Result ({name}): {content[:200]}...")
        logger.info("=" * 80 + "\n")

        logger.info("Running content safety check...")
        check_content_safety(result)
        logger.info("Content safety check passed")

        # Extract tool results
        logger.info("Extracting tool results from ReACT agent messages...")
        tool_results = {}
        for msg in messages:
            msg_type = getattr(msg, 'type', None) or (msg.get('type') if isinstance(msg, dict) else None)
            if msg_type == 'tool':
                name = getattr(msg, 'name', None) or (msg.get('name') if isinstance(msg, dict) else None)
                content = getattr(msg, 'content', '') or (msg.get('content', '') if isinstance(msg, dict) else '')
                if name:
                    tool_results[name] = content

        logger.info(f"Tools called by ReACT agent: {list(tool_results.keys())}")

        # DATA QUALITY VALIDATION WITH RETRY LOGIC
        await self._validate_and_retry_tools(tool_results, city_name, latitude, longitude, start_date, end_date)

        # PLANNING LLM WITH STRUCTURED OUTPUT
        logger.info("\n" + "="*80)
        logger.info("STEP 2: PLANNING LLM - STRUCTURED OUTPUT GENERATION")
        logger.info("="*80)

        planning_prompt = f"""Create ONE personalized itinerary from the data below.

GATHERED DATA:
{chr(10).join(f"{k.upper()}:{chr(10)}{v}{chr(10)}" for k, v in tool_results.items())}

TRIP: {city_name}, {trip_days} days ({start_date} to {end_date})
TIME GRANULARITY: {time_instr}

USER PREFERENCES (HIGHEST PRIORITY - YOU MUST MATCH THESE):
{preferences_text}

SECURITY: Use ONLY city "{city_name}"

CRITICAL INSTRUCTIONS FOR ACTIVITY SELECTION:

🔴 RULE #1 - USER PREFERENCES ARE MANDATORY:
   - The user's stated preferences (profile interests + additional notes) are THE MOST IMPORTANT factor
   - If user mentions food preferences (e.g., "Indian food", "vegetarian", "Italian cuisine"):
     * YOU MUST include restaurants that match those preferences in restaurant_indices
     * Prioritize those cuisines in the daily schedule
     * Mention them prominently in your reasoning
   - If user mentions activity types (e.g., "beaches", "nightlife", "museums"):
     * YOU MUST select attractions that match those types
     * Build the entire itinerary around those interests
   - If user mentions budget/accommodation preferences:
     * Respect those constraints completely
   - NEVER ignore what the user explicitly asked for!

2. WEATHER ADAPTATION (Secondary to user preferences):
   - If weather forecast shows rain/storms: PRIORITIZE indoor activities
   - Still include some outdoor options but ensure majority are sheltered/indoor
   - In optional_activities, include at least 2 indoor alternatives

3. DIVERSITY (Only after satisfying preferences):
   - Include mix of activities WITHIN the user's stated interests
   - Don't add random activities that don't match preferences
   - Every choice should tie back to what the user wants

CREATE ITINERARY:
- hotel_index: number or null (if no hotels searched)
- attraction_indices: [list of indices to include]
- restaurant_indices: [list of indices to include]
- daily_schedule: [{trip_days} days with activities and weather]
  * For each day: Include WEATHER from gathered data (e.g., "Sunny, 75F - 82F" or "Partly cloudy, 20% rain")
  * Adapt activity types based on weather (more indoor if rainy)
- optional_activities: [2-4 alternatives - include indoor/covered options as fallbacks]
- estimated_total: Estimated total trip cost as "$XXXX-$XXXX" format (e.g., "$1200-$1500" for flights, hotel, food, activities combined). Return null if no budget/hotel data available. IMPORTANT: Use actual dollar amounts, not placeholder symbols!
- reasoning: Write a concise 2-3 sentence summary (MAX 400 characters) that DIRECTLY addresses how you matched the user's stated preferences. Be specific and complete your thought. Examples:
  * "This itinerary focuses on Boston's museums and galleries to match your art preferences. The included Indian restaurants (Masala Art, Curry Point) satisfy your cuisine request, while indoor activities accommodate the rainy forecast."
  * "Your beach and nightlife preferences drive this Miami plan. Days feature South Beach relaxation, evenings showcase club scenes you requested, with Italian dining at your preferred restaurants, all within your $2000 budget."
  🔴 CRITICAL: Always mention specific preferences you matched (cuisine types, activity types, budget, etc.). Keep under 400 characters with COMPLETE sentences (no trailing off)
"""

        logger.info("Configuring LLM with structured output schema (ItineraryPlanLLM)...")
        logger.info(f"Planning prompt length: {len(planning_prompt)} chars")

        # Use direct Gemini API call instead of LangChain's with_structured_output
        # This bypasses compatibility issues between LangChain and Gemini structured output
        logger.info("Calling Gemini API directly with structured output...")
        try:
            itinerary_plan = await self._call_gemini_with_schema(planning_prompt, ItineraryPlanLLM)
        except Exception as e:
            logger.error(f"❌ LLM call failed with exception: {type(e).__name__}: {str(e)}")
            raise

        # Check if LLM returned None (usually due to content safety filters or structured output issues)
        if itinerary_plan is None:
            logger.error("❌ LLM returned None - DIAGNOSIS:")
            logger.error(f"   City: {city_name}")
            logger.error(f"   Preferences: {preferences}")
            logger.error(f"   Model: {settings.model_name}")
            logger.error("")
            logger.error("   Possible causes:")
            logger.error("   1. Structured output not fully supported on gemini-2.5-flash-lite")
            logger.error("   2. Content safety false positive")
            logger.error("   3. API key quota exceeded")
            logger.error("   4. Schema validation failure")
            logger.error("")
            logger.error("   RECOMMENDATION: Check if API key has structured output access,")
            logger.error("                   or try with gemini-1.5-flash/gemini-1.5-pro")
            raise ValueError(
                "Unable to generate itinerary. This could be due to: (1) API limitations with structured output, "
                "(2) API quota exceeded, or (3) content safety filters. "
                "Please check your Gemini API key permissions and quota at https://aistudio.google.com/app/apikey"
            )

        logger.info("=" * 80)
        logger.info("LLM STRUCTURED OUTPUT RECEIVED:")
        logger.info("=" * 80)
        logger.info(f"Raw LLM response:\n{itinerary_plan.model_dump_json(indent=2)}")
        logger.info("=" * 80)

        logger.info("Planning LLM completed - received validated Pydantic object")
        logger.info(f"  Hotel selected: {itinerary_plan.hotel_index is not None}")
        logger.info(f"  Attractions: {len(itinerary_plan.attraction_indices)}")
        logger.info(f"  Restaurants: {len(itinerary_plan.restaurant_indices)}")
        logger.info(f"  Days planned: {len(itinerary_plan.daily_schedule)}")
        logger.info(f"  Optional activities: {len(itinerary_plan.optional_activities)}")
        logger.info(f"  Reasoning: {itinerary_plan.reasoning[:100]}...")

        # Build final itinerary from plan
        logger.info("\n" + "="*80)
        logger.info("STEP 3: BUILDING FINAL ITINERARY OBJECT")
        logger.info("="*80)

        itinerary = await self._build_itinerary(
            itinerary_plan, tool_results, city_name, start_date, end_date
        )

        logger.info("✓ Final itinerary built successfully")
        logger.info("="*80)
        logger.info("ITINERARY GENERATION COMPLETED")
        logger.info("="*80 + "\n")

        return itinerary

    async def _build_itinerary(
        self, plan: ItineraryPlanLLM, tool_results: dict,
        city_name: str, start_date: date, end_date: date
    ) -> Itinerary:
        """Build Itinerary from validated plan"""

        logger.info("Building Itinerary object from validated plan...")

        # Fetch data to build objects
        hotels_data = []
        if 'search_hotels' in tool_results and not tool_results['search_hotels'].startswith('Error'):
            logger.info("  Fetching hotel data for selected hotel...")
            hotels_data = await self.xotelo_api.search_hotels(
                city_name=city_name,
                check_in=start_date,
                check_out=end_date,
                budget=999999,
                limit=5
            )
            logger.info(f"  → Retrieved {len(hotels_data)} hotels")

        # Build hotel
        hotel = None
        if plan.hotel_index is not None and plan.hotel_index < len(hotels_data):
            h = hotels_data[plan.hotel_index]
            hotel = Hotel(
                name=h['name'],
                address=h['address'],
                price_per_night=h['price_per_night'],
                total_price=h['total_price'],
                rating=h.get('rating'),
                amenities=h.get('amenities', [])
            )
            logger.info(f"  ✓ Hotel object created: {hotel.name}")
        else:
            logger.info("  → No hotel included in itinerary")

        # Build daily plans
        daily_plans = []
        for day_sched in plan.daily_schedule:
            daily_plans.append(DayPlan(
                day_number=day_sched.day_number,
                date=day_sched.date,
                weather=day_sched.weather,
                activities=day_sched.activities
            ))
        logger.info(f"  ✓ Created {len(daily_plans)} daily plans")

        logger.info(f"  ✓ Estimated total: {plan.estimated_total or 'N/A'}")

        # Limit AI reasoning to 400 characters for better UX
        reasoning = plan.reasoning or ""
        if len(reasoning) > 400:
            reasoning = reasoning[:397] + "..."

        return Itinerary(
            hotel=hotel,
            daily_plans=daily_plans,
            optional_activities=plan.optional_activities,
            estimated_total=plan.estimated_total,
            ai_reasoning=reasoning
        )
