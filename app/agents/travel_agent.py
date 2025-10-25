"""
ReACT Travel Agent - LLM does ALL the reasoning

NO manual preprocessing - LLM extracts budget, decides tags, allocates budget
"""
from typing import Optional
from datetime import date, datetime
import logging
from pathlib import Path
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
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=0.2,
            google_api_key=settings.gemini_api_key,
            safety_settings=configure_safety_settings()
        )
        logger.info(f"✓ APIs initialized: Xotelo, Google Places, Weather")
        logger.info(f"✓ LLM configured: {settings.model_name} (temp=0.2)")

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
        start_date: date, end_date: date, preferences: Optional[str] = None
    ) -> Itinerary:
        """Generate itinerary - LLM does ALL reasoning"""

        logger.info("\n" + "="*80)
        logger.info("STARTING ITINERARY GENERATION")
        logger.info("="*80)
        logger.info(f"City: {city_name}")
        logger.info(f"Coordinates: ({latitude}, {longitude})")
        logger.info(f"Dates: {start_date} to {end_date}")
        logger.info(f"Preferences: {preferences or 'None'}")

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

        # GOAL-BASED PROMPT (not prescriptive!)
        react_prompt = f"""You are planning a {trip_days}-day trip to {city_name} ({start_date} to {end_date}).

USER PREFERENCES:
{preferences or "No specific preferences - plan a general tourism itinerary"}

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
4. ALWAYS call search_restaurants (people need to eat!)
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
USER PREFERENCES: {preferences or "General tourism"}

SECURITY: Use ONLY city "{city_name}"

CRITICAL INSTRUCTIONS FOR ACTIVITY SELECTION:
1. WEATHER ADAPTATION: If weather forecast shows rain/storms (e.g., "rain", "drizzle", "thunderstorm"):
   - PRIORITIZE indoor activities: museums, restaurants, shopping centers, theaters, aquariums, galleries, indoor parks
   - Still include some outdoor options but ensure majority are sheltered/indoor
   - In optional_activities, ALWAYS include at least 2 indoor alternatives (museum, mall, gallery, theater, aquarium, etc.)

2. PREFERENCE MATCHING:
   - Check if selected attractions/restaurants match user preferences
   - If there's a mismatch (e.g., user wants "family-friendly" but results are nightclubs, or user wants "nightlife" but results are museums):
     - Balance with activities that match stated preferences
     - In optional_activities, include 2-3 activities that better align with user preferences
     - ALWAYS suggest indoor options as fallbacks

3. DIVERSITY: Include mix of:
   - Outdoor attractions (if weather permits)
   - Indoor entertainment (museums, galleries, shopping, restaurants)
   - Natural/parks (if not raining)
   - Cultural experiences
   - Dining experiences

CREATE ITINERARY:
- hotel_index: number or null (if no hotels searched)
- attraction_indices: [list of indices to include]
- restaurant_indices: [list of indices to include]
- daily_schedule: [{trip_days} days with activities and weather]
  * For each day: Include WEATHER from gathered data (e.g., "Sunny, 75F - 82F" or "Partly cloudy, 20% rain")
  * Adapt activity types based on weather (more indoor if rainy)
- optional_activities: [2-4 alternatives - include indoor/covered options as fallbacks]
- estimated_total: Estimated total trip cost as "$XXXX-$XXXX" format (e.g., "$1200-$1500" for flights, hotel, food, activities combined). Return null if no budget/hotel data available. IMPORTANT: Use actual dollar amounts, not placeholder symbols!
- reasoning: Brief explanation
"""

        logger.info("Configuring LLM with structured output schema (ItineraryPlanLLM)...")
        logger.info(f"Planning prompt length: {len(planning_prompt)} chars")

        # Use structured output - guaranteed valid ItineraryPlanLLM object
        planning_llm = self.llm.with_structured_output(ItineraryPlanLLM)

        logger.info("Calling planning LLM with structured output...")
        itinerary_plan = await safe_llm_call(
            planning_llm.ainvoke,
            [HumanMessage(content=planning_prompt)]
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

        return Itinerary(
            hotel=hotel,
            daily_plans=daily_plans,
            optional_activities=plan.optional_activities,
            estimated_total=plan.estimated_total
        )
