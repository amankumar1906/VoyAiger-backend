"""Restaurant Agent - finds dining options within budget"""
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from ..config import settings
from ..tools.google_places import GooglePlacesAPI
from ..schemas.response import Restaurant
from ..schemas.agent import RestaurantAgentOutput
from ..utils.content_safety import check_content_safety, configure_safety_settings


class RestaurantAgent:
    """Agent for finding restaurants using Google Places API"""

    def __init__(self):
        self.places_api = GooglePlacesAPI()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=settings.model_temperature,
            google_api_key=settings.gemini_api_key,
            safety_settings=configure_safety_settings()
        )

    async def close(self):
        """Close API clients"""
        await self.places_api.close()

    def _create_restaurant_search_tool(self, city: str, budget: float):
        """Create restaurant search tool for the agent"""
        async def search_restaurants_wrapper(query: str) -> str:
            """Search for restaurants in the specified city"""
            try:
                restaurants = await self.places_api.search_restaurants(
                    city_name=city,
                    budget=budget,
                    limit=15
                )

                if not restaurants:
                    return f"No restaurants found in {city}"

                # Format results for LLM
                results = []
                for restaurant in restaurants:
                    results.append(
                        f"Restaurant: {restaurant['name']}\n"
                        f"Address: {restaurant['address']}\n"
                        f"Rating: {restaurant.get('rating', 'N/A')}\n"
                        f"Price level: {restaurant.get('price_level', 'N/A')}\n"
                        f"Estimated cost per meal: ${restaurant['estimated_cost_per_meal']}\n"
                    )

                return "\n---\n".join(results)
            except Exception as e:
                return f"Error searching restaurants: {str(e)}"

        return Tool(
            name="search_restaurants",
            func=search_restaurants_wrapper,
            description=f"Search for restaurants in {city} within ${budget} budget. Returns list of restaurants with ratings and estimated meal costs."
        )

    async def find_restaurants(
        self,
        city: str,
        budget: float,
        trip_duration_days: int
    ) -> RestaurantAgentOutput:
        """
        Find restaurant options within budget

        Args:
            city: Destination city
            budget: Allocated budget for restaurants
            trip_duration_days: Number of days for the trip

        Returns:
            RestaurantAgentOutput with up to 3 restaurant options
        """
        # Create tool
        restaurant_tool = self._create_restaurant_search_tool(city, budget)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a dining and restaurant expert. Your task is to find the best 3 restaurant options within the given budget.

Search for restaurants and select 3 diverse options. Consider:
- Different price ranges within budget
- Variety of cuisines
- Highly rated establishments
- Good value for money
- Total budget of ${budget} for all dining

Return EXACTLY 3 restaurant recommendations."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        # Create agent
        agent = create_tool_calling_agent(self.llm, [restaurant_tool], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[restaurant_tool], verbose=True)

        # Execute agent
        avg_budget_per_meal = budget / (trip_duration_days * 3)  # 3 meals per day estimate
        result = await agent_executor.ainvoke({
            "input": f"Find 3 diverse restaurant options in {city} for a {trip_duration_days}-day trip within a total budget of ${budget}. Average budget per meal is approximately ${avg_budget_per_meal:.2f}. Include variety in price and cuisine."
        })

        # Content safety check
        check_content_safety(result)

        # Get structured data from API
        restaurants_data = await self.places_api.search_restaurants(
            city_name=city,
            budget=budget,
            limit=10
        )

        if not restaurants_data:
            raise Exception("No restaurants found in this city.")

        # Convert to Restaurant schema (select best 3 with variety)
        restaurants = []
        selected_count = 0

        # Sort by price level to get variety
        restaurants_data_sorted = sorted(
            restaurants_data,
            key=lambda x: x.get('price_level', 2)
        )

        for restaurant_data in restaurants_data_sorted:
            if selected_count >= 3:
                break

            # Determine cuisine from types
            types = restaurant_data.get('types', [])
            cuisine = "International"
            for t in types:
                if 'italian' in t.lower():
                    cuisine = "Italian"
                    break
                elif 'chinese' in t.lower():
                    cuisine = "Chinese"
                    break
                elif 'japanese' in t.lower():
                    cuisine = "Japanese"
                    break
                elif 'mexican' in t.lower():
                    cuisine = "Mexican"
                    break
                elif 'indian' in t.lower():
                    cuisine = "Indian"
                    break
                elif 'french' in t.lower():
                    cuisine = "French"
                    break

            restaurants.append(Restaurant(
                name=restaurant_data["name"],
                address=restaurant_data["address"],
                estimated_cost_per_meal=restaurant_data["estimated_cost_per_meal"],
                cuisine=cuisine,
                rating=restaurant_data.get("rating")
            ))
            selected_count += 1

        if not restaurants:
            raise Exception("No restaurants fit within budget. Please increase your budget.")

        return RestaurantAgentOutput(
            options=restaurants,
            total_allocated_budget=budget
        )
