"""Attractions Agent - finds tourist attractions within budget"""
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from ..config import settings
from ..tools.google_places import GooglePlacesAPI
from ..schemas.response import Attraction
from ..schemas.agent import AttractionsAgentOutput
from ..utils.content_safety import check_content_safety, configure_safety_settings


class AttractionsAgent:
    """Agent for finding tourist attractions using Google Places API"""

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

    def _create_attractions_search_tool(self, city: str, budget: float):
        """Create attractions search tool for the agent"""
        async def search_attractions_wrapper(query: str) -> str:
            """Search for tourist attractions in the specified city"""
            try:
                attractions = await self.places_api.search_attractions(
                    city_name=city,
                    budget=budget,
                    limit=15
                )

                if not attractions:
                    return f"No attractions found in {city}"

                # Format results for LLM
                results = []
                for attraction in attractions:
                    # Estimate entry price based on type
                    types = attraction.get('types', [])
                    estimated_price = 0
                    if 'museum' in types:
                        estimated_price = 15
                    elif 'amusement_park' in types or 'zoo' in types:
                        estimated_price = 30
                    elif 'park' in types or 'church' in types:
                        estimated_price = 0
                    else:
                        estimated_price = 10

                    results.append(
                        f"Attraction: {attraction['name']}\n"
                        f"Address: {attraction['address']}\n"
                        f"Rating: {attraction.get('rating', 'N/A')}\n"
                        f"Estimated price: ${estimated_price}\n"
                        f"Types: {', '.join(types[:3])}\n"
                    )

                return "\n---\n".join(results)
            except Exception as e:
                return f"Error searching attractions: {str(e)}"

        return Tool(
            name="search_attractions",
            func=search_attractions_wrapper,
            description=f"Search for tourist attractions in {city} within ${budget} budget. Returns list of attractions with ratings and estimated prices."
        )

    async def find_attractions(
        self,
        city: str,
        budget: float,
        trip_duration_days: int
    ) -> AttractionsAgentOutput:
        """
        Find attraction options within budget

        Args:
            city: Destination city
            budget: Allocated budget for attractions
            trip_duration_days: Number of days for the trip

        Returns:
            AttractionsAgentOutput with up to 3 attraction options
        """
        # Create tool
        attractions_tool = self._create_attractions_search_tool(city, budget)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a travel attractions expert. Your task is to find the best 3 tourist attractions within the given budget.

Search for attractions and select 3 diverse options. Consider:
- Mix of free and paid attractions
- Different types (museums, parks, landmarks, etc.)
- Highly rated places
- Must fit within budget of ${budget}

Return EXACTLY 3 attraction recommendations."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        # Create agent
        agent = create_tool_calling_agent(self.llm, [attractions_tool], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[attractions_tool], verbose=True)

        # Execute agent
        result = await agent_executor.ainvoke({
            "input": f"Find 3 diverse tourist attractions in {city} for a {trip_duration_days}-day trip within a budget of ${budget}. Include a mix of different types of attractions."
        })

        # Content safety check
        check_content_safety(result)

        # Get structured data from API
        attractions_data = await self.places_api.search_attractions(
            city_name=city,
            budget=budget,
            limit=10
        )

        if not attractions_data:
            raise Exception("No attractions found in this city.")

        # Convert to Attraction schema (select best 3)
        attractions = []
        selected_count = 0

        for attraction_data in attractions_data:
            if selected_count >= 3:
                break

            # Estimate price based on type
            types = attraction_data.get('types', [])
            estimated_price = 0
            category = "attraction"

            if 'museum' in types:
                estimated_price = 15
                category = "museum"
            elif 'amusement_park' in types or 'zoo' in types:
                estimated_price = 30
                category = "amusement_park"
            elif 'park' in types:
                estimated_price = 0
                category = "park"
            elif 'church' in types or 'place_of_worship' in types:
                estimated_price = 0
                category = "landmark"
            else:
                estimated_price = 10
                category = "tourist_attraction"

            # Check if within budget
            if sum(a.price for a in attractions) + estimated_price <= budget:
                attractions.append(Attraction(
                    name=attraction_data["name"],
                    address=attraction_data["address"],
                    price=estimated_price,
                    rating=attraction_data.get("rating"),
                    category=category
                ))
                selected_count += 1

        if not attractions:
            raise Exception("No attractions fit within budget. Please increase your budget.")

        return AttractionsAgentOutput(
            options=attractions,
            total_allocated_budget=budget
        )
