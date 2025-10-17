"""Hotel Agent - finds hotel options within budget"""
from typing import List
from datetime import date
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from ..config import settings
from ..tools.xotelo_api import XoteloAPI
from ..schemas.response import Hotel
from ..schemas.agent import HotelAgentOutput
from ..utils.content_safety import check_content_safety, configure_safety_settings


class HotelAgent:
    """Agent for finding hotels using Xotelo API"""

    def __init__(self):
        self.xotelo_api = XoteloAPI()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=settings.model_temperature,
            google_api_key=settings.gemini_api_key,
            safety_settings=configure_safety_settings()
        )

    async def close(self):
        """Close API clients"""
        await self.xotelo_api.close()

    def _create_hotel_search_tool(self, city: str, check_in: date, check_out: date, budget: float):
        """Create hotel search tool for the agent"""
        async def search_hotels_wrapper(query: str) -> str:
            """Search for hotels in the specified city"""
            try:
                hotels = await self.xotelo_api.search_hotels(
                    city_name=city,
                    check_in=check_in,
                    check_out=check_out,
                    budget=budget,
                    limit=10
                )

                if not hotels:
                    return f"No hotels found within budget of ${budget}"

                # Format results for LLM
                results = []
                for hotel in hotels:
                    results.append(
                        f"Hotel: {hotel['name']}\n"
                        f"Address: {hotel['address']}\n"
                        f"Price per night: ${hotel['price_per_night']:.2f}\n"
                        f"Total price: ${hotel['total_price']:.2f}\n"
                        f"Rating: {hotel.get('rating', 'N/A')}\n"
                        f"Amenities: {', '.join(hotel.get('amenities', []))}\n"
                    )

                return "\n---\n".join(results)
            except Exception as e:
                return f"Error searching hotels: {str(e)}"

        return Tool(
            name="search_hotels",
            func=search_hotels_wrapper,
            description=f"Search for hotels in {city} from {check_in} to {check_out} within ${budget} budget. Returns list of available hotels with pricing and details."
        )

    async def find_hotels(
        self,
        city: str,
        check_in: date,
        check_out: date,
        budget: float
    ) -> HotelAgentOutput:
        """
        Find hotel options within budget

        Args:
            city: Destination city
            check_in: Check-in date
            check_out: Check-out date
            budget: Allocated budget for hotels

        Returns:
            HotelAgentOutput with up to 3 hotel options
        """
        # Create tool
        hotel_tool = self._create_hotel_search_tool(city, check_in, check_out, budget)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hotel recommendation expert. Your task is to find the best 3 hotel options within the given budget.

Search for hotels and select 3 diverse options that offer good value. Consider:
- Different price points within budget
- Location and ratings
- Amenities offered
- Overall value for money

Return EXACTLY 3 hotel recommendations that fit within the budget of ${budget}."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        # Create agent
        agent = create_tool_calling_agent(self.llm, [hotel_tool], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[hotel_tool], verbose=True)

        # Execute agent
        nights = (check_out - check_in).days
        result = await agent_executor.ainvoke({
            "input": f"Find 3 hotel options in {city} for {nights} nights (from {check_in} to {check_out}) within a budget of ${budget}. Provide diverse options at different price points."
        })

        # Content safety check
        check_content_safety(result)

        # Parse and validate output
        # For now, we'll make a direct API call to get structured data
        # In production, you'd parse the LLM output more carefully
        hotels_data = await self.xotelo_api.search_hotels(
            city_name=city,
            check_in=check_in,
            check_out=check_out,
            budget=budget,
            limit=3
        )

        if not hotels_data:
            raise Exception("No hotels found within budget. Please increase your budget.")

        # Convert to Hotel schema
        hotels = []
        for hotel_data in hotels_data[:3]:
            hotels.append(Hotel(
                name=hotel_data["name"],
                address=hotel_data["address"],
                price_per_night=hotel_data["price_per_night"],
                total_price=hotel_data["total_price"],
                rating=hotel_data.get("rating"),
                amenities=hotel_data.get("amenities", [])
            ))

        return HotelAgentOutput(
            options=hotels,
            total_allocated_budget=budget
        )
