"""Orchestrator Agent - coordinates all sub-agents and creates final itineraries"""
from typing import List
from datetime import date
from itertools import product
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import json
from .hotel_agent import HotelAgent
from .attractions_agent import AttractionsAgent
from .restaurant_agent import RestaurantAgent
from ..schemas.response import Itinerary, Hotel, Attraction, Restaurant
from ..schemas.agent import BudgetAllocation, ItineraryPlan
from ..config import settings
from ..utils.content_safety import check_content_safety, configure_safety_settings
from pydantic import ValidationError


class OrchestratorAgent:
    """
    Orchestrator agent that coordinates Hotel, Attractions, and Restaurant agents
    to create complete travel itineraries using LLM-based intelligent planning
    """

    def __init__(self):
        self.hotel_agent = HotelAgent()
        self.attractions_agent = AttractionsAgent()
        self.restaurant_agent = RestaurantAgent()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=0.7,
            google_api_key=settings.gemini_api_key,
            safety_settings=configure_safety_settings()
        )

    async def close(self):
        """Close all sub-agents"""
        await self.hotel_agent.close()
        await self.attractions_agent.close()
        await self.restaurant_agent.close()

    def _allocate_budget(self, total_budget: float, trip_duration_days: int) -> BudgetAllocation:
        """
        Allocate budget across categories

        Strategy:
        - Hotels: 50% of budget
        - Attractions: 20% of budget
        - Restaurants: 25% of budget
        - Contingency: 5% of budget

        Args:
            total_budget: Total budget in USD
            trip_duration_days: Number of days

        Returns:
            BudgetAllocation object
        """
        hotel_budget = total_budget * 0.50
        attractions_budget = total_budget * 0.20
        restaurants_budget = total_budget * 0.25
        contingency = total_budget * 0.05

        return BudgetAllocation(
            hotel_budget=hotel_budget,
            attractions_budget=attractions_budget,
            restaurants_budget=restaurants_budget,
            contingency=contingency
        )

    async def generate_itineraries(
        self,
        city: str,
        budget: float,
        start_date: date,
        end_date: date
    ) -> List[Itinerary]:
        """
        Generate up to 3 complete itinerary options

        Args:
            city: Destination city
            budget: Total budget in USD
            start_date: Trip start date
            end_date: Trip end date

        Returns:
            List of up to 3 Itinerary objects

        Raises:
            Exception: If budget is insufficient or agents fail
        """
        # Calculate trip duration
        trip_duration_days = (end_date - start_date).days

        # Allocate budget
        allocation = self._allocate_budget(budget, trip_duration_days)

        # Sequential execution of agents
        print(f"📊 Budget Allocation:")
        print(f"  Hotels: ${allocation.hotel_budget:.2f}")
        print(f"  Attractions: ${allocation.attractions_budget:.2f}")
        print(f"  Restaurants: ${allocation.restaurants_budget:.2f}")
        print(f"  Contingency: ${allocation.contingency:.2f}")

        # 1. Get hotel options
        print(f"\n🏨 Finding hotels...")
        try:
            hotel_output = await self.hotel_agent.find_hotels(
                city=city,
                check_in=start_date,
                check_out=end_date,
                budget=allocation.hotel_budget
            )
        except Exception as e:
            raise Exception(f"Hotel search failed: {str(e)}. Please increase your budget.")

        # Check if hotel costs exceed budget
        if hotel_output.options:
            min_hotel_cost = min(h.total_price for h in hotel_output.options)
            if min_hotel_cost > allocation.hotel_budget:
                raise Exception(
                    f"Minimum hotel cost (${min_hotel_cost:.2f}) exceeds allocated budget (${allocation.hotel_budget:.2f}). "
                    f"Please increase your budget."
                )

        # 2. Get attraction options
        print(f"\n🎡 Finding attractions...")
        try:
            attractions_output = await self.attractions_agent.find_attractions(
                city=city,
                budget=allocation.attractions_budget,
                trip_duration_days=trip_duration_days
            )
        except Exception as e:
            raise Exception(f"Attractions search failed: {str(e)}. Please increase your budget.")

        # 3. Get restaurant options
        print(f"\n🍽️  Finding restaurants...")
        try:
            restaurant_output = await self.restaurant_agent.find_restaurants(
                city=city,
                budget=allocation.restaurants_budget,
                trip_duration_days=trip_duration_days
            )
        except Exception as e:
            raise Exception(f"Restaurant search failed: {str(e)}. Please increase your budget.")

        # Use LLM to create intelligent itinerary combinations
        print(f"\n🎯 Creating intelligent itinerary combinations...")
        itineraries = await self._create_intelligent_itineraries(
            city=city,
            budget=budget,
            start_date=start_date,
            end_date=end_date,
            trip_duration_days=trip_duration_days,
            hotels=hotel_output.options,
            attractions=attractions_output.options,
            restaurants=restaurant_output.options
        )

        if not itineraries:
            raise Exception(
                "Cannot create any itineraries within your budget. Please increase your budget."
            )

        print(f"\n✅ Generated {len(itineraries)} itinerary options")
        return itineraries[:3]  # Return max 3

    def _format_options_for_llm(
        self,
        hotels: List[Hotel],
        attractions: List[Attraction],
        restaurants: List[Restaurant]
    ) -> str:
        """Format all options into structured text for LLM"""
        parts = ["# Available Options\n\n"]

        # Hotels
        parts.append("## Hotels:\n")
        for i, hotel in enumerate(hotels):
            parts.append(
                f"{i}. {hotel.name} - ${hotel.price_per_night:.2f}/night "
                f"(Total: ${hotel.total_price:.2f}) - Rating: {hotel.rating or 'N/A'}\n"
            )

        # Attractions
        parts.append("\n## Attractions:\n")
        for i, attr in enumerate(attractions):
            parts.append(
                f"{i}. {attr.name} - ${attr.price:.2f} - "
                f"Rating: {attr.rating or 'N/A'} - Category: {attr.category or 'N/A'}\n"
            )

        # Restaurants
        parts.append("\n## Restaurants:\n")
        for i, rest in enumerate(restaurants):
            parts.append(
                f"{i}. {rest.name} - ${rest.estimated_cost_per_meal:.2f}/meal - "
                f"Cuisine: {rest.cuisine or 'N/A'} - Rating: {rest.rating or 'N/A'}\n"
            )

        return "".join(parts)

    async def _create_intelligent_itineraries(
        self,
        city: str,
        budget: float,
        start_date: date,
        end_date: date,
        trip_duration_days: int,
        hotels: List[Hotel],
        attractions: List[Attraction],
        restaurants: List[Restaurant]
    ) -> List[Itinerary]:
        """Use LLM to create intelligent itinerary combinations"""

        options_text = self._format_options_for_llm(hotels, attractions, restaurants)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert travel planner creating personalized itineraries.

Analyze the available options and create 3 DIFFERENT itinerary recommendations.

RULES:
1. Each itinerary has EXACTLY 1 hotel (for all nights)
2. Each itinerary can have MULTIPLE attractions (select 1-3 based on budget/time)
3. Each itinerary should have 2-3 different restaurants (used across meals)
4. Total cost must be ≤ ${budget}
5. Create 3 styles: Budget-friendly, Balanced, Premium

Calculate costs:
- Hotel total = shown total price
- Attractions total = sum of selected attraction prices
- Restaurants total = (avg meal cost × 3 meals/day × {days} days)

Return JSON array with 3 objects:
[
  {{
    "style": "budget-friendly",
    "hotel_index": 0,
    "attraction_indices": [0, 1],
    "restaurant_indices": [0, 1],
    "reasoning": "Why this combination"
  }},
  ...
]

ONLY return valid JSON, no other text."""),
            ("human", """Create 3 itineraries for {city} from {start_date} to {end_date}.
Budget: ${budget}
Duration: {days} days

{options}""")
        ])

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "city": city,
                "budget": budget,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": trip_duration_days,
                "options": options_text
            })

            # Content safety check
            check_content_safety(response)

            # Parse JSON response
            response_text = response.content if hasattr(response, 'content') else str(response)
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_text = response_text[start_idx:end_idx]
            raw_plans = json.loads(json_text)

            # Validate with Pydantic
            validated_plans = []
            for raw_plan in raw_plans[:3]:
                try:
                    validated_plan = ItineraryPlan(**raw_plan)
                    validated_plans.append(validated_plan)
                except ValidationError as e:
                    print(f"⚠️ Plan validation failed: {e}. Skipping this plan.")
                    continue

            if not validated_plans:
                raise ValueError("No valid plans after Pydantic validation")

            # Convert to Itinerary objects
            itineraries = []
            for plan in validated_plans:
                hotel_idx = plan.hotel_index
                if hotel_idx >= len(hotels):
                    continue

                hotel = hotels[hotel_idx]

                # Get selected attractions
                selected_attractions = [attractions[i] for i in plan.attraction_indices if i < len(attractions)]

                # Get selected restaurants
                selected_restaurants = [restaurants[i] for i in plan.restaurant_indices if i < len(restaurants)]

                if not selected_restaurants:
                    selected_restaurants = restaurants[:2]

                # Calculate total cost
                total_cost = (
                    hotel.total_price +
                    sum(a.price for a in selected_attractions) +
                    (sum(r.estimated_cost_per_meal for r in selected_restaurants) / len(selected_restaurants)) * 3 * trip_duration_days
                )

                if total_cost <= budget:
                    itineraries.append(Itinerary(
                        hotel=hotel,
                        attractions=selected_attractions,
                        restaurants=selected_restaurants,
                        total_cost=total_cost,
                        remaining_budget=budget - total_cost
                    ))

            return itineraries

        except Exception as e:
            print(f"⚠️ LLM planning failed: {e}. Using fallback logic.")
            return self._create_fallback_itineraries(budget, trip_duration_days, hotels, attractions, restaurants)

    def _create_fallback_itineraries(
        self,
        budget: float,
        days: int,
        hotels: List[Hotel],
        attractions: List[Attraction],
        restaurants: List[Restaurant]
    ) -> List[Itinerary]:
        """Fallback: Simple combination logic if LLM fails"""
        itineraries = []

        for h_idx in range(min(3, len(hotels))):
            hotel = hotels[h_idx]
            remaining = budget - hotel.total_price

            # Select attractions
            selected_attractions = []
            for attr in attractions:
                if sum(a.price for a in selected_attractions) + attr.price <= remaining * 0.3:
                    selected_attractions.append(attr)
                    if len(selected_attractions) >= 2:
                        break

            selected_restaurants = restaurants[:2] if len(restaurants) >= 2 else restaurants

            total_cost = (
                hotel.total_price +
                sum(a.price for a in selected_attractions) +
                (sum(r.estimated_cost_per_meal for r in selected_restaurants) / len(selected_restaurants)) * 3 * days
            )

            if total_cost <= budget:
                itineraries.append(Itinerary(
                    hotel=hotel,
                    attractions=selected_attractions,
                    restaurants=selected_restaurants,
                    total_cost=total_cost,
                    remaining_budget=budget - total_cost
                ))

            if len(itineraries) >= 3:
                break

        return itineraries
