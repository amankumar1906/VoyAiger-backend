"""Orchestrator Agent - coordinates all sub-agents and creates final itineraries"""
from typing import List
from datetime import date
from itertools import product
from .hotel_agent import HotelAgent
from .attractions_agent import AttractionsAgent
from .restaurant_agent import RestaurantAgent
from ..schemas.response import Itinerary
from ..schemas.agent import BudgetAllocation


class OrchestratorAgent:
    """
    Orchestrator agent that coordinates Hotel, Attractions, and Restaurant agents
    to create complete travel itineraries
    """

    def __init__(self):
        self.hotel_agent = HotelAgent()
        self.attractions_agent = AttractionsAgent()
        self.restaurant_agent = RestaurantAgent()

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

        # Create itinerary combinations
        print(f"\n🎯 Creating itinerary combinations...")
        itineraries = []

        # We have 3 hotels, 3 attractions, 3 restaurants
        # Create 3 diverse combinations
        combinations = [
            (0, 0, 0),  # Cheapest options
            (1, 1, 1),  # Mid-range options
            (2, 2, 2),  # Premium options
        ]

        for hotel_idx, attr_idx, rest_idx in combinations:
            # Safety check for indices
            if (hotel_idx >= len(hotel_output.options) or
                attr_idx >= len(attractions_output.options) or
                rest_idx >= len(restaurant_output.options)):
                continue

            hotel = hotel_output.options[hotel_idx]
            attractions = [attractions_output.options[attr_idx]]
            restaurants = [restaurant_output.options[rest_idx]]

            # Calculate total cost
            total_cost = (
                hotel.total_price +
                sum(a.price for a in attractions) +
                (sum(r.estimated_cost_per_meal for r in restaurants) * trip_duration_days * 3)  # 3 meals/day
            )

            # Check if within budget
            if total_cost <= budget:
                remaining = budget - total_cost

                itineraries.append(Itinerary(
                    hotel=hotel,
                    attractions=attractions,
                    restaurants=restaurants,
                    total_cost=total_cost,
                    remaining_budget=remaining
                ))

        # If no combinations fit, try cheaper options
        if not itineraries:
            # Try to find at least one combination that fits
            for h_idx in range(len(hotel_output.options)):
                for a_idx in range(len(attractions_output.options)):
                    for r_idx in range(len(restaurant_output.options)):
                        hotel = hotel_output.options[h_idx]
                        attractions = [attractions_output.options[a_idx]]
                        restaurants = [restaurant_output.options[r_idx]]

                        total_cost = (
                            hotel.total_price +
                            sum(a.price for a in attractions) +
                            (sum(r.estimated_cost_per_meal for r in restaurants) * trip_duration_days * 3)
                        )

                        if total_cost <= budget:
                            remaining = budget - total_cost
                            itineraries.append(Itinerary(
                                hotel=hotel,
                                attractions=attractions,
                                restaurants=restaurants,
                                total_cost=total_cost,
                                remaining_budget=remaining
                            ))

                            if len(itineraries) >= 3:
                                break
                    if len(itineraries) >= 3:
                        break
                if len(itineraries) >= 3:
                    break

        if not itineraries:
            raise Exception(
                "Cannot create any itineraries within your budget. Please increase your budget."
            )

        print(f"\n✅ Generated {len(itineraries)} itinerary options")
        return itineraries[:3]  # Return max 3
