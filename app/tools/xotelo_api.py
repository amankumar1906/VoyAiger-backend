"""Xotelo API wrapper for hotel searches"""
import httpx
from typing import List, Dict
from datetime import date
from ..config import settings


class XoteloAPI:
    """Wrapper for Xotelo Hotel API"""

    BASE_URL = "https://api.xotelo.com/api/v2"

    def __init__(self):
        self.api_key = settings.xotelo_api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    async def search_hotels(
        self,
        city_name: str,
        check_in: date,
        check_out: date,
        budget: float,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for hotels in a city

        Args:
            city_name: Name of the city
            check_in: Check-in date
            check_out: Check-out date
            budget: Budget allocated for hotels
            limit: Maximum number of results

        Returns:
            List of hotel dictionaries
        """
        url = f"{self.BASE_URL}/hotels/search"

        # Calculate nights
        nights = (check_out - check_in).days
        max_price_per_night = budget / nights if nights > 0 else budget

        params = {
            "api_key": self.api_key,
            "location": city_name,
            "checkin": check_in.isoformat(),
            "checkout": check_out.isoformat(),
            "limit": limit,
            "currency": "USD"
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for hotel in data.get("hotels", []):
                price_per_night = hotel.get("price", {}).get("amount", 0)

                # Filter by budget
                if price_per_night <= max_price_per_night:
                    results.append({
                        "hotel_id": hotel.get("id"),
                        "name": hotel.get("name"),
                        "address": hotel.get("address", ""),
                        "price_per_night": price_per_night,
                        "total_price": price_per_night * nights,
                        "rating": hotel.get("rating"),
                        "amenities": hotel.get("amenities", []),
                        "stars": hotel.get("stars"),
                        "image_url": hotel.get("image_url"),
                        "nights": nights
                    })

            return results[:limit]
        except httpx.HTTPStatusError as e:
            raise Exception(f"Xotelo API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"Failed to search hotels: {str(e)}")

    async def get_hotel_details(self, hotel_id: str) -> Dict:
        """
        Get detailed information about a hotel

        Args:
            hotel_id: Hotel ID from Xotelo

        Returns:
            Hotel details dictionary
        """
        url = f"{self.BASE_URL}/hotels/{hotel_id}"
        params = {"api_key": self.api_key}

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            return data.get("hotel", {})
        except Exception as e:
            raise Exception(f"Failed to get hotel details: {str(e)}")
