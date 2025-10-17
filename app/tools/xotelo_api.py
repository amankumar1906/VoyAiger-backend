"""Xotelo API wrapper for hotel searches via RapidAPI"""
import httpx
from typing import List, Dict
from datetime import date
from ..config import settings


class XoteloAPI:
    """Wrapper for Xotelo Hotel API via RapidAPI"""

    BASE_URL = "https://xotelo-hotel-prices.p.rapidapi.com"

    def __init__(self):
        self.api_key = settings.xotelo_api_key  # This is your RapidAPI key
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": "xotelo-hotel-prices.p.rapidapi.com"
            }
        )

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
        Search for hotels in a city using Xotelo /search endpoint

        Args:
            city_name: Name of the city
            check_in: Check-in date
            check_out: Check-out date
            budget: Budget allocated for hotels
            limit: Maximum number of results

        Returns:
            List of hotel dictionaries
        """
        url = f"{self.BASE_URL}/api/search"

        # Calculate nights
        nights = (check_out - check_in).days
        max_price_per_night = budget / nights if nights > 0 else budget

        # Xotelo /api/search endpoint parameters (based on RapidAPI docs)
        params = {
            "location": city_name,
            "location_type": "accommodation",
            "chk_in": check_in.strftime("%Y-%m-%d"),
            "chk_out": check_out.strftime("%Y-%m-%d")
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            # Parse Xotelo response format
            hotels_data = data if isinstance(data, list) else data.get("result", [])

            for hotel in hotels_data[:limit * 2]:  # Get more to filter by budget
                # Extract price (Xotelo returns price per night)
                price_per_night = hotel.get("price", 0)
                if isinstance(price_per_night, dict):
                    price_per_night = price_per_night.get("amount", 0)

                # Filter by budget
                if price_per_night > 0 and price_per_night <= max_price_per_night:
                    results.append({
                        "hotel_id": hotel.get("hotel_key", hotel.get("id")),
                        "name": hotel.get("name", "Unknown Hotel"),
                        "address": hotel.get("address", f"{city_name}"),
                        "price_per_night": float(price_per_night),
                        "total_price": float(price_per_night) * nights,
                        "rating": hotel.get("rating", hotel.get("stars")),
                        "amenities": hotel.get("amenities", []),
                        "stars": hotel.get("stars"),
                        "image_url": hotel.get("image", hotel.get("image_url")),
                        "nights": nights
                    })

                if len(results) >= limit:
                    break

            return results
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
