"""Google Places API wrapper for attractions and restaurants"""
import httpx
from typing import List, Dict, Optional
from ..config import settings


class GooglePlacesAPI:
    """Wrapper for Google Places API"""

    BASE_URL = "https://maps.googleapis.com/maps/api/place"

    def __init__(self):
        self.api_key = settings.google_places_api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    async def search_city(self, city_name: str) -> Optional[Dict]:
        """
        Validate and get city information

        Args:
            city_name: Name of the city

        Returns:
            City information dict or None if not found
        """
        url = f"{self.BASE_URL}/findplacefromtext/json"
        params = {
            "input": city_name,
            "inputtype": "textquery",
            "fields": "place_id,name,formatted_address,geometry",
            "key": self.api_key
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "OK" and data.get("candidates"):
                return data["candidates"][0]
            return None
        except Exception as e:
            raise Exception(f"Failed to validate city: {str(e)}")

    async def search_attractions(
        self,
        city_name: str,
        budget: float,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for tourist attractions in a city

        Args:
            city_name: Name of the city
            budget: Budget allocated for attractions
            limit: Maximum number of results

        Returns:
            List of attraction dictionaries
        """
        # First get city location
        city_info = await self.search_city(city_name)
        if not city_info:
            raise Exception(f"City '{city_name}' not found")

        location = city_info["geometry"]["location"]
        lat, lng = location["lat"], location["lng"]

        # Search for tourist attractions
        url = f"{self.BASE_URL}/nearbysearch/json"
        params = {
            "location": f"{lat},{lng}",
            "radius": 10000,  # 10km radius
            "type": "tourist_attraction",
            "key": self.api_key
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            if data.get("status") == "OK":
                for place in data.get("results", [])[:limit]:
                    results.append({
                        "place_id": place.get("place_id"),
                        "name": place.get("name"),
                        "address": place.get("vicinity", ""),
                        "rating": place.get("rating"),
                        "types": place.get("types", []),
                        "location": place.get("geometry", {}).get("location")
                    })

            return results
        except Exception as e:
            raise Exception(f"Failed to search attractions: {str(e)}")

    async def search_restaurants(
        self,
        city_name: str,
        budget: float,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for restaurants in a city

        Args:
            city_name: Name of the city
            budget: Budget allocated for restaurants
            limit: Maximum number of results

        Returns:
            List of restaurant dictionaries
        """
        # First get city location
        city_info = await self.search_city(city_name)
        if not city_info:
            raise Exception(f"City '{city_name}' not found")

        location = city_info["geometry"]["location"]
        lat, lng = location["lat"], location["lng"]

        # Search for restaurants
        url = f"{self.BASE_URL}/nearbysearch/json"
        params = {
            "location": f"{lat},{lng}",
            "radius": 5000,  # 5km radius
            "type": "restaurant",
            "key": self.api_key
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            if data.get("status") == "OK":
                for place in data.get("results", [])[:limit]:
                    # Estimate price level (1-4 from Google, convert to USD estimate)
                    price_level = place.get("price_level", 2)
                    estimated_cost = price_level * 25  # Rough estimate: $25 per level

                    results.append({
                        "place_id": place.get("place_id"),
                        "name": place.get("name"),
                        "address": place.get("vicinity", ""),
                        "rating": place.get("rating"),
                        "price_level": price_level,
                        "estimated_cost_per_meal": estimated_cost,
                        "types": place.get("types", []),
                        "location": place.get("geometry", {}).get("location")
                    })

            return results
        except Exception as e:
            raise Exception(f"Failed to search restaurants: {str(e)}")

    async def get_place_details(self, place_id: str) -> Dict:
        """
        Get detailed information about a place

        Args:
            place_id: Google Place ID

        Returns:
            Place details dictionary
        """
        url = f"{self.BASE_URL}/details/json"
        params = {
            "place_id": place_id,
            "fields": "name,formatted_address,rating,price_level,photos,reviews,opening_hours,website",
            "key": self.api_key
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "OK":
                return data.get("result", {})
            return {}
        except Exception as e:
            raise Exception(f"Failed to get place details: {str(e)}")
