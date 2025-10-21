"""Google Places API wrapper for attractions and restaurants (New API)"""
import httpx
from typing import List, Dict, Optional
from ..config import settings


class GooglePlacesAPI:
    """Wrapper for Google Places API (New)"""

    BASE_URL = "https://places.googleapis.com/v1"

    def __init__(self):
        self.api_key = settings.google_places_api_key
        self.client = httpx.AsyncClient(timeout=30.0)
        self._city_cache = {}  # Cache city lookups to avoid repeated API calls

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    async def search_city(self, city_name: str) -> Optional[Dict]:
        """
        Validate and get city information using Text Search (New)

        Uses caching to avoid repeated API calls for the same city

        Args:
            city_name: Name of the city

        Returns:
            City information dict or None if not found
        """
        # Check cache first
        if city_name in self._city_cache:
            return self._city_cache[city_name]

        url = f"{self.BASE_URL}/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location"
        }
        body = {
            "textQuery": city_name
        }

        try:
            response = await self.client.post(url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data.get("places") and len(data["places"]) > 0:
                place = data["places"][0]
                city_info = {
                    "place_id": place.get("id"),
                    "name": place.get("displayName", {}).get("text", city_name),
                    "formatted_address": place.get("formattedAddress"),
                    "location": place.get("location")
                }
                # Cache the result
                self._city_cache[city_name] = city_info
                return city_info
            return None
        except Exception as e:
            raise Exception(f"Failed to validate city: {str(e)}")

    async def search_attractions_by_types(
        self,
        latitude: float,
        longitude: float,
        types: List[str],
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for attractions by dynamic types using Nearby Search (New)

        Args:
            latitude: City latitude coordinate
            longitude: City longitude coordinate
            types: List of place types to search for (e.g., ["beach", "night_club", "tourist_attraction"])
            limit: Maximum number of results

        Returns:
            List of attraction dictionaries with price_level
        """
        lat, lng = latitude, longitude

        # Search for tourist attractions using new API
        url = f"{self.BASE_URL}/places:searchNearby"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.priceLevel,places.types"
        }
        body = {
            "includedTypes": types,  # Dynamic types from LLM
            "maxResultCount": min(limit, 20),  # API limit is 20
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": lat,
                        "longitude": lng
                    },
                    "radius": 10000.0  # 10km radius
                }
            }
        }

        try:
            response = await self.client.post(url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()

            results = []
            if data.get("places"):
                for place in data["places"][:limit]:
                    # Map price_level enum to integer (0-4)
                    price_level_map = {
                        "PRICE_LEVEL_FREE": 0,
                        "PRICE_LEVEL_INEXPENSIVE": 1,
                        "PRICE_LEVEL_MODERATE": 2,
                        "PRICE_LEVEL_EXPENSIVE": 3,
                        "PRICE_LEVEL_VERY_EXPENSIVE": 4
                    }
                    price_level = price_level_map.get(place.get("priceLevel"), None)

                    results.append({
                        "place_id": place.get("id"),
                        "name": place.get("displayName", {}).get("text", "Unknown"),
                        "address": place.get("formattedAddress", ""),
                        "rating": place.get("rating"),
                        "price_level": price_level,
                        "types": place.get("types", [])
                    })

            return results
        except Exception as e:
            raise Exception(f"Failed to search attractions: {str(e)}")

    async def search_restaurants(
        self,
        latitude: float,
        longitude: float,
        budget: float,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for restaurants using Nearby Search (New)

        Args:
            latitude: City latitude coordinate
            longitude: City longitude coordinate
            budget: Budget allocated for restaurants (not used with price_level)
            limit: Maximum number of results

        Returns:
            List of restaurant dictionaries with price_level
        """
        lat, lng = latitude, longitude

        # Search for restaurants using new API
        url = f"{self.BASE_URL}/places:searchNearby"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.priceLevel,places.types"
        }
        body = {
            "includedTypes": ["restaurant"],
            "maxResultCount": min(limit, 20),  # API limit is 20
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": lat,
                        "longitude": lng
                    },
                    "radius": 5000.0  # 5km radius
                }
            }
        }

        try:
            response = await self.client.post(url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()

            results = []
            if data.get("places"):
                for place in data["places"][:limit]:
                    # Map price_level enum to integer (0-4)
                    price_level_map = {
                        "PRICE_LEVEL_FREE": 0,
                        "PRICE_LEVEL_INEXPENSIVE": 1,
                        "PRICE_LEVEL_MODERATE": 2,
                        "PRICE_LEVEL_EXPENSIVE": 3,
                        "PRICE_LEVEL_VERY_EXPENSIVE": 4
                    }
                    price_level = price_level_map.get(place.get("priceLevel"), None)

                    results.append({
                        "place_id": place.get("id"),
                        "name": place.get("displayName", {}).get("text", "Unknown"),
                        "address": place.get("formattedAddress", ""),
                        "rating": place.get("rating"),
                        "price_level": price_level,
                        "types": place.get("types", [])
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
