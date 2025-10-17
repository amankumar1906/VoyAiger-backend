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
        Search for hotels in a city using 2-step process:
        1. /search to find hotels by city
        2. /rates to get prices for each hotel

        Args:
            city_name: Name of the city
            check_in: Check-in date
            check_out: Check-out date
            budget: Budget allocated for hotels
            limit: Maximum number of results

        Returns:
            List of hotel dictionaries with pricing
        """
        # Step 1: Search for hotels in the city
        search_url = f"{self.BASE_URL}/api/search"
        search_params = {
            "query": city_name,
            "location_type": "accommodation"
        }

        try:
            # Get hotel list
            search_response = await self.client.get(search_url, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()

            if search_data.get("error"):
                raise Exception(f"Search error: {search_data['error']}")

            hotels_list = search_data.get("result", {}).get("list", [])

            if not hotels_list:
                return []

            # Calculate nights and max price
            nights = (check_out - check_in).days
            max_price_per_night = budget / nights if nights > 0 else budget

            results = []

            # Step 2: Get rates for each hotel (up to limit)
            for hotel in hotels_list[:limit * 2]:  # Fetch more to filter by budget
                hotel_key = hotel.get("hotel_key")
                if not hotel_key:
                    continue

                # Get rates for this hotel
                rates_url = f"{self.BASE_URL}/api/rates"
                rates_params = {
                    "hotel_key": hotel_key,
                    "chk_in": check_in.strftime("%Y-%m-%d"),
                    "chk_out": check_out.strftime("%Y-%m-%d")
                }

                try:
                    rates_response = await self.client.get(rates_url, params=rates_params)
                    rates_response.raise_for_status()
                    rates_data = rates_response.json()

                    if rates_data.get("error"):
                        continue  # Skip hotels with rate errors

                    # Extract price from rates response
                    rates_result = rates_data.get("result", {})
                    price_per_night = rates_result.get("price", 0)

                    # Filter by budget
                    if price_per_night > 0 and price_per_night <= max_price_per_night:
                        results.append({
                            "hotel_id": hotel_key,
                            "name": hotel.get("name", "Unknown Hotel"),
                            "address": hotel.get("street_address", hotel.get("short_place_name", city_name)),
                            "price_per_night": float(price_per_night),
                            "total_price": float(price_per_night) * nights,
                            "rating": None,  # Not provided by search endpoint
                            "amenities": [],
                            "stars": None,
                            "image_url": hotel.get("image"),
                            "nights": nights
                        })

                    if len(results) >= limit:
                        break

                except Exception:
                    continue  # Skip hotels that fail rate lookup

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
