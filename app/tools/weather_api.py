"""Open-Meteo Weather API integration (100% free, no API key needed)"""
import httpx
from typing import List, Dict
from datetime import date


class WeatherAPI:
    """
    Free weather API using Open-Meteo
    Docs: https://open-meteo.com/en/docs
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def geocode_city(self, city_name: str) -> Dict:
        """
        Get coordinates for a city

        Args:
            city_name: City name (e.g., "Miami")

        Returns:
            Dict with latitude and longitude

        Raises:
            Exception: If city not found
        """
        params = {
            "name": city_name,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        response = await self.client.get(self.GEOCODING_URL, params=params)
        response.raise_for_status()

        data = response.json()

        if "results" not in data or not data["results"]:
            raise Exception(f"City '{city_name}' not found")

        result = data["results"][0]

        return {
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "name": result["name"],
            "country": result.get("country", ""),
            "state": result.get("admin1", "")
        }

    async def get_forecast(self, city_name: str, start_date: date, end_date: date) -> List[Dict]:
        """
        Get weather forecast for a city and date range

        Args:
            city_name: City name
            start_date: Start date of trip
            end_date: End date of trip

        Returns:
            List of daily forecasts with date, weather, and temperature

        Example return:
        [
            {
                "date": "2025-11-01",
                "weather_description": "Partly cloudy",
                "temperature_max": 78,
                "temperature_min": 65,
                "precipitation_probability": 20
            },
            ...
        ]
        """
        # Get city coordinates
        location = await self.geocode_city(city_name)

        # Request weather forecast
        params = {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "daily": [
                "weathercode",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_probability_max"
            ],
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",  # Default to EST, adjust if needed
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }

        response = await self.client.get(self.BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        # Parse forecast data
        forecasts = []
        daily = data.get("daily", {})

        dates = daily.get("time", [])
        weathercodes = daily.get("weathercode", [])
        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        precip_prob = daily.get("precipitation_probability_max", [])

        for i in range(len(dates)):
            weather_desc = self._weathercode_to_description(weathercodes[i])

            forecasts.append({
                "date": dates[i],
                "weather_description": weather_desc,
                "temperature_max": round(temps_max[i]),
                "temperature_min": round(temps_min[i]),
                "precipitation_probability": precip_prob[i] if precip_prob else 0
            })

        return forecasts

    def _weathercode_to_description(self, code: int) -> str:
        """
        Convert WMO weather code to human-readable description

        WMO codes: https://open-meteo.com/en/docs
        """
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Foggy",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Light rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Light snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Light rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Light snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with light hail",
            99: "Thunderstorm with heavy hail"
        }

        return weather_codes.get(code, "Unknown")

    def format_forecast_for_llm(self, forecasts: List[Dict]) -> str:
        """
        Format forecast data for LLM consumption

        Args:
            forecasts: List of forecast dictionaries

        Returns:
            Formatted string for LLM
        """
        if not forecasts:
            return "No weather data available"

        lines = []
        for forecast in forecasts:
            temp_max = forecast["temperature_max"]
            temp_min = forecast["temperature_min"]
            weather = forecast["weather_description"]
            precip = forecast["precipitation_probability"]

            lines.append(
                f"{forecast['date']}: {weather}, {temp_min}°F - {temp_max}°F"
                + (f", {precip}% chance of rain" if precip > 30 else "")
            )

        return "\n".join(lines)
