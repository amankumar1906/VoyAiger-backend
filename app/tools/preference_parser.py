"""Preference parsing tool - extracts structured data from unstructured text"""
import re
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class ParsedPreferences(BaseModel):
    """Structured output from preference parsing"""
    budget: Optional[float] = Field(None, description="Extracted budget in USD")
    tags: List[str] = Field(default_factory=list, description="Activity tags for Google Places API")
    interests: List[str] = Field(default_factory=list, description="General interests (nightlife, family, culture, etc.)")
    notes: str = Field("", description="Any additional context")


class PreferenceParser:
    """Parse unstructured preferences into structured tags and budget"""

    # Mapping from interest keywords to Google Places tags
    INTEREST_TAG_MAPPING = {
        # Nightlife
        'nightlife': ['night_club', 'bar', 'live_music', 'casino'],
        'party': ['night_club', 'bar', 'dance_club'],
        'bars': ['bar', 'cocktail_bar', 'wine_bar'],
        'clubs': ['night_club', 'dance_club'],

        # Family
        'family': ['amusement_park', 'zoo', 'aquarium', 'playground', 'park'],
        'kids': ['amusement_park', 'zoo', 'aquarium', 'playground'],
        'children': ['amusement_park', 'zoo', 'aquarium', 'playground'],

        # Culture
        'culture': ['museum', 'art_gallery', 'theater', 'historical_landmark'],
        'museum': ['museum', 'art_gallery'],
        'art': ['art_gallery', 'museum'],
        'history': ['museum', 'historical_landmark', 'monument'],

        # Food
        'foodie': ['restaurant', 'bakery', 'cafe', 'food'],
        'food': ['restaurant', 'cafe', 'food'],
        'dining': ['restaurant', 'cafe'],

        # Beach/Outdoor
        'beach': ['beach', 'water_sports', 'coastal'],
        'outdoor': ['park', 'hiking', 'camping', 'nature'],
        'nature': ['park', 'hiking', 'nature_reserve'],
        'hiking': ['hiking_area', 'park', 'trail'],

        # Romance
        'romantic': ['restaurant', 'spa', 'park', 'scenic_viewpoint'],
        'honeymoon': ['spa', 'restaurant', 'scenic_viewpoint'],

        # Adventure
        'adventure': ['amusement_park', 'water_sports', 'hiking', 'zip_line'],
        'sports': ['stadium', 'sports_club', 'water_sports'],

        # Shopping
        'shopping': ['shopping_mall', 'clothing_store', 'department_store'],

        # Relaxation
        'relaxing': ['spa', 'park', 'beach', 'cafe'],
        'spa': ['spa', 'wellness_center'],
    }

    @classmethod
    def extract_budget(cls, text: str) -> Optional[float]:
        """
        Extract budget from text using regex

        Examples:
            "$1500" -> 1500.0
            "1500 dollars" -> 1500.0
            "budget of 1500" -> 1500.0
            "$1,500" -> 1500.0
        """
        if not text:
            return None

        # Patterns to match budget mentions
        patterns = [
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $1500, $1,500
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars|usd|bucks)',  # 1500 dollars
            r'budget\s*(?:of|is)?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # budget of $1500
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                # Extract number and remove commas
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue

        return None

    @classmethod
    def extract_tags(cls, text: str) -> List[str]:
        """
        Extract activity tags from text based on interest keywords

        Args:
            text: User preferences text

        Returns:
            List of Google Places API tags
        """
        if not text:
            return []

        text_lower = text.lower()
        tags = set()
        matched_interests = []

        # Check for each interest keyword
        for interest, interest_tags in cls.INTEREST_TAG_MAPPING.items():
            if interest in text_lower:
                tags.update(interest_tags)
                matched_interests.append(interest)

        return list(tags)

    @classmethod
    def extract_interests(cls, text: str) -> List[str]:
        """
        Extract high-level interests from text

        Returns:
            List of interest categories (e.g., ['nightlife', 'culture'])
        """
        if not text:
            return []

        text_lower = text.lower()
        interests = []

        for interest in cls.INTEREST_TAG_MAPPING.keys():
            if interest in text_lower:
                interests.append(interest)

        return interests

    @classmethod
    def parse(cls, preferences_text: Optional[str]) -> ParsedPreferences:
        """
        Parse unstructured preferences into structured format

        Args:
            preferences_text: Optional user preferences text

        Returns:
            ParsedPreferences object with budget, tags, and interests

        Example:
            Input: "I love nightlife and have $1500 budget"
            Output: ParsedPreferences(
                budget=1500.0,
                tags=['night_club', 'bar', 'live_music', 'casino'],
                interests=['nightlife'],
                notes="I love nightlife and have $1500 budget"
            )
        """
        if not preferences_text:
            return ParsedPreferences()

        budget = cls.extract_budget(preferences_text)
        tags = cls.extract_tags(preferences_text)
        interests = cls.extract_interests(preferences_text)

        return ParsedPreferences(
            budget=budget,
            tags=tags,
            interests=interests,
            notes=preferences_text[:200]  # Store first 200 chars
        )

    @classmethod
    def format_for_llm(cls, parsed: ParsedPreferences) -> str:
        """
        Format parsed preferences for LLM consumption

        Args:
            parsed: ParsedPreferences object

        Returns:
            Formatted string for LLM prompt
        """
        lines = []

        if parsed.budget:
            lines.append(f"Budget: ${parsed.budget:,.0f}")
        else:
            lines.append("Budget: Not specified (skip hotel search)")

        if parsed.interests:
            lines.append(f"Interests: {', '.join(parsed.interests)}")

        if parsed.tags:
            lines.append(f"Activity Tags: {', '.join(parsed.tags[:10])}")  # Limit to 10 tags

        if parsed.notes:
            lines.append(f"Original: {parsed.notes}")

        return "\n".join(lines) if lines else "No preferences specified"
