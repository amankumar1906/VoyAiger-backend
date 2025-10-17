"""Content safety checks for LLM outputs"""
from typing import Dict, Any
import google.generativeai as genai
from ..config import settings


class ContentSafetyError(Exception):
    """Raised when content fails safety checks"""
    def __init__(self, message: str, safety_ratings: Dict = None):
        self.message = message
        self.safety_ratings = safety_ratings or {}
        super().__init__(self.message)


def configure_safety_settings():
    """
    Configure Gemini safety settings

    Returns:
        Safety settings dictionary for Gemini
    """
    return {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
    }


def check_content_safety(response: Any) -> bool:
    """
    Check if LLM response passes content safety filters

    Args:
        response: LLM response object from Gemini

    Returns:
        bool: True if safe, raises ContentSafetyError if unsafe

    Raises:
        ContentSafetyError: If content is flagged as unsafe
    """
    # For LangChain responses, extract the underlying response
    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata

        # Check for safety ratings in metadata
        if 'safety_ratings' in metadata:
            safety_ratings = metadata['safety_ratings']

            # Check if any rating is MEDIUM or HIGH
            for rating in safety_ratings:
                category = rating.get('category', 'UNKNOWN')
                probability = rating.get('probability', 'UNKNOWN')

                if probability in ['MEDIUM', 'HIGH']:
                    raise ContentSafetyError(
                        f"Content flagged for {category} with probability {probability}",
                        safety_ratings=safety_ratings
                    )

    # For direct Gemini responses
    if hasattr(response, 'prompt_feedback'):
        prompt_feedback = response.prompt_feedback

        if hasattr(prompt_feedback, 'block_reason'):
            raise ContentSafetyError(
                f"Content blocked: {prompt_feedback.block_reason}",
                safety_ratings=getattr(prompt_feedback, 'safety_ratings', {})
            )

    # Check candidates for safety ratings
    if hasattr(response, 'candidates'):
        for candidate in response.candidates:
            if hasattr(candidate, 'safety_ratings'):
                for rating in candidate.safety_ratings:
                    if rating.probability in ['MEDIUM', 'HIGH']:
                        raise ContentSafetyError(
                            f"Content flagged for {rating.category.name} with probability {rating.probability.name}",
                            safety_ratings=[{
                                'category': rating.category.name,
                                'probability': rating.probability.name
                            }]
                        )

    return True


def validate_agent_output(output: str) -> bool:
    """
    Additional validation for agent outputs

    Args:
        output: Agent output text

    Returns:
        bool: True if valid

    Raises:
        ContentSafetyError: If output contains inappropriate content
    """
    # Basic checks for inappropriate content patterns
    suspicious_patterns = [
        'hack', 'exploit', 'illegal', 'weapon', 'drug',
        'violence', 'suicide', 'self-harm'
    ]

    output_lower = output.lower()

    for pattern in suspicious_patterns:
        if pattern in output_lower:
            # This is a basic filter - in production, use more sophisticated methods
            raise ContentSafetyError(
                f"Output contains potentially unsafe content related to: {pattern}"
            )

    return True
