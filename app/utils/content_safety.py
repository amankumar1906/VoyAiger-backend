"""Content safety checks for LLM outputs"""
from typing import Dict, Any
from langchain_google_genai import HarmBlockThreshold, HarmCategory
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

    Uses BLOCK_ONLY_HIGH to avoid false positives on normal travel requests.
    MEDIUM threshold was too strict and flagged legitimate content.

    Returns:
        Safety settings dictionary for Gemini
    """
    return {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
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


async def safe_llm_call(llm_func, *args, **kwargs):
    """
    Wrapper for LLM calls with automatic safety checking

    Args:
        llm_func: Async LLM function to call (e.g., llm.ainvoke)
        *args: Positional arguments for llm_func
        **kwargs: Keyword arguments for llm_func

    Returns:
        LLM response if safe

    Raises:
        ContentSafetyError: If response fails safety checks

    Example:
        response = await safe_llm_call(llm.ainvoke, [HumanMessage(content=prompt)])
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info("🔄 Making LLM API call...")
        # Call the LLM function
        response = await llm_func(*args, **kwargs)

        logger.info(f"📥 LLM API response received: {type(response)}")
        logger.info(f"   Response is None: {response is None}")

        if response is not None:
            logger.info(f"   Response attributes: {dir(response)}")
            if hasattr(response, '__dict__'):
                logger.info(f"   Response dict: {response.__dict__}")

        # Check if response is None (can happen with Gemini's content filters or structured output failures)
        if response is None:
            logger.error("❌ LLM returned None - possible causes:")
            logger.error("   1. Invalid/expired API key")
            logger.error("   2. API quota exceeded")
            logger.error("   3. Structured output not supported on this API tier")
            logger.error("   4. Content blocked by safety filters")
            logger.error("   5. Model endpoint issue")
            # This is likely a false positive from Gemini's safety filters
            # Return None and let the caller handle it with better context
            return None

        # Run content safety check
        check_content_safety(response)

        # Additional validation for text content
        if hasattr(response, 'content'):
            validate_agent_output(response.content)

        logger.info("✅ LLM call completed successfully")
        return response

    except Exception as e:
        logger.error(f"❌ Exception during LLM call: {type(e).__name__}: {str(e)}")
        logger.error(f"   Exception details: {repr(e)}")
        # If it's already a ContentSafetyError, re-raise it
        if isinstance(e, ContentSafetyError):
            raise
        # Log and re-raise other exceptions
        raise
