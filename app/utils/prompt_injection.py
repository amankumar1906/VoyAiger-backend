"""Prompt injection detection and prevention utilities"""
from typing import List, Tuple


class PromptInjectionDetector:
    """Detect and prevent prompt injection attempts in user input"""

    # Patterns that indicate prompt injection attempts
    INJECTION_PATTERNS = [
        # Direct instruction override
        'ignore previous instructions',
        'ignore all previous',
        'ignore the above',
        'disregard',
        'forget',
        'new instructions',

        # Role manipulation
        'system:',
        'assistant:',
        'user:',
        'ai:',
        'human:',

        # Special tokens (ChatML, etc.)
        '<|im_start|>',
        '<|im_end|>',
        '<|endoftext|>',
        '[INST]',
        '[/INST]',

        # Markdown/formatting exploits
        '###',
        '```system',
        '```instruction',

        # Direct role assertions
        'you are now',
        'act as',
        'pretend to be',
        'roleplay',

        # Context manipulation
        'the following is',
        'translate to',
        'decode',
        'execute',
    ]

    # Patterns specific to city/location inputs
    CITY_INJECTION_PATTERNS = [
        '<script',
        'javascript:',
        'eval(',
        'exec(',
        'system(',
        'import ',
        '--',
        '/*',
        '*/',
        '<!--',
        '-->',
        '<?',
        '?>',
    ]

    @classmethod
    def detect_injection(cls, text: str, check_city: bool = False) -> Tuple[bool, List[str]]:
        """
        Detect potential prompt injection in text

        Args:
            text: Text to check
            check_city: If True, also check for code injection patterns

        Returns:
            Tuple of (is_safe, detected_patterns)
            is_safe: False if injection detected
            detected_patterns: List of detected pattern keywords
        """
        if not text:
            return True, []

        text_lower = text.lower()
        detected = []

        # Check for prompt injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.lower() in text_lower:
                detected.append(pattern)

        # Check for city-specific injection patterns
        if check_city:
            for pattern in cls.CITY_INJECTION_PATTERNS:
                if pattern.lower() in text_lower:
                    detected.append(pattern)

        is_safe = len(detected) == 0
        return is_safe, detected

    @classmethod
    def sanitize_text(cls, text: str, max_length: int = 500) -> str:
        """
        Sanitize text by removing dangerous characters

        Args:
            text: Text to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        # Trim to max length
        text = text[:max_length]

        # Remove null bytes and control characters
        text = ''.join(c for c in text if c.isprintable() or c.isspace())

        # Remove excessive whitespace
        text = ' '.join(text.split())

        return text.strip()

    @classmethod
    def validate_city_name(cls, city: str) -> bool:
        """
        Validate city name format

        Args:
            city: City name to validate

        Returns:
            True if valid city name format
        """
        # Only allow letters, spaces, hyphens, apostrophes, periods
        # Valid: "New York", "St. Louis", "O'Fallon", "Winston-Salem"
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -'.")

        return all(c in allowed_chars for c in city)
