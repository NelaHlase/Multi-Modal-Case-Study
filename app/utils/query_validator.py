from typing import Optional, Union
from PIL import Image


def validate_query(query: str) -> Optional[str]:
    """
    Validate and sanitize search query
    Args:
        query: Input search query
    Returns:
        Sanitized query or None if invalid
    Raises:
        ValueError: If query is invalid
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    sanitized = query.strip()
    if len(sanitized) > 500:
        raise ValueError("Query exceeds maximum length of 500 characters")
    return sanitized