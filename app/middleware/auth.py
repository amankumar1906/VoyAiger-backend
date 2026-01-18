"""Authentication middleware for protected routes"""
from fastapi import HTTPException, Cookie, Depends, Request
from typing import Optional, Dict, Any
from app.utils.auth import decode_access_token
from app.utils.database import get_user_by_id


async def get_current_user(request: Request, access_token: Optional[str] = Cookie(None)) -> Dict[str, Any]:
    """
    Dependency to get the current authenticated user from JWT cookie

    Args:
        request: FastAPI request object
        access_token: JWT token from HttpOnly cookie

    Returns:
        Current user data

    Raises:
        HTTPException: If token is missing, invalid, or user not found
    """
    import logging
    logger = logging.getLogger(__name__)

    # Log authentication attempt with request details
    user_agent = request.headers.get("user-agent", "unknown")
    origin = request.headers.get("origin", "unknown")
    cookie_header = request.headers.get("cookie", "")
    referer = request.headers.get("referer", "unknown")

    logger.info(f"Auth attempt - Path: {request.url.path}, Origin: {origin}, Referer: {referer}")
    logger.info(f"User-Agent: {user_agent}")
    logger.info(f"Cookie header present: {bool(cookie_header)}, Access token from cookie: {bool(access_token)}")
    if cookie_header:
        # Log cookie names (not values for security)
        cookie_names = [c.split('=')[0].strip() for c in cookie_header.split(';')]
        logger.info(f"Cookies received: {cookie_names}")
    else:
        logger.warning(f"NO COOKIES RECEIVED - This is the mobile Chrome bug!")

    # If no cookie, check Authorization header as fallback (for backward compatibility during migration)
    if not access_token:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            access_token = auth_header.split(" ")[1]
            logger.info("Using token from Authorization header as fallback")

    if not access_token:
        logger.warning(f"Missing authentication token - Path: {request.url.path}, Origin: {origin}")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Unauthorized",
                "message": "Missing authentication token",
                "details": {}
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Decode and verify token
    payload = decode_access_token(access_token)
    if not payload:
        logger.warning(f"Invalid or expired token - Path: {request.url.path}, Origin: {origin}")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Unauthorized",
                "message": "Invalid or expired authentication token",
                "details": {}
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Get user from database
    user_id = payload.get("sub")
    if not user_id:
        logger.warning(f"Invalid token payload (missing sub) - Path: {request.url.path}")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Unauthorized",
                "message": "Invalid token payload",
                "details": {}
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = await get_user_by_id(user_id)
    if not user:
        logger.warning(f"User not found for ID: {user_id} - Path: {request.url.path}")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Unauthorized",
                "message": "User not found",
                "details": {}
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    logger.info(f"Authentication successful for user {user.get('email', 'unknown')} - Path: {request.url.path}")
    return user


def require_auth(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Dependency shorthand for requiring authentication

    Args:
        user: Current user from get_current_user dependency

    Returns:
        Current user data
    """
    return user
