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
    # If no cookie, check Authorization header as fallback (for backward compatibility during migration)
    if not access_token:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            access_token = auth_header.split(" ")[1]

    if not access_token:
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
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Unauthorized",
                "message": "User not found",
                "details": {}
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

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
