"""Authentication middleware for protected routes"""
from fastapi import HTTPException, Header, Depends
from typing import Optional, Dict, Any
from app.utils.auth import decode_access_token
from app.utils.database import get_user_by_id


async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Dependency to get the current authenticated user from JWT token

    Args:
        authorization: Authorization header with Bearer token

    Returns:
        Current user data

    Raises:
        HTTPException: If token is missing, invalid, or user not found
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Unauthorized",
                "message": "Missing authentication token",
                "details": {}
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Unauthorized",
                "message": "Invalid authentication token format. Expected 'Bearer <token>'",
                "details": {}
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = parts[1]

    # Decode and verify token
    payload = decode_access_token(token)
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
