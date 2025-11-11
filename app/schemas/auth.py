"""Authentication schemas for requests and responses"""
from pydantic import BaseModel, EmailStr, Field, validator
import re


class UserRegisterRequest(BaseModel):
    """User registration request schema"""
    name: str = Field(..., min_length=1, max_length=255, description="User's full name")
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, max_length=100, description="User's password (min 8 characters)")

    @validator('password')
    def validate_password_strength(cls, v):
        """Ensure password has minimum security requirements"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserLoginRequest(BaseModel):
    """User login request schema"""
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")


class TokenResponse(BaseModel):
    """JWT token response schema"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type (always 'bearer')")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class UserResponse(BaseModel):
    """User data response schema"""
    id: str
    name: str
    email: str
    created_at: str

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    """Complete authentication response with token and user data"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse
