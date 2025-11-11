"""User database model"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    """User model matching Supabase users table schema"""
    id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    password_hash: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserInDB(BaseModel):
    """User model as stored in database"""
    id: str
    name: str
    email: str
    password_hash: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
