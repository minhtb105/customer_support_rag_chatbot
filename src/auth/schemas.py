"""Pydantic schemas for auth"""
from __future__ import annotations
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, EmailStr

Role = Literal["user", "doctor", "pharmacist", "specialist", "admin"]

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_\-\.]+$")
    email: Optional[str] = Field(None, description="optional")
    password: str = Field(..., min_length=6, max_length=100)
    full_name: Optional[str] = None
    role: Optional[Role] = Field(None, description="Chỉ admin được tạo expert/admin; user thường mặc định user")

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserOut"

class UserOut(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str
    is_active: bool
    is_verified: bool
    created_at: str
    last_login: Optional[str] = None

class UserUpdateRequest(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[Role] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None

class RefreshRequest(BaseModel):
    refresh_token: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=6)

class NotificationOut(BaseModel):
    id: str
    user_id: str
    type: str
    title: str
    body: Optional[str] = None
    review_id: Optional[str] = None
    is_read: bool
    created_at: str

TokenResponse.model_rebuild()
