"""FastAPI dependencies — get_current_user, require_role"""
from __future__ import annotations
from typing import Optional, List
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

try:
    from src.auth.security import decode_token
    from src.auth.db import get_user_by_id
except ImportError:
    from auth.security import decode_token  # type: ignore
    from auth.db import get_user_by_id  # type: ignore

# HTTPAuthorizationCredentials — auto_error=False để đọc cookie fallback
bearer_scheme = HTTPBearer(auto_error=False)

def _extract_token(request: Request, credentials: Optional[HTTPAuthorizationCredentials]) -> Optional[str]:
    # 1. Authorization header
    if credentials and credentials.credentials:
        return credentials.credentials
    # 2. httpOnly cookie
    token = request.cookies.get("access_token")
    if token:
        # cookie may be "Bearer xxx" or just xxx
        if token.startswith("Bearer "):
            return token[7:]
        return token
    # 3. cookie named 'access_token' without Bearer
    return None

def get_current_user(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    token = _extract_token(request, credentials)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated: missing token")
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if not user.get("is_active"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User deactivated")
    # expert chưa verify thì chặn (trừ user/admin)
    if user.get("role") in ("doctor", "pharmacist", "specialist") and not user.get("is_verified"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Expert account not verified by admin yet")
    return user

def get_current_user_optional(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    try:
        return get_current_user(request, credentials)
    except HTTPException:
        return None

def require_role(*roles: str):
    """Dependency factory: require one of roles"""
    def _checker(user=Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Requires role {roles}, your role is {user['role']}")
        return user
    return _checker

def require_expert(user=Depends(get_current_user)):
    if user["role"] not in ("doctor", "pharmacist", "specialist", "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Expert role required")
    return user

def require_admin(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user

def enforce_user_ownership(path_user_id: str, current_user: dict):
    """Helper: user chỉ được xem own, expert/admin được xem all"""
    if current_user["role"] in ("admin", "doctor", "pharmacist", "specialist"):
        return
    if path_user_id != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden: can only access your own data")
