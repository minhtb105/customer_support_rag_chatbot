"""Auth router — /v1/auth/*  (register, login httpOnly cookie, refresh, me, logout, admin user mgmt)"""
from __future__ import annotations
import re
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Response, Request, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

try:
    from src.config import ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS, COOKIE_SECURE, COOKIE_SAMESITE, VALID_ROLES, EXPERT_ROLES
    from src.auth.schemas import RegisterRequest, LoginRequest, TokenResponse, UserOut, UserUpdateRequest, RefreshRequest
    from src.auth.security import hash_password, verify_password, create_access_token, create_refresh_token, decode_token
    from src.auth.db import (
        get_user_by_username, get_user_by_email, get_user_by_id, create_user, update_user, delete_user, list_users,
        store_refresh_token, get_refresh_token, revoke_refresh_token, revoke_all_user_tokens,
        create_notification, list_notifications, count_unread, mark_notification_read, mark_all_read
    )
    from src.auth.dependencies import get_current_user, require_admin
except ImportError:
    from config import ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS, COOKIE_SECURE, COOKIE_SAMESITE, VALID_ROLES, EXPERT_ROLES  # type: ignore
    from auth.schemas import RegisterRequest, LoginRequest, TokenResponse, UserOut, UserUpdateRequest, RefreshRequest  # type: ignore
    from auth.security import hash_password, verify_password, create_access_token, create_refresh_token, decode_token  # type: ignore
    from auth.db import (  # type: ignore
        get_user_by_username, get_user_by_email, get_user_by_id, create_user, update_user, delete_user, list_users,
        store_refresh_token, get_refresh_token, revoke_refresh_token, revoke_all_user_tokens,
        create_notification, list_notifications, count_unread, mark_notification_read, mark_all_read
    )
    from auth.dependencies import get_current_user, require_admin  # type: ignore

router = APIRouter(prefix="/v1/auth", tags=["auth"])

def _to_user_out(u: dict) -> dict:
    return {
        "id": u["id"],
        "username": u["username"],
        "email": u.get("email"),
        "full_name": u.get("full_name"),
        "role": u["role"],
        "is_active": bool(u.get("is_active", 1)),
        "is_verified": bool(u.get("is_verified", 1)),
        "created_at": u.get("created_at"),
        "last_login": u.get("last_login"),
    }

def _set_auth_cookies(response: Response, access_token: str, refresh_token: str):
    # access_token httpOnly
    max_age_access = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    max_age_refresh = REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600
    # Chrome requires Secure when SameSite=None
    samesite = COOKIE_SAMESITE
    secure = COOKIE_SECURE
    # Nếu SameSite=None nhưng không Secure thì fallback lax
    if samesite == "none" and not secure:
        samesite = "lax"
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=secure, samesite=samesite, max_age=max_age_access, path="/")
    response.set_cookie(key="refresh_token", value=refresh_token, httponly=True, secure=secure, samesite=samesite, max_age=max_age_refresh, path="/")

def _clear_auth_cookies(response: Response):
    response.delete_cookie(key="access_token", path="/")
    response.delete_cookie(key="refresh_token", path="/")

@router.post("/register", response_model=UserOut, status_code=201)
def register(payload: RegisterRequest, request: Request, current_user: Optional[dict] = Depends(lambda: None)):
    # Try to get current user optionally for role elevation
    # If no token, only allow role=user
    # If token present and admin, allow any role
    # We manually parse token instead of Depends(get_current_user) to allow anonymous register
    from fastapi.security import HTTPBearer
    token = None
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header[7:]
    else:
        token = request.cookies.get("access_token")
        if token and token.startswith("Bearer "):
            token = token[7:]
    requester = None
    if token:
        pl = decode_token(token)
        if pl and pl.get("sub"):
            requester = get_user_by_id(pl["sub"])

    desired_role = payload.role or "user"
    if desired_role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role {desired_role}")
    # Non-admin cannot self-register as expert/admin
    if desired_role in EXPERT_ROLES or desired_role == "admin":
        if not requester or requester.get("role") != "admin":
            # For expert self-register, we allow but mark is_verified=False and require admin approval
            # For admin, forbid
            if desired_role == "admin":
                raise HTTPException(status_code=403, detail="Only admin can create admin accounts")
            # else expert: allow but unverified
            is_verified = False
        else:
            is_verified = True
            # admin creating expert -> verified directly
            if desired_role in EXPERT_ROLES:
                is_verified = True
    else:
        is_verified = True
        desired_role = "user"

    if get_user_by_username(payload.username):
        raise HTTPException(status_code=409, detail="Username already exists")
    if payload.email and get_user_by_email(payload.email):
        raise HTTPException(status_code=409, detail="Email already exists")

    hashed = hash_password(payload.password)
    user = create_user(
        username=payload.username,
        email=payload.email,
        hashed_password=hashed,
        full_name=payload.full_name,
        role=desired_role,
        is_verified=is_verified,
    )
    # If expert unverified, notify admins
    if not is_verified:
        # notify all admins
        for admin in list_users(limit=100, role="admin"):
            create_notification(admin["id"], type="expert_pending", title=f"Expert pending verification: {payload.username}", body=f"User {payload.username} registered as {desired_role}, needs admin verification", review_id=None)
    return _to_user_out(user)

@router.post("/login")
def login(response: Response, form: OAuth2PasswordRequestForm = Depends()):
    # form.username / form.password
    user = get_user_by_username(form.username)
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    if not user.get("is_active"):
        raise HTTPException(status_code=403, detail="Account deactivated")
    if user.get("role") in EXPERT_ROLES and not user.get("is_verified"):
        raise HTTPException(status_code=403, detail="Expert account pending admin verification")
    # update last_login
    update_user(user["id"], last_login=datetime.utcnow().isoformat())
    # create tokens
    access_token = create_access_token({"sub": user["id"], "role": user["role"], "username": user["username"]})
    refresh_token = create_refresh_token({"sub": user["id"]})
    # store refresh token
    expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    store_refresh_token(refresh_token, user["id"], expires_at)
    _set_auth_cookies(response, access_token, refresh_token)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": _to_user_out(get_user_by_id(user["id"])),
    }

@router.post("/login-json")
def login_json(payload: LoginRequest, response: Response):
    user = get_user_by_username(payload.username)
    if not user or not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    if not user.get("is_active"):
        raise HTTPException(status_code=403, detail="Account deactivated")
    if user.get("role") in EXPERT_ROLES and not user.get("is_verified"):
        raise HTTPException(status_code=403, detail="Expert account pending admin verification")
    update_user(user["id"], last_login=datetime.utcnow().isoformat())
    access_token = create_access_token({"sub": user["id"], "role": user["role"], "username": user["username"]})
    refresh_token = create_refresh_token({"sub": user["id"]})
    expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    store_refresh_token(refresh_token, user["id"], expires_at)
    _set_auth_cookies(response, access_token, refresh_token)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": _to_user_out(get_user_by_id(user["id"])),
    }

@router.post("/refresh")
def refresh_token_endpoint(request: Request, response: Response, payload: Optional[RefreshRequest] = None):
    # try cookie first, then body
    token = request.cookies.get("refresh_token")
    if not token and payload and payload.refresh_token:
        token = payload.refresh_token
    # also check header? no
    if not token:
        raise HTTPException(status_code=401, detail="Missing refresh_token")
    pl = decode_token(token)
    if not pl or pl.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    stored = get_refresh_token(token)
    if not stored or stored.get("revoked"):
        raise HTTPException(status_code=401, detail="Refresh token revoked")
    # check expiry
    try:
        exp = datetime.fromisoformat(stored["expires_at"])
        if exp < datetime.utcnow():
            revoke_refresh_token(token)
            raise HTTPException(status_code=401, detail="Refresh token expired")
    except Exception:
        pass
    user = get_user_by_id(pl["sub"])
    if not user or not user.get("is_active"):
        raise HTTPException(status_code=401, detail="User not found or deactivated")
    # rotate: revoke old, issue new
    revoke_refresh_token(token)
    new_access = create_access_token({"sub": user["id"], "role": user["role"], "username": user["username"]})
    new_refresh = create_refresh_token({"sub": user["id"]})
    expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    store_refresh_token(new_refresh, user["id"], expires_at)
    _set_auth_cookies(response, new_access, new_refresh)
    return {"access_token": new_access, "refresh_token": new_refresh, "token_type": "bearer", "user": _to_user_out(user)}

@router.post("/logout")
def logout(request: Request, response: Response, current_user=Depends(get_current_user)):
    token = request.cookies.get("refresh_token")
    if token:
        revoke_refresh_token(token)
    # also revoke access? we rely on short expiry; optionally revoke all
    _clear_auth_cookies(response)
    return {"msg": "Logged out"}

@router.get("/me", response_model=UserOut)
def me(current_user=Depends(get_current_user)):
    return _to_user_out(current_user)

# ---------- Admin user mgmt ----------
@router.get("/users", dependencies=[Depends(require_admin)])
def admin_list_users(limit: int = 100, offset: int = 0, role: Optional[str] = None):
    users = list_users(limit=limit, offset=offset, role=role)
    return [_to_user_out(u) for u in users]

@router.get("/users/{user_id}", dependencies=[Depends(require_admin)])
def admin_get_user(user_id: str):
    u = get_user_by_id(user_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_user_out(u)

@router.patch("/users/{user_id}", dependencies=[Depends(require_admin)])
def admin_update_user(user_id: str, payload: UserUpdateRequest):
    u = get_user_by_id(user_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    updates = {}
    if payload.email is not None:
        updates["email"] = payload.email
    if payload.full_name is not None:
        updates["full_name"] = payload.full_name
    if payload.role is not None:
        if payload.role not in VALID_ROLES:
            raise HTTPException(status_code=400, detail="Invalid role")
        updates["role"] = payload.role
    if payload.is_active is not None:
        updates["is_active"] = 1 if payload.is_active else 0
    if payload.is_verified is not None:
        updates["is_verified"] = 1 if payload.is_verified else 0
    if updates:
        u = update_user(user_id, **updates)
    # if verifying expert, notify
    if payload.is_verified is True and u:
        create_notification(u["id"], type="expert_verified", title="Account verified", body=f"Your {u['role']} account has been verified by admin", review_id=None)
    return _to_user_out(u)  # type: ignore

@router.post("/users/{user_id}/verify", dependencies=[Depends(require_admin)])
def admin_verify_expert(user_id: str):
    u = get_user_by_id(user_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    if u["role"] not in EXPERT_ROLES:
        raise HTTPException(status_code=400, detail="User is not expert role")
    u = update_user(user_id, is_verified=1)
    create_notification(user_id, type="expert_verified", title="Account verified", body="Your expert account has been verified", review_id=None)
    return _to_user_out(u)  # type: ignore

@router.delete("/users/{user_id}", dependencies=[Depends(require_admin)])
def admin_delete_user(user_id: str):
    if not get_user_by_id(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    delete_user(user_id)
    return {"msg": "Deleted"}

# ---------- Notifications ----------
@router.get("/notifications")
def get_notifications(request: Request, current_user=Depends(get_current_user), limit: int = 50, unread_only: bool = False):
    notifs = list_notifications(current_user["id"], limit=limit, unread_only=unread_only)
    # normalize is_read bool
    for n in notifs:
        n["is_read"] = bool(n["is_read"])
    return {"notifications": notifs, "unread_count": count_unread(current_user["id"])}

@router.post("/notifications/{notif_id}/read")
def read_notification(notif_id: str, current_user=Depends(get_current_user)):
    ok = mark_notification_read(notif_id, current_user["id"])
    if not ok:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"msg": "Marked read"}

@router.post("/notifications/read-all")
def read_all(current_user=Depends(get_current_user)):
    mark_all_read(current_user["id"])
    return {"msg": "All marked read"}
