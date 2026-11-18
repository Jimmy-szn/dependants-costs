import os

from dotenv import load_dotenv
from fastapi import HTTPException, Request, status
from supabase import create_client

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

if not supabase_url or not supabase_anon_key:
    raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be configured")

supabase_auth = create_client(supabase_url, supabase_anon_key)


def _admin_ids() -> set[str]:
    return {
        value.strip()
        for value in os.getenv("ADMIN_USER_IDS", "").split(",")
        if value.strip()
    }


def authenticate_admin(email: str, password: str) -> tuple[str, int]:
    admin_ids = _admin_ids()
    if not admin_ids:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin access is not configured",
        )

    try:
        login_client = create_client(supabase_url, supabase_anon_key)
        result = login_client.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        ) from exc

    user = result.user if result else None
    session = result.session if result else None
    if not user or not user.id or not session or not session.access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to create an authenticated session",
        )

    if str(user.id) not in admin_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    return session.access_token, int(session.expires_in or 3600)


def require_admin(request: Request) -> str:
    admin_ids = _admin_ids()
    if not admin_ids:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin access is not configured",
        )

    access_token = request.cookies.get("admin_access_token")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        result = supabase_auth.auth.get_user(access_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired admin session",
        ) from exc

    user = result.user if result else None
    if not user or not user.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired admin session",
        )

    if str(user.id) not in admin_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    return str(user.id)
