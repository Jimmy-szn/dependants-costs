import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.admin_service import get_ai_results, get_overview
from app.auth import authenticate_admin, require_admin

router = APIRouter(prefix="/admin", tags=["admin"])


class AdminLoginRequest(BaseModel):
    email: str
    password: str


def _secure_cookie(request: Request) -> bool:
    return (
        request.url.scheme == "https"
        or os.getenv("ADMIN_COOKIE_SECURE", "").lower() == "true"
    )


@router.get("", include_in_schema=False)
def admin_page():
    page = Path(__file__).resolve().parent.parent / "demo" / "admin.html"
    return FileResponse(page)


@router.post("/login")
def login(body: AdminLoginRequest, request: Request, response: Response):
    access_token, expires_in = authenticate_admin(body.email, body.password)
    response.set_cookie(
        key="admin_access_token",
        value=access_token,
        max_age=expires_in,
        httponly=True,
        secure=_secure_cookie(request),
        samesite="lax",
        path="/admin",
    )
    return {"status": "ok"}


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response):
    response.delete_cookie(
        key="admin_access_token",
        httponly=True,
        secure=_secure_cookie(request),
        samesite="lax",
        path="/admin",
    )


@router.get("/overview")
def overview(admin_id: str = Depends(require_admin)):
    try:
        overview_data = get_overview()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        ai_results_data = get_ai_results()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"overview": overview_data, "ai_results": ai_results_data}
