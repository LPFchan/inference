"""Desktop client download endpoints (/downloads/latest.json, /downloads/{filename})."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from grimoire.auth import require_api

router = APIRouter()

DOWNLOADS_DIR = Path(os.environ.get("GRIMOIRE_DOWNLOADS_DIR", "/var/lib/grimoire/downloads"))
ALLOWED_EXTENSIONS = (".dmg", ".exe", ".flatpak", ".tar.gz", ".txt")


def _safe_file(filename: str) -> Path:
    if not filename or "/" in filename or "\\" in filename or ".." in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not filename.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=404, detail="File not found")
    base = DOWNLOADS_DIR.resolve()
    target = (base / filename).resolve()
    if target.parent != base or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return target


@router.get("/downloads/latest.json")
async def downloads_latest(request: Request):
    """Return metadata for the currently hosted desktop app installers."""
    require_api(request)
    manifest_path = DOWNLOADS_DIR / "latest.json"
    if not manifest_path.is_file():
        return {"version": None, "macos": None, "windows": None, "linux": None}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": None, "macos": None, "windows": None, "linux": None}
    if not isinstance(data, dict):
        return {"version": None, "macos": None, "windows": None, "linux": None}
    return data


@router.get("/downloads/{filename}")
async def downloads_file(request: Request, filename: str):
    """Stream an installer artifact stored in /var/lib/grimoire/downloads."""
    require_api(request)
    target = _safe_file(filename)
    return FileResponse(
        path=target,
        filename=target.name,
        media_type="application/octet-stream",
        headers={"Cache-Control": "private, max-age=3600"},
    )
