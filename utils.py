"""
utils.py — Shared utility helpers used across the application.
"""

from __future__ import annotations

import os
import re
import logging
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from flask import request

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename / path safety
# ---------------------------------------------------------------------------

_SAFE_FILENAME_RE = re.compile(r"[^\w\s\-.]")


def secure_name(filename: str) -> str:
    """Sanitise an upload filename, preserving extension.

    Strips path traversal, special chars, and limits length.
    """
    filename = os.path.basename(filename)
    stem, _, ext = filename.rpartition(".")
    stem = _SAFE_FILENAME_RE.sub("", stem).strip()[:64] or "file"
    ext  = ext.lower()[:8]
    return f"{stem}.{ext}"


def allowed_image(filename: str) -> bool:
    """Return True if the filename has an allowed image extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in config.ALLOWED_IMAGE_EXTENSIONS


def timestamped_filename(prefix: str, ext: str) -> str:
    """Return a timestamped filename like 'unknown_20240415_143022.mp4'."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


# ---------------------------------------------------------------------------
# IP address (respects Cloudflare / proxy headers)
# ---------------------------------------------------------------------------

def get_client_ip() -> str:
    """Return the real client IP, honouring CF-Connecting-IP and X-Forwarded-For.

    These headers are only trusted when a trusted proxy is actually configured
    (PROXY_COUNT > 0). When the app is reachable directly, they are fully
    attacker-controlled — trusting them would let a client present a fresh
    fake IP on every request to evade the per-IP login lockout and to forge
    audit-log entries. In that case we fall back to the real socket peer.
    """
    if getattr(config, "PROXY_COUNT", 0) > 0:
        # Cloudflare always sets CF-Connecting-IP
        cf_ip = request.headers.get("CF-Connecting-IP")
        if cf_ip:
            return cf_ip.strip()
        xff = request.headers.get("X-Forwarded-For", "")
        if xff:
            # Take the first IP in the chain (original client)
            return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


# ---------------------------------------------------------------------------
# Audit log helper (thin wrapper so callers don't import database directly)
# ---------------------------------------------------------------------------

def audit(
    action: str,
    user_id: int | None = None,
    username: str = "",
    target_type: str = "",
    target_id: str = "",
    detail: str = "",
    ip_address: str = "",
) -> None:
    """Write an entry to the audit log. Fails silently to avoid crashing routes."""
    try:
        from database import get_db
        import models as m
        db = get_db()
        m.add_audit(
            db,
            action=action,
            user_id=user_id,
            username=username,
            target_type=target_type,
            target_id=str(target_id),
            detail=detail,
            ip_address=ip_address,
        )
    except Exception as exc:
        logger.warning("audit() failed: %s", exc)


# ---------------------------------------------------------------------------
# Storage usage
# ---------------------------------------------------------------------------

def dir_size_mb(path: str) -> float:
    """Return total size of a directory in megabytes."""
    total = 0
    try:
        for p in Path(path).rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except OSError:
        pass
    return round(total / (1024 * 1024), 2)


def disk_usage_percent(path: str = ".") -> float:
    """Return disk usage percentage for the filesystem containing `path`."""
    try:
        usage = shutil.disk_usage(Path(path).resolve())
        return round(usage.used / usage.total * 100, 1) if usage.total else 0.0
    except (OSError, RuntimeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

def utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_dt(dt_str: str) -> str:
    """Convert ISO 8601 UTC string to a human-readable local-ish representation."""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, TypeError):
        return dt_str or ""


# ---------------------------------------------------------------------------
# Hashing (for etag / dedup purposes, not passwords)
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()
