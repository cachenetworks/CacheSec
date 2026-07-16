"""
config.py — Centralised configuration loaded from environment / .env file.

All other modules import from here. Never hardcode secrets elsewhere.
"""

import os
import sys
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (same directory as this file)
_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env")


def _bool(key: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes")


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
FLASK_ENV: str = os.getenv("FLASK_ENV", "production")
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = _int("PORT", 5000)
DEBUG: bool = FLASK_ENV == "development"

# ---------------------------------------------------------------------------
# Session signing key.
#
# Flask session cookies are *signed* (not encrypted) with SECRET_KEY. Anyone
# who knows this value can forge a cookie for any user — including one with
# role=admin — and bypass login entirely. The old default was the public
# literal "CHANGE_ME_IN_PRODUCTION", which means anyone who has ever seen this
# source code could forge an admin session. We now refuse to run in production
# with a missing/placeholder key, and auto-generate an ephemeral one in dev.
# ---------------------------------------------------------------------------
_INSECURE_SECRET_KEYS = {"", "CHANGE_ME_IN_PRODUCTION", "changeme", "secret"}
SECRET_KEY: str = os.getenv("SECRET_KEY", "").strip()
if SECRET_KEY in _INSECURE_SECRET_KEYS:
    if DEBUG:
        SECRET_KEY = secrets.token_urlsafe(48)
        print(
            "WARNING: SECRET_KEY is unset — generated a random ephemeral key for "
            "development. Sessions will be invalidated on every restart. Set "
            "SECRET_KEY in your .env for a stable key.",
            file=sys.stderr,
        )
    else:
        raise RuntimeError(
            "SECRET_KEY is unset or left at an insecure default. Refusing to "
            "start: a known signing key lets anyone forge an admin session "
            "cookie. Generate a strong key with:\n"
            "    python -c \"import secrets; print(secrets.token_urlsafe(48))\"\n"
            "and set it as SECRET_KEY in your .env (or FLASK_ENV=development for "
            "local testing)."
        )

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(_BASE_DIR / "cachesec.db"))

# ---------------------------------------------------------------------------
# Face recognition
# ---------------------------------------------------------------------------
RECOGNITION_THRESHOLD: float = _float("RECOGNITION_THRESHOLD", 0.4)
FRAME_SKIP: int = _int("FRAME_SKIP", 3)
CAMERA_INDEX: int = _int("CAMERA_INDEX", 0)
FRAME_WIDTH: int = _int("FRAME_WIDTH", 640)
FRAME_HEIGHT: int = _int("FRAME_HEIGHT", 480)
UNKNOWN_COOLDOWN_SECONDS: int = _int("UNKNOWN_COOLDOWN_SECONDS", 10)
NIGHT_VISION_MODE: str = os.getenv("NIGHT_VISION_MODE", "auto").strip().lower()
if NIGHT_VISION_MODE not in {"auto", "force_off"}:
    NIGHT_VISION_MODE = "auto"
CAMERA_PREFERRED_SOURCE: str = os.getenv("CAMERA_PREFERRED_SOURCE", "none").strip().lower()
if CAMERA_PREFERRED_SOURCE not in {"webcam", "kinect", "ip", "tapo", "none"}:
    CAMERA_PREFERRED_SOURCE = "none"
USB_CAMERA_INDICES: str = os.getenv("USB_CAMERA_INDICES", "").strip()
USB_CAMERA_AUTO_DISCOVER: bool = _bool("USB_CAMERA_AUTO_DISCOVER", False)
USB_CAMERA_SCAN_LIMIT: int = _int("USB_CAMERA_SCAN_LIMIT", 0)
MULTI_CAMERA_DETECTION_ENABLED: bool = _bool("MULTI_CAMERA_DETECTION_ENABLED", False)
IP_CAMERA_URL: str = os.getenv("IP_CAMERA_URL", "").strip()
IP_CAMERA_URLS: str = os.getenv("IP_CAMERA_URLS", "").strip()
IP_CAMERA_RTSP_TRANSPORT: str = os.getenv("IP_CAMERA_RTSP_TRANSPORT", "tcp").strip().lower()
if IP_CAMERA_RTSP_TRANSPORT not in {"tcp", "udp", "udp_multicast", "http"}:
    IP_CAMERA_RTSP_TRANSPORT = "tcp"
IP_CAMERA_ONVIF_NIGHT_MODE: str = os.getenv("IP_CAMERA_ONVIF_NIGHT_MODE", "disabled").strip().lower()
if IP_CAMERA_ONVIF_NIGHT_MODE not in {"disabled", "detect", "force_off"}:
    IP_CAMERA_ONVIF_NIGHT_MODE = "disabled"
IP_CAMERA_ONVIF_HOST: str = os.getenv("IP_CAMERA_ONVIF_HOST", "").strip()
IP_CAMERA_ONVIF_PORT: int = _int("IP_CAMERA_ONVIF_PORT", 0)
IP_CAMERA_ONVIF_USERNAME: str = os.getenv("IP_CAMERA_ONVIF_USERNAME", "").strip()
IP_CAMERA_ONVIF_PASSWORD: str = os.getenv("IP_CAMERA_ONVIF_PASSWORD", "")
IP_CAMERA_ONVIF_WSDL_DIR: str = os.getenv("IP_CAMERA_ONVIF_WSDL_DIR", "").strip()
PERSON_DETECTION_BACKEND: str = os.getenv("PERSON_DETECTION_BACKEND", "disabled").strip().lower()
if PERSON_DETECTION_BACKEND not in {"disabled", "detectron2"}:
    PERSON_DETECTION_BACKEND = "disabled"
PERSON_DETECTION_THRESHOLD: float = _float("PERSON_DETECTION_THRESHOLD", 0.75)
PERSON_DETECTION_DEVICE: str = os.getenv("PERSON_DETECTION_DEVICE", "auto").strip().lower()
if PERSON_DETECTION_DEVICE not in {"auto", "cpu", "cuda"}:
    PERSON_DETECTION_DEVICE = "auto"
OBJECT_DETECTION_BACKEND: str = os.getenv("OBJECT_DETECTION_BACKEND", PERSON_DETECTION_BACKEND).strip().lower()
if OBJECT_DETECTION_BACKEND not in {"disabled", "detectron2"}:
    OBJECT_DETECTION_BACKEND = "disabled"
OBJECT_DETECTION_MODE: str = os.getenv("OBJECT_DETECTION_MODE", "people_pets").strip().lower()
if OBJECT_DETECTION_MODE not in {"person", "people_pets", "all"}:
    OBJECT_DETECTION_MODE = "people_pets"
OBJECT_DETECTION_THRESHOLD: float = _float("OBJECT_DETECTION_THRESHOLD", PERSON_DETECTION_THRESHOLD)
OBJECT_DETECTION_DEVICE: str = os.getenv("OBJECT_DETECTION_DEVICE", PERSON_DETECTION_DEVICE).strip().lower()
if OBJECT_DETECTION_DEVICE not in {"auto", "cpu", "cuda"}:
    OBJECT_DETECTION_DEVICE = "auto"
MOVING_OBJECT_DETECTION_ENABLED: bool = _bool("MOVING_OBJECT_DETECTION_ENABLED", False)
MOVING_OBJECT_MIN_AREA: int = _int("MOVING_OBJECT_MIN_AREA", 2500)
MOVING_OBJECT_THRESHOLD: int = _int("MOVING_OBJECT_THRESHOLD", 25)

# ---------------------------------------------------------------------------
# TP-Link Tapo cameras (RTSP only)
# ---------------------------------------------------------------------------
# Tapo C-series cameras are used as a regular RTSP source. The username and
# password are the "Camera Account" set in the Tapo app under
# Advanced Settings → Camera Account (NOT the cloud login).
TAPO_HOST: str = os.getenv("TAPO_HOST", "").strip()
TAPO_USERNAME: str = os.getenv("TAPO_USERNAME", "admin").strip()
TAPO_PASSWORD: str = os.getenv("TAPO_PASSWORD", "")
TAPO_STREAM: str = os.getenv("TAPO_STREAM", "stream1").strip().lower()
if TAPO_STREAM not in {"stream1", "stream2"}:
    TAPO_STREAM = "stream1"

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
RECORDINGS_DIR: str = os.getenv("RECORDINGS_DIR", str(_BASE_DIR / "recordings"))
SNAPSHOTS_DIR: str = os.getenv("SNAPSHOTS_DIR", str(_BASE_DIR / "snapshots"))
MIN_RECORDING_SECONDS: int = _int("MIN_RECORDING_SECONDS", 15)
MAX_RECORDING_SECONDS: int = _int("MAX_RECORDING_SECONDS", 5400)  # 1h30m
SAVE_RECORDINGS_LOCALLY: bool = _bool("SAVE_RECORDINGS_LOCALLY", True)
RECORD_AUDIO_ENABLED: bool = _bool("RECORD_AUDIO_ENABLED", False)
RECORD_AUDIO_DEVICE: str = os.getenv("RECORD_AUDIO_DEVICE", "auto").strip() or "auto"
VIDEO_ENCODER: str = os.getenv("VIDEO_ENCODER", "auto").strip().lower()
if VIDEO_ENCODER not in {"auto", "libx264", "h264_nvenc", "hevc_nvenc", "h264_qsv"}:
    VIDEO_ENCODER = "auto"
VIDEO_ENCODER_PRESET: str = os.getenv("VIDEO_ENCODER_PRESET", "fast").strip() or "fast"
VIDEO_ENCODER_QUALITY: int = _int("VIDEO_ENCODER_QUALITY", 26)

# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------
DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
DISCORD_COOLDOWN_SECONDS: int = _int("DISCORD_COOLDOWN_SECONDS", 60)
DISCORD_MENTION_EVERYONE: bool = _bool("DISCORD_MENTION_EVERYONE", False)

# ---------------------------------------------------------------------------
# Kinect
# ---------------------------------------------------------------------------
KINECT_ENABLED: bool = _bool("KINECT_ENABLED", True)   # auto-detect on startup
KINECT_TILT:    int  = _int("KINECT_TILT", 0)          # motor tilt degrees (-27 to +27)
# Keep this off for SLS: a separate motor/LED handle can block libfreenect's
# sync video/depth stream on some hosts.
KINECT_MOTOR_ENABLED: bool = _bool("KINECT_MOTOR_ENABLED", False)
KINECT_NIGHT_VISION_ENABLED: bool = _bool("KINECT_NIGHT_VISION_ENABLED", True)

# ---------------------------------------------------------------------------
# SLS / skeleton overlay
# ---------------------------------------------------------------------------
SLS_ENABLED: bool = _bool("SLS_ENABLED", True)
SLS_MODE: str = os.getenv("SLS_MODE", "night").strip().lower()
if SLS_MODE not in {"night", "always"}:
    SLS_MODE = "night"
SLS_MAX_PEOPLE: int = _int("SLS_MAX_PEOPLE", 4)

# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------
SOUND_ENABLED: bool = _bool("SOUND_ENABLED", True)
SOUND_GPIO_PIN: int = _int("SOUND_GPIO_PIN", 18)

# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", str(_BASE_DIR / "uploads" / "faces"))
MAX_CONTENT_LENGTH: int = _int("MAX_CONTENT_LENGTH", 5 * 1024 * 1024)  # 5 MB
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# ---------------------------------------------------------------------------
# Session / Security
# ---------------------------------------------------------------------------
SESSION_LIFETIME_MINUTES: int = _int("SESSION_LIFETIME_MINUTES", 60)
SESSION_COOKIE_SECURE: bool = _bool("SESSION_COOKIE_SECURE", True)
LOGIN_MAX_ATTEMPTS: int = _int("LOGIN_MAX_ATTEMPTS", 5)
LOGIN_LOCKOUT_SECONDS: int = _int("LOGIN_LOCKOUT_SECONDS", 300)

# ---------------------------------------------------------------------------
# Proxy / Cloudflare
# ---------------------------------------------------------------------------
PROXY_COUNT: int = _int("PROXY_COUNT", 1)
_raw_hosts = os.getenv("ALLOWED_HOSTS", "")
ALLOWED_HOSTS: list[str] = [h.strip() for h in _raw_hosts.split(",") if h.strip()]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE: str = os.getenv("LOG_FILE", str(_BASE_DIR / "logs" / "cachesec.log"))

# ---------------------------------------------------------------------------
# Ensure required directories exist at import time
# ---------------------------------------------------------------------------
for _d in (RECORDINGS_DIR, SNAPSHOTS_DIR, UPLOAD_FOLDER, str(_BASE_DIR / "logs")):
    Path(_d).mkdir(parents=True, exist_ok=True)
