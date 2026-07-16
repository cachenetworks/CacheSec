"""
database.py — SQLite connection management and schema initialisation.

All raw SQL lives here. Call `init_db()` once at startup to ensure
every table exists. Use `get_db()` inside Flask request context or
call `get_raw_db()` from background threads.
"""

import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# SQLite WAL mode improves concurrent read performance (background thread + web)
_PRAGMAS = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;
"""

_SCHEMA = """
-- -------------------------------------------------------------------------
-- Roles
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS roles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,   -- 'admin', 'operator', 'viewer'
    description TEXT
);

-- -------------------------------------------------------------------------
-- Dashboard users
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    display_name    TEXT,
    email           TEXT    UNIQUE COLLATE NOCASE,
    password_hash   TEXT    NOT NULL,
    role_id         INTEGER NOT NULL REFERENCES roles(id),
    is_active       INTEGER NOT NULL DEFAULT 1,   -- 0 = disabled
    remember_token  TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    created_by      INTEGER REFERENCES users(id)
);

-- -------------------------------------------------------------------------
-- Login rate-limiting / lockout
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS login_attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    identifier  TEXT    NOT NULL,   -- username or IP
    attempted_at TEXT   NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    success     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_login_attempts_ident
    ON login_attempts(identifier, attempted_at);

-- -------------------------------------------------------------------------
-- Enrolled people (authorized subjects)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS enrolled_people (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    notes       TEXT,
    is_active   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    created_by  INTEGER REFERENCES users(id)
);

-- -------------------------------------------------------------------------
-- Enrolled face images (original uploads, for display)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS enrolled_images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES enrolled_people(id) ON DELETE CASCADE,
    filename    TEXT    NOT NULL,   -- relative to UPLOAD_FOLDER
    uploaded_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    uploaded_by INTEGER REFERENCES users(id)
);

-- -------------------------------------------------------------------------
-- Face embeddings (one row per enrolled image after processing)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS face_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES enrolled_people(id) ON DELETE CASCADE,
    image_id    INTEGER REFERENCES enrolled_images(id) ON DELETE CASCADE,
    embedding   BLOB    NOT NULL,   -- numpy array serialised with numpy.save
    model_name  TEXT    NOT NULL DEFAULT 'buffalo_l',
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- -------------------------------------------------------------------------
-- Detection events
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT    NOT NULL,   -- 'unknown' | 'recognized'
    person_id       INTEGER REFERENCES enrolled_people(id),
    person_name     TEXT,               -- denormalised snapshot
    confidence      REAL,               -- recognition score (0-1, higher=better)
    snapshot_path   TEXT,
    recording_path  TEXT,
    recording_start TEXT,
    recording_end   TEXT,
    webhook_sent    INTEGER NOT NULL DEFAULT 0,
    webhook_error   TEXT,
    occurred_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    ended_at        TEXT,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_occurred ON events(occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_type     ON events(event_type);

-- -------------------------------------------------------------------------
-- Recordings (separate table for richer metadata)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recordings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        INTEGER REFERENCES events(id) ON DELETE SET NULL,
    filename        TEXT    NOT NULL,
    file_size_bytes INTEGER,
    duration_seconds REAL,
    started_at      TEXT    NOT NULL,
    ended_at        TEXT,
    deleted         INTEGER NOT NULL DEFAULT 0,
    deleted_by      INTEGER REFERENCES users(id),
    deleted_at      TEXT
);

-- -------------------------------------------------------------------------
-- Application settings (key-value)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS app_settings (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_by  INTEGER REFERENCES users(id)
);

-- -------------------------------------------------------------------------
-- Per-person access schedules
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS access_schedules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES enrolled_people(id) ON DELETE CASCADE,
    day_of_week INTEGER NOT NULL,   -- 0=Mon … 6=Sun (Python weekday())
    time_start  TEXT    NOT NULL,   -- "HH:MM" 24-hour
    time_end    TEXT    NOT NULL,   -- "HH:MM" 24-hour
    is_active   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_schedule_person ON access_schedules(person_id);

-- -------------------------------------------------------------------------
-- Audit log
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
    username    TEXT,                   -- denormalised so it survives user deletion
    action      TEXT    NOT NULL,       -- e.g. 'USER_CREATED', 'SETTINGS_CHANGED'
    target_type TEXT,                   -- e.g. 'user', 'enrolled_person', 'recording'
    target_id   TEXT,
    detail      TEXT,                   -- free-form JSON or description
    ip_address  TEXT,
    occurred_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_audit_occurred ON audit_log(occurred_at);
"""

_DEFAULT_ROLES = [
    ("admin",    "Full system access"),
    ("operator", "View dashboard, events, recordings, live feed; cannot manage users"),
    ("viewer",   "Read-only access to limited pages"),
]


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(_PRAGMAS)
    return conn


def init_db() -> None:
    """Create all tables and seed default data. Safe to call multiple times."""
    logger.info("Initialising database at %s", config.DATABASE_PATH)
    conn = _connect(config.DATABASE_PATH)
    try:
        conn.executescript(_SCHEMA)

        # Seed roles
        for name, desc in _DEFAULT_ROLES:
            conn.execute(
                "INSERT OR IGNORE INTO roles(name, description) VALUES (?, ?)",
                (name, desc),
            )

        # Seed default settings
        _default_settings = {
            "recognition_threshold":    str(config.RECOGNITION_THRESHOLD),
            "frame_skip":               str(config.FRAME_SKIP),
            "camera_index":             str(config.CAMERA_INDEX),
            "night_vision_mode":        config.NIGHT_VISION_MODE,
            "setup_cameras":            "",
            "camera_preferred_source":  config.CAMERA_PREFERRED_SOURCE,
            "usb_camera_indices":       config.USB_CAMERA_INDICES,
            "usb_camera_auto_discover": "true" if config.USB_CAMERA_AUTO_DISCOVER else "false",
            "usb_camera_scan_limit":    str(config.USB_CAMERA_SCAN_LIMIT),
            "multi_camera_detection_enabled": "true" if config.MULTI_CAMERA_DETECTION_ENABLED else "false",
            "ip_camera_url":            config.IP_CAMERA_URL,
            "ip_camera_urls":           config.IP_CAMERA_URLS,
            "ip_camera_rtsp_transport": config.IP_CAMERA_RTSP_TRANSPORT,
            "ip_camera_onvif_night_mode": config.IP_CAMERA_ONVIF_NIGHT_MODE,
            "ip_camera_onvif_host":       config.IP_CAMERA_ONVIF_HOST,
            "ip_camera_onvif_port":       str(config.IP_CAMERA_ONVIF_PORT or ""),
            "ip_camera_onvif_username":   config.IP_CAMERA_ONVIF_USERNAME,
            "ip_camera_onvif_password":   config.IP_CAMERA_ONVIF_PASSWORD,
            "ip_camera_onvif_wsdl_dir":   config.IP_CAMERA_ONVIF_WSDL_DIR,
            "unknown_cooldown_seconds": str(config.UNKNOWN_COOLDOWN_SECONDS),
            "person_detection_backend": config.PERSON_DETECTION_BACKEND,
            "person_detection_threshold": str(config.PERSON_DETECTION_THRESHOLD),
            "person_detection_device":  config.PERSON_DETECTION_DEVICE,
            "object_detection_backend": config.OBJECT_DETECTION_BACKEND,
            "object_detection_mode":    config.OBJECT_DETECTION_MODE,
            "object_detection_threshold": str(config.OBJECT_DETECTION_THRESHOLD),
            "object_detection_device":  config.OBJECT_DETECTION_DEVICE,
            "moving_object_detection_enabled": "true" if config.MOVING_OBJECT_DETECTION_ENABLED else "false",
            "moving_object_min_area":   str(config.MOVING_OBJECT_MIN_AREA),
            "moving_object_threshold":  str(config.MOVING_OBJECT_THRESHOLD),
            "min_recording_seconds":    str(config.MIN_RECORDING_SECONDS),
            "max_recording_seconds":    str(config.MAX_RECORDING_SECONDS),
            "record_all_mode":          "false",
            "save_recordings_locally":  "true" if config.SAVE_RECORDINGS_LOCALLY else "false",
            "record_audio_enabled":     "true" if config.RECORD_AUDIO_ENABLED else "false",
            "video_encoder":            config.VIDEO_ENCODER,
            "video_encoder_preset":     config.VIDEO_ENCODER_PRESET,
            "video_encoder_quality":    str(config.VIDEO_ENCODER_QUALITY),
            "discord_cooldown_seconds": str(config.DISCORD_COOLDOWN_SECONDS),
            "discord_mention_everyone": "true" if config.DISCORD_MENTION_EVERYONE else "false",
            "sound_enabled":            "true" if config.SOUND_ENABLED else "false",
            "discord_webhook_url":      config.DISCORD_WEBHOOK_URL,
        }
        for k, v in _default_settings.items():
            conn.execute(
                "INSERT OR IGNORE INTO app_settings(key, value) VALUES (?, ?)", (k, v)
            )

        conn.commit()
        logger.info("Database initialised OK")
    finally:
        conn.close()


def get_raw_db() -> sqlite3.Connection:
    """Return a new connection. Caller is responsible for closing it.

    Use this from background threads (camera loop, recorder, etc.).
    """
    return _connect(config.DATABASE_PATH)


@contextmanager
def raw_db_ctx():
    """Context manager that opens, yields, commits, and closes a connection."""
    conn = get_raw_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Flask integration — stores connection on flask.g
# ---------------------------------------------------------------------------
def get_db():
    """Return the per-request SQLite connection (Flask context only)."""
    from flask import g
    if "db" not in g:
        g.db = _connect(config.DATABASE_PATH)
    return g.db


def close_db(e=None):
    """Teardown handler — close the per-request connection."""
    from flask import g
    db = g.pop("db", None)
    if db is not None:
        db.close()


def get_setting(key: str, default: str = "") -> str:
    """Read a single app setting from the database (uses raw connection)."""
    with raw_db_ctx() as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str, user_id: int | None = None) -> None:
    """Write a single app setting."""
    with raw_db_ctx() as conn:
        conn.execute(
            """INSERT INTO app_settings(key, value, updated_at, updated_by)
               VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'), ?)
               ON CONFLICT(key) DO UPDATE SET
                   value      = excluded.value,
                   updated_at = excluded.updated_at,
                   updated_by = excluded.updated_by""",
            (key, value, user_id),
        )
