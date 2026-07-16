"""
setup_wizard.py — First-run setup wizard.

Steps:
    admin   → no users in DB
    cameras → user adds zero or more cameras (one optionally flagged "detection")
    discord → optional Discord webhook
    done    → flag setup_complete=true and bounce to dashboard

Cameras are accumulated in the `setup_cameras` JSON blob during the wizard.
On completion we expand them into the existing settings keys
(`camera_preferred_source`, `tapo_*`, `ip_camera_url`, `ip_camera_urls`,
`camera_index`) so the rest of the app keeps working unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from flask import (
    Blueprint, render_template, redirect, url_for, request,
    flash, session, jsonify,
)

import config
import models
from auth import hash_password, login_user
from database import get_db, set_setting, get_setting
from utils import audit, get_client_ip

logger = logging.getLogger(__name__)

setup_bp = Blueprint("setup", __name__, url_prefix="/setup")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def setup_complete() -> bool:
    try:
        return get_setting("setup_complete", "").strip().lower() == "true"
    except Exception:
        return False


def _has_users() -> bool:
    try:
        return bool(models.get_all_users(get_db()))
    except Exception:
        return False


def _cameras_decided() -> bool:
    return get_setting("setup_cameras_done", "").strip().lower() == "true"


def _discord_decided() -> bool:
    if get_setting("setup_skip_discord", "").strip().lower() == "true":
        return True
    return bool(get_setting("discord_webhook_url", ""))


def _wizard_state() -> str:
    if not _has_users():
        return "admin"
    if not _cameras_decided():
        return "cameras"
    if not _discord_decided():
        return "discord"
    return "done"


def _start_camera_loop() -> None:
    try:
        from camera import get_camera_loop
        get_camera_loop().start()
    except Exception as exc:
        logger.warning("Camera loop start failed: %s", exc)


# ---------------------------------------------------------------------------
# Camera list helpers (stored as JSON in a single setting during setup)
# ---------------------------------------------------------------------------

def _load_cams() -> list[dict]:
    raw = get_setting("setup_cameras", "")
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_cams(cams: list[dict]) -> None:
    set_setting("setup_cameras", json.dumps(cams), user_id=session.get("user_id"))
    try:
        from go2rtc_config import regenerate_config, reload_go2rtc
        regenerate_config()
        reload_go2rtc()
    except Exception as exc:
        logger.debug("go2rtc regen after camera save failed: %s", exc)


def _cam_describe(cam: dict) -> str:
    t = cam.get("type", "")
    if t == "tapo":
        return f"Tapo · {cam.get('host','')}"
    if t == "ip":
        return f"IP / RTSP · {cam.get('url','')[:48]}"
    if t == "kinect":
        return "Kinect v1"
    if t == "webcam":
        return f"USB / Pi · /dev/video{cam.get('index', 0)}"
    return t


def _cam_title(cam: dict, ordinal: int = 1) -> str:
    label = (cam.get("label") or "").strip()
    if label:
        return label
    t = cam.get("type", "")
    if t == "tapo":
        return "Tapo Camera"
    if t == "ip":
        return f"IP Camera {ordinal}"
    if t == "kinect":
        return "Kinect"
    if t == "webcam":
        return f"USB / Pi Camera {cam.get('index', 0)}"
    return "Camera"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@setup_bp.route("/", methods=["GET"])
def wizard():
    if setup_complete():
        return redirect(url_for("admin.dashboard"))

    state = _wizard_state()
    if state == "done":
        set_setting("setup_complete", "true", user_id=session.get("user_id"))
        _start_camera_loop()
        return redirect(url_for("admin.dashboard"))

    raw_cams = _load_cams() if state == "cameras" else []
    cams = [
        {**c, "describe": _cam_describe(c), "title": _cam_title(c, i)}
        for i, c in enumerate(raw_cams, start=1)
    ]
    resp = render_template(
        "setup/wizard.html",
        step=state,
        cams=cams,
    )
    from flask import make_response
    response = make_response(resp)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


# --------- ADMIN ----------

@setup_bp.route("/admin", methods=["POST"])
def submit_admin():
    if _has_users():
        return redirect(url_for("setup.wizard"))

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    confirm  = request.form.get("password_confirm") or ""

    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
        flash("Username must be 3-32 chars (letters, digits, _ . -).", "danger")
        return redirect(url_for("setup.wizard"))
    if len(password) < 8:
        flash("Password must be at least 8 characters.", "danger")
        return redirect(url_for("setup.wizard"))
    if password != confirm:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("setup.wizard"))

    db = get_db()
    role = models.get_role_by_name(db, "admin")
    if not role:
        flash("Default admin role missing — DB not initialised correctly.", "danger")
        return redirect(url_for("setup.wizard"))

    uid = models.create_user(
        db,
        username=username,
        password_hash=hash_password(password),
        role_id=role["id"],
        display_name=username,
    )
    user_row = models.get_user_by_id(db, uid)
    if user_row:
        login_user(user_row)

    audit("SETUP_ADMIN_CREATED", user_id=uid, username=username,
          ip_address=get_client_ip())
    flash(f"Welcome, {username}.", "success")
    return redirect(url_for("setup.wizard"))


# --------- CAMERAS ----------

@setup_bp.route("/cameras/add", methods=["POST"])
def add_camera():
    if not _has_users():
        return redirect(url_for("setup.wizard"))

    cam_type = (request.form.get("cam_type") or "").strip().lower()
    if cam_type not in {"webcam", "ip", "tapo", "kinect"}:
        flash("Pick a camera type.", "danger")
        return redirect(url_for("setup.wizard"))

    label = (request.form.get("label") or "").strip()
    cam: dict = {"id": uuid.uuid4().hex[:8], "type": cam_type, "label": label}

    if cam_type == "webcam":
        try:
            idx = int(request.form.get("index") or "0")
        except ValueError:
            flash("Camera index must be a number.", "danger")
            return redirect(url_for("setup.wizard"))
        cam["index"] = max(0, idx)
    elif cam_type == "ip":
        url = (request.form.get("url") or "").strip()
        if not url:
            flash("Stream URL is required.", "danger")
            return redirect(url_for("setup.wizard"))
        if not re.match(r"^(rtsp|rtsps|http|https)://", url):
            flash("URL must start with rtsp://, http://, or https://", "danger")
            return redirect(url_for("setup.wizard"))
        transport = (request.form.get("transport") or "tcp").strip().lower()
        if transport not in {"tcp", "udp", "udp_multicast", "http"}:
            transport = "tcp"
        cam["url"] = url
        cam["transport"] = transport
    elif cam_type == "tapo":
        host = (request.form.get("host") or "").strip()
        username = (request.form.get("username") or "admin").strip() or "admin"
        password = request.form.get("password") or ""
        stream = (request.form.get("stream") or "stream1").strip().lower()
        if stream not in {"stream1", "stream2"}:
            stream = "stream1"
        if not host or not password:
            flash("Tapo host and password are required.", "danger")
            return redirect(url_for("setup.wizard"))
        cam.update(host=host, username=username, password=password, stream=stream)
    # Kinect needs no fields.

    cam["detection"] = False  # detection is opt-in per camera
    cams = _load_cams()
    cams.append(cam)
    _save_cams(cams)
    flash(f"Added: {_cam_describe(cam)}", "success")
    return redirect(url_for("setup.wizard"))


@setup_bp.route("/cameras/remove", methods=["POST"])
def remove_camera():
    cam_id = (request.form.get("id") or "").strip()
    cams = _load_cams()
    _save_cams([c for c in cams if c.get("id") != cam_id])
    return redirect(url_for("setup.wizard"))


@setup_bp.route("/cameras/toggle-detection", methods=["POST"])
def toggle_detection():
    """Flip detection on/off for one camera independently of the others."""
    cam_id = (request.form.get("id") or "").strip()
    cams = _load_cams()
    for c in cams:
        if c.get("id") == cam_id:
            c["detection"] = not bool(c.get("detection"))
            break
    _save_cams(cams)
    return redirect(url_for("setup.wizard"))


@setup_bp.route("/cameras/done", methods=["POST"])
def cameras_done():
    if not _has_users():
        return redirect(url_for("setup.wizard"))

    cams = _load_cams()
    _expand_cameras_into_settings(cams)
    set_setting("setup_cameras_done", "true", user_id=session.get("user_id"))
    audit("SETUP_CAMERAS_CONFIGURED",
          user_id=session.get("user_id"),
          username=session.get("username", ""),
          detail=f"count={len(cams)}", ip_address=get_client_ip())
    return redirect(url_for("setup.wizard"))


def _expand_cameras_into_settings(cams: list[dict]) -> None:
    """Translate the wizard's camera list into the runtime setting keys
    used by camera.py.

    Model: zero or more cameras, each with `detection` independently on/off.
    The first detection-on camera becomes the "primary detection feed"; the
    rest become live-only sources. If no camera has detection on, we write
    `camera_preferred_source = none` so the camera loop never tries to open
    a device the user didn't explicitly add.
    """
    uid = session.get("user_id")
    detection_cams = [c for c in cams if c.get("detection")]
    primary = detection_cams[0] if detection_cams else None

    if primary is None:
        set_setting("camera_preferred_source", "none", user_id=uid)
    else:
        t = primary.get("type", "webcam")
        set_setting("camera_preferred_source", t, user_id=uid)
        if t == "tapo":
            set_setting("tapo_host", primary.get("host", ""), user_id=uid)
            set_setting("tapo_username", primary.get("username", "admin"), user_id=uid)
            set_setting("tapo_password", primary.get("password", ""), user_id=uid)
            set_setting("tapo_stream", primary.get("stream", "stream1"), user_id=uid)
        elif t == "ip":
            set_setting("ip_camera_url", primary.get("url", ""), user_id=uid)
            set_setting("ip_camera_rtsp_transport",
                        primary.get("transport", "tcp"), user_id=uid)
        elif t == "webcam":
            set_setting("camera_index", str(primary.get("index", 0)), user_id=uid)

    # Everything that's not the primary detection cam is live-only.
    extras = [c for c in cams if c is not primary]
    extras_lines: list[str] = []
    extra_indices: list[str] = []
    for c in extras:
        if c.get("type") == "ip":
            line = f"{c.get('label','')}|{c.get('url','')}" if c.get("label") else c.get("url", "")
            if line:
                extras_lines.append(line)
        elif c.get("type") == "tapo":
            host = c.get("host", "")
            user = c.get("username", "admin")
            pw = c.get("password", "")
            stream = c.get("stream", "stream1")
            from urllib.parse import quote
            url = f"rtsp://{quote(user, safe='')}:{quote(pw, safe='')}@{host}:554/{stream}"
            line = f"{c.get('label','')}|{url}" if c.get("label") else url
            extras_lines.append(line)
        elif c.get("type") == "webcam":
            extra_indices.append(str(c.get("index", 0)))

    set_setting("ip_camera_urls", "\n".join(extras_lines), user_id=uid)
    set_setting("usb_camera_indices", ",".join(extra_indices), user_id=uid)


# --------- DISCORD ----------

@setup_bp.route("/discord", methods=["POST"])
def submit_discord():
    if not _has_users():
        return redirect(url_for("setup.wizard"))
    # Don't allow jumping straight from admin → done with a stale form.
    if not _cameras_decided():
        flash("Please finish the camera step first.", "warning")
        return redirect(url_for("setup.wizard"))

    uid = session.get("user_id")
    skip = (request.form.get("skip") or "").strip().lower() == "true"
    if skip:
        set_setting("setup_skip_discord", "true", user_id=uid)
    else:
        webhook = (request.form.get("discord_webhook_url") or "").strip()
        if webhook and not webhook.startswith("https://discord.com/api/webhooks/"):
            flash("That doesn't look like a Discord webhook URL.", "danger")
            return redirect(url_for("setup.wizard"))
        if not webhook:
            flash("Webhook URL is required (or click Skip).", "danger")
            return redirect(url_for("setup.wizard"))
        set_setting("discord_webhook_url", webhook, user_id=uid)

    set_setting("setup_complete", "true", user_id=uid)
    audit("SETUP_COMPLETE", user_id=uid,
          username=session.get("username", ""),
          ip_address=get_client_ip())
    flash("Setup complete.", "success")
    _start_camera_loop()
    return redirect(url_for("admin.dashboard"))


# ---------------------------------------------------------------------------
# Manual camera test (only when the user clicks the Test button)
# ---------------------------------------------------------------------------

@setup_bp.route("/test-camera", methods=["POST"])
def test_camera():
    if not _has_users():
        return jsonify(ok=False, error="Create the admin account first."), 403

    cam_type = (request.form.get("cam_type") or "").strip().lower()
    try:
        if cam_type == "webcam":
            try:
                idx = int(request.form.get("index") or "0")
            except ValueError:
                idx = 0
            return _probe_webcam(idx)
        if cam_type == "ip":
            return _probe_url((request.form.get("url") or "").strip())
        if cam_type == "tapo":
            from urllib.parse import quote
            host = (request.form.get("host") or "").strip()
            user = (request.form.get("username") or "admin").strip() or "admin"
            pw = request.form.get("password") or ""
            stream = (request.form.get("stream") or "stream1").strip().lower()
            if stream not in {"stream1", "stream2"}:
                stream = "stream1"
            if not host or not pw:
                return jsonify(ok=False, error="Host and password required.")
            return _probe_url(f"rtsp://{quote(user,safe='')}:{quote(pw,safe='')}@{host}:554/{stream}")
        if cam_type == "kinect":
            return _probe_kinect()
        return jsonify(ok=False, error="Pick a camera type first.")
    except Exception as exc:
        logger.exception("Camera probe failed")
        return jsonify(ok=False, error=str(exc))


def _probe_webcam(idx: int):
    import cv2
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(idx)
    try:
        if not cap.isOpened():
            return jsonify(ok=False, error=f"Cannot open /dev/video{idx}.")
        ok, _ = cap.read()
        if not ok:
            return jsonify(ok=False, error="Camera opened but returned no frames.")
        return jsonify(ok=True, message=f"Frame received from /dev/video{idx}.")
    finally:
        cap.release()


def _probe_url(url: str):
    import cv2, os
    if not url:
        return jsonify(ok=False, error="URL is empty.")
    prev = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return jsonify(ok=False, error="Could not open the stream (auth, URL, or network).")
        for _ in range(20):
            ok, _ = cap.read()
            if ok:
                return jsonify(ok=True, message="Stream opened and produced frames.")
        return jsonify(ok=False, error="Stream opened but produced no frames in time.")
    finally:
        cap.release()
        if prev is None:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev


def _probe_kinect():
    try:
        from kinect import kinect_available
    except Exception as exc:
        return jsonify(ok=False, error=f"Kinect module unavailable: {exc}")
    if not kinect_available():
        return jsonify(ok=False, error="Kinect device not detected on USB.")
    return jsonify(ok=True, message="Kinect detected.")
