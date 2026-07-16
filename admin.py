"""
admin.py — Flask Blueprint for the admin dashboard.

Routes:
  /                  → dashboard overview
  /live              → live camera feed
  /events            → all events
  /events/unknown    → unknown detections
  /events/recognized → recognized detections
  /recordings        → saved recordings
  /recordings/<id>/delete
  /enrolled          → enrolled people list
  /enrolled/add
  /enrolled/<id>     → person detail + images
  /enrolled/<id>/edit
  /enrolled/<id>/delete
  /enrolled/<id>/upload
  /enrolled/<id>/retrain
  /users             → user management (admin only)
  /users/add
  /users/<id>/edit
  /users/<id>/delete
  /users/<id>/toggle
  /settings          → app settings
  /audit             → audit log
  /health            → system health
  /api/status        → JSON status for dashboard widgets
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from flask import (
    Blueprint, render_template, redirect, url_for, request,
    flash, session, send_from_directory, Response, jsonify,
    abort, current_app,
)

import config
import models
from auth import (
    login_required, admin_required, operator_required, viewer_required,
    hash_password, current_user,
)
from database import get_db, set_setting, get_setting
from utils import (
    secure_name, allowed_image, timestamped_filename,
    dir_size_mb, disk_usage_percent, audit, get_client_ip,
)

logger = logging.getLogger(__name__)

admin_bp = Blueprint("admin", __name__)


def _storage_disk_path() -> str:
    """Pick a local CacheSec storage path for filesystem usage stats."""
    return config.RECORDINGS_DIR or config.SNAPSHOTS_DIR or config.UPLOAD_FOLDER or "."


def _sync_camera_settings_from_setup_cameras(raw: str, user_id: int | None) -> None:
    """Normalize the unified camera list and update legacy runtime settings."""
    try:
        data = json.loads(raw) if raw else []
    except Exception:
        data = []
    if not isinstance(data, list):
        data = []

    cams: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        cam_type = str(item.get("type") or "").strip().lower()
        if cam_type not in {"webcam", "ip", "tapo", "kinect"}:
            continue
        cam = {
            "id": str(item.get("id") or uuid.uuid4().hex[:8])[:32],
            "type": cam_type,
            "label": str(item.get("label") or "").strip(),
            "detection": bool(item.get("detection")),
        }
        if cam_type == "webcam":
            try:
                cam["index"] = max(0, int(item.get("index", 0)))
            except (TypeError, ValueError):
                cam["index"] = 0
        elif cam_type == "ip":
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            transport = str(item.get("transport") or "tcp").strip().lower()
            if transport not in {"tcp", "udp", "udp_multicast", "http"}:
                transport = "tcp"
            options = item.get("options") if isinstance(item.get("options"), dict) else {}
            cam["url"] = url
            cam["transport"] = transport
            cam["options"] = {
                key: str(options.get(key) or "").strip()
                for key in ("referer", "origin", "user_agent")
                if str(options.get(key) or "").strip()
            }
        elif cam_type == "tapo":
            host = str(item.get("host") or "").strip()
            password = str(item.get("password") or "")
            if not host or not password:
                continue
            stream = str(item.get("stream") or "stream1").strip().lower()
            if stream not in {"stream1", "stream2"}:
                stream = "stream1"
            cam.update(
                host=host,
                username=str(item.get("username") or "admin").strip() or "admin",
                password=password,
                stream=stream,
            )
        cams.append(cam)

    detection_cams = [cam for cam in cams if cam.get("detection")]
    primary = detection_cams[0] if detection_cams else None

    set_setting("setup_cameras", json.dumps(cams), user_id=user_id)
    set_setting("setup_cameras_done", "true" if cams else "false", user_id=user_id)
    set_setting(
        "multi_camera_detection_enabled",
        "true" if len(detection_cams) > 1 else "false",
        user_id=user_id,
    )

    if primary is None:
        set_setting("camera_preferred_source", "none", user_id=user_id)
        set_setting("ip_camera_url", "", user_id=user_id)
    else:
        primary_type = primary.get("type", "webcam")
        set_setting("camera_preferred_source", primary_type, user_id=user_id)
        if primary_type == "webcam":
            set_setting("camera_index", str(primary.get("index", 0)), user_id=user_id)
            set_setting("ip_camera_url", "", user_id=user_id)
        elif primary_type == "ip":
            set_setting("ip_camera_url", primary.get("url", ""), user_id=user_id)
            set_setting("ip_camera_rtsp_transport", primary.get("transport", "tcp"), user_id=user_id)
        elif primary_type == "tapo":
            set_setting("tapo_host", primary.get("host", ""), user_id=user_id)
            set_setting("tapo_username", primary.get("username", "admin"), user_id=user_id)
            set_setting("tapo_password", primary.get("password", ""), user_id=user_id)
            set_setting("tapo_stream", primary.get("stream", "stream1"), user_id=user_id)

    extras = [cam for cam in cams if cam is not primary]
    extra_ip_lines: list[str] = []
    extra_usb_indices: list[str] = []
    for cam in extras:
        cam_type = cam.get("type")
        label = str(cam.get("label") or "").strip()
        if cam_type == "webcam":
            extra_usb_indices.append(str(cam.get("index", 0)))
        elif cam_type == "ip":
            parts = []
            if label:
                parts.append(label)
            parts.append(str(cam.get("url") or ""))
            options = cam.get("options") if isinstance(cam.get("options"), dict) else {}
            for key in ("referer", "origin", "user_agent"):
                if options.get(key):
                    parts.append(f"{key}={options[key]}")
            extra_ip_lines.append("|".join(parts))
        elif cam_type == "tapo":
            host = cam.get("host", "")
            user = quote(cam.get("username", "admin"), safe="")
            pw = quote(cam.get("password", ""), safe="")
            stream = cam.get("stream", "stream1")
            url = f"rtsp://{user}:{pw}@{host}:554/{stream}"
            extra_ip_lines.append(f"{label}|{url}" if label else url)

    set_setting("ip_camera_urls", "\n".join(extra_ip_lines), user_id=user_id)
    set_setting("usb_camera_indices", ",".join(extra_usb_indices), user_id=user_id)


# ---------------------------------------------------------------------------
# Dashboard overview
# ---------------------------------------------------------------------------

@admin_bp.route("/")
@viewer_required
def dashboard():
    db   = get_db()
    unkn = models.count_events_today(db, "unknown")
    recg = models.count_events_today(db, "recognized")
    enrolled_count = len(models.get_all_enrolled(db))
    user_count     = len(models.get_all_users(db))
    recordings     = models.get_all_recordings(db)
    recent_events  = models.get_recent_events(db, limit=10)

    from camera import get_camera_status
    from recorder import get_recorder
    cam_status = get_camera_status()
    rec_state  = get_recorder().get_state()

    storage_mb = dir_size_mb(config.RECORDINGS_DIR) + dir_size_mb(config.SNAPSHOTS_DIR)
    disk_pct   = disk_usage_percent(_storage_disk_path())

    return render_template(
        "admin/dashboard.html",
        unknown_today=unkn,
        recognized_today=recg,
        enrolled_count=enrolled_count,
        user_count=user_count,
        recording_count=len([r for r in recordings]),
        cam_status=cam_status,
        rec_state=rec_state,
        recent_events=recent_events,
        storage_mb=round(storage_mb, 1),
        disk_pct=disk_pct,
    )


# ---------------------------------------------------------------------------
# Live feed
# ---------------------------------------------------------------------------

@admin_bp.route("/live")
@operator_required
def live_feed():
    from camera import get_live_sources
    return render_template("admin/live.html", live_sources=get_live_sources())


@admin_bp.route("/live/stream")
@admin_bp.route("/live/stream/<source_id>")
@operator_required
def live_stream(source_id: str = "primary"):
    from camera import generate_mjpeg
    return Response(
        generate_mjpeg(source_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@admin_bp.route("/api/camera/control", methods=["POST"])
@operator_required
def api_camera_control():
    action = (request.json or {}).get("action", "").strip().lower()
    source_id = (request.json or {}).get("source_id", "primary").strip().lower()
    if action not in {"up", "down", "left", "right", "stop"}:
        return jsonify({"ok": False, "error": "Unsupported action"}), 400

    if source_id in {"", "primary"}:
        from camera import get_camera_status
        source = (get_camera_status().get("source") or "").lower()
    elif source_id == "kinect":
        source = "kinect"
    elif source_id.startswith("kinect"):
        source = "kinect"
    elif source_id.startswith("tapo"):
        source = "tapo"
    elif source_id.startswith("ip"):
        source = "ip"
    else:
        source = "unknown"

    if source == "kinect":
        from kinect import get_kinect
        index = 0
        if source_id.startswith("kinect") and source_id != "kinect":
            try:
                index = max(0, int(source_id.replace("kinect", "")) - 1)
            except ValueError:
                index = 0
        ks = get_kinect(index)
        if action in {"left", "right"}:
            return jsonify({
                "ok": False,
                "error": "Kinect only tilts up/down — it has no left/right pan motor.",
            }), 400
        if not ks.motor_available:
            return jsonify({
                "ok": False,
                "error": "Kinect motor is disabled. Enable 'Kinect Motor / Tilt Control' in "
                         "Settings and save — the camera restarts automatically to pick it up.",
            }), 400
        if action == "up":
            ks.set_tilt((ks.get_tilt_angle() or 0.0) + 5.0)
        elif action == "down":
            ks.set_tilt((ks.get_tilt_angle() or 0.0) - 5.0)
        return jsonify({
            "ok": True,
            "source": source,
            "source_id": source_id,
            "action": action,
            "tilt_degrees": ks.get_tilt_angle(),
        })

    return jsonify({
        "ok": False,
        "error": f"Source '{source_id}' ({source}) does not expose move controls in CacheSec yet.",
    }), 400


# ---------------------------------------------------------------------------
# go2rtc reverse proxy — exposes go2rtc's HTTP API (including the streaming
# /api/stream.mp4 fMP4 endpoint) under the same origin so a browser only
# talks to CacheSec / Cloudflare. WebSockets are not proxied here; for
# WebSocket-based MSE, point Cloudflare Tunnel at go2rtc:1984 directly.
# ---------------------------------------------------------------------------

@admin_bp.route("/go2rtc/<path:subpath>")
@operator_required
def go2rtc_proxy(subpath: str):
    import os, urllib.request, urllib.error
    from urllib.parse import urlencode
    host = os.environ.get("GO2RTC_HOST", "go2rtc")
    port = os.environ.get("GO2RTC_HTTP_PORT", "1984")
    qs = urlencode(request.args.to_dict(flat=False), doseq=True)
    url = f"http://{host}:{port}/{subpath}"
    if qs:
        url = f"{url}?{qs}"
    try:
        upstream = urllib.request.urlopen(url, timeout=30)
    except (urllib.error.URLError, OSError) as exc:
        return Response(f"go2rtc unavailable: {exc}", status=502)

    def stream():
        try:
            while True:
                chunk = upstream.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            upstream.close()

    headers = {}
    ct = upstream.headers.get("Content-Type")
    if ct:
        headers["Content-Type"] = ct
    cl = upstream.headers.get("Content-Length")
    if cl:
        headers["Content-Length"] = cl
    return Response(stream(), status=upstream.status, headers=headers)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@admin_bp.route("/events")
@operator_required
def events():
    db     = get_db()
    rows   = models.get_recent_events(db, limit=200)
    return render_template("admin/events.html", events=rows, filter_type="all")


@admin_bp.route("/events/unknown")
@operator_required
def events_unknown():
    db   = get_db()
    rows = get_db().execute(
        "SELECT * FROM events WHERE event_type='unknown' ORDER BY occurred_at DESC LIMIT 200"
    ).fetchall()
    return render_template("admin/events.html", events=rows, filter_type="unknown")


@admin_bp.route("/events/recognized")
@operator_required
def events_recognized():
    rows = get_db().execute(
        "SELECT * FROM events WHERE event_type='recognized' ORDER BY occurred_at DESC LIMIT 200"
    ).fetchall()
    return render_template("admin/events.html", events=rows, filter_type="recognized")


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------

@admin_bp.route("/recordings")
@operator_required
def recordings():
    rows = models.get_all_recordings(get_db())
    return render_template("admin/recordings.html", recordings=rows)


@admin_bp.route("/recordings/<int:rec_id>/delete", methods=["POST"])
@admin_required
def delete_recording(rec_id: int):
    db  = get_db()
    row = db.execute("SELECT * FROM recordings WHERE id=?", (rec_id,)).fetchone()
    if not row:
        abort(404)
    fpath = Path(config.RECORDINGS_DIR) / row["filename"]
    try:
        fpath.unlink(missing_ok=True)
    except OSError as exc:
        flash(f"File delete error: {exc}", "warning")

    models.soft_delete_recording(db, rec_id, deleted_by=session["user_id"])
    if row["event_id"]:
        models.update_event(db, row["event_id"], recording_path="")
    audit(
        "RECORDING_DELETED",
        user_id=session["user_id"],
        username=session["username"],
        target_type="recording",
        target_id=str(rec_id),
        ip_address=get_client_ip(),
    )
    flash("Recording deleted.", "success")
    return redirect(url_for("admin.recordings"))


@admin_bp.route("/recordings/delete-batch", methods=["POST"])
@admin_required
def delete_recordings_batch():
    raw_ids = request.form.getlist("recording_ids")
    rec_ids: list[int] = []
    for raw_id in raw_ids:
        try:
            rec_ids.append(int(raw_id))
        except (TypeError, ValueError):
            continue

    rec_ids = sorted(set(rec_ids))
    if not rec_ids:
        flash("No recordings selected.", "warning")
        return redirect(url_for("admin.recordings"))

    db = get_db()
    deleted = 0
    file_errors: list[str] = []
    for rec_id in rec_ids:
        row = db.execute(
            "SELECT * FROM recordings WHERE id=? AND deleted=0", (rec_id,)
        ).fetchone()
        if not row:
            continue

        fpath = Path(config.RECORDINGS_DIR) / Path(row["filename"]).name
        try:
            fpath.unlink(missing_ok=True)
        except OSError as exc:
            file_errors.append(f"{row['filename']}: {exc}")

        models.soft_delete_recording(db, rec_id, deleted_by=session["user_id"])
        if row["event_id"]:
            models.update_event(db, row["event_id"], recording_path="")
        deleted += 1

    if deleted:
        audit(
            "RECORDINGS_BATCH_DELETED",
            user_id=session["user_id"],
            username=session["username"],
            target_type="recording",
            target_id=",".join(str(i) for i in rec_ids),
            detail=f"{deleted} recording(s) deleted",
            ip_address=get_client_ip(),
        )
        flash(f"{deleted} recording(s) deleted.", "success")
    else:
        flash("No matching recordings were deleted.", "warning")

    if file_errors:
        flash("Some files could not be removed: " + "; ".join(file_errors[:3]), "warning")

    return redirect(url_for("admin.recordings"))


@admin_bp.route("/recordings/<path:filename>")
@operator_required
def serve_recording(filename: str):
    # Prevent path traversal
    safe = Path(filename).name
    ext  = safe.rsplit(".", 1)[-1].lower()
    mime = "video/mp4" if ext == "mp4" else "video/x-msvideo"
    # conditional=True enables Range requests — required for browser seek/scrub
    return send_from_directory(
        config.RECORDINGS_DIR, safe, mimetype=mime, conditional=True
    )


@admin_bp.route("/snapshots/<path:filename>")
@operator_required
def serve_snapshot(filename: str):
    safe = Path(filename).name
    return send_from_directory(config.SNAPSHOTS_DIR, safe)


# ---------------------------------------------------------------------------
# Enrolled people
# ---------------------------------------------------------------------------

@admin_bp.route("/enrolled")
@operator_required
def enrolled():
    rows = models.get_all_enrolled(get_db())
    return render_template("admin/enrolled.html", people=rows)


@admin_bp.route("/enrolled/add", methods=["GET", "POST"])
@admin_required
def enrolled_add():
    if request.method == "POST":
        name  = request.form.get("name", "").strip()
        notes = request.form.get("notes", "").strip()
        if not name:
            flash("Name is required.", "danger")
            return render_template("admin/enrolled_form.html", person=None)

        person_id = models.create_enrolled_person(
            get_db(), name=name, notes=notes, created_by=session["user_id"]
        )
        audit(
            "ENROLLED_PERSON_ADDED",
            user_id=session["user_id"],
            username=session["username"],
            target_type="enrolled_person",
            target_id=str(person_id),
            detail=f"name={name}",
            ip_address=get_client_ip(),
        )
        flash(f"Person '{name}' added. Now upload face images.", "success")
        return redirect(url_for("admin.enrolled_detail", person_id=person_id))

    return render_template("admin/enrolled_form.html", person=None)


@admin_bp.route("/enrolled/<int:person_id>")
@operator_required
def enrolled_detail(person_id: int):
    db     = get_db()
    person = models.get_enrolled_by_id(db, person_id)
    if not person:
        abort(404)
    images    = models.get_images_for_person(db, person_id)
    schedules = models.get_schedules_for_person(db, person_id)
    return render_template("admin/enrolled_detail.html", person=person,
                           images=images, schedules=schedules,
                           day_names=models.DAY_NAMES)


@admin_bp.route("/enrolled/<int:person_id>/schedule/add", methods=["POST"])
@admin_required
def schedule_add(person_id: int):
    db = get_db()
    if not models.get_enrolled_by_id(db, person_id):
        abort(404)
    try:
        day   = int(request.form["day_of_week"])
        start = request.form["time_start"].strip()
        end   = request.form["time_end"].strip()
        if not (0 <= day <= 6) or not start or not end:
            raise ValueError
        models.create_schedule(db, person_id, day, start, end)
        flash("Schedule added.", "success")
    except (KeyError, ValueError):
        flash("Invalid schedule entry.", "danger")
    return redirect(url_for("admin.enrolled_detail", person_id=person_id))


@admin_bp.route("/enrolled/<int:person_id>/schedule/<int:sched_id>/delete", methods=["POST"])
@admin_required
def schedule_delete(person_id: int, sched_id: int):
    db = get_db()
    models.delete_schedule(db, sched_id)
    flash("Schedule removed.", "success")
    return redirect(url_for("admin.enrolled_detail", person_id=person_id))


@admin_bp.route("/enrolled/<int:person_id>/edit", methods=["GET", "POST"])
@admin_required
def enrolled_edit(person_id: int):
    db     = get_db()
    person = models.get_enrolled_by_id(db, person_id)
    if not person:
        abort(404)

    if request.method == "POST":
        name  = request.form.get("name", "").strip()
        notes = request.form.get("notes", "").strip()
        active = 1 if request.form.get("is_active") else 0
        if not name:
            flash("Name is required.", "danger")
        else:
            models.update_enrolled_person(db, person_id, name=name, notes=notes, is_active=active)
            audit("ENROLLED_PERSON_UPDATED", user_id=session["user_id"],
                  username=session["username"], target_type="enrolled_person",
                  target_id=str(person_id), ip_address=get_client_ip())
            flash("Person updated.", "success")
            return redirect(url_for("admin.enrolled_detail", person_id=person_id))

    return render_template("admin/enrolled_form.html", person=person)


@admin_bp.route("/enrolled/<int:person_id>/delete", methods=["POST"])
@admin_required
def enrolled_delete(person_id: int):
    db     = get_db()
    person = models.get_enrolled_by_id(db, person_id)
    if not person:
        abort(404)

    # Delete uploaded images from disk
    for img in models.get_images_for_person(db, person_id):
        try:
            Path(config.UPLOAD_FOLDER, img["filename"]).unlink(missing_ok=True)
        except OSError:
            pass

    models.delete_enrolled_person(db, person_id)
    audit("ENROLLED_PERSON_DELETED", user_id=session["user_id"],
          username=session["username"], target_type="enrolled_person",
          target_id=str(person_id), detail=f"name={person['name']}",
          ip_address=get_client_ip())

    from recognition import reload_gallery
    reload_gallery()

    flash(f"Person '{person['name']}' deleted.", "success")
    return redirect(url_for("admin.enrolled"))


@admin_bp.route("/enrolled/<int:person_id>/upload", methods=["POST"])
@admin_required
def enrolled_upload(person_id: int):
    db     = get_db()
    person = models.get_enrolled_by_id(db, person_id)
    if not person:
        abort(404)

    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        flash("No files selected.", "warning")
        return redirect(url_for("admin.enrolled_detail", person_id=person_id))

    saved = 0
    errors = []

    for file in files:
        if not file.filename:
            continue
        if not allowed_image(file.filename):
            errors.append(f"{file.filename}: unsupported file type")
            continue

        fname  = f"person{person_id}_{timestamped_filename('face', 'jpg')}"
        fpath  = Path(config.UPLOAD_FOLDER) / fname

        try:
            file.save(str(fpath))
        except OSError as exc:
            errors.append(f"{file.filename}: save error ({exc})")
            continue

        # Generate embedding
        try:
            from recognition import get_recognizer, serialise_embedding
            rec  = get_recognizer()
            embs = rec.embed_image_bytes(fpath.read_bytes())
            if not embs:
                fpath.unlink(missing_ok=True)
                errors.append(f"{file.filename}: no face detected in image")
                continue

            image_id = models.add_enrolled_image(
                db, person_id=person_id, filename=fname,
                uploaded_by=session["user_id"],
            )
            for emb in embs:
                models.add_embedding(
                    db, person_id=person_id, image_id=image_id,
                    embedding_bytes=serialise_embedding(emb),
                )
            saved += 1

        except Exception as exc:
            logger.exception("Embedding error for %s", file.filename)
            errors.append(f"{file.filename}: embedding error ({exc})")
            fpath.unlink(missing_ok=True)

    if saved:
        from recognition import reload_gallery
        reload_gallery()
        audit("FACE_IMAGES_UPLOADED", user_id=session["user_id"],
              username=session["username"], target_type="enrolled_person",
              target_id=str(person_id), detail=f"{saved} image(s) uploaded",
              ip_address=get_client_ip())
        flash(f"{saved} image(s) uploaded and embedded.", "success")

    for err in errors:
        flash(err, "warning")

    return redirect(url_for("admin.enrolled_detail", person_id=person_id))


@admin_bp.route("/enrolled/<int:person_id>/image/<int:image_id>/delete", methods=["POST"])
@admin_required
def enrolled_image_delete(person_id: int, image_id: int):
    db  = get_db()
    img = db.execute("SELECT * FROM enrolled_images WHERE id=? AND person_id=?",
                     (image_id, person_id)).fetchone()
    if not img:
        abort(404)
    try:
        Path(config.UPLOAD_FOLDER, img["filename"]).unlink(missing_ok=True)
    except OSError:
        pass
    models.delete_enrolled_image(db, image_id)
    from recognition import reload_gallery
    reload_gallery()
    flash("Image deleted.", "success")
    return redirect(url_for("admin.enrolled_detail", person_id=person_id))


@admin_bp.route("/enrolled/<int:person_id>/capture_preview")
@admin_required
def enrolled_capture_preview(person_id: int):
    """Return the current camera frame as a JPEG for the capture UI preview."""
    from camera import get_latest_jpeg
    jpeg = get_latest_jpeg()
    if not jpeg:
        abort(503)
    return Response(jpeg, mimetype="image/jpeg")


@admin_bp.route("/enrolled/<int:person_id>/capture", methods=["POST"])
@admin_required
def enrolled_capture(person_id: int):
    """
    Grab the current live frame, save it as a face image, generate embedding,
    and enroll it. Returns JSON so the page can update without a full reload.
    """
    import cv2
    import numpy as np
    from camera import get_latest_jpeg
    from recognition import get_recognizer, serialise_embedding, reload_gallery

    db     = get_db()
    person = models.get_enrolled_by_id(db, person_id)
    if not person:
        return jsonify({"ok": False, "error": "Person not found"}), 404

    jpeg = get_latest_jpeg()
    if not jpeg:
        return jsonify({"ok": False, "error": "No camera frame available — is the camera running?"}), 503

    # Decode frame
    nparr = np.frombuffer(jpeg, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"ok": False, "error": "Could not decode camera frame"}), 500

    # Detect and embed
    rec   = get_recognizer()
    faces = rec.detect(frame)
    if not faces:
        return jsonify({"ok": False, "error": "No face detected in frame. Move closer to the camera."}), 400

    if len(faces) > 1:
        return jsonify({"ok": False, "error": f"{len(faces)} faces detected — only one person should be in frame."}), 400

    face = faces[0]
    if face.embedding is None:
        return jsonify({"ok": False, "error": "Face detected but embedding failed."}), 500

    # Draw a tight box on the saved image so it's clear what was captured
    annotated = frame.copy()
    x1, y1, x2, y2 = face.bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)

    # Save the annotated capture as the enrolled image
    fname = f"person{person_id}_{timestamped_filename('capture', 'jpg')}"
    fpath = Path(config.UPLOAD_FOLDER) / fname
    cv2.imwrite(str(fpath), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Store in DB
    image_id = models.add_enrolled_image(
        db, person_id=person_id, filename=fname,
        uploaded_by=session["user_id"],
    )
    models.add_embedding(
        db, person_id=person_id, image_id=image_id,
        embedding_bytes=serialise_embedding(face.embedding),
    )
    reload_gallery()

    audit(
        "FACE_CAPTURED",
        user_id=session["user_id"],
        username=session["username"],
        target_type="enrolled_person",
        target_id=str(person_id),
        detail=f"Live capture: {fname}  det_score={face.det_score:.3f}",
        ip_address=get_client_ip(),
    )

    return jsonify({
        "ok":        True,
        "filename":  fname,
        "det_score": round(face.det_score, 3),
        "bbox":      list(face.bbox),
        "image_url": url_for("admin.serve_face", filename=fname),
    })


@admin_bp.route("/enrolled/<int:person_id>/retrain", methods=["POST"])
@admin_required
def enrolled_retrain(person_id: int):
    """Re-generate embeddings for all images of a person."""
    db     = get_db()
    person = models.get_enrolled_by_id(db, person_id)
    if not person:
        abort(404)

    models.delete_embeddings_for_person(db, person_id)
    images = models.get_images_for_person(db, person_id)

    from recognition import get_recognizer, serialise_embedding
    rec = get_recognizer()
    rebuilt = 0

    for img in images:
        fpath = Path(config.UPLOAD_FOLDER) / img["filename"]
        if not fpath.exists():
            continue
        try:
            embs = rec.embed_image_bytes(fpath.read_bytes())
            for emb in embs:
                models.add_embedding(
                    db, person_id=person_id, image_id=img["id"],
                    embedding_bytes=serialise_embedding(emb),
                )
            rebuilt += 1
        except Exception as exc:
            logger.warning("Retrain error for image %s: %s", img["filename"], exc)

    from recognition import reload_gallery
    reload_gallery()
    audit("EMBEDDINGS_REBUILT", user_id=session["user_id"],
          username=session["username"], target_type="enrolled_person",
          target_id=str(person_id), detail=f"{rebuilt} image(s) reprocessed",
          ip_address=get_client_ip())
    flash(f"Retrained {rebuilt} image(s).", "success")
    return redirect(url_for("admin.enrolled_detail", person_id=person_id))


# ---------------------------------------------------------------------------
# User management (admin only)
# ---------------------------------------------------------------------------

@admin_bp.route("/users")
@admin_required
def users():
    rows = models.get_all_users(get_db())
    return render_template("admin/users.html", users=rows)


@admin_bp.route("/users/add", methods=["GET", "POST"])
@admin_required
def user_add():
    db    = get_db()
    roles = models.get_all_roles(db)

    if request.method == "POST":
        username     = request.form.get("username", "").strip()
        display_name = request.form.get("display_name", "").strip()
        email        = request.form.get("email", "").strip()
        password     = request.form.get("password", "")
        role_id      = request.form.get("role_id", type=int)

        errors = []
        if not username:
            errors.append("Username is required.")
        if not password or len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if not role_id:
            errors.append("Role is required.")
        if models.get_user_by_username(db, username):
            errors.append("Username already exists.")

        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template("admin/user_form.html", user=None, roles=roles)

        uid = models.create_user(
            db, username=username, password_hash=hash_password(password),
            role_id=role_id, display_name=display_name, email=email,
            created_by=session["user_id"],
        )
        audit("USER_CREATED", user_id=session["user_id"],
              username=session["username"], target_type="user",
              target_id=str(uid), detail=f"username={username}",
              ip_address=get_client_ip())
        flash(f"User '{username}' created.", "success")
        return redirect(url_for("admin.users"))

    return render_template("admin/user_form.html", user=None, roles=roles)


@admin_bp.route("/users/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def user_edit(user_id: int):
    db    = get_db()
    user  = models.get_user_by_id(db, user_id)
    roles = models.get_all_roles(db)
    if not user:
        abort(404)

    if request.method == "POST":
        display_name = request.form.get("display_name", "").strip()
        email        = request.form.get("email", "").strip()
        role_id      = request.form.get("role_id", type=int)
        new_password = request.form.get("new_password", "").strip()

        updates = dict(display_name=display_name, email=email or None, role_id=role_id)
        if new_password:
            if len(new_password) < 8:
                flash("Password must be at least 8 characters.", "danger")
                return render_template("admin/user_form.html", user=user, roles=roles)
            updates["password_hash"] = hash_password(new_password)

        models.update_user(db, user_id, **updates)
        audit("USER_UPDATED", user_id=session["user_id"],
              username=session["username"], target_type="user",
              target_id=str(user_id), ip_address=get_client_ip())
        flash("User updated.", "success")
        return redirect(url_for("admin.users"))

    return render_template("admin/user_form.html", user=user, roles=roles)


@admin_bp.route("/users/<int:user_id>/toggle", methods=["POST"])
@admin_required
def user_toggle(user_id: int):
    db   = get_db()
    user = models.get_user_by_id(db, user_id)
    if not user:
        abort(404)
    if user_id == session["user_id"]:
        flash("You cannot disable your own account.", "danger")
        return redirect(url_for("admin.users"))

    new_state = 0 if user["is_active"] else 1
    models.update_user(db, user_id, is_active=new_state)
    label = "enabled" if new_state else "disabled"
    audit(f"USER_{label.upper()}", user_id=session["user_id"],
          username=session["username"], target_type="user",
          target_id=str(user_id), ip_address=get_client_ip())
    flash(f"User '{user['username']}' {label}.", "success")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def user_delete(user_id: int):
    db   = get_db()
    user = models.get_user_by_id(db, user_id)
    if not user:
        abort(404)
    if user_id == session["user_id"]:
        flash("You cannot delete your own account.", "danger")
        return redirect(url_for("admin.users"))

    models.delete_user(db, user_id)
    audit("USER_DELETED", user_id=session["user_id"],
          username=session["username"], target_type="user",
          target_id=str(user_id), detail=f"username={user['username']}",
          ip_address=get_client_ip())
    flash(f"User '{user['username']}' deleted.", "success")
    return redirect(url_for("admin.users"))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@admin_bp.route("/settings", methods=["GET", "POST"])
@admin_required
def settings():
    db = get_db()

    _setting_keys = [
        "recognition_threshold",
        "frame_skip",
        "night_vision_mode",
        "ghost_hunting_mode",
        "kinect_motor_enabled",
        "setup_cameras",
        "camera_preferred_source",
        "camera_index",
        "usb_camera_indices",
        "usb_camera_auto_discover",
        "usb_camera_scan_limit",
        "multi_camera_detection_enabled",
        "ip_camera_url",
        "ip_camera_urls",
        "ip_camera_rtsp_transport",
        "ip_camera_onvif_night_mode",
        "ip_camera_onvif_host",
        "ip_camera_onvif_port",
        "ip_camera_onvif_username",
        "ip_camera_onvif_password",
        "ip_camera_onvif_wsdl_dir",
        "tapo_host",
        "tapo_username",
        "tapo_password",
        "tapo_stream",
        "unknown_cooldown_seconds",
        "object_detection_backend",
        "object_detection_mode",
        "object_detection_threshold",
        "object_detection_device",
        "moving_object_detection_enabled",
        "moving_object_min_area",
        "moving_object_threshold",
        "min_recording_seconds",
        "max_recording_seconds",
        "record_all_mode",
        "save_recordings_locally",
        "record_audio_enabled",
        "video_encoder",
        "video_encoder_preset",
        "video_encoder_quality",
        "discord_cooldown_seconds",
        "discord_mention_everyone",
        "discord_webhook_url",
        "sound_enabled",
    ]

    if request.method == "POST":
        before = {k: get_setting(k) for k in _setting_keys}
        for key in _setting_keys:
            values = request.form.getlist(key)
            val = values[-1].strip() if values else ""
            set_setting(key, val, user_id=session["user_id"])
        _sync_camera_settings_from_setup_cameras(
            request.form.get("setup_cameras", ""),
            session["user_id"],
        )
        after = {k: get_setting(k) for k in _setting_keys}

        reload_keys = {
            "frame_skip",
            "night_vision_mode",
            "setup_cameras",
            "camera_preferred_source",
            "camera_index",
            "usb_camera_indices",
            "usb_camera_auto_discover",
            "usb_camera_scan_limit",
            "multi_camera_detection_enabled",
            "kinect_motor_enabled",
            "ip_camera_url",
            "ip_camera_urls",
            "ip_camera_rtsp_transport",
            "ip_camera_onvif_night_mode",
            "ip_camera_onvif_host",
            "ip_camera_onvif_port",
            "ip_camera_onvif_username",
            "ip_camera_onvif_password",
            "ip_camera_onvif_wsdl_dir",
            "object_detection_backend",
            "object_detection_mode",
            "object_detection_threshold",
            "object_detection_device",
            "moving_object_detection_enabled",
            "moving_object_min_area",
            "moving_object_threshold",
        }
        changed_keys = {k for k in _setting_keys if before.get(k) != after.get(k)}
        if (
            "record_all_mode" in changed_keys
            and before.get("record_all_mode", "false").strip().lower() == "true"
            and after.get("record_all_mode", "false").strip().lower() != "true"
        ):
            try:
                from recorder import get_recorder

                get_recorder().stop_continuous_recording("primary")
            except Exception as exc:
                logger.warning("Could not stop continuous recording after settings change: %s", exc)
        if changed_keys & reload_keys:
            try:
                from go2rtc_config import regenerate_config, reload_go2rtc

                regenerate_config()
                reload_go2rtc()
            except Exception as exc:
                logger.warning("go2rtc config reload failed after settings change: %s", exc)
            try:
                from camera import get_camera_loop

                get_camera_loop().restart_async()
            except Exception as exc:
                logger.warning("Camera runtime reload failed after settings change: %s", exc)

        audit("SETTINGS_CHANGED", user_id=session["user_id"],
              username=session["username"], ip_address=get_client_ip())
        flash("Settings saved. Camera changes are applied in-process without restarting Docker.", "success")
        return redirect(url_for("admin.settings"))

    current = {k: get_setting(k) for k in _setting_keys}
    return render_template("admin/settings.html", settings=current)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@admin_bp.route("/audit")
@admin_required
def audit_log():
    rows = models.get_audit_log(get_db(), limit=300)
    return render_template("admin/audit.html", entries=rows)


# ---------------------------------------------------------------------------
# System health
# ---------------------------------------------------------------------------

@admin_bp.route("/health")
@operator_required
def health():
    from camera import get_camera_status
    from recorder import get_recorder
    import platform, sys

    cam_status = get_camera_status()
    rec_state  = get_recorder().get_state()
    disk_pct   = disk_usage_percent(_storage_disk_path())
    storage_mb = dir_size_mb(config.RECORDINGS_DIR) + dir_size_mb(config.SNAPSHOTS_DIR)

    return render_template(
        "admin/health.html",
        cam_status=cam_status,
        rec_state=rec_state,
        disk_pct=disk_pct,
        storage_mb=round(storage_mb, 1),
        python_version=sys.version,
        platform_info=platform.platform(),
        recordings_dir=config.RECORDINGS_DIR,
        snapshots_dir=config.SNAPSHOTS_DIR,
        uploads_dir=config.UPLOAD_FOLDER,
    )


# ---------------------------------------------------------------------------
# JSON API for dashboard widgets
# ---------------------------------------------------------------------------

@admin_bp.route("/api/status")
@login_required
def api_status():
    db = get_db()
    from camera import get_camera_status
    from recorder import get_recorder

    cam  = get_camera_status()
    rec  = get_recorder().get_state()
    unkn = models.count_events_today(db, "unknown")
    recg = models.count_events_today(db, "recognized")

    # Streaming-camera count (independent of detection state) — used by the
    # topbar badge so "Live" reflects whether the user has any feed at all.
    from camera import get_live_sources
    live_sources = get_live_sources()
    detection_count = sum(1 for s in live_sources if s.get("detection"))

    return jsonify({
        "camera_running":      cam.get("running", False),
        "camera_error":        cam.get("error", ""),
        "camera_source":       cam.get("source", "webcam"),
        "live_camera_count":   len(live_sources),
        "detection_camera_count": detection_count,
        "night_vision":        cam.get("night_vision", False),
        "sls_enabled":         cam.get("sls_enabled", False),
        "sls_active":          cam.get("sls_active", False),
        "ghost_hunting_mode":  cam.get("ghost_hunting_mode", False),
        "depth":               cam.get("depth", False),
        "is_recording":        rec.is_recording,
        "recording_duration":  rec.duration_seconds,
        "record_all_mode":     (get_setting("record_all_mode", "false").strip().lower() == "true"),
        "multi_camera_detection": cam.get("multi_camera_detection", False),
        "multi_camera_sources": cam.get("multi_camera_sources", 0),
        "unknown_today":       unkn,
        "recognized_today":    recg,
        "enrolled_count":      len(models.get_all_enrolled(db)),
        "disk_pct":            disk_usage_percent(_storage_disk_path()),
    })


@admin_bp.route("/api/sensors")
@login_required
def api_sensors():
    from camera import get_camera_status, get_live_sources

    cam = get_camera_status()
    live_sources = get_live_sources()

    try:
        webcam_index = int(get_setting("camera_index", str(config.CAMERA_INDEX)) or config.CAMERA_INDEX)
    except (TypeError, ValueError):
        webcam_index = config.CAMERA_INDEX
    webcam_device = f"/dev/video{max(0, webcam_index)}"
    webcam_present = Path(webcam_device).exists()
    gpio_present = Path("/dev/gpiochip0").exists()
    audio_present = Path("/dev/snd").exists()
    usb_bus_present = Path("/dev/bus/usb").exists()

    kinect_parts = {"camera": False, "audio": False, "motor": False}
    try:
        out = subprocess.run(
            ["lsusb"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout
        text = out.lower()
        kinect_parts["camera"] = "045e:02ae" in text
        kinect_parts["audio"] = "045e:02ad" in text
        kinect_parts["motor"] = "045e:02b0" in text
    except Exception:
        pass

    return jsonify({
        "camera_running": cam.get("running", False),
        "camera_error": cam.get("error", ""),
        "camera_source": cam.get("source", "webcam"),
        "night_vision": cam.get("night_vision", False),
        "sls_enabled": cam.get("sls_enabled", False),
        "sls_active": cam.get("sls_active", False),
        "ghost_hunting_mode": cam.get("ghost_hunting_mode", False),
        "depth_available": cam.get("depth", False),
        "webcam_device": webcam_device,
        "webcam_present": webcam_present,
        "gpio_present": gpio_present,
        "audio_present": audio_present,
        "usb_bus_present": usb_bus_present,
        "kinect_enabled": bool(config.KINECT_ENABLED),
        "kinect_camera_present": kinect_parts["camera"],
        "kinect_audio_present": kinect_parts["audio"],
        "kinect_motor_present": kinect_parts["motor"],
        "kinect_fully_powered": all(kinect_parts.values()),
        "record_all_mode": (get_setting("record_all_mode", "false").strip().lower() == "true"),
        "multi_camera_detection": cam.get("multi_camera_detection", False),
        "multi_camera_sources": cam.get("multi_camera_sources", 0),
        "object_detection_backend": get_setting("object_detection_backend", config.OBJECT_DETECTION_BACKEND),
        "object_detection_mode": get_setting("object_detection_mode", config.OBJECT_DETECTION_MODE),
        "moving_object_detection": get_setting("moving_object_detection_enabled", "false").strip().lower() == "true",
        "ip_primary_configured": bool(get_setting("ip_camera_url", config.IP_CAMERA_URL).strip()),
        "ip_extra_count": len(_split_nonempty(get_setting("ip_camera_urls", config.IP_CAMERA_URLS))),
        "live_source_count": len(live_sources),
        "live_sources": live_sources,
    })


def _split_nonempty(raw: str) -> list[str]:
    values: list[str] = []
    for chunk in (raw or "").replace(",", "\n").splitlines():
        entry = chunk.strip()
        if entry and not entry.startswith("#"):
            values.append(entry)
    return values


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

@admin_bp.route("/heatmap")
@operator_required
def heatmap_page():
    from heatmap import get_stats
    stats = get_stats()
    return render_template("admin/heatmap.html", stats=stats)


@admin_bp.route("/api/heatmap.png")
@login_required
def heatmap_png():
    from heatmap import render_heatmap
    png = render_heatmap(width=640, height=480)
    return Response(png, mimetype="image/png",
                    headers={"Cache-Control": "no-store"})


@admin_bp.route("/api/heatmap/stats")
@login_required
def heatmap_stats():
    from heatmap import get_stats
    return jsonify(get_stats())


@admin_bp.route("/api/heatmap/reset", methods=["POST"])
@admin_required
def heatmap_reset():
    from heatmap import reset
    reset()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Serve uploaded face images
# ---------------------------------------------------------------------------

@admin_bp.route("/faces/<path:filename>")
@operator_required
def serve_face(filename: str):
    safe = Path(filename).name
    return send_from_directory(config.UPLOAD_FOLDER, safe)
