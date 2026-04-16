"""
camera.py — Camera capture and main detection loop.

Responsibilities:
  1. Open the camera — Kinect v1 (preferred) or USB webcam (fallback).
  2. Continuously read frames.
  3. Run face detection every FRAME_SKIP frames.
  4. Match detected faces against enrolled gallery.
  5. Route results to recorder.py and sound.py.
  6. Save events and snapshots to the database.
  7. Expose the latest JPEG frame for the live web stream (MJPEG).
  8. Handle camera disconnects and attempt reconnection.
  9. Apply night-vision filter when frame is dark.

Kinect vs webcam night-vision:
  - Kinect: switches to the native IR stream (hardware IR projector illuminates
    the room in complete darkness; no software processing needed beyond the
    green tint). Face detection runs on the IR frame directly.
  - Webcam fallback: software gamma lift + CLAHE + green tint on the RGB frame.
    Quality is limited by how much light the sensor can collect.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import config
from recognition import get_recognizer, DetectedFace
from recorder import get_recorder
from utils import timestamped_filename

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state for live MJPEG stream
# ---------------------------------------------------------------------------
_latest_frame_lock  = threading.Lock()
_latest_jpeg: bytes = b""
_camera_status      = {
    "running":      False,
    "error":        "",
    "night_vision": False,
    "source":       "webcam",   # "webcam" or "kinect"
}

# ---------------------------------------------------------------------------
# Night-vision parameters
# ---------------------------------------------------------------------------
# Brightness threshold below which night-vision filter activates (0-255).
# Hysteresis band prevents rapid switching: activate below threshold,
# deactivate above threshold + NIGHT_VISION_HYSTERESIS.
NIGHT_VISION_THRESHOLD   = 100  # activate if mean brightness < 100/255
NIGHT_VISION_HYSTERESIS  = 25   # deactivate only when > 125/255
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def get_latest_jpeg() -> bytes:
    with _latest_frame_lock:
        return _latest_jpeg


def _set_latest_jpeg(frame: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _latest_frame_lock:
            global _latest_jpeg
            _latest_jpeg = buf.tobytes()


def get_camera_status() -> dict:
    return dict(_camera_status)


# ---------------------------------------------------------------------------
# Night-vision filter
# ---------------------------------------------------------------------------

def _apply_night_vision(frame: np.ndarray) -> np.ndarray:
    """
    Night-vision filter for genuinely dark frames:
      1. Gamma correction — aggressively brightens dark pixels while
         leaving bright pixels mostly untouched (gamma < 1 = brighten).
      2. CLAHE on the Y channel — stretches local contrast after brightening.
      3. Green phosphor tint.
    Works even when the frame is near-black because gamma lift happens
    before CLAHE, giving the equaliser actual signal to work with.
    """
    # -- Step 1: extreme gamma lift for near-pitch-black frames (gamma=0.12)
    inv_gamma = 1.0 / 0.12
    lut = np.array([
        min(255, int((i / 255.0) ** inv_gamma * 255))
        for i in range(256)
    ], dtype=np.uint8)
    brightened = cv2.LUT(frame, lut)

    # -- Step 2: convert to grayscale and discard colour noise —
    #    in true darkness colour channels are just sensor noise.
    #    Working in grayscale gives a cleaner NV look.
    gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)

    # -- Step 3: aggressive CLAHE (high clipLimit = more local contrast)
    clahe_hard = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    gray_eq = clahe_hard.apply(gray)

    # -- Step 4: light denoise to kill salt-and-pepper sensor noise
    gray_dn = cv2.fastNlMeansDenoising(gray_eq, h=10, templateWindowSize=7, searchWindowSize=21)

    # -- Step 5: green phosphor tint — map grayscale into green channel only
    zeros = np.zeros_like(gray_dn)
    # slight blue tint for that classic scope look
    blue  = (gray_dn.astype(np.uint16) * 30 // 100).astype(np.uint8)
    return cv2.merge([blue, gray_dn, zeros])


def _apply_ir_tint(frame: np.ndarray) -> np.ndarray:
    """
    Apply green phosphor tint to a Kinect IR frame (already bright grayscale).
    Much simpler than the webcam NV filter — no gamma boost needed.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    zeros = np.zeros_like(gray_eq)
    blue  = (gray_eq.astype(np.uint16) * 30 // 100).astype(np.uint8)
    return cv2.merge([blue, gray_eq, zeros])


_night_vision_active = False


# ---------------------------------------------------------------------------
# Unknown-detection state machine
# ---------------------------------------------------------------------------

class _UnknownTracker:
    """
    Tracks a continuous unknown-person event to prevent duplicate DB rows
    and require a sustained presence before triggering an alert.

    State machine:
      IDLE      → face appears → CONFIRMING (accumulate confirm_secs)
      CONFIRMING → face held for confirm_secs → ACTIVE (create event, start recording)
      CONFIRMING → face disappears → back to IDLE (false positive / glance)
      ACTIVE    → face gone   → COOLDOWN (signal recorder to stop after its gap)
      COOLDOWN  → cooldown_secs elapsed → IDLE (ready for next event)

    confirm_secs prevents a single-frame or brief appearance from triggering
    an alert — the person must be in frame continuously for ~3-4 seconds.
    """

    CONFIRM_SECS = 3.5   # must be seen this long before alert fires

    def __init__(self):
        self.active              = False
        self.confirming          = False      # seen but not yet confirmed
        self.confirm_start       = 0.0        # when we first saw this unknown
        self.event_id: int | None = None
        self.last_seen           = 0.0
        self.last_event_time     = 0.0
        self.cooldown_secs       = config.UNKNOWN_COOLDOWN_SECONDS

    def reset(self):
        self.active        = False
        self.confirming    = False
        self.confirm_start = 0.0
        self.event_id      = None
        self.last_seen     = 0.0
        self.last_event_time = time.monotonic()

    def in_cooldown(self) -> bool:
        return (time.monotonic() - self.last_event_time) < self.cooldown_secs

    def is_confirmed(self) -> bool:
        """True once the unknown has been in frame long enough to trigger."""
        return self.confirming and (time.monotonic() - self.confirm_start) >= self.CONFIRM_SECS

    def is_expired(self) -> bool:
        return (
            self.active
            and (time.monotonic() - self.last_seen) >= self.cooldown_secs
        )


# ---------------------------------------------------------------------------
# Camera thread
# ---------------------------------------------------------------------------

class CameraLoop:
    def __init__(self):
        self._stop_flag  = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="CameraLoop"
        )
        self._thread.start()
        logger.info("CameraLoop started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("CameraLoop stopped")

    # ------------------------------------------------------------------

    def _open_camera(self) -> cv2.VideoCapture | None:
        idx = config.CAMERA_INDEX
        # Try V4L2 backend first (most reliable on Pi OS)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            logger.error("Cannot open camera index %d", idx)
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimal latency

        # Start in auto-exposure (aperture priority) with neutral settings.
        # Night vision will switch to manual max when darkness is detected.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # 3 = aperture priority auto
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)      # neutral (default=0)
        cap.set(cv2.CAP_PROP_CONTRAST, 32)       # default
        cap.set(cv2.CAP_PROP_GAIN, 0)            # let auto handle it

        # Warm up: read and discard a few frames so the sensor settles
        for _ in range(5):
            cap.read()

        # Log what we actually got
        exp  = cap.get(cv2.CAP_PROP_EXPOSURE)
        gain = cap.get(cv2.CAP_PROP_GAIN)
        logger.info("Camera opened (index=%d, %dx%d, exposure=%.0f, gain=%.0f)",
                    idx, config.FRAME_WIDTH, config.FRAME_HEIGHT, exp, gain)
        return cap

    def _run(self) -> None:
        global _night_vision_active

        recognizer = get_recognizer()
        recorder   = get_recorder()
        recorder.start_background()

        tracker        = _UnknownTracker()
        frame_count    = 0
        reconnect_wait = 2

        _camera_status["running"] = True
        _camera_status["error"]   = ""

        # ---- Try Kinect first, fall back to webcam ----
        from kinect import get_kinect, kinect_available, KinectLED
        kinect = get_kinect()
        use_kinect = config.KINECT_ENABLED and kinect_available()

        if use_kinect:
            if kinect.start():
                _camera_status["source"] = "kinect"
                kinect.set_led(KinectLED.GREEN)
                if config.KINECT_TILT != 0:
                    kinect.set_tilt(config.KINECT_TILT)
                logger.info("Using Kinect as camera source")
                cap = None
            else:
                logger.warning("Kinect detected but failed to start: %s — falling back to webcam",
                               kinect.error)
                use_kinect = False

        if not use_kinect:
            _camera_status["source"] = "webcam"
            cap = self.__try_open()
            if cap is None:
                _camera_status["running"] = False
                recorder.stop_background()
                return

        try:
            while not self._stop_flag.is_set():

                # ---- Read frame from appropriate source ----
                if use_kinect:
                    frame = self._read_kinect_frame(kinect)
                    if frame is None:
                        time.sleep(0.03)
                        continue
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        logger.warning("Webcam read failed — reconnecting in %ds", reconnect_wait)
                        _camera_status["error"] = "Camera disconnected"
                        cap.release()
                        time.sleep(reconnect_wait)
                        cap = self.__try_open()
                        if cap is None:
                            time.sleep(reconnect_wait)
                            continue
                        _camera_status["error"] = ""
                        frame_count = 0
                        continue

                frame_count += 1

                # ---- Night-vision ----
                if use_kinect:
                    # Skip brightness check for a few frames after a mode switch
                    # to let the Kinect flush its old-stream buffer (3-frame flush
                    # in kinect.py) before we evaluate brightness again.
                    kinect_settle = getattr(self, '_kinect_settle', 0)
                    if kinect_settle > 0:
                        self._kinect_settle = kinect_settle - 1
                    else:
                        gray_mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                        if not _night_vision_active and gray_mean < NIGHT_VISION_THRESHOLD:
                            _night_vision_active = True
                            _camera_status["night_vision"] = True
                            kinect.set_mode("ir")
                            kinect.set_led(KinectLED.BLINK_GREEN)
                            self._kinect_settle = 6   # skip 6 frames after switch
                            logger.info("Kinect → IR mode (brightness=%.1f)", gray_mean)
                        elif _night_vision_active and gray_mean > (NIGHT_VISION_THRESHOLD + NIGHT_VISION_HYSTERESIS):
                            _night_vision_active = False
                            _camera_status["night_vision"] = False
                            kinect.set_mode("rgb")
                            kinect.set_led(KinectLED.GREEN)
                            self._kinect_settle = 6
                            logger.info("Kinect → RGB mode (brightness=%.1f)", gray_mean)

                    # kinect.read_frame() already returns green-tinted BGR in IR mode
                    # and normal BGR in RGB mode — no further processing needed
                    display_frame = frame
                else:
                    # Webcam: software NV filter
                    gray_mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                    if not _night_vision_active and gray_mean < NIGHT_VISION_THRESHOLD:
                        _night_vision_active = True
                        _camera_status["night_vision"] = True
                        _set_camera_exposure(cap, night=True)
                        logger.info("Night-vision ON (brightness=%.1f)", gray_mean)
                    elif _night_vision_active and gray_mean > (NIGHT_VISION_THRESHOLD + NIGHT_VISION_HYSTERESIS):
                        _night_vision_active = False
                        _camera_status["night_vision"] = False
                        _set_camera_exposure(cap, night=False)
                        logger.info("Night-vision OFF (brightness=%.1f)", gray_mean)

                    display_frame = _apply_night_vision(frame) if _night_vision_active else frame

                # Always push raw frame to recorder
                recorder.push_frame(frame.copy())

                # Run detection every FRAME_SKIP frames
                if frame_count % max(1, config.FRAME_SKIP) != 0:
                    _set_latest_jpeg(display_frame)
                    continue

                threshold = _live_threshold()

                # Detect faces — on raw frame for accuracy
                # For Kinect IR, the IR frame works well for SCRFD detection
                faces: list[DetectedFace] = recognizer.detect(frame)

                if not faces:
                    if tracker.active:
                        recorder.signal_unknown_gone()
                        tracker.reset()
                    _set_latest_jpeg(display_frame)
                    continue

                unknown_in_frame = False
                annotated = display_frame.copy()

                # Grab Kinect depth frame for spoof check (best-effort)
                depth_raw = None
                if use_kinect:
                    depth_raw = kinect.read_depth()

                fh, fw = frame.shape[:2]
                for face in faces:
                    if face.embedding is None:
                        continue

                    # Record every detection in the heatmap
                    from heatmap import record_detection
                    record_detection(face.bbox, frame_w=fw, frame_h=fh)

                    # --- Spoof check ---
                    x1, y1, x2, y2 = face.bbox
                    face_crop = frame[y1:y2, x1:x2]
                    from spoof import is_live
                    live, spoof_reason = is_live(face_crop, face.bbox, depth_raw)
                    if not live:
                        logger.info("Spoof detected (bbox=%s): %s", face.bbox, spoof_reason)
                        _draw_face(annotated, face, "SPOOF", 0.0, color=(0, 165, 255))
                        continue

                    # --- Mask check ---
                    masked, mask_reason = _check_mask(face_crop)
                    if masked:
                        logger.info("Mask detected (bbox=%s): %s", face.bbox, mask_reason)
                        _draw_face(annotated, face, "MASKED", 0.0, color=(255, 165, 0))
                        unknown_in_frame = True
                        continue

                    match = recognizer.match(face.embedding, threshold=threshold)

                    if match:
                        # Check access schedule — treat out-of-hours as unknown
                        from database import raw_db_ctx
                        import models as m
                        with raw_db_ctx() as db:
                            allowed = m.is_person_allowed_now(db, match.person_id)
                        if allowed:
                            _draw_face(annotated, face, match.person_name,
                                       match.score, color=(0, 255, 0))
                            _log_recognized(face, match)
                            # Cancel any pending unknown tracker — this is a known person
                            if tracker.active or tracker.confirming:
                                logger.info("Known person recognised — cancelling unknown tracker")
                                if tracker.active:
                                    recorder.signal_unknown_gone()
                                tracker.reset()
                        else:
                            unknown_in_frame = True
                            _draw_face(annotated, face,
                                       f"{match.person_name} (NO ACCESS)",
                                       match.score, color=(0, 165, 255))
                    else:
                        unknown_in_frame = True
                        _draw_face(annotated, face, "UNKNOWN", 0.0, color=(0, 0, 220))

                if unknown_in_frame:
                    tracker.last_seen = time.monotonic()
                    if tracker.active:
                        # Already confirmed and recording — keep signalling
                        recorder.signal_unknown_visible(tracker.event_id)
                    elif not tracker.in_cooldown():
                        if not tracker.confirming:
                            # First sighting — start the confirmation timer
                            tracker.confirming    = True
                            tracker.confirm_start = time.monotonic()
                            logger.debug("Unknown face seen — confirming (need %.1fs)", tracker.CONFIRM_SECS)
                        elif tracker.is_confirmed():
                            # Held long enough — fire the alert
                            event_id = _create_unknown_event(frame)
                            tracker.active          = True
                            tracker.confirming      = False
                            tracker.event_id        = event_id
                            tracker.last_event_time = time.monotonic()
                            recorder.signal_unknown_visible(event_id)
                            _alert_unknown(event_id, frame)
                        # else: still accumulating confirm time — show "VERIFYING" label
                        if tracker.confirming and not tracker.active:
                            elapsed = time.monotonic() - tracker.confirm_start
                            remaining = max(0, tracker.CONFIRM_SECS - elapsed)
                            # Overlay a "verifying" countdown on the annotated frame
                            cv2.putText(annotated,
                                        f"Verifying... {remaining:.1f}s",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 200, 255), 2)
                else:
                    if tracker.active:
                        recorder.signal_unknown_gone()
                        tracker.reset()
                    elif tracker.confirming:
                        # Disappeared before confirmation — reset silently
                        tracker.confirming    = False
                        tracker.confirm_start = 0.0
                        logger.debug("Unknown face gone before confirmation — ignoring")

                _set_latest_jpeg(annotated)

        finally:
            if use_kinect:
                try:
                    kinect.set_led(KinectLED.BLINK_RED)
                    kinect.stop()
                except Exception:
                    pass
            elif cap is not None:
                cap.release()
            recorder.stop_background()
            _camera_status["running"] = False

    def _read_kinect_frame(self, kinect) -> np.ndarray | None:
        """Read a frame from Kinect in whichever mode is currently active."""
        return kinect.read_frame()

    def __try_open(self) -> cv2.VideoCapture | None:
        for attempt in range(5):
            if self._stop_flag.is_set():
                return None
            cap = self._open_camera()
            if cap:
                return cap
            logger.info("Retrying camera open (attempt %d/5)", attempt + 1)
            time.sleep(2)
        _camera_status["error"] = "Camera unavailable"
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_mask(face_crop: np.ndarray) -> tuple[bool, str]:
    """
    Detect if the lower face is covered (mask, scarf, balaclava).

    Splits the face crop into upper (eyes/forehead) and lower (nose/mouth)
    halves and compares skin-tone pixel density. A real unmasked face has
    skin tone in both halves. A masked face has skin tone concentrated only
    in the upper half.

    Returns (masked: bool, reason: str).
    """
    if face_crop is None or face_crop.size == 0:
        return False, "no_crop"

    h, w = face_crop.shape[:2]
    if h < 20 or w < 20:
        return False, "too_small"

    # Convert to YCrCb — skin tone range is well-defined in this space
    ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
    # Standard skin-tone range in YCrCb
    lower = np.array([0,   133, 77],  dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)

    mid = h // 2
    upper_half = skin_mask[:mid, :]
    lower_half = skin_mask[mid:, :]

    upper_density = float(np.sum(upper_half > 0)) / max(upper_half.size, 1)
    lower_density = float(np.sum(lower_half > 0)) / max(lower_half.size, 1)

    # Masked if upper half has decent skin signal but lower half is mostly absent
    if upper_density > 0.15 and lower_density < 0.08:
        return True, f"lower_skin={lower_density:.2f}"

    return False, "ok"


def _set_camera_exposure(cap: cv2.VideoCapture, night: bool) -> None:
    """
    Switch camera between night (max exposure/gain) and day (auto) modes.
    V4L2 auto_exposure: 1 = manual, 3 = aperture-priority auto.
    """
    try:
        import subprocess, shutil
        if night:
            # Manual mode, max exposure, max gain
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, 5000)
            cap.set(cv2.CAP_PROP_GAIN, 100)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 64)
            # Also set via v4l2-ctl — some cameras ignore OpenCV props
            v4l2 = shutil.which("v4l2-ctl")
            if v4l2:
                dev = f"/dev/video{config.CAMERA_INDEX}"
                subprocess.run(
                    [v4l2, "-d", dev,
                     "--set-ctrl=auto_exposure=1",
                     "--set-ctrl=exposure_time_absolute=5000",
                     "--set-ctrl=gain=100",
                     "--set-ctrl=brightness=64"],
                    capture_output=True, timeout=2
                )
            logger.info("Camera exposure: NIGHT (manual max)")
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            cap.set(cv2.CAP_PROP_GAIN, 0)
            v4l2 = shutil.which("v4l2-ctl")
            if v4l2:
                dev = f"/dev/video{config.CAMERA_INDEX}"
                subprocess.run(
                    [v4l2, "-d", dev, "--set-ctrl=auto_exposure=3"],
                    capture_output=True, timeout=2
                )
            logger.info("Camera exposure: DAY (auto)")
    except Exception as exc:
        logger.debug("Exposure switch failed (non-fatal): %s", exc)


def _live_threshold() -> float:
    from database import get_setting
    try:
        return float(get_setting("recognition_threshold",
                                  str(config.RECOGNITION_THRESHOLD)))
    except (ValueError, TypeError):
        return config.RECOGNITION_THRESHOLD


def _draw_face(
    frame: np.ndarray,
    face: DetectedFace,
    label: str,
    score: float,
    color: tuple,
) -> None:
    x1, y1, x2, y2 = face.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label}" + (f" {score:.2f}" if score > 0 else "")
    cv2.putText(frame, text, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)



def _log_recognized(face: DetectedFace, match) -> None:
    """Log a recognized-person event (throttled to avoid DB spam)."""
    # Simple in-memory throttle: one DB write per person per 30s
    now = time.monotonic()
    key = match.person_id
    last = _recognized_throttle.get(key, 0)
    if now - last < 60:
        return
    _recognized_throttle[key] = now

    try:
        from database import raw_db_ctx
        import models as m
        with raw_db_ctx() as db:
            m.create_event(
                db,
                event_type="recognized",
                person_id=match.person_id,
                person_name=match.person_name,
                confidence=round(match.score, 4),
            )
    except Exception as exc:
        logger.warning("Failed to log recognized event: %s", exc)


_recognized_throttle: dict[int, float] = {}


def _save_snapshot(frame: np.ndarray) -> str:
    """Save a JPEG snapshot and return just the filename (not the full path)."""
    Path(config.SNAPSHOTS_DIR).mkdir(parents=True, exist_ok=True)
    fname = timestamped_filename("unknown", "jpg")
    fpath = str(Path(config.SNAPSHOTS_DIR) / fname)
    cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return fname  # only the filename — route reconstructs the full path


def _create_unknown_event(frame: np.ndarray) -> int:
    """Write an unknown event to the DB and return the new event_id."""
    snapshot_path = ""
    try:
        snapshot_path = _save_snapshot(frame)
    except Exception as exc:
        logger.warning("Snapshot save failed: %s", exc)

    try:
        from database import raw_db_ctx
        import models as m
        with raw_db_ctx() as db:
            event_id = m.create_event(
                db,
                event_type="unknown",
                snapshot_path=snapshot_path,
                occurred_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        return event_id
    except Exception as exc:
        logger.error("Failed to create unknown event: %s", exc)
        return -1


def _alert_unknown(event_id: int, frame: np.ndarray) -> None:
    """Fire sound and Discord notification for a new unknown event."""
    from sound import play_access_denied
    play_access_denied()

    # snapshot_path in DB is just the filename; build the full path for Discord
    snapshot_full_path = ""
    try:
        from database import raw_db_ctx
        import models as m
        with raw_db_ctx() as db:
            ev = m.get_event_by_id(db, event_id)
            if ev and ev["snapshot_path"]:
                snapshot_full_path = str(
                    Path(config.SNAPSHOTS_DIR) / Path(ev["snapshot_path"]).name
                )
    except Exception:
        pass

    from discord_notify import notify_unknown
    notify_unknown(
        event_id=event_id,
        occurred_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        recording_started=True,
        snapshot_path=snapshot_full_path,
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_camera_loop: CameraLoop | None = None


def get_camera_loop() -> CameraLoop:
    global _camera_loop
    if _camera_loop is None:
        _camera_loop = CameraLoop()
    return _camera_loop


def generate_mjpeg():
    """Generator for Flask MJPEG streaming endpoint."""
    while True:
        frame = get_latest_jpeg()
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.05)  # ~20 FPS max to browser
