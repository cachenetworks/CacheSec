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
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np

import config
from onvif_control import OnvifNightVisionController, OnvifNightVisionSettings
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
    "source":       "webcam",   # "webcam", "ip", or "kinect"
    "sls_enabled":  config.SLS_ENABLED,
    "sls_active":   False,
    "depth":        False,
    "multi_camera_detection": False,
    "multi_camera_sources": 0,
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


def _kinect_ambient_brightness(kinect, mode: str, frame: np.ndarray) -> float:
    """
    Mean brightness used to decide whether the Kinect should switch between
    RGB (day) and IR (night) mode.

    In rgb mode `frame` is the plain camera image, so its grayscale mean
    tracks room brightness directly. In ir mode `frame` is the CLAHE +
    unsharp-mask enhanced display image (see kinect.py's _ir_to_bgr) — that
    processing deliberately stretches contrast to fill the full 0-255 range,
    which flattens out its mean regardless of how bright the room actually
    is. Measuring brightness on that enhanced frame means the "room got
    bright again, switch back to RGB" check almost never fires, so it looks
    stuck in night vision. Read the raw (unenhanced) IR sensor frame instead
    — ambient light still raises it on top of the IR projector's dot
    pattern, so it tracks real room brightness.
    """
    if mode == "ir":
        raw = kinect.read_raw_ir()
        if raw is not None:
            return float(np.mean(raw))
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


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


def get_live_sources() -> list[dict]:
    """Return one live source for each configured camera.

    The setup wizard stores the canonical camera list in `setup_cameras`,
    while the Settings page still edits the older runtime keys
    (`ip_camera_url`, `ip_camera_urls`, `tapo_*`, etc.). Merge both so Live
    Feed does not disappear or keep stale titles after the user edits cameras
    from Settings.
    """
    sources: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(source: dict, key: tuple[str, str]) -> None:
        if not key[1] or key in seen:
            return
        seen.add(key)
        sources.append(source)

    cams = _load_setup_cameras()
    for ordinal, cam in enumerate(cams, start=1):
        cam_id = str(cam.get("id") or "").strip()
        cam_type = str(cam.get("type") or "").strip().lower()
        source_id = f"cam_{cam_id}" if cam_id else f"setup_{ordinal}"
        label = _wizard_cam_label(cam, ordinal)
        detail = _wizard_cam_describe(cam)
        if cam_type in {"ip", "tapo"}:
            url = _setup_camera_stream_url(cam)
            if not _is_supported_camera_url(url):
                continue
            # HLS/snapshot HTTP feeds are more reliable through CacheSec's
            # ffmpeg-backed MJPEG path than go2rtc's fMP4 endpoint.
            go2rtc_stream = ""
            if cam_id and not (_looks_like_hls_url(url) or _looks_like_snapshot_url(url)):
                go2rtc_stream = f"cam_{cam_id}"
            add({
                "id": source_id,
                "label": label,
                "kind": cam_type,
                "active": False,
                "detail": detail,
                "go2rtc_stream": go2rtc_stream,
                "detection": bool(cam.get("detection")),
            }, (cam_type, _normalize_camera_url(url)))
        elif cam_type == "webcam":
            idx = int(cam.get("index", 0))
            primary_idx = _primary_camera_index()
            add({
                "id": f"webcam{idx}" if idx != primary_idx else "webcam",
                "label": label,
                "kind": "webcam",
                "active": False,
                "detail": f"/dev/video{idx}",
                "go2rtc_stream": "",
                "detection": bool(cam.get("detection")),
            }, ("webcam", str(idx)))
        elif cam_type == "kinect":
            add({
                "id": "kinect",
                "label": label,
                "kind": "kinect",
                "active": False,
                "detail": "Kinect v1",
                "go2rtc_stream": "",
                "detection": bool(cam.get("detection")),
            }, ("kinect", "0"))
    if cams:
        return sources

    preferred = _live_camera_preferred_source()
    primary_idx = _primary_camera_index()
    if preferred == "webcam":
        add({
            "id": "webcam",
            "label": "USB / Pi Camera",
            "kind": "webcam",
            "active": False,
            "detail": f"/dev/video{primary_idx}",
            "go2rtc_stream": "",
            "detection": True,
        }, ("webcam", str(primary_idx)))
    elif preferred == "kinect":
        add({
            "id": "kinect",
            "label": "Kinect",
            "kind": "kinect",
            "active": False,
            "detail": "Kinect v1",
            "go2rtc_stream": "",
            "detection": True,
        }, ("kinect", "0"))

    try:
        from tapo_control import tapo_configured, tapo_rtsp_url, tapo_settings

        if tapo_configured():
            tapo = tapo_settings()
            url = tapo_rtsp_url(tapo)
            add({
                "id": "tapo",
                "label": tapo.get("label") or "Tapo Camera",
                "kind": "tapo",
                "active": False,
                "detail": f"{tapo.get('host', '')} ({tapo.get('stream', 'stream1')})",
                "go2rtc_stream": "",
                "detection": preferred == "tapo",
            }, ("tapo", _normalize_camera_url(url)))
    except Exception:
        pass

    for idx, item in enumerate(_configured_ip_sources(include_primary=True), start=1):
        url = str(item["url"])
        raw_label = str(item.get("label") or "").strip()
        label = raw_label or f"IP Camera {idx}"
        scheme = urlsplit(url).scheme.lower()
        if scheme == "usb":
            try:
                usb_index = int((urlsplit(url).netloc or "0").strip() or "0")
            except ValueError:
                usb_index = 0
            add({
                "id": "webcam" if usb_index == primary_idx else f"webcam{usb_index}",
                "label": raw_label or f"USB / Pi Camera {usb_index}",
                "kind": "webcam",
                "active": False,
                "detail": f"/dev/video{usb_index}",
                "go2rtc_stream": "",
                "detection": preferred == "webcam" and usb_index == primary_idx,
            }, ("webcam", str(usb_index)))
            continue
        if scheme == "kinect":
            add({
                "id": "kinect",
                "label": raw_label or "Kinect",
                "kind": "kinect",
                "active": False,
                "detail": "Kinect v1",
                "go2rtc_stream": "",
                "detection": preferred == "kinect",
            }, ("kinect", "0"))
            continue
        if scheme == "tapo":
            continue
        add({
            "id": f"ip{idx}",
            "label": label,
            "kind": "ip",
            "active": False,
            "detail": _display_source_url(url),
            "go2rtc_stream": "",
            "detection": preferred == "ip" and idx == 1,
        }, ("ip", _normalize_camera_url(url)))
    return sources


def _load_setup_cameras() -> list[dict]:
    import json
    try:
        raw = _live_setting("setup_cameras", "")
        cams = json.loads(raw) if raw else []
        return cams if isinstance(cams, list) else []
    except Exception:
        return []


def _wizard_cam_label(cam: dict, ordinal: int = 1) -> str:
    label = str(cam.get("label") or "").strip()
    if label:
        return label
    t = str(cam.get("type") or "").strip().lower()
    if t == "tapo":
        return "Tapo Camera"
    if t == "ip":
        return f"IP Camera {ordinal}"
    if t == "webcam":
        return f"USB / Pi Camera {cam.get('index', 0)}"
    if t == "kinect":
        return "Kinect"
    return "Camera"


def _setup_camera_stream_url(cam: dict) -> str:
    t = str(cam.get("type") or "").strip().lower()
    if t == "ip":
        return _normalize_camera_url(str(cam.get("url") or ""))
    if t == "tapo":
        host = str(cam.get("host") or "").strip()
        user = str(cam.get("username") or "admin") or "admin"
        password = str(cam.get("password") or "")
        stream = str(cam.get("stream") or "stream1").strip().lower()
        if stream not in {"stream1", "stream2"}:
            stream = "stream1"
        if not host or not password:
            return ""
        return f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}@{host}:554/{stream}"
    return ""


def _wizard_cam_describe(cam: dict) -> str:
    t = cam.get("type", "")
    if t == "tapo":
        return f"Tapo · {cam.get('host', '')}"
    if t == "ip":
        url = cam.get("url", "")
        return f"IP · {url[:60]}"
    if t == "webcam":
        return f"USB / Pi · /dev/video{cam.get('index', 0)}"
    if t == "kinect":
        return "Kinect v1"
    return t


class FFmpegFrameCapture:
    """Minimal VideoCapture-like wrapper for HLS streams via system ffmpeg."""

    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        fps: int = 15,
        http_options: dict[str, str] | None = None,
    ):
        self.url = url
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self._frame_bytes = self.width * self.height * 3
        self._proc: subprocess.Popen[bytes] | None = None

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logger.warning("ffmpeg is not available for HLS camera support")
            return

        vf = (
            f"fps={self.fps},"
            f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
            f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2"
        )
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "warning",
            "-nostdin",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-rw_timeout", "5000000",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "2",
        ]
        cmd += _ffmpeg_http_input_args(url, http_options)
        cmd += [
            "-i", url,
            "-an",
            "-vf", vf,
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-f", "rawvideo",
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self._frame_bytes * 2,
            )
        except Exception as exc:
            logger.warning("Failed to start ffmpeg for %s: %s", _redact_url(url), exc)
            self._proc = None

    def isOpened(self) -> bool:
        return bool(self._proc and self._proc.poll() is None and self._proc.stdout)

    def set(self, _prop_id: int, _value: float) -> bool:
        return False

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def _read_exact(self, size: int) -> bytes:
        if not self._proc or not self._proc.stdout:
            return b""
        data = bytearray()
        while len(data) < size:
            chunk = self._proc.stdout.read(size - len(data))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.isOpened():
            return False, None
        payload = self._read_exact(self._frame_bytes)
        if len(payload) != self._frame_bytes:
            self.release()
            return False, None
        frame = np.frombuffer(payload, dtype=np.uint8)
        return True, frame.reshape((self.height, self.width, 3))

    def release(self) -> None:
        proc = self._proc
        self._proc = None
        if not proc:
            return
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        try:
            proc.wait(timeout=1)
        except Exception:
            pass


CaptureHandle = cv2.VideoCapture | FFmpegFrameCapture


@dataclass(slots=True)
class _CameraSourceSpec:
    id: str
    label: str
    kind: str
    detail: str = ""
    url: str = ""
    index: int | None = None
    options: dict[str, str] = field(default_factory=dict)
    primary: bool = False


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

    # -- Step 5: return as grayscale BGR (all channels equal)
    return cv2.cvtColor(gray_dn, cv2.COLOR_GRAY2BGR)



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


class _DetectionSourceRuntime:
    def __init__(self, spec: _CameraSourceSpec, recognizer, recorder):
        self.spec = spec
        self.recognizer = recognizer
        self.recorder = recorder
        self.tracker = _UnknownTracker()
        self.cap: CaptureHandle | None = None
        self.frame_count = 0
        self.last_open_attempt = 0.0
        self.last_snapshot_read = 0.0
        self.kinect_source = None
        try:
            from person_detection import MotionDetector

            self.motion_detector = MotionDetector()
        except Exception:
            self.motion_detector = None

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def step(self) -> bool:
        frame = self._read_frame()
        if frame is None:
            return False

        display_frame = frame
        self.recorder.push_frame(display_frame.copy(), source_id=self.spec.id)
        self.frame_count += 1
        frame_skip = max(1, _live_int_setting("frame_skip", config.FRAME_SKIP))
        if self.frame_count % frame_skip != 0:
            return True

        _process_detection_frame(
            recognizer=self.recognizer,
            frame=frame,
            display_frame=display_frame,
            tracker=self.tracker,
            recorder=self.recorder,
            threshold=_live_threshold(),
            source_id=self.spec.id,
            source_label=self.spec.label,
            record_all_mode=False,
            depth_raw=self._read_depth(),
            audio_source=self.spec.url if self.spec.kind == "ip" else "",
            motion_detector=self.motion_detector,
        )
        return True

    def _read_frame(self) -> np.ndarray | None:
        if self.spec.kind == "ip" and _looks_like_snapshot_url(self.spec.url):
            now = time.monotonic()
            if now - self.last_snapshot_read < 1.0:
                return None
            self.last_snapshot_read = now
            return _read_snapshot_frame(self.spec.url, self.spec.label, self.spec.options)

        if self.spec.kind == "kinect":
            return self._read_kinect_frame()

        if self.cap is None or not self.cap.isOpened():
            if not self._open_capture():
                return None

        ok, frame = self.cap.read()
        if ok and frame is not None:
            return frame

        logger.warning("Auxiliary camera read failed for %s; reconnecting", self.spec.label)
        self.close()
        return None

    def _open_capture(self) -> bool:
        now = time.monotonic()
        if now - self.last_open_attempt < 5.0:
            return False
        self.last_open_attempt = now

        if self.spec.kind == "ip":
            transport = _live_setting(
                "ip_camera_rtsp_transport",
                config.IP_CAMERA_RTSP_TRANSPORT,
            ).strip().lower()
            self.cap = _open_stream_capture(
                self.spec.url,
                self.spec.label,
                transport=transport,
                http_options=self.spec.options,
            )
        elif self.spec.kind == "webcam" and self.spec.index is not None:
            cap = cv2.VideoCapture(int(self.spec.index))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
            else:
                cap.release()
                self.cap = None
        else:
            self.cap = None

        if self.cap is None or not self.cap.isOpened():
            logger.warning("Auxiliary camera unavailable: %s", self.spec.label)
            self.close()
            return False

        logger.info("Auxiliary camera opened for detection: %s", self.spec.label)
        return True

    def _read_kinect_frame(self) -> np.ndarray | None:
        if self.spec.index is None:
            return None
        try:
            from kinect import get_kinect

            if self.kinect_source is None:
                self.kinect_source = get_kinect(self.spec.index)
            if not self.kinect_source.available and not self.kinect_source.start():
                return None
            frame = self.kinect_source.read_frame()
            if frame is None:
                return None
            if not _night_vision_forced_off():
                mode = self.kinect_source.get_mode()
                gray_mean = _kinect_ambient_brightness(self.kinect_source, mode, frame)
                if mode != "ir" and gray_mean < NIGHT_VISION_THRESHOLD:
                    self.kinect_source.set_mode("ir")
                elif mode == "ir" and gray_mean > (NIGHT_VISION_THRESHOLD + NIGHT_VISION_HYSTERESIS):
                    self.kinect_source.set_mode("rgb")
            elif self.kinect_source.get_mode() != "rgb":
                self.kinect_source.set_mode("rgb")
            return frame
        except Exception as exc:
            logger.debug("Auxiliary Kinect read failed for %s: %s", self.spec.label, exc)
            return None

    def _read_depth(self) -> np.ndarray | None:
        if self.spec.kind != "kinect" or self.kinect_source is None:
            return None
        try:
            return self.kinect_source.read_depth()
        except Exception:
            return None


class _MultiCameraDetectionWorker:
    def __init__(self, recognizer, recorder, preferred_source: str):
        self.recognizer = recognizer
        self.recorder = recorder
        self.preferred_source = preferred_source
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self._runtimes: list[_DetectionSourceRuntime] = []

    def start(self) -> None:
        enabled = _live_bool_setting(
            "multi_camera_detection_enabled",
            config.MULTI_CAMERA_DETECTION_ENABLED,
        )
        if not enabled:
            _camera_status["multi_camera_detection"] = False
            _camera_status["multi_camera_sources"] = 0
            return

        specs = _live_auxiliary_source_specs(self.preferred_source)
        self._runtimes = [
            _DetectionSourceRuntime(spec, self.recognizer, self.recorder)
            for spec in specs
        ]
        _camera_status["multi_camera_detection"] = bool(self._runtimes)
        _camera_status["multi_camera_sources"] = len(self._runtimes)
        if not self._runtimes:
            return

        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="MultiCameraDetection",
        )
        self._thread.start()
        logger.info("Multi-camera detection started for %d source(s)", len(self._runtimes))

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        for runtime in self._runtimes:
            runtime.close()
        self._runtimes = []
        _camera_status["multi_camera_detection"] = False
        _camera_status["multi_camera_sources"] = 0

    def _run(self) -> None:
        while not self._stop_flag.is_set():
            did_work = False
            for runtime in list(self._runtimes):
                if self._stop_flag.is_set():
                    break
                did_work = runtime.step() or did_work
            time.sleep(0.02 if did_work else 0.2)


# ---------------------------------------------------------------------------
# Camera thread
# ---------------------------------------------------------------------------

class CameraLoop:
    def __init__(self):
        self._stop_flag  = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.RLock()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_flag.clear()
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="CameraLoop"
            )
            self._thread.start()
            logger.info("CameraLoop started")

    def stop(self, timeout: float = 5.0) -> None:
        with self._lifecycle_lock:
            self._stop_flag.set()
            if self._thread:
                self._thread.join(timeout=timeout)
            logger.info("CameraLoop stopped")

    def restart(self) -> None:
        with self._lifecycle_lock:
            self.stop(timeout=8.0)
            self.start()

    def restart_async(self) -> None:
        threading.Thread(
            target=self.restart,
            daemon=True,
            name="CameraLoopReload",
        ).start()

    # ------------------------------------------------------------------

    def _open_camera(self) -> cv2.VideoCapture | None:
        idx = _primary_camera_index()
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

    def _open_ip_camera(self) -> CaptureHandle | None:
        url = _live_setting("ip_camera_url", config.IP_CAMERA_URL).strip()
        if not url:
            logger.error("IP camera source selected but IP_CAMERA_URL is empty")
            _camera_status["error"] = "IP camera URL is not configured"
            return None

        transport = _live_setting(
            "ip_camera_rtsp_transport", config.IP_CAMERA_RTSP_TRANSPORT
        ).strip().lower()
        if transport not in {"tcp", "udp", "udp_multicast", "http"}:
            transport = "tcp"

        url = _normalize_camera_url(url)
        if not _is_supported_camera_url(url):
            logger.error("Unsupported IP camera URL: %s", _redact_url(url))
            _camera_status["error"] = "IP camera URL must use rtsp://, http://, or https://"
            return None

        # Prefer the go2rtc relay so the camera only sees a single connection.
        relay = _go2rtc_relay_for_primary("ip")
        open_url = relay or url

        cap = _open_stream_capture(open_url, "IP camera", transport=transport)
        if cap is None:
            logger.error("Cannot open IP camera stream: %s", _redact_url(open_url))
            _camera_status["error"] = "IP camera unavailable"
            return None

        logger.info("IP camera opened: %s%s",
                    _redact_url(open_url), " (via go2rtc)" if relay else "")
        return cap

    def _open_tapo_camera(self) -> CaptureHandle | None:
        from tapo_control import tapo_rtsp_url, tapo_settings
        s = tapo_settings()
        if not s["host"] or not s["password"]:
            logger.error("Tapo source selected but host/password are not configured")
            _camera_status["error"] = "Tapo camera credentials not configured"
            return None

        relay = _go2rtc_relay_for_primary("tapo")
        url = relay or tapo_rtsp_url(s)

        cap = _open_stream_capture(url, "Tapo camera", transport="tcp")
        if cap is None:
            logger.error("Cannot open Tapo camera stream at %s", _redact_url(url))
            _camera_status["error"] = "Tapo camera unavailable"
            return None

        logger.info("Tapo camera opened: %s (%s)%s",
                    s["host"], s["stream"], " via go2rtc" if relay else "")
        return cap

    def _run(self) -> None:
        global _night_vision_active

        recognizer = get_recognizer()
        recorder   = get_recorder()
        recorder.start_background()

        multi_detector: _MultiCameraDetectionWorker | None = None
        tracker        = _UnknownTracker()
        try:
            from person_detection import MotionDetector

            primary_motion_detector = MotionDetector()
        except Exception:
            primary_motion_detector = None
        frame_count    = 0
        reconnect_wait = 2
        record_all_mode = _live_bool_setting("record_all_mode", False)
        record_all_last_check = time.monotonic()
        record_all_last_signal = 0.0

        _camera_status["running"] = True
        _camera_status["error"]   = ""
        _night_vision_active = False
        _camera_status["night_vision"] = False
        _camera_status["sls_enabled"] = config.SLS_ENABLED
        _camera_status["sls_active"] = False
        _camera_status["ghost_hunting_mode"] = False
        _camera_status["depth"] = False

        # ---- Open preferred source, with fallback ----
        from kinect import get_kinect, kinect_available, KinectLED
        kinect = get_kinect()
        use_kinect = False
        cap = None
        preferred_source = _live_camera_preferred_source()
        ip_source_url = _normalize_camera_url(_live_setting("ip_camera_url", config.IP_CAMERA_URL))
        ip_onvif = _build_ip_onvif_controller(ip_source_url) if preferred_source == "ip" else None

        initial_ip_night = ip_onvif.initial_state() if ip_onvif is not None else None
        if initial_ip_night is not None:
            _night_vision_active = initial_ip_night
            _camera_status["night_vision"] = initial_ip_night

        def primary_source_label() -> str:
            if preferred_source == "ip":
                return "ip"
            if preferred_source == "tapo":
                return "tapo"
            return "webcam"

        def try_open_primary_source() -> CaptureHandle | None:
            return self.__try_open_primary(preferred_source)

        def start_kinect_source() -> bool:
            global _night_vision_active
            if not (config.KINECT_ENABLED and kinect_available()):
                return False
            if kinect.start():
                _camera_status["source"] = "kinect"
                if config.KINECT_TILT != 0:
                    kinect.set_tilt(config.KINECT_TILT)
                logger.info("Using Kinect as camera source")
                cap = None

                if _ghost_hunting_mode_enabled():
                    logger.info("Ghost Hunting Mode enabled — starting Kinect in IR/SLS mode")
                    _night_vision_active = True
                    _camera_status["night_vision"] = True
                    _camera_status["sls_active"] = True
                    kinect.set_mode("ir")
                    kinect.set_led(KinectLED.BLINK_GREEN)
                elif _night_vision_forced_off():
                    _night_vision_active = False
                    _camera_status["night_vision"] = False
                    _camera_status["sls_active"] = False
                    kinect.set_mode("rgb")
                    kinect.set_led(KinectLED.GREEN)
                elif config.SLS_ENABLED and config.SLS_MODE == "always":
                    logger.info("SLS always mode enabled — starting Kinect in IR/depth mode")
                    _night_vision_active = True
                    _camera_status["night_vision"] = True
                    _camera_status["sls_active"] = True
                    kinect.set_mode("ir")
                    kinect.set_led(KinectLED.BLINK_GREEN)
                else:
                    # Check brightness of first frame — if already dark, start in IR
                    first = kinect.read_frame()
                    if first is not None:
                        gray_mean = float(np.mean(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)))
                        if gray_mean < NIGHT_VISION_THRESHOLD:
                            logger.info("Room already dark (%.1f) — starting Kinect in IR mode", gray_mean)
                            _night_vision_active = True
                            _camera_status["night_vision"] = True
                            kinect.set_mode("ir")
                            kinect.set_led(KinectLED.BLINK_GREEN)
                        else:
                            kinect.set_led(KinectLED.GREEN)
                    else:
                        kinect.set_led(KinectLED.GREEN)
                return True
            else:
                logger.warning("Kinect detected but failed to start: %s — falling back to webcam",
                               kinect.error)
                return False

        def start_kinect_one_source() -> CaptureHandle | None:
            """Experimental fallback for Xbox One Kinect RGB via V4L2 index."""
            raw_index = os.environ.get("KINECT_ONE_CAMERA_INDEX", "-1").strip()
            try:
                index = int(raw_index)
            except ValueError:
                index = -1
            if index < 0:
                return None
            cap2 = cv2.VideoCapture(index)
            if not cap2.isOpened():
                cap2.release()
                logger.warning("Kinect One index %s not available", index)
                return None
            cap2.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            _camera_status["source"] = "kinect_one"
            _camera_status["night_vision"] = False
            _camera_status["sls_active"] = False
            _camera_status["depth"] = False
            logger.info("Using experimental Kinect One RGB source at /dev/video%s", index)
            return cap2

        def switch_to_kinect_night(gray_mean: float, force: bool = False) -> bool:
            global _night_vision_active
            if not force and (not config.KINECT_NIGHT_VISION_ENABLED or _night_vision_forced_off()):
                return False
            logger.info("Main camera dark (brightness=%.1f) - trying Kinect IR night vision", gray_mean)
            if not start_kinect_source():
                return False
            if cap is not None:
                cap.release()
            _night_vision_active = True
            _camera_status["night_vision"] = True
            _camera_status["source"] = "kinect"
            kinect.set_mode("ir")
            kinect.set_led(KinectLED.BLINK_GREEN)
            self._kinect_settle = 10
            logger.info("Switched to Kinect IR night vision")
            return True

        def switch_to_webcam_day(gray_mean: float):
            global _night_vision_active
            new_cap = try_open_primary_source()
            if new_cap is None:
                return None
            try:
                kinect.stop()
            except Exception:
                pass
            _night_vision_active = False
            _camera_status["night_vision"] = False
            _camera_status["sls_active"] = False
            _camera_status["depth"] = False
            _camera_status["source"] = primary_source_label()
            logger.info(
                "Kinect bright (brightness=%.1f) - switched back to %s camera",
                gray_mean, primary_source_label(),
            )
            return new_cap

        if preferred_source == "none":
            _camera_status["source"] = "none"
            _camera_status["error"] = ""
            logger.info("No detection camera configured — camera loop idle. "
                        "Add a camera in Settings and enable detection to start.")
            _camera_status["running"] = False
            recorder.stop_background()
            return

        if preferred_source in {"webcam", "ip", "tapo"}:
            _camera_status["source"] = primary_source_label()
            cap = try_open_primary_source()
            if cap is None:
                # Don't silently fall back to a different device — the user
                # picked this one explicitly. Surface the error and stop.
                logger.error("Preferred %s camera unavailable; not falling back.", primary_source_label())
                _camera_status["running"] = False
                recorder.stop_background()
                return
            logger.info("Using %s as camera source", primary_source_label())
        elif preferred_source == "kinect":
            use_kinect = start_kinect_source()
            if not use_kinect:
                cap = start_kinect_one_source()
                if cap is None:
                    logger.error("Kinect requested but unavailable; Kinect One fallback unavailable too.")
                    _camera_status["running"] = False
                    recorder.stop_background()
                    return

        if cap is None and not use_kinect:
            _camera_status["running"] = False
            recorder.stop_background()
            return

        multi_detector = _MultiCameraDetectionWorker(recognizer, recorder, preferred_source)
        multi_detector.start()

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
                        cap = try_open_primary_source()
                        if cap is None:
                            time.sleep(reconnect_wait)
                            continue
                        _camera_status["error"] = ""
                        frame_count = 0
                        continue

                frame_count += 1

                ghost_mode = _ghost_hunting_mode_enabled()
                _camera_status["ghost_hunting_mode"] = ghost_mode

                # ---- Night-vision ----
                if use_kinect:
                    if ghost_mode:
                        if not _night_vision_active or kinect.get_mode() != "ir":
                            _night_vision_active = True
                            _camera_status["night_vision"] = True
                            kinect.set_mode("ir")
                            kinect.set_led(KinectLED.BLINK_GREEN)
                    elif _night_vision_forced_off():
                        if _night_vision_active or kinect.get_mode() != "rgb":
                            _night_vision_active = False
                            _camera_status["night_vision"] = False
                            _camera_status["sls_active"] = False
                            _camera_status["depth"] = False
                            kinect.set_mode("rgb")
                            kinect.set_led(KinectLED.GREEN)
                    elif config.SLS_ENABLED and config.SLS_MODE == "always":
                        if not _night_vision_active or kinect.get_mode() != "ir":
                            _night_vision_active = True
                            _camera_status["night_vision"] = True
                            kinect.set_mode("ir")
                            kinect.set_led(KinectLED.BLINK_GREEN)
                    else:
                        # Skip brightness check for a few frames after a mode switch
                        # to let the Kinect flush its old-stream buffer before we
                        # evaluate brightness again.
                        kinect_settle = getattr(self, '_kinect_settle', 0)
                        if kinect_settle > 0:
                            self._kinect_settle = kinect_settle - 1
                        else:
                            gray_mean = _kinect_ambient_brightness(kinect, kinect.get_mode(), frame)
                            if not _night_vision_active and gray_mean < NIGHT_VISION_THRESHOLD:
                                _night_vision_active = True
                                _camera_status["night_vision"] = True
                                kinect.set_mode("ir")
                                kinect.set_led(KinectLED.BLINK_GREEN)
                                self._kinect_settle = 6   # skip 6 frames after switch
                                logger.info("Kinect → IR mode (brightness=%.1f)", gray_mean)
                            elif _night_vision_active and gray_mean > (NIGHT_VISION_THRESHOLD + NIGHT_VISION_HYSTERESIS):
                                if (
                                    preferred_source in {"webcam", "ip", "tapo"}
                                    and config.KINECT_NIGHT_VISION_ENABLED
                                ):
                                    new_cap = switch_to_webcam_day(gray_mean)
                                    if new_cap is not None:
                                        cap = new_cap
                                        use_kinect = False
                                        frame_count = 0
                                        continue
                                _night_vision_active = False
                                _camera_status["night_vision"] = False
                                kinect.set_mode("rgb")
                                kinect.set_led(KinectLED.GREEN)
                                self._kinect_settle = 6
                                logger.info("Kinect → RGB mode (brightness=%.1f)", gray_mean)

                    # kinect.read_frame() returns plain BGR in RGB mode and
                    # CLAHE-enhanced grayscale-as-BGR in IR mode — no further processing
                    display_frame = frame

                    # SLS skeleton overlay — uses Kinect depth data.
                    sls_active = bool(
                        config.SLS_ENABLED
                        and (
                            ghost_mode
                            or (
                                not _night_vision_forced_off()
                                and (_night_vision_active or config.SLS_MODE == "always")
                            )
                        )
                    )
                    _camera_status["sls_active"] = sls_active
                    if sls_active:
                        depth_for_skel = kinect.read_depth()
                        _camera_status["depth"] = depth_for_skel is not None
                        if depth_for_skel is not None:
                            from skeleton import overlay_skeletons
                            display_frame = overlay_skeletons(
                                display_frame.copy(),
                                depth_for_skel,
                                max_people=config.SLS_MAX_PEOPLE,
                            )
                    else:
                        _camera_status["depth"] = kinect.read_depth() is not None
                else:
                    _camera_status["sls_active"] = False
                    _camera_status["depth"] = False
                    if preferred_source == "tapo":
                        # Tapo cameras flip their own IR-cut filter at night;
                        # don't apply the software green-tint filter.
                        if _night_vision_active:
                            _night_vision_active = False
                            _camera_status["night_vision"] = False
                        display_frame = frame
                    elif preferred_source == "ip":
                        # IP cameras can expose IR-cut control via ONVIF. When
                        # enabled, use the same brightness detection as the
                        # webcam path, but send the switch command to the
                        # camera instead of drawing a fake green NV filter.
                        if _night_vision_forced_off():
                            if _night_vision_active:
                                _night_vision_active = False
                                _camera_status["night_vision"] = False
                        elif ip_onvif is not None and ip_onvif.detects_darkness():
                            gray_mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                            if not _night_vision_active and gray_mean < NIGHT_VISION_THRESHOLD:
                                if ip_onvif.set_night_vision(True):
                                    _night_vision_active = True
                                    _camera_status["night_vision"] = True
                                    logger.info("IP camera ONVIF night vision ON (brightness=%.1f)", gray_mean)
                            elif _night_vision_active and gray_mean > (NIGHT_VISION_THRESHOLD + NIGHT_VISION_HYSTERESIS):
                                if ip_onvif.set_night_vision(False):
                                    _night_vision_active = False
                                    _camera_status["night_vision"] = False
                                    logger.info("IP camera ONVIF night vision OFF (brightness=%.1f)", gray_mean)
                        elif _night_vision_active:
                            _night_vision_active = False
                            _camera_status["night_vision"] = False
                        display_frame = frame
                    else:
                        # Webcam: software NV filter
                        gray_mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                        if ghost_mode:
                            if switch_to_kinect_night(gray_mean, force=True):
                                use_kinect = True
                                cap = None
                                frame_count = 0
                                continue
                        elif _night_vision_forced_off():
                            if _night_vision_active:
                                _night_vision_active = False
                                _camera_status["night_vision"] = False
                                _set_camera_exposure(cap, night=False)
                        elif not _night_vision_active and gray_mean < NIGHT_VISION_THRESHOLD:
                            if switch_to_kinect_night(gray_mean):
                                use_kinect = True
                                cap = None
                                frame_count = 0
                                continue
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

                # Record the operator-visible image: normal frames in daylight,
                # night-vision/SLS-enhanced frames when those modes are active.
                recorder.push_frame(display_frame.copy(), source_id="primary")

                now = time.monotonic()
                # Keep this setting dynamic so it can be toggled at runtime.
                if now - record_all_last_check >= 2.0:
                    new_mode = _live_bool_setting("record_all_mode", False)
                    if record_all_mode and not new_mode:
                        recorder.stop_continuous_recording("primary")
                    record_all_mode = new_mode
                    record_all_last_check = now

                if record_all_mode and (now - record_all_last_signal) >= 1.0:
                    # Continuous recording mode. None means "no specific event id".
                    recorder.signal_unknown_visible(
                        None,
                        source_id="primary",
                        source_label="Primary Detection Feed",
                        audio_source=ip_source_url if preferred_source == "ip" else "",
                    )
                    record_all_last_signal = now

                # Run detection every FRAME_SKIP frames
                if frame_count % max(1, _live_int_setting("frame_skip", config.FRAME_SKIP)) != 0:
                    _set_latest_jpeg(display_frame)
                    continue

                depth_raw = kinect.read_depth() if use_kinect else None
                annotated = _process_detection_frame(
                    recognizer=recognizer,
                    frame=frame,
                    display_frame=display_frame,
                    tracker=tracker,
                    recorder=recorder,
                    threshold=_live_threshold(),
                    source_id="primary",
                    source_label="Primary Detection Feed",
                    record_all_mode=record_all_mode,
                    depth_raw=depth_raw,
                    audio_source=ip_source_url if preferred_source == "ip" else "",
                    motion_detector=primary_motion_detector,
                )
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
            if multi_detector is not None:
                multi_detector.stop()
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

    def __try_open_primary(self, preferred_source: str) -> CaptureHandle | None:
        if preferred_source == "ip":
            for attempt in range(5):
                if self._stop_flag.is_set():
                    return None
                cap = self._open_ip_camera()
                if cap:
                    return cap
                logger.info("Retrying IP camera open (attempt %d/5)", attempt + 1)
                time.sleep(2)
            _camera_status["error"] = "IP camera unavailable"
            return None
        if preferred_source == "tapo":
            for attempt in range(5):
                if self._stop_flag.is_set():
                    return None
                cap = self._open_tapo_camera()
                if cap:
                    return cap
                logger.info("Retrying Tapo camera open (attempt %d/5)", attempt + 1)
                time.sleep(2)
            _camera_status["error"] = "Tapo camera unavailable"
            return None
        return self.__try_open()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _live_setting(key: str, default: str = "") -> str:
    try:
        from database import get_setting
        value = get_setting(key, default)
    except Exception:
        return default
    return value if value is not None else default


def _live_int_setting(key: str, default: int = 0) -> int:
    raw = _live_setting(key, str(default)).strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _live_bool_setting(key: str, default: bool = False) -> bool:
    raw = _live_setting(key, "true" if default else "false")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _primary_camera_index() -> int:
    return max(0, _live_int_setting("camera_index", config.CAMERA_INDEX))


def _live_camera_preferred_source() -> str:
    """Returns the configured primary detection source.

    "none" means the user hasn't enabled detection on any camera; the loop
    must NOT silently fall back to /dev/video0.
    """
    source = _live_setting("camera_preferred_source", config.CAMERA_PREFERRED_SOURCE)
    source = source.strip().lower()
    return source if source in {"webcam", "ip", "kinect", "tapo", "none"} else "none"


def _go2rtc_relay_for_primary(cam_type: str) -> str:
    """Previously routed the detection loop through go2rtc to share a single
    camera connection. That caused MSE stutters in the browser: when
    OpenCV closed its RTSP read between frames, go2rtc dropped the producer
    (no consumers left), which restarted the browser's MSE buffer.

    Now detection pulls the camera directly. go2rtc only serves the
    browser-facing fMP4 + AAC transcode. The Tapo / IP camera sees two
    connections (one from cachesec, one from go2rtc), but both are stable.
    """
    return ""


def _live_auxiliary_source_specs(preferred: str | None = None) -> list[_CameraSourceSpec]:
    preferred = preferred or _live_camera_preferred_source()
    setup_cams = _load_setup_cameras()
    if setup_cams:
        return _setup_detection_source_specs(setup_cams)

    specs: list[_CameraSourceSpec] = []

    # USB indices are only exposed when the user explicitly configured them
    # in `usb_camera_indices`. Don't probe `/dev/video0` just because
    # CAMERA_INDEX has a default value.
    usb_indices = _configured_usb_indices()
    primary_idx = _primary_camera_index()
    for index in usb_indices:
        if preferred == "webcam" and index == primary_idx:
            continue
        source_id = "webcam" if index == primary_idx else f"webcam{index}"
        specs.append(_CameraSourceSpec(
            id=source_id,
            label=f"USB / Pi Camera {index}",
            kind="webcam",
            detail=f"index {index}",
            index=index,
        ))

    specs.extend(_kinect_source_specs(preferred))

    # Tapo as a first-class source whenever credentials are configured.
    try:
        from tapo_control import tapo_configured, tapo_settings, tapo_rtsp_url
        if tapo_configured():
            s = tapo_settings()
            specs.append(_CameraSourceSpec(
                id="tapo",
                label=s.get("label") or "Tapo Camera",
                kind="tapo",
                detail=f"{s['host']} ({s['stream']})",
                url=tapo_rtsp_url(s),
            ))
    except Exception:
        pass

    for idx, item in enumerate(_configured_ip_sources(include_primary=preferred != "ip"), start=1):
        url = str(item["url"])
        scheme = urlsplit(url).scheme.lower()
        if scheme == "usb":
            try:
                usb_index = int((urlsplit(url).netloc or "0").strip() or "0")
            except ValueError:
                usb_index = 0
            specs.append(_CameraSourceSpec(
                id=f"webcam{usb_index}",
                label=str(item["label"] or f"USB / Pi Camera {usb_index}"),
                kind="webcam",
                detail=f"index {usb_index}",
                index=usb_index,
            ))
        elif scheme == "kinect":
            specs.append(_CameraSourceSpec(
                id="kinect",
                label=str(item["label"] or "Kinect"),
                kind="kinect",
                detail="RGB/IR (hardware)",
                index=0,
            ))
        elif scheme == "tapo":
            try:
                from tapo_control import tapo_settings, tapo_rtsp_url
                s = tapo_settings()
                specs.append(_CameraSourceSpec(
                    id="tapo",
                    label=str(item["label"] or s.get("label") or "Tapo Camera"),
                    kind="tapo",
                    detail=f"{s['host']} ({s['stream']})",
                    url=tapo_rtsp_url(s),
                ))
            except Exception:
                continue
        else:
            specs.append(_CameraSourceSpec(
                id=f"ip{idx}",
                label=str(item["label"]),
                kind="ip",
                detail=_display_source_url(url),
                url=url,
                options=dict(item.get("options") or {}),
            ))
    return specs


def _setup_detection_source_specs(cams: list[dict]) -> list[_CameraSourceSpec]:
    detection_cams = [cam for cam in cams if cam.get("detection")]
    if len(detection_cams) <= 1:
        return []

    specs: list[_CameraSourceSpec] = []
    for ordinal, cam in enumerate(detection_cams[1:], start=2):
        spec = _setup_camera_detection_spec(cam, ordinal)
        if spec is not None:
            specs.append(spec)
    return specs


def _setup_camera_detection_spec(cam: dict, ordinal: int) -> _CameraSourceSpec | None:
    cam_id = str(cam.get("id") or f"setup_{ordinal}").strip()
    source_id = f"cam_{cam_id}" if not cam_id.startswith("cam_") else cam_id
    cam_type = str(cam.get("type") or "").strip().lower()
    label = _wizard_cam_label(cam, ordinal)
    if cam_type == "webcam":
        try:
            index = max(0, int(cam.get("index", 0)))
        except (TypeError, ValueError):
            index = 0
        return _CameraSourceSpec(
            id=source_id,
            label=label,
            kind="webcam",
            detail=f"index {index}",
            index=index,
        )
    if cam_type == "kinect":
        return _CameraSourceSpec(
            id=source_id,
            label=label,
            kind="kinect",
            detail="RGB/IR (hardware)",
            index=0,
        )
    if cam_type in {"ip", "tapo"}:
        url = _setup_camera_stream_url(cam)
        if not _is_supported_camera_url(url):
            return None
        options = cam.get("options") if isinstance(cam.get("options"), dict) else {}
        return _CameraSourceSpec(
            id=source_id,
            label=label,
            kind="ip",
            detail=_display_source_url(url),
            url=url,
            options=dict(options),
        )
    return None


def _configured_usb_indices() -> list[int]:
    raw = _live_setting("usb_camera_indices", config.USB_CAMERA_INDICES)
    values: list[int] = []
    seen: set[int] = set()

    if _live_bool_setting("usb_camera_auto_discover", config.USB_CAMERA_AUTO_DISCOVER):
        for index in _auto_discovered_usb_indices():
            if index not in seen:
                seen.add(index)
                values.append(index)

    for chunk in raw.replace(",", "\n").splitlines():
        text = chunk.strip()
        if not text or text.startswith("#"):
            continue
        try:
            index = int(text)
        except ValueError:
            logger.warning("Ignoring invalid USB camera index: %s", text)
            continue
        if index < 0 or index in seen:
            continue
        seen.add(index)
        values.append(index)
    return values


def _auto_discovered_usb_indices() -> list[int]:
    if os.name != "posix":
        return []
    limit = max(0, _live_int_setting("usb_camera_scan_limit", config.USB_CAMERA_SCAN_LIMIT))
    indices: list[int] = []
    for path in sorted(Path("/dev").glob("video*")):
        suffix = path.name.removeprefix("video")
        if not suffix.isdigit():
            continue
        index = int(suffix)
        if limit and index >= limit:
            continue
        indices.append(index)
    return indices


def _kinect_source_specs(preferred: str | None = None) -> list[_CameraSourceSpec]:
    if not config.KINECT_ENABLED:
        return []
    try:
        from kinect import kinect_count

        count = kinect_count()
    except Exception:
        count = 0
    preferred = preferred or _live_camera_preferred_source()
    specs: list[_CameraSourceSpec] = []
    for index in range(count):
        if preferred == "kinect" and index == 0:
            continue
        source_id = "kinect" if index == 0 else f"kinect{index + 1}"
        label = "Kinect v1" if index == 0 else f"Kinect v1 #{index + 1}"
        specs.append(_CameraSourceSpec(
            id=source_id,
            label=label,
            kind="kinect",
            detail="RGB/IR (hardware)",
            index=index,
        ))
    return specs


def _night_vision_forced_off() -> bool:
    mode = _live_setting("night_vision_mode", config.NIGHT_VISION_MODE)
    return mode.strip().lower() == "force_off"


def _ghost_hunting_mode_enabled() -> bool:
    """Ghost Hunting Mode: keep the Kinect in IR + SLS skeleton view at all
    times instead of only switching on when the room is dark. This is the
    "SLS camera" visual effect from paranormal TV shows — a depth-based
    stick-figure overlay on person-shaped blobs — not a scientific spirit
    or EMF detector. Takes priority over night_vision_mode=force_off since
    the user explicitly asked for it."""
    return _live_bool_setting("ghost_hunting_mode", False)


def _build_ip_onvif_controller(stream_url: str) -> OnvifNightVisionController | None:
    mode = _live_setting("ip_camera_onvif_night_mode", config.IP_CAMERA_ONVIF_NIGHT_MODE)
    mode = mode.strip().lower()
    if mode not in {"disabled", "detect", "force_off"}:
        mode = "disabled"
    if mode == "disabled":
        return None

    settings = OnvifNightVisionSettings(
        mode=mode,
        host=_live_setting("ip_camera_onvif_host", config.IP_CAMERA_ONVIF_HOST).strip(),
        port=_live_int_setting("ip_camera_onvif_port", config.IP_CAMERA_ONVIF_PORT),
        username=_live_setting("ip_camera_onvif_username", config.IP_CAMERA_ONVIF_USERNAME),
        password=_live_setting("ip_camera_onvif_password", config.IP_CAMERA_ONVIF_PASSWORD),
        wsdl_dir=_live_setting("ip_camera_onvif_wsdl_dir", config.IP_CAMERA_ONVIF_WSDL_DIR).strip(),
        stream_url=stream_url,
        force_persistence=False,
    )
    return OnvifNightVisionController(settings)


def _configured_ip_sources(include_primary: bool | None = None) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    # The primary IP detection camera is also surfaced here so the live page
    # has one tile per configured IP camera regardless of detection state.
    if include_primary is None:
        include_primary = True

    primary = _normalize_camera_url(_live_setting("ip_camera_url", config.IP_CAMERA_URL))
    if include_primary and _is_supported_camera_url(primary):
        entries.append({"label": "IP Camera", "url": primary, "options": {}})

    raw = _live_setting("ip_camera_urls", config.IP_CAMERA_URLS)
    for chunk in raw.splitlines():
        item = chunk.strip()
        if not item or item.startswith("#"):
            continue
        parsed = _parse_camera_source_entry(item, len(entries) + 1)
        if parsed:
            entries.append(parsed)
    return entries


def _parse_camera_source_entry(item: str, index: int) -> dict[str, object] | None:
    parts = [part.strip() for part in item.split("|") if part.strip()]
    if not parts:
        return None

    options_start = len(parts)
    for idx, part in enumerate(parts):
        candidate = _normalize_camera_url(part)
        # URL query strings contain "=" frequently; only treat as an option
        # segment when the token is not itself a supported URL.
        if "=" in part and not _is_supported_camera_url(candidate):
            options_start = idx
            break
    main_parts = parts[:options_start]
    option_parts = parts[options_start:]

    label = f"IP Camera {index}"
    url = ""
    if main_parts:
        first = _normalize_camera_url(main_parts[0])
        if _is_supported_camera_url(first):
            url = first
        else:
            label = main_parts[0]
            if len(main_parts) >= 2:
                url = _normalize_camera_url(main_parts[1])

    if not _is_supported_camera_url(url):
        return None

    options: dict[str, str] = {}
    for opt in option_parts:
        key, _, value = opt.partition("=")
        key = key.strip().lower().replace("-", "_")
        value = value.strip()
        if not value:
            continue
        if key in {"referer", "origin", "user_agent"}:
            options[key] = value

    return {"label": label, "url": url, "options": options}


def _normalize_camera_url(raw: str) -> str:
    url = (raw or "").strip()
    if not url or "://" in url:
        return url
    host = url.split("/", 1)[0]
    if "." in host or ":" in host or host.lower() == "localhost":
        return f"http://{url}"
    return url


def _is_supported_camera_url(url: str) -> bool:
    scheme = urlsplit(url).scheme.lower()
    return scheme in {"rtsp", "rtsps", "http", "https", "usb", "kinect", "tapo"}


def _set_ffmpeg_capture_options(url: str, transport: str | None = None) -> None:
    if not url.lower().startswith(("rtsp://", "rtsps://")):
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        return
    if not transport:
        transport = _live_setting(
            "ip_camera_rtsp_transport", config.IP_CAMERA_RTSP_TRANSPORT
        ).strip().lower()
    if transport not in {"tcp", "udp", "udp_multicast", "http"}:
        transport = "tcp"
    # OpenCV forwards these options to its ffmpeg backend. TCP is generally
    # more reliable for wireless RTSP cameras; nobuffer keeps latency down.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|fflags;nobuffer|max_delay;500000|stimeout;5000000"
    )


def _looks_like_hls_url(url: str) -> bool:
    parts = urlsplit(url)
    path = parts.path.lower()
    query = parts.query.lower()
    return path.endswith((".m3u8", ".m3u")) or ".m3u8" in path or "m3u8" in query


def _default_http_camera_user_agent() -> str:
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    )


def _default_http_camera_options(url: str) -> dict[str, str]:
    options = {"user_agent": _default_http_camera_user_agent()}
    host = urlsplit(url).netloc.lower()
    if "surfline" in host:
        options["referer"] = "https://www.surfline.com/"
        options["origin"] = "https://www.surfline.com"
    return options


def _http_camera_options(
    url: str,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    options = _default_http_camera_options(url)
    if overrides:
        for key, value in overrides.items():
            if value:
                options[key] = value
    return options


def _ffmpeg_http_input_args(
    url: str,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    if not url.lower().startswith(("http://", "https://")):
        return []
    options = _http_camera_options(url, overrides)
    args: list[str] = []
    user_agent = options.get("user_agent", "").strip()
    if user_agent:
        args += ["-user_agent", user_agent]
    header_lines: list[str] = []
    referer = options.get("referer", "").strip()
    origin = options.get("origin", "").strip()
    if referer:
        header_lines.append(f"Referer: {referer}")
    if origin:
        header_lines.append(f"Origin: {origin}")
    if header_lines:
        args += ["-headers", "\r\n".join(header_lines) + "\r\n"]
    return args


def _prime_capture(cap: CaptureHandle, attempts: int = 10) -> bool:
    for _ in range(max(1, attempts)):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True
        time.sleep(0.1)
    return False


def _open_hls_capture(
    url: str,
    label: str,
    http_options: dict[str, str] | None = None,
) -> FFmpegFrameCapture | None:
    cap = FFmpegFrameCapture(
        url,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        fps=15,
        http_options=http_options,
    )
    if not cap.isOpened():
        cap.release()
        logger.warning("ffmpeg HLS open failed for %s: %s", label, _redact_url(url))
        return None
    if not _prime_capture(cap, attempts=5):
        cap.release()
        logger.warning(
            "ffmpeg HLS produced no frames for %s: %s. "
            "If this is a web player feed, try referer/origin headers in IP_CAMERA_URLS.",
            label,
            _redact_url(url),
        )
        return None
    logger.info("Opened HLS stream via ffmpeg for %s: %s", label, _redact_url(url))
    return cap


def _open_stream_capture(
    url: str,
    label: str,
    transport: str | None = None,
    http_options: dict[str, str] | None = None,
) -> CaptureHandle | None:
    url = _normalize_camera_url(url)
    if not _is_supported_camera_url(url):
        return None

    # Prefer system ffmpeg for HLS playlists because OpenCV network support
    # varies a lot between builds, while ffmpeg handles .m3u8 reliably.
    if _looks_like_hls_url(url):
        return _open_hls_capture(url, label, http_options=http_options)

    _set_ffmpeg_capture_options(url, transport)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not _prime_capture(cap, attempts=10):
        cap.release()
        return None
    return cap


def _looks_like_snapshot_url(url: str) -> bool:
    path = urlsplit(url).path.lower()
    if path.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return True
    lowered = url.lower()
    return any(token in lowered for token in ("snapshot", "image", "still", "jpg"))


def _redact_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        if "@" not in parts.netloc:
            return url
        host = parts.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parts.scheme, f"***:***@{host}", parts.path, parts.query, parts.fragment))
    except Exception:
        return "<invalid-url>"


def _display_source_url(url: str) -> str:
    try:
        parts = urlsplit(_redact_url(url))
        suffix = "?..." if parts.query else ""
        text = urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")) + suffix
        return text if len(text) <= 120 else text[:117] + "..."
    except Exception:
        text = _redact_url(url)
        return text if len(text) <= 120 else text[:117] + "..."


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
                dev = f"/dev/video{_primary_camera_index()}"
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
                dev = f"/dev/video{_primary_camera_index()}"
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


def _process_detection_frame(
    recognizer,
    frame: np.ndarray,
    display_frame: np.ndarray,
    tracker: _UnknownTracker,
    recorder,
    threshold: float,
    source_id: str = "primary",
    source_label: str = "Primary Detection Feed",
    record_all_mode: bool = False,
    depth_raw: np.ndarray | None = None,
    audio_source: str = "",
    motion_detector=None,
) -> np.ndarray:
    faces: list[DetectedFace] = recognizer.detect(frame)

    object_detections = []
    object_mode = _live_setting("object_detection_mode", config.OBJECT_DETECTION_MODE).strip().lower()
    if object_mode not in {"person", "people_pets", "all"}:
        object_mode = "people_pets"
    if not faces or object_mode in {"people_pets", "all"}:
        try:
            from person_detection import get_object_detector

            object_detector = get_object_detector()
            if object_detector.is_enabled():
                object_detections = object_detector.detect(frame)
        except Exception as exc:
            logger.debug("Object detector unavailable for %s: %s", source_label, exc)

    if _live_bool_setting("moving_object_detection_enabled", config.MOVING_OBJECT_DETECTION_ENABLED):
        try:
            if motion_detector is not None:
                object_detections.extend(motion_detector.detect(
                    frame,
                    min_area=_live_int_setting(
                        "moving_object_min_area",
                        config.MOVING_OBJECT_MIN_AREA,
                    ),
                    threshold=_live_int_setting(
                        "moving_object_threshold",
                        config.MOVING_OBJECT_THRESHOLD,
                    ),
                ))
        except Exception as exc:
            logger.debug("Motion detector unavailable for %s: %s", source_label, exc)

    if not faces and not object_detections:
        if tracker.active and not record_all_mode:
            recorder.signal_unknown_gone(source_id)
            tracker.reset()
        return display_frame

    unknown_in_frame = False
    known_allowed_in_frame = False
    annotated = display_frame.copy()

    fh, fw = frame.shape[:2]

    for face in faces:
        if face.embedding is None:
            continue

        from heatmap import record_detection
        record_detection(face.bbox, frame_w=fw, frame_h=fh)

        x1, y1, x2, y2 = face.bbox
        face_crop = frame[y1:y2, x1:x2]
        from spoof import is_live
        live, spoof_reason = is_live(face_crop, face.bbox, depth_raw)
        if not live:
            logger.info("Spoof detected on %s (bbox=%s): %s", source_label, face.bbox, spoof_reason)
            _draw_face(annotated, face, "SPOOF", 0.0, color=(0, 165, 255))
            continue

        masked, mask_reason = _check_mask(face_crop)
        if masked:
            logger.info("Mask detected on %s (bbox=%s): %s", source_label, face.bbox, mask_reason)
            _draw_face(annotated, face, "MASKED", 0.0, color=(255, 165, 0))
            unknown_in_frame = True
            continue

        match = recognizer.match(face.embedding, threshold=threshold)

        if match:
            from database import raw_db_ctx
            import models as m
            with raw_db_ctx() as db:
                allowed = m.is_person_allowed_now(db, match.person_id)
            if allowed:
                known_allowed_in_frame = True
                _draw_face(annotated, face, match.person_name,
                           match.score, color=(0, 255, 0))
                _log_recognized(face, match, source_label)
                if tracker.active or tracker.confirming:
                    logger.info("Known person recognised on %s - cancelling unknown tracker", source_label)
                    if tracker.active and not record_all_mode:
                        recorder.signal_unknown_gone(source_id)
                    tracker.reset()
            else:
                unknown_in_frame = True
                _draw_face(annotated, face,
                           f"{match.person_name} (NO ACCESS)",
                           match.score, color=(0, 165, 255))
        else:
            unknown_in_frame = True
            _draw_face(annotated, face, "UNKNOWN", 0.0, color=(0, 0, 220))

    if object_detections:
        from heatmap import record_detection
        from person_detection import draw_object_detection

        for detection in object_detections:
            record_detection(detection.bbox, frame_w=fw, frame_h=fh)
            draw_object_detection(annotated, detection)
        if not known_allowed_in_frame:
            unknown_in_frame = True

    if unknown_in_frame:
        tracker.last_seen = time.monotonic()
        if tracker.active:
            recorder.signal_unknown_visible(
                tracker.event_id,
                source_id=source_id,
                source_label=source_label,
                audio_source=audio_source,
            )
        elif not tracker.in_cooldown():
            if not tracker.confirming:
                tracker.confirming    = True
                tracker.confirm_start = time.monotonic()
                logger.debug(
                    "Unknown seen on %s - confirming (need %.1fs)",
                    source_label,
                    tracker.CONFIRM_SECS,
                )
            elif tracker.is_confirmed():
                event_id = _create_unknown_event(frame, source_label=source_label)
                tracker.active          = True
                tracker.confirming      = False
                tracker.event_id        = event_id
                tracker.last_event_time = time.monotonic()
                recorder.signal_unknown_visible(
                    event_id,
                    source_id=source_id,
                    source_label=source_label,
                    audio_source=audio_source,
                )
                _alert_unknown(event_id, frame)
            if tracker.confirming and not tracker.active:
                elapsed = time.monotonic() - tracker.confirm_start
                remaining = max(0, tracker.CONFIRM_SECS - elapsed)
                cv2.putText(annotated,
                            f"Verifying... {remaining:.1f}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 200, 255), 2)
    else:
        if tracker.active and not record_all_mode:
            recorder.signal_unknown_gone(source_id)
            tracker.reset()
        elif tracker.confirming:
            tracker.confirming    = False
            tracker.confirm_start = 0.0
            logger.debug("Unknown gone from %s before confirmation - ignoring", source_label)

    return annotated


def _log_recognized(face: DetectedFace, match, source_label: str = "") -> None:
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
                notes=f"Camera: {source_label}" if source_label else "",
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


def _create_unknown_event(frame: np.ndarray, source_label: str = "") -> int:
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
                notes=f"Camera: {source_label}" if source_label else "",
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


def generate_mjpeg(source_id: str = "primary"):
    """Generator for Flask MJPEG streaming endpoint."""
    if source_id == "primary":
        while True:
            # Fall back to a "No feed" placeholder instead of yielding
            # nothing — otherwise a not-yet-started/stopped camera loop
            # just leaves the browser's <img> blank forever with no
            # indication anything is wrong.
            frame = get_latest_jpeg() or get_error_jpeg_bytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.05)  # ~20 FPS max to browser
        return

    setup_match = _setup_camera_for_source_id(source_id)
    if setup_match is not None:
        cam, ordinal = setup_match
        cam_type = str(cam.get("type") or "").strip().lower()
        label = _wizard_cam_label(cam, ordinal)
        if cam_type == "ip":
            options = cam.get("options") if isinstance(cam.get("options"), dict) else {}
            yield from _generate_ip_mjpeg(
                _setup_camera_stream_url(cam),
                label,
                http_options=dict(options),
            )
            return
        if cam_type == "tapo":
            yield from _generate_capture_mjpeg(_setup_camera_stream_url(cam), label=label)
            return

    webcam_index = _parse_webcam_source_id(source_id)
    if webcam_index is not None:
        yield from _generate_capture_mjpeg(webcam_index, label=f"USB webcam {webcam_index}")
        return

    kinect_index = _parse_kinect_source_id(source_id)
    if kinect_index is not None:
        yield from _generate_kinect_mjpeg(kinect_index)
        return

    if source_id == "tapo":
        try:
            from tapo_control import tapo_rtsp_url, tapo_configured
        except Exception:
            yield from _generate_error_mjpeg()
            return
        if not tapo_configured():
            yield from _generate_error_mjpeg()
            return
        url = tapo_rtsp_url()
        if not url:
            yield from _generate_error_mjpeg()
            return
        yield from _generate_capture_mjpeg(url, label="Tapo camera")
        return

    if source_id.startswith("ip"):
        try:
            idx = int(source_id[2:]) - 1
        except ValueError:
            idx = -1
        sources = _configured_ip_sources()
        if 0 <= idx < len(sources):
            source = sources[idx]
            yield from _generate_ip_mjpeg(
                source["url"],
                source["label"],
                http_options=source.get("options"),
            )
            return

    yield from _generate_error_mjpeg()


def _setup_camera_for_source_id(source_id: str) -> tuple[dict, int] | None:
    for ordinal, cam in enumerate(_load_setup_cameras(), start=1):
        cam_id = str(cam.get("id") or "").strip()
        if (cam_id and source_id == f"cam_{cam_id}") or source_id == f"setup_{ordinal}":
            return cam, ordinal
    return None


def _parse_webcam_source_id(source_id: str) -> int | None:
    if source_id == "webcam":
        return _primary_camera_index()
    if not source_id.startswith("webcam"):
        return None
    suffix = source_id[6:]
    if not suffix:
        return _primary_camera_index()
    try:
        return int(suffix)
    except ValueError:
        return None


def _parse_kinect_source_id(source_id: str) -> int | None:
    if source_id == "kinect":
        return 0
    if not source_id.startswith("kinect"):
        return None
    suffix = source_id[6:]
    if not suffix:
        return 0
    try:
        index = int(suffix) - 1
    except ValueError:
        return None
    return index if index >= 0 else None


def _generate_kinect_mjpeg(index: int = 0):
    try:
        from kinect import get_kinect, kinect_available
    except Exception:
        logger.warning("Kinect module unavailable for live stream")
        yield from _generate_error_mjpeg()
        return

    if not kinect_available(index):
        logger.warning("Kinect live feed requested but Kinect #%d is not available", index + 1)
        yield from _generate_error_mjpeg()
        return

    kinect = get_kinect(index)
    if not kinect.available and not kinect.start():
        logger.warning("Kinect live feed could not start: %s", kinect.error)
        yield from _generate_error_mjpeg()
        return

    while True:
        frame = kinect.read_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.03)


def _generate_ip_mjpeg(
    url: str,
    label: str,
    http_options: dict[str, str] | None = None,
):
    if _looks_like_snapshot_url(url):
        yield from _generate_snapshot_mjpeg(url, label, http_options=http_options)
        return
    yield from _generate_capture_mjpeg(url, label=label, http_options=http_options)


def _generate_capture_mjpeg(
    source,
    label: str = "camera",
    http_options: dict[str, str] | None = None,
):
    if isinstance(source, str):
        source = _normalize_camera_url(source)
        if not _is_supported_camera_url(source):
            logger.warning("Live feed has unsupported URL for %s: %s", label, _redact_url(source))
            yield from _generate_error_mjpeg()
            return
        cap = _open_stream_capture(source, label, http_options=http_options)
    else:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        logger.warning("Live feed could not open %s", label)
        yield from _generate_error_mjpeg()
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.2)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.03)
    finally:
        cap.release()


def _generate_snapshot_mjpeg(
    url: str,
    label: str,
    http_options: dict[str, str] | None = None,
):
    while True:
        try:
            frame = _read_snapshot_frame(url, label, http_options)
            if frame is None:
                raise ValueError("snapshot did not decode as an image")
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        except Exception as exc:
            logger.debug("Snapshot live feed failed for %s: %s", label, exc)
            yield from _generate_error_mjpeg(single=True)
        time.sleep(1.0)


def _read_snapshot_frame(
    url: str,
    label: str,
    http_options: dict[str, str] | None = None,
) -> np.ndarray | None:
    try:
        options = _http_camera_options(url, http_options)
        headers = {"User-Agent": options.get("user_agent", "CacheSec/1.0")}
        if options.get("referer"):
            headers["Referer"] = options["referer"]
        if options.get("origin"):
            headers["Origin"] = options["origin"]
        req = Request(url, headers=headers)
        with urlopen(req, timeout=5) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.debug("Snapshot fetch failed for %s: %s", label, exc)
        return None


def get_error_jpeg_bytes() -> bytes:
    img = np.zeros((240, 426, 3), dtype=np.uint8)
    cv2.putText(img, "No feed", (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else b""


def _generate_error_mjpeg(single: bool = False):
    payload = get_error_jpeg_bytes()
    while True:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        if single:
            return
        time.sleep(1.0)


def _capture_single_frame(source, http_options: dict | None = None) -> np.ndarray | None:
    """
    Open a capture just long enough to grab one frame, then release it.

    Used for live-feed grid thumbnails so having N cameras on screen doesn't
    turn into N continuous ~30fps capture+encode loops (see generate_mjpeg,
    which is right for the single-camera focus view but far too expensive
    to run per-tile for every camera at once) — a thumbnail only needs a
    fresh frame every few seconds.
    """
    if isinstance(source, str):
        cap = _open_stream_capture(source, "snapshot", http_options=http_options)
    else:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _encode_jpeg_bytes(frame: np.ndarray | None, quality: int = 70) -> bytes | None:
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def get_snapshot_jpeg(source_id: str = "primary") -> bytes | None:
    """
    Return one encoded JPEG frame for a live-feed grid thumbnail, without
    opening a persistent capture loop for it. Mirrors generate_mjpeg's
    source-id dispatch, but grabs a single frame per call instead of
    streaming continuously.
    """
    if source_id == "primary":
        return get_latest_jpeg() or None

    setup_match = _setup_camera_for_source_id(source_id)
    if setup_match is not None:
        cam, _ordinal = setup_match
        cam_type = str(cam.get("type") or "").strip().lower()
        url = _setup_camera_stream_url(cam)
        if cam_type == "ip":
            options = cam.get("options") if isinstance(cam.get("options"), dict) else {}
            if _looks_like_snapshot_url(url):
                return _encode_jpeg_bytes(_read_snapshot_frame(url, "ip camera", http_options=dict(options)))
            return _encode_jpeg_bytes(_capture_single_frame(url, http_options=dict(options)))
        if cam_type == "tapo":
            return _encode_jpeg_bytes(_capture_single_frame(url))

    webcam_index = _parse_webcam_source_id(source_id)
    if webcam_index is not None:
        return _encode_jpeg_bytes(_capture_single_frame(webcam_index))

    kinect_index = _parse_kinect_source_id(source_id)
    if kinect_index is not None:
        try:
            from kinect import get_kinect
        except Exception:
            return None
        kinect = get_kinect(kinect_index)
        # Don't call kinect.start() here — it can block for up to 4s waiting
        # for the first frame, which is fine once for a persistent stream
        # connection but far too slow to repeat on every few-second
        # thumbnail poll. Only serve a frame if it's already running
        # (as the active detection source, or from an open focus-view
        # stream) — otherwise fail fast so the poll doesn't stack up.
        if not kinect.available:
            return None
        return _encode_jpeg_bytes(kinect.read_frame())

    if source_id == "tapo":
        try:
            from tapo_control import tapo_rtsp_url, tapo_configured
        except Exception:
            return None
        if not tapo_configured():
            return None
        url = tapo_rtsp_url()
        if not url:
            return None
        return _encode_jpeg_bytes(_capture_single_frame(url))

    if source_id.startswith("ip"):
        try:
            idx = int(source_id[2:]) - 1
        except ValueError:
            idx = -1
        sources = _configured_ip_sources()
        if 0 <= idx < len(sources):
            source = sources[idx]
            if _looks_like_snapshot_url(source["url"]):
                return _encode_jpeg_bytes(
                    _read_snapshot_frame(source["url"], source["label"], http_options=source.get("options"))
                )
            return _encode_jpeg_bytes(
                _capture_single_frame(source["url"], http_options=source.get("options"))
            )

    return None
