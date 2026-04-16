"""
kinect.py — Kinect v1 (Xbox 360) driver for CacheSec.

Provides a unified camera source that:
  - Streams RGB in normal (day) mode
  - Streams IR in night-vision mode (hardware IR projector, true darkness)
  - Exposes depth data for optional range filtering
  - Falls back gracefully if Kinect is not connected / not powered

Hardware requirements:
  - Xbox 360 Kinect (model 1414 or 1473)
  - USB data cable to Pi
  - 12V power supply (without it only the motor enumerates at 045e:02b0)

All 3 PIDs must appear in lsusb when fully powered:
  045e:02ae  camera
  045e:02ad  audio
  045e:02b0  motor

Software:
  sudo apt install libfreenect-dev freenect
  pip install freenect

Mode switching:
  The Kinect RGB and IR sensors share the same image pipeline — only one
  stream can be active at a time. We use freenect's sync API which manages
  its own internal context per format, making clean switching simple:
  just call sync_get_video with a different format constant.
  A 3-frame flush is performed on switch to drain the old stream buffer.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LED states
# ---------------------------------------------------------------------------

class KinectLED(IntEnum):
    OFF          = 0
    GREEN        = 1
    RED          = 2
    YELLOW       = 3
    BLINK_YELLOW = 4
    BLINK_GREEN  = 5
    BLINK_RED    = 6


# ---------------------------------------------------------------------------
# Lazy freenect import
# ---------------------------------------------------------------------------

try:
    import freenect as _fn
    _FREENECT_AVAILABLE = True
except ImportError:
    _fn = None
    _FREENECT_AVAILABLE = False
    logger.warning("freenect not installed — Kinect support disabled. "
                   "Install: sudo apt install libfreenect-dev && pip install freenect")


def kinect_available() -> bool:
    """True if freenect is installed AND a fully-powered Kinect is connected."""
    if not _FREENECT_AVAILABLE or _fn is None:
        return False
    try:
        ctx = _fn.init()
        n   = _fn.num_devices(ctx)
        _fn.shutdown(ctx)
        return n > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Thread-safe frame store
# ---------------------------------------------------------------------------

class _FrameStore:
    def __init__(self):
        self._lock  = threading.Lock()
        self._rgb   = None
        self._ir    = None
        self._depth = None

    def set_rgb(self, arr):
        with self._lock:
            self._rgb = arr

    def set_ir(self, arr):
        with self._lock:
            self._ir = arr

    def set_depth(self, arr):
        with self._lock:
            self._depth = arr

    def get_rgb(self):
        with self._lock:
            return None if self._rgb is None else self._rgb.copy()

    def get_ir(self):
        with self._lock:
            return None if self._ir is None else self._ir.copy()

    def get_depth(self):
        with self._lock:
            return None if self._depth is None else self._depth.copy()


# ---------------------------------------------------------------------------
# KinectSource
# ---------------------------------------------------------------------------

class KinectSource:
    """
    Thread-safe Kinect v1 frame source using freenect sync API.

    Modes
    -----
    rgb  — colour camera (640×480 BGR).  Used in daylight.
    ir   — infrared camera (640×480 greyscale + green tint).  Used at night.
           The Kinect's built-in IR projector illuminates the scene; works in
           complete darkness without any external illuminator.

    Both modes also capture depth frames continuously.

    Usage
    -----
        ks = KinectSource()
        if ks.start():
            frame = ks.read_frame()   # BGR, correct for current mode
            depth = ks.read_depth()
            ks.set_mode("ir")
            ks.set_tilt(10)
    """

    TILT_MIN = -27
    TILT_MAX =  27

    def __init__(self):
        self._store      = _FrameStore()
        self._mode       = "rgb"
        self._mode_lock  = threading.Lock()
        self._pending_mode_switch = False   # flush flag
        self._tilt       = 0
        self._stop_flag  = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready      = False
        self._error      = ""
        # Motor/LED dev opened separately (doesn't need full init)
        self._motor_ctx  = None
        self._motor_dev  = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._ready

    @property
    def error(self) -> str:
        return self._error

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start frame capture thread. Returns True if Kinect is available."""
        if not _FREENECT_AVAILABLE:
            self._error = "freenect not installed"
            return False
        if not kinect_available():
            self._error = "No Kinect detected — check USB and 12V power supply"
            logger.warning("KinectSource: %s", self._error)
            return False

        # Open a motor/LED handle on the main thread
        self._open_motor()

        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="KinectLoop"
        )
        self._thread.start()

        # Wait up to 4 s for first frame
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            if self._ready:
                logger.info("KinectSource ready (mode=%s)", self._mode)
                return True
            time.sleep(0.05)

        self._error = "Kinect opened but no frames received — is 12V connected?"
        logger.warning("KinectSource: %s", self._error)
        return False

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=4.0)
        self._close_motor()
        logger.info("KinectSource stopped")

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_frame(self) -> np.ndarray | None:
        """
        Return the latest frame as a BGR numpy array (640×480×3).
        In rgb mode:  colour image.
        In ir mode:   green-tinted IR image (same shape, ready for display
                      and face detection).
        Returns None if no frame is available yet.
        """
        if self._mode == "ir":
            return self._ir_to_bgr(self._store.get_ir())
        else:
            raw = self._store.get_rgb()
            if raw is None:
                return None
            import cv2
            return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

    def read_raw_ir(self) -> np.ndarray | None:
        """Return latest IR frame as 8-bit grayscale, or None."""
        raw = self._store.get_ir()
        if raw is None:
            return None
        # freenect VIDEO_IR_8BIT → already uint8; VIDEO_IR_10BIT → scale down
        if raw.dtype != np.uint8:
            raw = np.clip(raw.astype(np.float32) / 4.0, 0, 255).astype(np.uint8)
        return raw

    def read_depth(self) -> np.ndarray | None:
        """Return latest raw depth frame (uint16, 640×480), or None."""
        return self._store.get_depth()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch between 'rgb' and 'ir'. Thread-safe."""
        if mode not in ("rgb", "ir"):
            raise ValueError(f"mode must be 'rgb' or 'ir', got {mode!r}")
        with self._mode_lock:
            if self._mode != mode:
                self._mode = mode
                self._pending_mode_switch = True
        logger.info("Kinect mode change requested → %s", mode)

    def get_mode(self) -> str:
        with self._mode_lock:
            return self._mode

    def set_tilt(self, degrees: float) -> None:
        """Tilt motor. Range -27 (down) to +27 (up) degrees."""
        degrees = max(self.TILT_MIN, min(self.TILT_MAX, float(degrees)))
        self._tilt = degrees
        if self._motor_dev is not None and _fn is not None:
            try:
                _fn.set_tilt_degs(self._motor_dev, degrees)
            except Exception as exc:
                logger.warning("Kinect tilt error: %s", exc)

    def set_led(self, state: KinectLED) -> None:
        if self._motor_dev is not None and _fn is not None:
            try:
                _fn.set_led(self._motor_dev, int(state))
            except Exception as exc:
                logger.warning("Kinect LED error: %s", exc)

    def get_tilt_angle(self) -> float | None:
        if self._motor_dev is None or _fn is None:
            return None
        try:
            _fn.update_tilt_state(self._motor_dev)
            state = _fn.get_tilt_state(self._motor_dev)
            return float(_fn.get_tilt_degs(state))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Depth utilities
    # ------------------------------------------------------------------

    @staticmethod
    def depth_to_mm(raw: np.ndarray) -> np.ndarray:
        """Convert raw DEPTH_11BIT values to millimetres (float32)."""
        valid = raw < 2047
        k1, k2, k3 = 1.1863, 2842.5, 0.1236
        mm = np.where(
            valid,
            k3 * np.tan(raw.astype(np.float32) / k2 + k1) * 1000.0,
            0.0,
        )
        return mm.astype(np.float32)

    @staticmethod
    def person_in_range(depth_raw: np.ndarray,
                        max_mm: float = 3000.0,
                        min_coverage: float = 0.005) -> bool:
        """True if ≥min_coverage fraction of pixels are within max_mm."""
        valid = (depth_raw > 0) & (depth_raw < 2047)
        mm    = KinectSource.depth_to_mm(depth_raw)
        close = (mm > 200) & (mm < max_mm) & valid
        return float(close.sum()) / max(depth_raw.size, 1) >= min_coverage

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ir_to_bgr(gray: np.ndarray | None) -> np.ndarray | None:
        """
        Convert raw Kinect IR to a military-style green phosphor NV frame.

        Pipeline:
          1. Scale 10-bit → 8-bit if needed
          2. Median blur to kill the Kinect structured-light dot pattern
          3. Gaussian blur to smooth sensor noise before CLAHE
          4. CLAHE for local contrast stretch
          5. Unsharp mask to recover edge detail lost in blur
          6. Green phosphor tint (BGR: 20% blue, 100% green, 0% red)
        """
        if gray is None:
            return None
        import cv2

        # 1. Scale 10-bit → 8-bit
        if gray.dtype != np.uint8:
            gray = np.clip(gray.astype(np.float32) / 4.0, 0, 255).astype(np.uint8)

        # 2. Kill the Kinect dot-pattern noise with median blur
        gray = cv2.medianBlur(gray, 5)

        # 3. Light Gaussian blur before CLAHE for smoother result
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4. CLAHE — stronger clip for more punch
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        gray  = clahe.apply(gray)

        # 5. Unsharp mask — recover edge detail lost in blur steps
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        gray   = cv2.filter2D(gray, -1, kernel)

        # 6. Return as grayscale BGR (all channels equal)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _open_motor(self) -> None:
        """Open a device handle for motor/LED control."""
        if _fn is None:
            return
        try:
            ctx = _fn.init()
            dev = _fn.open_device(ctx, 0)
            self._motor_ctx = ctx
            self._motor_dev = dev
            _fn.set_led(dev, int(KinectLED.GREEN))
            _fn.set_tilt_degs(dev, self._tilt)
        except Exception as exc:
            logger.warning("Kinect motor handle failed (non-fatal): %s", exc)

    def _close_motor(self) -> None:
        if _fn is None:
            return
        try:
            if self._motor_dev is not None:
                _fn.set_led(self._motor_dev, int(KinectLED.OFF))
                _fn.close_device(self._motor_dev)
            if self._motor_ctx is not None:
                _fn.shutdown(self._motor_ctx)
        except Exception:
            pass
        self._motor_dev = None
        self._motor_ctx = None

    # ------------------------------------------------------------------
    # Frame capture loop — uses sync API exclusively
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Background thread. Uses freenect sync_get_* which manages its own
        internal libfreenect context — no manual start_video/stop_video needed.

        Mode switching: on mode change we flush 3 frames from the new format
        to drain any buffered frames from the old stream before marking ready.
        """
        assert _fn is not None

        # Determine format constants
        RGB_FMT   = _fn.VIDEO_RGB
        # Try 8-bit IR first (lower bandwidth), fall back to 10-bit
        IR_FMT    = getattr(_fn, 'VIDEO_IR_8BIT',
                    getattr(_fn, 'VIDEO_IR_10BIT', 3))
        DEPTH_FMT = _fn.DEPTH_11BIT

        current_mode  = None
        flush_needed  = 0   # frames to discard after mode switch

        logger.info("Kinect frame loop started")

        while not self._stop_flag.is_set():
            with self._mode_lock:
                desired_mode  = self._mode
                switch_flag   = self._pending_mode_switch
                if switch_flag:
                    self._pending_mode_switch = False

            # On mode change, flush frames to drain old stream and let the
            # IR projector spin up. The Kinect IR structured-light projector
            # activates as soon as the IR sync context starts grabbing, but
            # needs ~10 frames (~330ms at 30fps) to stabilise.
            if desired_mode != current_mode or switch_flag:
                flush_needed = 10 if desired_mode == "ir" else 5
                current_mode = desired_mode
                logger.info("Kinect stream: switching to %s (flushing %d frames)",
                            current_mode, flush_needed)
                # Brief pause so the old sync context fully releases before
                # the new one starts — prevents the IR projector being blocked
                time.sleep(0.15)
                # Update LED to reflect mode
                if self._motor_dev is not None:
                    led = KinectLED.BLINK_GREEN if current_mode == "ir" else KinectLED.GREEN
                    try:
                        _fn.set_led(self._motor_dev, int(led))
                    except Exception:
                        pass

            # --- Grab video frame ---
            try:
                fmt  = IR_FMT if current_mode == "ir" else RGB_FMT
                arr, _ts = _fn.sync_get_video(index=0, format=fmt)
            except Exception as exc:
                logger.debug("Kinect video grab error: %s", exc)
                time.sleep(0.05)
                continue

            if arr is None:
                time.sleep(0.03)
                continue

            if flush_needed > 0:
                flush_needed -= 1
                continue   # discard flush frames

            # Store frame
            if current_mode == "ir":
                self._store.set_ir(arr)
            else:
                self._store.set_rgb(arr)
            self._ready = True

            # --- Grab depth frame (best-effort, same sync context) ---
            try:
                depth, _ = _fn.sync_get_depth(index=0, format=DEPTH_FMT)
                if depth is not None:
                    self._store.set_depth(depth)
            except Exception:
                pass

        logger.info("Kinect frame loop ended")
        self._ready = False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_kinect: KinectSource | None = None
_kinect_lock = threading.Lock()


def get_kinect() -> KinectSource:
    global _kinect
    with _kinect_lock:
        if _kinect is None:
            _kinect = KinectSource()
    return _kinect
