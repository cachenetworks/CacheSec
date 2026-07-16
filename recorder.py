"""
recorder.py — Video recording logic for unknown detection events.

Recording contract:
  - Start when an unknown face is detected.
  - Keep recording while the unknown person remains visible.
  - If they disappear before MIN_RECORDING_SECONDS, hold open until min is met.
  - If they reappear while recording is active, extend (same clip, same event).
  - Stop when they have been absent for UNKNOWN_COOLDOWN_SECONDS AND min duration met.
  - Never exceed MAX_RECORDING_SECONDS; force-stop and save if hit.
  - Timestamped filenames: unknown_YYYYMMDD_HHMMSS.mp4
  - Links the recording to the event row in the DB.

Codec strategy:
  OpenCV's bundled ffmpeg does not include libx264, so writing H.264 directly
  via VideoWriter fails on Pi OS. Instead:
    1. Record to a temporary .avi file using MJPEG (always works with OpenCV).
    2. After the recording ends, spawn a background thread that calls the
       system ffmpeg (which has libx264) to re-encode to H.264 .mp4.
    3. The .avi is deleted after successful re-encode.
  This gives browser-playable H.264 mp4 without blocking the recording loop.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class RecordingState(NamedTuple):
    is_recording: bool
    event_id: int | None
    filename: str           # final .mp4 filename (may still be encoding)
    duration_seconds: float
    source_id: str = "primary"
    source_label: str = "Primary Detection Feed"


class Recorder:
    """
    Thread-safe video recorder.

    Usage:
        recorder = Recorder()
        recorder.start_background()

        recorder.push_frame(frame)
        recorder.signal_unknown_visible(event_id)
        recorder.signal_unknown_gone()

        recorder.stop_background()
    """

    def __init__(self):
        self._frame_q:   queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue(maxsize=64)
        self._cmd_q:     queue.Queue[tuple]              = queue.Queue()
        self._state_lock = threading.Lock()

        self._is_recording      = False
        self._event_id: int | None = None
        self._writer: cv2.VideoWriter | None = None
        self._avi_path          = ""   # temp .avi being written
        self._mp4_filename      = ""   # final .mp4 name (set at start)
        self._start_time        = 0.0
        self._last_visible_time = 0.0
        self._min_duration      = config.MIN_RECORDING_SECONDS
        self._max_duration      = config.MAX_RECORDING_SECONDS
        self._save_locally      = config.SAVE_RECORDINGS_LOCALLY
        self._record_audio      = config.RECORD_AUDIO_ENABLED
        self._audio_path        = ""
        self._audio_process: subprocess.Popen | None = None
        self._audio_source_override = ""
        self._frame_count       = 0
        self._avi_fps           = 15.0
        self._first_frame_time  = 0.0
        self._last_frame_time   = 0.0
        self._continuous_recording = False
        self._source_id         = "primary"
        self._source_label      = "Primary Detection Feed"

        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def signal_unknown_visible(
        self,
        event_id: int | None,
        source_id: str = "primary",
        source_label: str = "Primary Detection Feed",
        audio_source: str = "",
    ) -> None:
        self._cmd_q.put(("visible", event_id, source_id, source_label, audio_source))

    def signal_unknown_gone(self, source_id: str = "primary") -> None:
        self._cmd_q.put(("gone", None, source_id, "", ""))

    def stop_continuous_recording(self, source_id: str = "primary") -> None:
        self._cmd_q.put(("stop_continuous", None, source_id, "", ""))

    def push_frame(self, frame: np.ndarray, source_id: str = "primary") -> None:
        try:
            self._frame_q.put_nowait((source_id, frame))
        except queue.Full:
            pass

    def get_state(self) -> RecordingState:
        with self._state_lock:
            dur = (time.monotonic() - self._start_time) if self._is_recording else 0.0
            return RecordingState(
                is_recording=self._is_recording,
                event_id=self._event_id,
                filename=self._mp4_filename,
                duration_seconds=round(dur, 1),
                source_id=self._source_id,
                source_label=self._source_label,
            )

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="Recorder")
        self._thread.start()
        logger.info("Recorder background thread started")

    def stop_background(self, timeout: float = 5.0) -> None:
        self._stop_flag.set()
        self._cmd_q.put(("stop", None, "", "", ""))
        if self._thread:
            self._thread.join(timeout=timeout)
        self._finalise_recording()
        logger.info("Recorder stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        unknown_visible = False

        while not self._stop_flag.is_set():
            # Drain command queue
            while True:
                try:
                    item = self._cmd_q.get_nowait()
                except queue.Empty:
                    break
                cmd = item[0]
                arg = item[1] if len(item) > 1 else None
                source_id = item[2] if len(item) > 2 and item[2] else "primary"
                source_label = item[3] if len(item) > 3 and item[3] else "Primary Detection Feed"
                audio_source = item[4] if len(item) > 4 else ""

                if cmd == "visible":
                    if not self._is_recording:
                        unknown_visible         = True
                        self._last_visible_time = time.monotonic()
                        self._start_recording(arg, source_id, source_label, audio_source)
                    elif self._source_id == source_id:
                        unknown_visible         = True
                        self._last_visible_time = time.monotonic()
                        if arg is not None and self._event_id != arg:
                            with self._state_lock:
                                self._event_id = arg

                elif cmd == "gone":
                    if self._source_id == source_id:
                        unknown_visible = False

                elif cmd == "stop_continuous":
                    if self._is_recording and self._source_id == source_id and self._continuous_recording:
                        unknown_visible = False
                        logger.info("Continuous recording disabled - finalising current clip")
                        self._finalise_recording()

                elif cmd == "stop":
                    return

            active_source_id = self._source_id

            if self._is_recording:
                try:
                    item = self._frame_q.get(timeout=0.05)
                    frame_source_id, frame = item if item is not None else ("", None)
                    if (
                        frame_source_id == active_source_id
                        and self._writer
                        and frame is not None
                    ):
                        frame = _normalise_recording_frame(frame)
                        audio_process = None
                        audio_source = ""
                        if (
                            self._record_audio
                            and self._audio_path
                            and self._audio_process is None
                            and self._frame_count == 0
                        ):
                            audio_process, audio_source = _start_audio_capture(
                                self._audio_path,
                                self._audio_source_override,
                            )
                            with self._state_lock:
                                if audio_process:
                                    self._audio_process = audio_process
                                    logger.info("Audio synced to first video frame: %s", audio_source)
                                else:
                                    self._record_audio = False
                                    self._audio_path = ""
                                    logger.warning("Audio disabled for this recording; capture did not start")
                        now = time.monotonic()
                        if self._frame_count == 0:
                            self._first_frame_time = now
                        self._writer.write(frame)
                        self._frame_count += 1
                        self._last_frame_time = now
                except queue.Empty:
                    pass

                elapsed = time.monotonic() - self._start_time
                absent  = time.monotonic() - self._last_visible_time

                max_duration = max(1, self._max_duration)
                min_duration = max(1, self._min_duration)

                if elapsed >= max_duration:
                    logger.warning("Max recording duration reached — stopping")
                    self._finalise_recording()

                elif (
                    not unknown_visible
                    and elapsed >= min_duration
                    and absent  >= 3.0   # 3-second settling gap after camera signals gone
                ):
                    logger.info("Unknown person absent %.1fs — stopping recording", absent)
                    self._finalise_recording()

            else:
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    time.sleep(0.05)

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def _start_recording(
        self,
        event_id: int | None,
        source_id: str = "primary",
        source_label: str = "Primary Detection Feed",
        audio_source: str = "",
    ) -> None:
        Path(config.RECORDINGS_DIR).mkdir(parents=True, exist_ok=True)
        ts          = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        source_fragment = _safe_filename_fragment(source_id)
        prefix = "continuous" if event_id is None else "unknown"
        if source_fragment and source_fragment != "primary":
            prefix = f"{prefix}_{source_fragment}"
        mp4_fname   = f"{prefix}_{ts}.mp4"
        avi_path    = str(Path(config.RECORDINGS_DIR) / f"{prefix}_{ts}.avi")
        min_duration = _int_setting("min_recording_seconds", config.MIN_RECORDING_SECONDS)
        max_duration = _int_setting("max_recording_seconds", config.MAX_RECORDING_SECONDS)
        save_locally = _bool_setting("save_recordings_locally", config.SAVE_RECORDINGS_LOCALLY)
        record_audio = _bool_setting("record_audio_enabled", config.RECORD_AUDIO_ENABLED)
        audio_path = str(Path(config.RECORDINGS_DIR) / f"{prefix}_{ts}.wav") if record_audio else ""

        # MJPEG into AVI — always works with OpenCV's bundled ffmpeg
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps    = 15.0
        size   = (config.FRAME_WIDTH, config.FRAME_HEIGHT)

        writer = cv2.VideoWriter(avi_path, fourcc, fps, size)
        if not writer.isOpened():
            logger.error("VideoWriter failed to open: %s", avi_path)
            writer.release()
            _delete_quietly(audio_path)
            return

        with self._state_lock:
            self._is_recording      = True
            self._event_id          = event_id
            self._writer            = writer
            self._avi_path          = avi_path
            self._mp4_filename      = mp4_fname
            self._start_time        = time.monotonic()
            self._last_visible_time = time.monotonic()
            self._min_duration      = min_duration
            self._max_duration      = max_duration
            self._save_locally      = save_locally
            self._record_audio      = record_audio
            self._audio_path        = audio_path
            self._audio_process     = None
            self._audio_source_override = audio_source
            self._frame_count       = 0
            self._avi_fps           = fps
            self._first_frame_time  = 0.0
            self._last_frame_time   = 0.0
            self._continuous_recording = event_id is None
            self._source_id         = source_id
            self._source_label      = source_label

        logger.info(
            "Recording started (MJPEG/AVI): %s  event=%s  source=%s  save_locally=%s  audio=%s",
            avi_path, str(event_id), source_label, save_locally, "pending" if record_audio else "off",
        )

        started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        mp4_path   = str(Path(config.RECORDINGS_DIR) / mp4_fname)
        fields = {"recording_start": started_at}
        notes = []
        if save_locally:
            fields["recording_path"] = mp4_path
        else:
            notes.append("Recording will be uploaded to Discord only.")
        if source_label and source_label != "Primary Detection Feed":
            notes.append(f"Camera: {source_label}.")
        if notes:
            fields["notes"] = " ".join(notes)
        self._update_event_recording(event_id, **fields)

    def _finalise_recording(self) -> None:
        with self._state_lock:
            if not self._is_recording:
                return
            writer      = self._writer
            event_id    = self._event_id
            avi_path    = self._avi_path
            mp4_fname   = self._mp4_filename
            start       = self._start_time
            save_locally = self._save_locally
            audio_path  = self._audio_path
            audio_process = self._audio_process
            frame_count = self._frame_count
            avi_fps     = self._avi_fps
            first_frame_time = self._first_frame_time
            last_frame_time = self._last_frame_time
            source_id   = self._source_id
            source_label = self._source_label
            self._is_recording  = False
            self._writer        = None
            self._event_id      = None
            self._avi_path      = ""
            self._mp4_filename  = ""
            self._audio_path    = ""
            self._audio_process = None
            self._audio_source_override = ""
            self._frame_count   = 0
            self._first_frame_time = 0.0
            self._last_frame_time = 0.0
            self._continuous_recording = False
            self._source_id     = "primary"
            self._source_label  = "Primary Detection Feed"

        elapsed_duration = round(time.monotonic() - start, 3)
        frame_duration = _frame_wall_duration(
            frame_count, first_frame_time, last_frame_time, avi_fps
        )
        duration = frame_duration or elapsed_duration
        if elapsed_duration > 0:
            duration = min(duration, elapsed_duration)

        if writer:
            writer.release()
        if audio_process:
            _stop_audio_capture(audio_process)
        if not _usable_audio_file(audio_path):
            _delete_quietly(audio_path)
            audio_path = ""

        if frame_count <= 0:
            logger.info("Recording stopped before any frames were written; discarding empty clip")
            _delete_quietly(avi_path)
            _delete_quietly(audio_path)
            return

        ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(
            "Raw recording closed: %s  source=%s  duration=%.1fs",
            avi_path, source_label or source_id, duration,
        )

        if event_id is not None:
            self._update_event_recording(event_id, recording_end=ended_at, ended_at=ended_at)

        # Re-encode in a daemon thread so we don't block anything
        mp4_path = str(Path(config.RECORDINGS_DIR) / mp4_fname)
        t = threading.Thread(
            target=self._reencode,
            args=(
                avi_path,
                mp4_path,
                mp4_fname,
                event_id,
                duration,
                ended_at,
                save_locally,
                audio_path,
                frame_count,
                avi_fps,
            ),
            daemon=True,
            name="Reencoder",
        )
        t.start()

        from discord_notify import clear_event_cooldown
        if event_id:
            clear_event_cooldown(event_id)

    def _reencode(
        self,
        avi_path: str,
        mp4_path: str,
        mp4_fname: str,
        event_id: int | None,
        duration: float,
        ended_at: str,
        save_locally: bool,
        audio_path: str = "",
        frame_count: int = 0,
        avi_fps: float = 15.0,
    ) -> None:
        """
        Re-encode the temp AVI to H.264 MP4 using system ffmpeg.
        Runs in a background thread after recording ends.
        """
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            # ffmpeg not found — just rename the AVI as a fallback
            logger.warning("ffmpeg not found; keeping .avi file as-is")
            avi_fname = Path(avi_path).name
            if save_locally:
                self._save_recording_row(
                    event_id=event_id,
                    filename=avi_fname,
                    file_size_bytes=_file_size(avi_path),
                    duration_seconds=duration,
                    started_at=ended_at,
                    ended_at=ended_at,
                )
                self._update_event_recording(event_id, recording_path=avi_path)
            else:
                self._upload_and_discard(avi_path, avi_fname, event_id, duration, ended_at)
            _delete_quietly(audio_path)
            return

        logger.info("Re-encoding %s → %s", Path(avi_path).name, Path(mp4_path).name)
        video_filter = _timeline_video_filter(frame_count, duration, avi_fps)
        logger.info(
            "Recording timeline: frames=%d real_duration=%.1fs filter=%s",
            frame_count,
            duration,
            video_filter or "none",
        )
        encoder = _select_video_encoder(ffmpeg)
        cmd = _build_reencode_command(
            ffmpeg=ffmpeg,
            avi_path=avi_path,
            mp4_path=mp4_path,
            audio_path=audio_path,
            video_filter=video_filter,
            encoder=encoder,
        )
        try:
            encode_timeout = _encode_timeout(duration)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=encode_timeout
            )
            if result.returncode != 0 and encoder != "libx264":
                logger.warning(
                    "ffmpeg %s encode failed; retrying with libx264:\n%s",
                    encoder,
                    result.stderr[-1000:],
                )
                encoder = "libx264"
                cmd = _build_reencode_command(
                    ffmpeg=ffmpeg,
                    avi_path=avi_path,
                    mp4_path=mp4_path,
                    audio_path=audio_path,
                    video_filter=video_filter,
                    encoder=encoder,
                )
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=encode_timeout
                )
            if result.returncode == 0:
                mp4_size = _file_size(mp4_path)
                logger.info(
                    "Re-encode complete: %s (%.1f MB, encoder=%s)",
                    mp4_fname,
                    mp4_size / 1e6,
                    encoder,
                )
                # Remove the temp AVI
                try:
                    os.unlink(avi_path)
                except OSError:
                    pass
                _delete_quietly(audio_path)
                if save_locally:
                    self._save_recording_row(
                        event_id=event_id,
                        filename=mp4_fname,
                        file_size_bytes=mp4_size,
                        duration_seconds=duration,
                        started_at=ended_at,
                        ended_at=ended_at,
                    )
                    self._update_event_recording(event_id, recording_path=mp4_path)
                else:
                    self._upload_and_discard(mp4_path, mp4_fname, event_id, duration, ended_at)
            else:
                logger.error("ffmpeg re-encode failed:\n%s", result.stderr[-2000:])
                # Fall back: keep the AVI, update DB to point at it
                avi_fname = Path(avi_path).name
                if save_locally:
                    self._save_recording_row(
                        event_id=event_id,
                        filename=avi_fname,
                        file_size_bytes=_file_size(avi_path),
                        duration_seconds=duration,
                        started_at=ended_at,
                        ended_at=ended_at,
                    )
                    self._update_event_recording(event_id, recording_path=avi_path)
                else:
                    _delete_quietly(mp4_path)
                    self._upload_and_discard(avi_path, avi_fname, event_id, duration, ended_at)
                _delete_quietly(audio_path)
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out re-encoding %s", avi_path)
            _delete_quietly(audio_path)
            if not save_locally:
                _delete_quietly(avi_path)
                _delete_quietly(mp4_path)
                self._update_event_recording(
                    event_id,
                    webhook_error="Recording upload skipped: ffmpeg timed out.",
                    notes="Recording was not saved locally and encoding timed out.",
                )
        except Exception as exc:
            logger.error("Re-encode error: %s", exc)
            _delete_quietly(audio_path)
            if not save_locally:
                _delete_quietly(avi_path)
                _delete_quietly(mp4_path)
                self._update_event_recording(
                    event_id,
                    webhook_error=f"Recording upload skipped: {exc}"[:500],
                    notes="Recording was not saved locally after encoding error.",
                )

    def _upload_and_discard(
        self,
        path: str,
        filename: str,
        event_id: int | None,
        duration: float,
        ended_at: str,
    ) -> None:
        """Upload the finished clip to Discord, then remove the local file."""
        try:
            from discord_notify import upload_recording
            success = upload_recording(
                event_id=event_id,
                recording_path=path,
                recording_filename=filename,
                duration_seconds=duration,
                ended_at=ended_at,
            )
            note = (
                f"Recording uploaded to Discord only: {filename}"
                if success else
                f"Recording was not saved locally; Discord upload failed: {filename}"
            )
            fields = {"notes": note}
            if success:
                fields["webhook_sent"] = 1
                fields["webhook_error"] = ""
            else:
                fields["webhook_sent"] = 0
            self._update_event_recording(event_id, **fields)
        finally:
            _delete_quietly(path)

    def _update_event_recording(self, event_id: int | None, **fields) -> None:
        if event_id is None:
            return
        try:
            from database import raw_db_ctx
            import models as m
            with raw_db_ctx() as db:
                m.update_event(db, event_id, **fields)
        except Exception as exc:
            logger.warning("Could not update event recording fields: %s", exc)

    def _save_recording_row(self, **fields) -> None:
        try:
            from database import raw_db_ctx
            import models as m
            with raw_db_ctx() as db:
                m.create_recording(db, **fields)
        except Exception as exc:
            logger.warning("Could not save recording row: %s", exc)


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _normalise_recording_frame(frame: np.ndarray) -> np.ndarray:
    target = (config.FRAME_WIDTH, config.FRAME_HEIGHT)
    if frame.shape[1] == target[0] and frame.shape[0] == target[1]:
        return frame
    return cv2.resize(frame, target)


def _safe_filename_fragment(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    return text.strip("_")[:40] or "primary"


def _select_video_encoder(ffmpeg: str) -> str:
    requested = _runtime_setting("video_encoder", config.VIDEO_ENCODER).strip().lower()
    if requested not in {"auto", "libx264", "h264_nvenc", "hevc_nvenc", "h264_qsv"}:
        requested = "auto"

    if requested != "auto":
        return requested if _ffmpeg_encoder_available(ffmpeg, requested) else "libx264"

    for candidate in ("h264_nvenc", "libx264"):
        if _ffmpeg_encoder_available(ffmpeg, candidate):
            return candidate
    return "libx264"


def _ffmpeg_encoder_available(ffmpeg: str, encoder: str) -> bool:
    if encoder == "libx264":
        return True
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return encoder in result.stdout


def _build_reencode_command(
    ffmpeg: str,
    avi_path: str,
    mp4_path: str,
    audio_path: str,
    video_filter: str,
    encoder: str,
) -> list[str]:
    cmd = [ffmpeg, "-y", "-i", avi_path]
    if audio_path:
        cmd += ["-i", audio_path]

    quality = max(0, min(51, _int_setting("video_encoder_quality", config.VIDEO_ENCODER_QUALITY)))
    preset = _runtime_setting("video_encoder_preset", config.VIDEO_ENCODER_PRESET).strip() or "fast"

    if encoder in {"h264_nvenc", "hevc_nvenc"}:
        cmd += [
            "-c:v", encoder,
            "-preset", preset,
            "-cq", str(quality),
            "-b:v", "0",
            "-movflags", "+faststart",
        ]
    elif encoder == "h264_qsv":
        cmd += [
            "-c:v", "h264_qsv",
            "-global_quality", str(quality),
            "-preset", preset,
            "-movflags", "+faststart",
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", preset if preset in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"} else "veryfast",
            "-crf", str(quality),
            "-movflags", "+faststart",
        ]

    if video_filter:
        cmd += ["-vf", video_filter]
    if audio_path:
        cmd += [
            "-af", "aresample=async=1:first_pts=0",
            "-c:a", "aac",
            "-b:a", "96k",
            "-ac", "1",
            "-shortest",
        ]
    else:
        cmd += ["-an"]
    cmd.append(mp4_path)
    return cmd


def _timeline_video_filter(frame_count: int, duration: float, avi_fps: float) -> str:
    """
    OpenCV writes AVI timestamps at a fixed FPS even if frames arrive faster or
    slower. Audio is captured in real time, so stretch/compress video PTS to
    the measured wall-clock duration before muxing audio.
    """
    if frame_count <= 1 or duration <= 0 or avi_fps <= 0:
        return ""

    encoded_duration = frame_count / avi_fps
    if encoded_duration <= 0:
        return ""

    factor = duration / encoded_duration
    if factor <= 0:
        return ""

    # Tiny differences are not worth filtering and can create extra work.
    if 0.98 <= factor <= 1.02:
        return ""

    output_fps = max(1, min(30, round(frame_count / duration)))
    return f"setpts={factor:.8f}*PTS,fps={output_fps}"


def _encode_timeout(duration: float) -> int:
    # Continuous recordings can be much longer than event clips. Give ffmpeg
    # enough time for CPU-only hosts while still bounding truly stuck encodes.
    return max(300, min(7200, int(max(1.0, duration) * 4) + 120))


def _frame_wall_duration(
    frame_count: int,
    first_frame_time: float,
    last_frame_time: float,
    avi_fps: float,
) -> float:
    if frame_count <= 0 or first_frame_time <= 0 or last_frame_time <= 0:
        return 0.0

    fallback_interval = 1.0 / avi_fps if avi_fps > 0 else 0.066
    if frame_count == 1:
        return round(fallback_interval, 3)

    span = max(0.0, last_frame_time - first_frame_time)
    avg_interval = span / max(frame_count - 1, 1)
    if avg_interval <= 0:
        avg_interval = fallback_interval
    return round(span + avg_interval, 3)


def _is_stream_url(value: str) -> bool:
    value = value.strip().lower()
    return value.startswith(("rtsp://", "rtsps://", "http://", "https://"))


def _runtime_setting(key: str, default: str = "") -> str:
    try:
        from database import get_setting
        value = get_setting(key, default)
    except Exception:
        return default
    return value if value is not None else default


def _start_audio_capture(
    audio_path: str,
    source_override: str = "",
) -> tuple[subprocess.Popen | None, str]:
    if not audio_path:
        return None, ""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("Audio recording requested but ffmpeg is not available")
        return None, ""

    source = source_override.strip() or _audio_input_source()
    if not source:
        logger.warning("Audio recording requested but no microphone was detected")
        return None, ""

    for candidate in _audio_capture_commands(ffmpeg, source, audio_path):
        label = candidate[0]
        cmd = candidate[1]
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.1)
            if process.poll() is None:
                logger.info("Audio capture started from input %s (%s)", source, label)
                return process, f"{source} {label}"

            stderr = ""
            if process.stderr:
                stderr = process.stderr.read()[-500:].strip()
            _delete_quietly(audio_path)
            logger.warning(
                "Audio capture attempt failed for %s (%s): %s",
                source, label, stderr or f"exit={process.returncode}",
            )
        except Exception as exc:
            _delete_quietly(audio_path)
            logger.warning("Audio capture attempt could not start for %s (%s): %s", source, label, exc)

    return None, ""


def _audio_capture_commands(ffmpeg: str, source: str, audio_path: str) -> list[tuple[str, list[str]]]:
    if _is_stream_url(source):
        cmd = [
            ffmpeg, "-y",
            "-hide_banner",
            "-loglevel", "warning",
        ]
        if source.lower().startswith("rtsp://"):
            transport = _runtime_setting("ip_camera_rtsp_transport", config.IP_CAMERA_RTSP_TRANSPORT)
            transport = transport.strip().lower()
            if transport not in {"tcp", "udp", "udp_multicast", "http"}:
                transport = "tcp"
            cmd += ["-rtsp_transport", transport]
        cmd += [
            "-i", source,
            "-vn",
            "-c:a", "pcm_s16le",
            "-ac", "1",
            "-ar", "48000",
            audio_path,
        ]
        return [("ip-camera-audio", cmd)]

    base = [
        ffmpeg, "-y",
        "-hide_banner",
        "-loglevel", "warning",
        "-f", "alsa",
        "-thread_queue_size", "512",
    ]
    sources = [source]
    if source.startswith("hw:"):
        sources.insert(0, "plughw:" + source[3:])

    commands: list[tuple[str, list[str]]] = []
    for src in sources:
        commands.append((
            "kinect-native-s32-4ch",
            base + [
                "-ac", "4",
                "-ar", "16000",
                "-sample_fmt", "s32",
                "-i", src,
                "-c:a", "pcm_s32le",
                audio_path,
            ],
        ))
        commands.append((
            "alsa-default",
            base + [
                "-i", src,
                "-c:a", "pcm_s32le",
                audio_path,
            ],
        ))
    return commands


def _stop_audio_capture(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def _usable_audio_file(path: str) -> bool:
    return bool(path) and Path(path).is_file() and _file_size(path) > 1024


def _audio_input_source() -> str:
    override = config.RECORD_AUDIO_DEVICE.strip()
    if override and override.lower() != "auto":
        return override

    camera_source = _runtime_setting("camera_preferred_source", config.CAMERA_PREFERRED_SOURCE)
    if camera_source.strip().lower() == "ip":
        ip_url = _runtime_setting("ip_camera_url", config.IP_CAMERA_URL).strip()
        if ip_url:
            return ip_url

    detected = _detect_alsa_audio_source()
    return detected or "default"


def _detect_alsa_audio_source() -> str:
    if os.name != "posix":
        return ""

    arecord = shutil.which("arecord")
    if not arecord:
        return "default"

    try:
        result = subprocess.run(
            [arecord, "-l"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return "default"

    if result.returncode != 0:
        return "default"

    devices: list[tuple[int, int, str]] = []
    pattern = re.compile(
        r"card\s+(\d+):\s*([^\[]+)\[([^\]]*)\],\s*device\s+(\d+):\s*([^\[]+)\[([^\]]*)\]",
        re.IGNORECASE,
    )
    for line in result.stdout.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        card = int(match.group(1))
        device = int(match.group(4))
        label = " ".join(part.strip() for part in match.groups()[1:] if part).lower()
        devices.append((card, device, label))

    if not devices:
        return ""

    def score(item: tuple[int, int, str]) -> int:
        label = item[2]
        if any(term in label for term in ("kinect", "xbox", "nui")):
            return 100
        if any(term in label for term in ("webcam", "camera")):
            return 80
        if "usb" in label:
            return 60
        if any(term in label for term in ("microphone", "mic")):
            return 40
        return 10

    card, device, label = max(devices, key=score)
    source = f"hw:{card},{device}"
    logger.info("Detected audio input %s (%s)", source, label)
    return source


def _delete_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _bool_setting(key: str, default: bool) -> bool:
    try:
        from database import get_setting
        value = get_setting(key, "true" if default else "false")
    except Exception:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int_setting(key: str, default: int) -> int:
    try:
        from database import get_setting
        value = int(get_setting(key, str(default)))
    except Exception:
        return default
    return value if value > 0 else default


_recorder: Recorder | None = None


def get_recorder() -> Recorder:
    global _recorder
    if _recorder is None:
        _recorder = Recorder()
    return _recorder
