"""
go2rtc_config.py — Generate go2rtc.yaml from the wizard's camera list.

go2rtc is a separate Docker service that pulls each RTSP camera once and
demuxes it to multiple consumers (the browser via WebSocket+MSE, and the
CacheSec detection loop via an internal RTSP relay).

This module writes /data/go2rtc/go2rtc.yaml from `setup_cameras` so the user
never has to touch the file by hand. CacheSec calls `regenerate_config()`
on startup and after every camera change.

Stream naming: every camera gets a stable id like `cam_<8-hex>`. The browser
talks to `/api/ws?src=cam_<id>` (proxied through Flask -> go2rtc:1984).
The detection loop pulls `rtsp://go2rtc:8554/cam_<id>`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

logger = logging.getLogger(__name__)

# go2rtc reads its config from /config/go2rtc.yaml inside its container.
# We bind-mount the same volume into CacheSec at /data/go2rtc.
GO2RTC_CONFIG_DIR = Path(os.environ.get("GO2RTC_CONFIG_DIR", "/data/go2rtc"))
GO2RTC_CONFIG_FILE = GO2RTC_CONFIG_DIR / "go2rtc.yaml"
GO2RTC_HOST = os.environ.get("GO2RTC_HOST", "go2rtc")
GO2RTC_HTTP_PORT = int(os.environ.get("GO2RTC_HTTP_PORT", "1984"))
GO2RTC_RTSP_PORT = int(os.environ.get("GO2RTC_RTSP_PORT", "8554"))


def stream_id_for(cam: dict) -> str:
    """Deterministic stream id used in go2rtc and in the browser URL."""
    cam_id = (cam.get("id") or "").strip()
    return f"cam_{cam_id}" if cam_id else ""


def relay_rtsp_url(cam: dict) -> str:
    """RTSP URL the CacheSec detection loop uses to pull from go2rtc."""
    sid = stream_id_for(cam)
    if not sid:
        return ""
    return f"rtsp://{GO2RTC_HOST}:{GO2RTC_RTSP_PORT}/{sid}"


def webrtc_ws_path(cam: dict) -> str:
    """Path the browser hits (proxied by Flask) to connect to MSE."""
    sid = stream_id_for(cam)
    return f"/go2rtc/api/ws?src={sid}" if sid else ""


def _source_url(cam: dict) -> str:
    """Translate one wizard camera record into a go2rtc source URL."""
    t = (cam.get("type") or "").lower()
    if t == "ip":
        return cam.get("url", "")
    if t == "tapo":
        host = cam.get("host", "")
        user = cam.get("username", "admin") or "admin"
        pw = cam.get("password", "") or ""
        stream = cam.get("stream", "stream1")
        if stream not in {"stream1", "stream2"}:
            stream = "stream1"
        if not host or not pw:
            return ""
        return f"rtsp://{quote(user, safe='')}:{quote(pw, safe='')}@{host}:554/{stream}"
    # webcam / kinect aren't proxied through go2rtc — they stay on the host
    # OpenCV path inside CacheSec.
    return ""


def _skip_go2rtc_source(url: str) -> bool:
    """Feeds that CacheSec serves through its ffmpeg MJPEG fallback."""
    parts = urlsplit(url)
    path = parts.path.lower()
    query = parts.query.lower()
    if (
        path.endswith((".m3u8", ".m3u", ".jpg", ".jpeg", ".png", ".webp"))
        or ".m3u8" in path
        or "m3u8" in query
    ):
        return True
    lowered = url.lower()
    return any(token in lowered for token in ("snapshot", "still"))


def regenerate_config() -> None:
    """Read the wizard's camera list and write go2rtc.yaml."""
    try:
        from database import get_setting
        raw = get_setting("setup_cameras", "")
        cams = json.loads(raw) if raw else []
        if not isinstance(cams, list):
            cams = []
    except Exception as exc:
        logger.warning("Could not read camera list for go2rtc: %s", exc)
        cams = []

    streams: dict[str, list[str]] = {}
    for cam in cams:
        sid = stream_id_for(cam)
        url = _source_url(cam)
        if not (sid and url) or _skip_go2rtc_source(url):
            continue
        # The first source is the raw RTSP from the camera. The second is
        # an ffmpeg-backed transcoder that converts the camera's audio
        # codec (Tapo: PCMA/G.711) into AAC, which is what browsers can
        # decode inside fMP4. Without this entry the fMP4 stream would
        # have video only.
        streams[sid] = [
            url,
            f"ffmpeg:{sid}#video=copy#audio=aac",
        ]

    config: dict[str, Any] = {
        # `debug` so we can see ffmpeg restarts and producer reconnects.
        "log": {"level": "debug", "format": "color"},
        "api": {"listen": ":1984"},
        "rtsp": {"listen": ":8554"},
        "webrtc": {
            # No external ICE candidates needed — Cloudflare deployment
            # uses MSE (WebSocket) only. WebRTC is left enabled for LAN
            # use but won't be required.
            "candidates": [],
        },
        "streams": streams,
    }

    GO2RTC_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    GO2RTC_CONFIG_FILE.write_text(_yaml_dump(config), encoding="utf-8")
    logger.info("Wrote go2rtc config with %d stream(s) to %s",
                len(streams), GO2RTC_CONFIG_FILE)


def _yaml_dump(obj: Any, indent: int = 0) -> str:
    """Tiny YAML emitter — avoids adding PyYAML as a runtime dependency.

    Only handles the dict/list/str/int/None shapes we produce here.
    """
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        if not obj:
            return pad + "{}\n"
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(_yaml_dump(v, indent + 1).rstrip("\n"))
            else:
                lines.append(f"{pad}{k}: {_scalar(v)}")
    elif isinstance(obj, list):
        if not obj:
            return pad + "[]\n"
        for item in obj:
            lines.append(f"{pad}- {_scalar(item)}")
    else:
        lines.append(f"{pad}{_scalar(obj)}")
    return "\n".join(lines) + "\n"


def _scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    # Quote when we have characters that YAML 1.1 dislikes bare.
    if s == "" or s[0] in "!&*[]{}|>'\"%@`#" or ":" in s or "\n" in s:
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'
    return s


def reload_go2rtc() -> None:
    """Tell go2rtc to re-read its config (POST /api/restart).

    go2rtc does NOT watch its config file, so this call is required after
    every camera change for it to pick up the new yaml.
    """
    try:
        import urllib.request
        url = f"http://{GO2RTC_HOST}:{GO2RTC_HTTP_PORT}/api/restart"
        req = urllib.request.Request(url, method="POST")
        urllib.request.urlopen(req, timeout=3).read()
        logger.info("go2rtc reload requested")
    except Exception as exc:
        # Most likely cause: go2rtc not running yet (cachesec booted first).
        # Camera regeneration on next setting change will retry.
        logger.debug("go2rtc reload failed (continuing): %s", exc)
