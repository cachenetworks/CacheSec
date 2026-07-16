# CacheSec Fix Snapshot

## Current Focus

- Unified Settings camera management.
- Live Feed camera grid/focus behavior.
- HLS `.m3u8` camera playback.
- Camera titles falling back to raw URLs when no friendly name is set.
- Keeping a lightweight backtrack note for future fixes.

## Fixed In This Pass

- Settings now uses one `Cameras` list instead of separate `Primary Detection Camera` and live-only sections.
- The unified camera list supports add, edit, remove, and per-camera Detection toggles.
- Settings saves the unified camera list to `setup_cameras` and syncs legacy runtime fields so Live Feed and the capture loop use the same source list.
- The first detection-enabled camera becomes the primary detection camera; additional detection-enabled cameras turn on multi-camera detection automatically.
- Live Feed and multi-camera detection now consume `setup_cameras` directly when it exists, so camera changes apply through the in-process reload path without restarting Docker.
- Turning off full/continuous recording now sends an explicit stop to the recorder so the current continuous clip finalizes immediately instead of waiting for unknown-person absent/min-duration logic.
- Continuous recordings that later get linked to an unknown event still remember they started as continuous, so the full-recording toggle can stop them.
- ffmpeg encode timeout now scales with clip duration, which avoids long full-recording clips being left as AVI/encoding failures after the old fixed 300 second timeout.
- Empty continuous clips are discarded if recording is turned off before any frame is written.
- Live Feed now prefers `setup_cameras`, with fallback to Settings-era keys such as `ip_camera_url`, `ip_camera_urls`, `tapo_*`, and USB/Kinect entries for older installs.
- Blank IP camera names now display as friendly titles like `IP Camera 1`; the raw URL stays in the detail line only.
- HLS and snapshot-style HTTP feeds now use CacheSec's ffmpeg-backed MJPEG fallback instead of the go2rtc fMP4 path.
- Live Feed grid tile clicks now reveal the focused feed card and provide working back/show-all behavior.
- Settings camera saves now attempt to regenerate/reload go2rtc before restarting the in-process camera loop.
- Setup wizard camera rows now use friendly titles instead of promoting the stream URL into the row heading.
- The setup wizard's saved `camera_index` setting is now respected by the primary USB camera path and sensor status.
- Multi-camera detection no longer re-adds the primary IP camera as an auxiliary IP source.

## Files Changed

- `camera.py`
- `recorder.py`
- `templates/admin/live.html`
- `templates/admin/settings.html`
- `admin.py`
- `database.py`
- `go2rtc_config.py`
- `setup_wizard.py`
- `templates/setup/wizard.html`
- `Fixes.md`

## Verification

- `python -m py_compile camera.py admin.py recorder.py database.py go2rtc_config.py setup_wizard.py`
- Settings page inline JavaScript checked with `node --check -`
- `git diff --check`

## Still Worth Checking Later

- ONVIF night-vision controls still apply to the primary IP camera only.
- HLS streams that require custom auth beyond Referer/Origin/User-Agent may still need extra header support.
