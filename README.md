# CacheSec — Face Recognition Security System

A production-ready, self-hosted face recognition security system built for
**Raspberry Pi 5**. Continuously monitors a camera, recognises enrolled people,
alerts on unknowns via Discord, records video evidence, and provides a full
admin dashboard accessible locally or remotely through Cloudflare Tunnel.

---

## Features

| Feature | Details |
|---------|---------|
| Face recognition | InsightFace `buffalo_l` (ONNX, ARM64-optimised) |
| Web dashboard | Flask + Bootstrap 5 dark theme |
| Auth | bcrypt passwords, session auth, rate limiting, lockout |
| RBAC | admin / operator / viewer roles |
| Alerts | Discord webhook with snapshot image attachment and optional video upload |
| Recording | Auto-start/stop with min/max duration enforcement; local saving and microphone audio are configurable |
| Multi-camera detection | Extra USB, Kinect, and IP cameras can run the same detection loop as the primary feed |
| Camera management | `+ Add Camera` modal with per-type fields (USB / Pi, IP / RTSP, Kinect, Tapo); per-row edit and remove for live-only feeds |
| TP-Link Tapo support | Native source type that consumes the camera's RTSP stream with Camera Account credentials |
| Optional object detection | Detectron2 can detect people, pets/animals, or all COCO objects when installed separately |
| Motion detection | Frame-difference moving-object detection can be enabled per camera |
| Sound | GPIO PWM buzzer (access-denied tone on unknown) |
| Database | SQLite with WAL mode |
| Deployment | Gunicorn + systemd or Docker + Cloudflare Tunnel |
| Audit log | All admin actions tracked |

---

## Directory Structure

```
cachesec/
├── app.py                  # Flask entry point + bootstrap
├── config.py               # All config from .env
├── database.py             # SQLite connection + schema init
├── models.py               # DB query helpers
├── auth.py                 # Login, RBAC, session management
├── admin.py                # Dashboard blueprint (all routes)
├── camera.py               # Camera capture + detection loop
├── recognition.py          # InsightFace embedding + matching
├── recorder.py             # Video recording state machine
├── tapo_control.py         # Tapo RTSP URL builder
├── discord_notify.py       # Discord webhook integration
├── sound.py                # GPIO buzzer abstraction
├── sounds.py               # Original GPIO tone primitives
├── utils.py                # Shared helpers
├── requirements.txt
├── .env.example            # Copy to .env and fill in
├── templates/
│   ├── base.html
│   ├── auth/login.html
│   ├── admin/              # All dashboard pages
│   └── errors/             # 403, 404, 429, 500
├── static/
│   ├── css/dashboard.css
│   ├── js/dashboard.js
│   └── img/
├── uploads/faces/          # Enrolled face images
├── recordings/             # Saved video clips
├── snapshots/              # Event snapshot JPEGs
├── logs/
└── systemd/
    ├── cachesec.service
    └── install.sh
```

---

## Quick Start

### 1 — Clone / copy the project

```bash
cd /home/cache
# (project already at /home/cache/cachesec)
```

### 2 — Create a virtual environment

```bash
cd /home/cache/cachesec
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Optional Detectron2 object detection is not installed by default because PyTorch
and Detectron2 wheels depend on your CPU/GPU, Python, and CUDA stack. Install a
matching PyTorch build first, then:

```bash
pip install --no-build-isolation -r requirements-detectron2.txt
```

For Docker builds, set `INSTALL_DETECTRON2=true`. CPU builds use the PyTorch CPU
wheel index by default. NVIDIA builds can set `TORCH_INDEX_URL` to the CUDA wheel
index that matches the host driver stack, then use `OBJECT_DETECTION_DEVICE=cuda`.
GPU video encoding is controlled separately with `VIDEO_ENCODER=h264_nvenc`.

> **Note on InsightFace model download:**  
> On first run, InsightFace downloads the `buffalo_l` model weights (~300 MB)
> from a CDN. Ensure your Pi has internet access. Models are cached in
> `~/.insightface/models/`. After that, the system works fully offline.

### 4 — Configure environment

```bash
cp .env.example .env
nano .env
```

**Required changes in `.env`:**

| Key | What to set |
|-----|-------------|
| `SECRET_KEY` | Generate with: `python3 -c "import secrets; print(secrets.token_hex(32))"` |
| `DISCORD_WEBHOOK_URL` | Your Discord channel webhook URL |
| `DISCORD_MENTION_EVERYONE` | Set `true` to include `@everyone` in unknown Discord alerts; defaults to `false` |
| `CAMERA_PREFERRED_SOURCE` | Use `webcam`, `kinect`, `ip`, or `tapo` |
| `KINECT_ONE_CAMERA_INDEX` | Optional experimental Xbox One Kinect RGB V4L2 index fallback (e.g. `2`) when `CAMERA_PREFERRED_SOURCE=kinect` and Kinect v1 is unavailable |
| `USB_CAMERA_AUTO_DISCOVER` | Automatically discover `/dev/video*` cameras for grid and detection |
| `USB_CAMERA_INDICES` | Optional comma-separated extra USB/V4L2 indices to show and detect on |
| `MULTI_CAMERA_DETECTION_ENABLED` | Set `true` to run detection on auxiliary USB, Kinect, and IP feeds |
| `IP_CAMERA_URL` | RTSP/HTTP/MJPEG stream URL when using an IP camera |
| `IP_CAMERA_ONVIF_NIGHT_MODE` | Optional: `detect` drives ONVIF `IrCutFilter` from darkness; `force_off` keeps IR-cut on |
| `NIGHT_VISION_MODE` | Use `force_off` to disable USB software night vision and Kinect IR switching |
| `OBJECT_DETECTION_BACKEND` | Optional: set `detectron2` after installing `requirements-detectron2.txt` |
| `OBJECT_DETECTION_MODE` | `person`, `people_pets`, or `all`; default `people_pets` detects people plus COCO animal classes |
| `OBJECT_DETECTION_DEVICE` | `auto`, `cpu`, or `cuda` |
| `MOVING_OBJECT_DETECTION_ENABLED` | Optional motion-box detection on all detection feeds |
| `VIDEO_ENCODER` | `auto`, `libx264`, `h264_nvenc`, `hevc_nvenc`, or `h264_qsv` |
| `IP_CAMERA_ONVIF_HOST` / `IP_CAMERA_ONVIF_PORT` | Optional ONVIF endpoint override if the control port differs from the stream URL |
| `TAPO_HOST` | LAN IP / hostname of the Tapo camera (e.g. `192.168.1.60`) |
| `TAPO_USERNAME` / `TAPO_PASSWORD` | "Camera Account" credentials set in the Tapo app under *Advanced Settings → Camera Account* (NOT the cloud login) |
| `TAPO_STREAM` | `stream1` (high) or `stream2` (low) |
| `SAVE_RECORDINGS_LOCALLY` | Set `false` to upload completed clips to Discord and remove local video files |
| `RECORD_AUDIO_ENABLED` | Optional microphone/IP-camera audio capture for recordings; defaults to `false` |
| `SESSION_COOKIE_SECURE` | `true` if behind Cloudflare HTTPS, `false` for LAN-only HTTP |

### 5 — Initialise and run (dev)

```bash
source venv/bin/activate
python app.py
```

Open `http://localhost:5000` in your browser.  
**Default credentials: `admin` / `changeme123`** — change immediately.

---

## Face Enrollment

1. Log in as `admin`.
2. Navigate to **Enrolled Faces → Add Person**.
3. Enter the person's name and save.
4. On the person detail page, click **Upload & Enroll** and select 3–10 clear
   face photos (good lighting, different angles).
5. CacheSec generates ONNX embeddings and stores them in the database.
6. The person is immediately active in the recognition gallery.

**Tips for good enrollment images:**
- Well-lit, front-facing shots work best.
- Include some slight angle variation (left/right/up/down ~15°).
- Avoid sunglasses or heavy obstructions.
- 5–8 images per person gives good accuracy.

---

## systemd Service

### Install (run as root)

```bash
sudo bash systemd/install.sh
```

### Manual setup

```bash
# Copy service file
sudo cp systemd/cachesec.service /etc/systemd/system/

# Edit User/Group/WorkingDirectory if needed
sudo nano /etc/systemd/system/cachesec.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable cachesec
sudo systemctl start cachesec
```

### Management commands

```bash
# Status
sudo systemctl status cachesec

# View live logs
sudo journalctl -u cachesec -f

# Restart (after config change)
sudo systemctl restart cachesec

# Stop
sudo systemctl stop cachesec

# Disable autostart
sudo systemctl disable cachesec
```

### GPIO group (for buzzer)

The service user needs to be in the `gpio` group:

```bash
sudo usermod -aG gpio cache
sudo usermod -aG audio cache   # required if recording microphone audio
# Log out and back in, or restart the service
```

---

## Docker

The repository now includes:

- `Dockerfile` â€” production image with Gunicorn, ffmpeg, and OpenCV runtime libs
- `docker-compose.yml` â€” local container deployment with persistent `/data` storage
- `.github/workflows/container.yml` â€” GitHub Actions workflow that builds and publishes a multi-arch image to GHCR

### Recommended image variants

Every variant bundles the **full feature set** — Kinect v1, Kinect v2, and
Detectron2 — all built into the same image from the same Dockerfile. Variants
only differ by platform/GPU target, not by which features are compiled in, so
there's a single image to pull per platform instead of one per feature:

- `cachesec:cpu` (also tagged `:latest`) — CPU-only runtime, `linux/amd64` + `linux/arm64`. Kinect v1 + Kinect v2 + Detectron2 (CPU torch).
- `cachesec:pi` — Raspberry Pi (Pi 4/Pi 5, `linux/arm64`). Kinect v1 + Kinect v2. **No Detectron2** — there's no GPU on a Pi and CPU-only inference there isn't practically usable, so it's skipped to keep the build fast and the image small.
- `cachesec:cuda11` — NVIDIA CUDA 11 runtime, `linux/amd64`. Kinect v1 + Kinect v2 + Detectron2 (CUDA 11 torch). Needs an NVIDIA GPU + `nvidia-container-toolkit` on the host to actually use the GPU at runtime.
- `cachesec:cuda12` — NVIDIA CUDA 12 runtime, `linux/amd64`. Same as `cuda11` but CUDA 12 torch wheels.

You can build them yourself from the same Dockerfile by changing build args:

```bash
# CPU (default): Kinect v1 + Kinect v2 + Detectron2, CPU torch
docker build -t cachesec:cpu \
  --build-arg INSTALL_KINECT=true \
  --build-arg INSTALL_KINECT2=true \
  --build-arg INSTALL_DETECTRON2=true .

# Raspberry Pi: Kinect v1 + Kinect v2, no Detectron2
docker build -t cachesec:pi \
  --build-arg INSTALL_KINECT=true \
  --build-arg INSTALL_KINECT2=true \
  --build-arg INSTALL_DETECTRON2=false .

# CUDA 11: Kinect v1 + Kinect v2 + Detectron2, CUDA 11 torch
docker build -t cachesec:cuda11 \
  --build-arg INSTALL_KINECT=true \
  --build-arg INSTALL_KINECT2=true \
  --build-arg INSTALL_DETECTRON2=true \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 .

# CUDA 12: Kinect v1 + Kinect v2 + Detectron2, CUDA 12 torch
docker build -t cachesec:cuda12 \
  --build-arg INSTALL_KINECT=true \
  --build-arg INSTALL_KINECT2=true \
  --build-arg INSTALL_DETECTRON2=true \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 .
```

If you don't need a feature (e.g. no Kinect hardware at all), set its
`INSTALL_*` build arg to `false` for your own local build — the CI-published
tags above always include everything for a given platform/GPU target.

GitHub Actions, GitLab CI, Woodpecker, and Gitea/Forgejo Actions
(`.github/workflows/container.yml`, `.gitlab-ci.yml`, `.woodpecker.yml`,
`.gitea/workflows/container.yml`) all publish the same 4 tags: `cpu` (+
`latest`), `pi`, `cuda11`, `cuda12`.

> **Migrating from older tags:** `:kinect`, `:kinect1`, `:kinect2`,
> `:detectron2`, `:detectron2-cpu`, `:detectron2-cuda11`, and
> `:detectron2-cuda12` are no longer published — everything they provided is
> now in `:cpu` (or `:cuda11`/`:cuda12` for the CUDA builds). Switch any
> deployment pinned to one of the old tags to the matching new one above.

### Local Docker Compose

1. Copy `.env.example` to `.env`
2. Start the stack:

```bash
docker compose up --build -d
```

The container stores its runtime state in `/data` inside the container, backed by
the named volume `cachesec-data`. That includes:

- SQLite database
- uploaded face images
- snapshots
- recordings
- logs
- downloaded ONNX models

Open `http://localhost:5000` after the container becomes healthy.

`docker-compose.yml` intentionally forces Docker-friendly defaults for a local
HTTP deployment:

- `CACHESEC_SESSION_COOKIE_SECURE=false`
- `CACHESEC_PROXY_COUNT=0`

If you run the container behind HTTPS and a trusted reverse proxy, override
those compose-only variables before starting the stack.

### Camera / GPIO passthrough

For IP cameras, no host device mapping is required.

For a local webcam or Pi camera, uncomment the `devices:` section in
`docker-compose.yml`.

By default the compose example maps host `/dev/video0` to container
`/dev/video0`. On Raspberry Pi/libcamera systems the capture node is often not
`/dev/video0`; use `v4l2-ctl --list-devices` on the host to find the actual
capture node, then set:

```bash
CACHESEC_VIDEO_DEVICE=/dev/video19
CAMERA_INDEX=0
```

Replace `/dev/video19` with the usable host device. Keeping `CAMERA_INDEX=0`
works because the selected host node is mapped to `/dev/video0` inside the
container.

For the GPIO buzzer on Raspberry Pi, also map `/dev/gpiochip0`.

### Kinect v1 in Docker

Kinect v1 / Xbox 360 Kinect is not a normal V4L2 webcam in this app. Do not
map it as `/dev/video0`; CacheSec accesses it through OpenKinect/libfreenect.

On the Raspberry Pi host, verify the Kinect is fully powered and enumerated:

```bash
lsusb | grep -i '045e'
```

A fully powered Kinect v1 should expose all three Microsoft USB functions:

```text
045e:02ae  camera
045e:02ad  audio
045e:02b0  motor
```

If only `045e:02b0` appears, the USB side is connected but the Kinect does not
have 12V power, so the camera cannot work.

Use USB passthrough for Docker:

```yaml
privileged: true
volumes:
  - ./data:/data
  - /dev/bus/usb:/dev/bus/usb
# Optional if you want ALSA microphone/audio capture from the host:
devices:
  - /dev/snd:/dev/snd
```

Set the camera source to Kinect:

```bash
CAMERA_PREFERRED_SOURCE=kinect
KINECT_ENABLED=true
KINECT_MOTOR_ENABLED=false
```

The camera source is also stored in the SQLite settings database. After the
first boot, changing `.env` may not override the existing value. Change the
camera source in the admin Settings page, or remove `/data/cachesec.db` if you
are intentionally resetting the deployment.

The container image must include `libfreenect` and the Python `freenect`
package. The Dockerfile installs Kinect support by default with
`INSTALL_KINECT=true`. CacheSec still auto-detects the device at runtime, so the
image can run without a Kinect attached.

The Kinect-enabled Docker build also installs `alsa-utils`,
`kinect-audio-setup`, `freenect`, `libfreenect-bin`, `libfreenect-dev`,
`libfreenect0.5`, and libusb runtime/build packages. `kinect-audio-setup` is a
Debian `contrib` package and downloads Microsoft's non-redistributable Kinect
audio firmware during package setup, so the image build needs network access.

If you are using a published GHCR image, rebuild/publish it after Dockerfile
changes, or build locally with:

```bash
docker compose build --no-cache
docker compose up -d
```

### TP-Link Tapo cameras

CacheSec uses Tapo C-series cameras as a regular RTSP source. No host device
passthrough is needed — Tapo cameras live on the LAN.

**One-time setup on the camera:**

1. Open the Tapo app and go to your camera → **Advanced Settings → Camera Account**.
2. Enable the camera account and set a username (default `admin`) + password.
   This is **not** the same as the Tapo cloud login.

**In CacheSec:**

1. Open **Settings**, click **Change** next to *Primary Detection Camera*.
2. Pick **Tapo**, fill in the IP and the camera-account user/password.
3. Save and restart the app.

Tapo cameras flip their own IR-cut filter automatically, so CacheSec disables
the software green-tint night-vision overlay on Tapo feeds.


To build a smaller image without Kinect support:

```bash
docker compose build --build-arg INSTALL_KINECT=false
```

### GitHub Container Registry

The GitHub Actions workflow publishes the image to:

```text
ghcr.io/<your-github-owner>/cachesec
```

It builds these image variants (see [Recommended image variants](#recommended-image-variants)
above for what each one bundles):

- `:cpu` / `:latest` and version tags — full feature set, `linux/amd64` + `linux/arm64`
- `:pi` — full feature set minus Detectron2, `linux/arm64` (Raspberry Pi)
- `:cuda11` / `:cuda12` — full feature set with CUDA 11/12 torch, `linux/amd64`

It runs on pushes to `main`/`master`, version tags like `v1.0.0`, and manual
dispatches. Pull requests build the image without pushing it.

### Xbox One Kinect (experimental fallback)

CacheSec now includes an **experimental** Xbox One Kinect fallback path for RGB
video only. If Kinect v1 cannot start and `CAMERA_PREFERRED_SOURCE=kinect`,
the app will try `KINECT_ONE_CAMERA_INDEX` as a direct V4L2 camera source.

- No tilt/motor controls in this mode
- No Kinect v1-style IR/depth/SLS path in this mode
- Intended for community testing when Xbox One Kinect adapters/drivers are present

---

## Cloudflare Tunnel Deployment

Cloudflare Tunnel lets you expose CacheSec's dashboard remotely without
opening any firewall ports. Traffic flows:

```
Browser → Cloudflare Edge → cloudflared (on Pi) → localhost:5000
```

### Step 1 — Install cloudflared

```bash
# Raspberry Pi OS (aarch64)
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared-linux-arm64.deb
```

### Step 2 — Authenticate with your Cloudflare account

```bash
cloudflared tunnel login
```

This opens a browser for OAuth. Authorise and a certificate is saved to
`~/.cloudflared/cert.pem`.

### Step 3 — Create a named tunnel

```bash
cloudflared tunnel create cachesec
```

Note the **Tunnel ID** shown (e.g. `abc123...`).

### Step 4 — Configure the tunnel

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: <YOUR_TUNNEL_ID>
credentials-file: /home/cache/.cloudflared/<YOUR_TUNNEL_ID>.json

ingress:
  - hostname: security.yourdomain.com
    service: http://127.0.0.1:5000
    originRequest:
      noTLSVerify: false
  - service: http_status:404
```

Replace `security.yourdomain.com` with your actual subdomain.

### Step 5 — Create DNS CNAME record

```bash
cloudflared tunnel route dns cachesec security.yourdomain.com
```

This creates a `CNAME` in your Cloudflare DNS pointing to the tunnel.

### Step 6 — Run cloudflared as a service

```bash
sudo cloudflared service install
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
```

### Step 7 — Update .env for HTTPS

```env
SESSION_COOKIE_SECURE=true
PROXY_COUNT=1
ALLOWED_HOSTS=security.yourdomain.com
```

Restart CacheSec:

```bash
sudo systemctl restart cachesec
```

### Cloudflare Access (Zero Trust) — recommended

Add an additional authentication layer in front of your dashboard so it
requires a Cloudflare identity check (Google/GitHub/email OTP) before
the Flask login page is even reached.

1. Go to **Cloudflare Zero Trust → Access → Applications**.
2. Add a **Self-hosted** application.
3. Set the domain to `security.yourdomain.com`.
4. Add a policy: allow only your email address.
5. Save. Cloudflare will now prompt for identity before forwarding to your Pi.

This provides defence-in-depth: Cloudflare Access blocks untrusted visitors
before they can attempt Flask login brute-force.

---

## Security Notes

| Topic | Implementation |
|-------|---------------|
| Password storage | bcrypt via passlib (cost factor 12) |
| Login lockout | 5 failed attempts → 5-minute lockout (DB-backed) |
| Session cookies | HttpOnly, SameSite=Lax, Secure (behind HTTPS) |
| RBAC | admin / operator / viewer enforced per route |
| File uploads | Extension whitelist, size limit, sanitised filenames |
| Path traversal | Filenames sanitised with `os.path.basename()` |
| SQL injection | Parameterised queries throughout |
| Security headers | CSP-compatible headers added on every response |
| Proxy trust | `ProxyFix` with `PROXY_COUNT=1` for Cloudflare |
| Secrets | Never hardcoded; always loaded from `.env` |
| Debug mode | Disabled in production (`FLASK_ENV=production`) |

---

## Face Recognition — Library Choice Rationale

**InsightFace with ONNX Runtime** was chosen over alternatives for these reasons:

| Library | Speed (Pi 5) | Accuracy | Install complexity |
|---------|-------------|----------|--------------------|
| **InsightFace + ONNX** | ~80–150ms/detection | Excellent (buffalo_l ~99.7% LFW) | pip install, weights auto-download |
| face_recognition (dlib) | ~400–800ms | Good | Requires dlib compilation (~30min on Pi) |
| DeepFace | ~300–600ms | Good | Heavy dependencies |
| OpenCV DNN | ~50ms | Fair | Built into OpenCV |

**Tuning options** (all via `.env` or Settings page):

- Swap to `buffalo_s` for ~2× speed at slight accuracy cost: change
  `model_name="buffalo_s"` in `recognition.py:get_recognizer()`
- Reduce `FRAME_WIDTH`/`FRAME_HEIGHT` to `320x240` for faster capture
- Increase `FRAME_SKIP` to `5–10` to further reduce CPU (sacrifices responsiveness)
- Lower `RECOGNITION_THRESHOLD` (e.g. `0.35`) to be stricter about matches

---

## Troubleshooting

**Camera not detected:**
```bash
# Check camera is visible
ls /dev/video*
# Test with OpenCV
python3 -c "import cv2; c=cv2.VideoCapture(0); print(c.isOpened()); c.release()"
```

**InsightFace model download fails:**
```bash
# Manual download
python3 -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)"
```

**GPIO buzzer not working:**
```bash
# Check group membership
groups cache
# Add if missing
sudo usermod -aG gpio cache
```

**Dashboard 500 errors:**
```bash
sudo journalctl -u cachesec -n 50 --no-pager
tail -f /home/cache/cachesec/logs/cachesec.log
```

**Slow recognition on Pi:**
- Increase `FRAME_SKIP` in `.env`
- Use `buffalo_s` model instead of `buffalo_l`
- Reduce detection resolution in `recognition.py` (`det_size=(160, 160)`)

---

## Privacy & Consent

This system is designed for use on **your own property** with people who have
**given explicit consent** to be enrolled. Key privacy defaults:

- All data stays on-device (no cloud processing)
- Embeddings are 512-dimensional float vectors; they cannot reconstruct a face
- Snapshots are stored locally; recordings can be stored locally or uploaded to Discord only
- Audit log tracks who enrolled or deleted face data
- Enrolled people can be deleted (removing images and embeddings) at any time

---

## License

MIT — Use responsibly and in accordance with local privacy laws.
