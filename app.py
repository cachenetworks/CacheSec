"""
app.py — Flask application factory and entry point.

Run with:
    python app.py                    # dev
    gunicorn -w 1 -b 127.0.0.1:5000 app:app   # production (single worker — SQLite)

Notes on single worker:
  SQLite + WAL mode handles concurrent reads well, but we keep a single
  Gunicorn worker to avoid multiple processes fighting over the camera.
  Background threads (camera, recorder) live inside this one process.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from datetime import timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, render_template, session, g
from werkzeug.middleware.proxy_fix import ProxyFix

import config
from database import init_db, close_db, get_db
from auth import auth_bp, current_user
from admin import admin_bp
from setup_wizard import setup_bp, setup_complete

# ---------------------------------------------------------------------------
# Logging setup (before app creation so it catches import errors too)
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    fmt   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_path = Path(config.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(
        RotatingFileHandler(
            str(log_path), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
    )

    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    # Quieten chatty libraries
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


_setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------
    app.secret_key = config.SECRET_KEY
    app.config.update(
        SESSION_COOKIE_HTTPONLY  = True,
        SESSION_COOKIE_SAMESITE  = "Lax",
        SESSION_COOKIE_SECURE    = config.SESSION_COOKIE_SECURE,
        PERMANENT_SESSION_LIFETIME = timedelta(minutes=config.SESSION_LIFETIME_MINUTES),
        MAX_CONTENT_LENGTH       = config.MAX_CONTENT_LENGTH,
        # Do not expose Jinja2 details in production
        PROPAGATE_EXCEPTIONS     = config.DEBUG,
        TEMPLATES_AUTO_RELOAD    = config.DEBUG,
    )

    # -----------------------------------------------------------------------
    # Trusted proxy (Cloudflare Tunnel sends X-Forwarded-For + CF-Connecting-IP)
    # -----------------------------------------------------------------------
    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for=config.PROXY_COUNT,
        x_proto=config.PROXY_COUNT,
        x_host=config.PROXY_COUNT,
    )

    # -----------------------------------------------------------------------
    # Security headers
    # -----------------------------------------------------------------------
    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"]    = "nosniff"
        response.headers["X-Frame-Options"]           = "DENY"
        response.headers["X-XSS-Protection"]          = "1; mode=block"
        response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"]        = "camera=(), microphone=(), geolocation=()"
        # Only set HSTS when behind HTTPS (Cloudflare handles TLS termination)
        if config.SESSION_COOKIE_SECURE:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        # Disable browser caching for static assets so UI/JS updates are always
        # visible without hard refresh after deployments or settings changes.
        try:
            from flask import request as _request
            if (_request.path or "").startswith("/static/"):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
        except Exception:
            pass
        return response

    # -----------------------------------------------------------------------
    # Allowed hosts check
    # -----------------------------------------------------------------------
    if config.ALLOWED_HOSTS:
        @app.before_request
        def check_host():
            from flask import request, abort
            host = request.host.split(":")[0]
            if host not in config.ALLOWED_HOSTS:
                abort(400)

    # -----------------------------------------------------------------------
    # Database lifecycle
    # -----------------------------------------------------------------------
    app.teardown_appcontext(close_db)

    # -----------------------------------------------------------------------
    # Template filters
    # -----------------------------------------------------------------------
    @app.template_filter("basename")
    def basename_filter(path: str) -> str:
        return os.path.basename(path) if path else ""

    # Template globals / context processors
    # -----------------------------------------------------------------------
    @app.context_processor
    def inject_globals():
        return {
            "current_user": current_user(),
            "app_name":     "CacheSec",
            "enumerate":    enumerate,
        }

    # -----------------------------------------------------------------------
    # Blueprints
    # -----------------------------------------------------------------------
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(setup_bp)

    # First-run setup gate: until setup_complete=true in the settings table,
    # all routes redirect to /setup. Static assets and the wizard itself stay
    # reachable so the page can render and submit forms.
    @app.before_request
    def _setup_gate():
        from flask import request, redirect, url_for
        path = request.path or ""
        if path.startswith("/setup") or path.startswith("/static"):
            return None
        if setup_complete():
            return None
        return redirect(url_for("setup.wizard"))

    # Root redirect
    @app.route("/")
    def index():
        from flask import redirect, url_for
        return redirect(url_for("admin.dashboard"))

    # -----------------------------------------------------------------------
    # Error pages
    # -----------------------------------------------------------------------
    @app.errorhandler(403)
    def forbidden(e):
        return render_template("errors/403.html"), 403

    @app.errorhandler(404)
    def not_found(e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(429)
    def rate_limited(e):
        return render_template("errors/429.html"), 429

    @app.errorhandler(500)
    def server_error(e):
        logger.exception("Internal server error")
        return render_template("errors/500.html"), 500

    return app


# ---------------------------------------------------------------------------
# Initialise DB and start background threads
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Run once at startup: init DB and (if setup is done) start camera."""
    init_db()
    try:
        from go2rtc_config import regenerate_config, reload_go2rtc
        regenerate_config()
        reload_go2rtc()
    except Exception as exc:
        logger.warning("go2rtc config regeneration failed (non-fatal): %s", exc)
    if setup_complete():
        _start_camera()
    else:
        logger.info("Setup wizard not yet completed — camera loop deferred")


def _start_camera() -> None:
    from camera import get_camera_loop
    loop = get_camera_loop()
    loop.start()
    logger.info("Camera detection loop started")


def _shutdown(*_) -> None:
    logger.info("Shutting down CacheSec...")
    try:
        from camera import get_camera_loop
        get_camera_loop().stop()
    except Exception:
        pass
    try:
        from sound import shutdown as sound_shutdown
        sound_shutdown()
    except Exception:
        pass
    sys.exit(0)


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

# ---------------------------------------------------------------------------
# Create app and bootstrap
# ---------------------------------------------------------------------------
app = create_app()

with app.app_context():
    _bootstrap()

# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(
        "Starting CacheSec on %s:%d (debug=%s)",
        config.HOST, config.PORT, config.DEBUG,
    )
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        use_reloader=False,   # Reloader would start two camera threads
        threaded=True,
    )
