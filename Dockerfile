FROM python:3.11-slim

ARG INSTALL_KINECT=true
ARG INSTALL_KINECT2=false
ARG INSTALL_DETECTRON2=false
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/data \
    HOST=0.0.0.0 \
    PORT=5000 \
    DATABASE_PATH=/data/cachesec.db \
    RECORDINGS_DIR=/data/recordings \
    SNAPSHOTS_DIR=/data/snapshots \
    UPLOAD_FOLDER=/data/uploads/faces \
    LOG_FILE=/data/logs/cachesec.log

WORKDIR /app

RUN set -eux; \
    if [ "$INSTALL_KINECT" = "true" ]; then \
        if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
            sed -i -E 's/^Components: .*/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources; \
        elif [ -f /etc/apt/sources.list ]; then \
            sed -i -E 's/ main($| )/ main contrib non-free non-free-firmware /' /etc/apt/sources.list; \
        fi; \
    fi; \
    apt-get update; \
    packages="\
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        tini \
    "; \
    if [ "$INSTALL_DETECTRON2" = "true" ]; then \
        packages="$packages \
            build-essential \
            git \
            ninja-build \
        "; \
    fi; \
    if [ "$INSTALL_KINECT" = "true" ]; then \
        packages="$packages \
            alsa-utils \
            build-essential \
            freenect \
            kinect-audio-setup \
            libfreenect-bin \
            libfreenect-dev \
            libfreenect0.5 \
            libusb-1.0-0 \
            libusb-1.0-0-dev \
        "; \
    fi; \
    if [ "$INSTALL_KINECT2" = "true" ]; then \
        packages="$packages \
            libusb-1.0-0 \
            libusb-1.0-0-dev \
            libglfw3 \
            libglfw3-dev \
            libjpeg62-turbo \
            libjpeg62-turbo-dev \
            libturbojpeg0 \
            libturbojpeg0-dev \
        "; \
    fi; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $packages; \
    if [ "$INSTALL_KINECT2" = "true" ]; then \
        set +e; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            libfreenect2-2 libfreenect2-dev freenect2-tools; \
        rc=$?; \
        set -e; \
        if [ "$rc" -ne 0 ]; then \
            echo "libfreenect2 packages not found in this distro snapshot; image keeps Kinect2 runtime prerequisites only."; \
        fi; \
    fi; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p \
        /data/logs \
        /data/recordings \
        /data/snapshots \
        /data/uploads/faces \
        /data/.cachesec/models

COPY requirements.txt requirements-kinect.txt requirements-detectron2.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && if [ "$INSTALL_KINECT" = "true" ]; then \
        python -m pip install -r requirements-kinect.txt; \
    fi \
    && if [ "$INSTALL_DETECTRON2" = "true" ]; then \
        python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"; \
        python -m pip install --upgrade setuptools wheel; \
        python -m pip install --no-build-isolation -r requirements-detectron2.txt; \
    fi

COPY . .

EXPOSE 5000

VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/', timeout=5)"

ENTRYPOINT ["tini", "--"]

CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "8", "-b", "0.0.0.0:5000", "--timeout", "0", "--graceful-timeout", "30", "--keep-alive", "5", "app:app"]
