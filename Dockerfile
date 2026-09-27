# Smart Sugarcane AI - one container serving the whole application.
#
# Build from the PROJECT ROOT, not from backend/:
#     docker build -t smart-sugarcane-ai .
#     docker run -p 7860:7860 smart-sugarcane-ai
#
# The React build is copied into frontend/dist, which app/main.py mounts as a
# single-page app at "/". One origin serves the interface and the API together,
# so there is no CORS configuration to get wrong and the app reaches the backend
# at a relative /api path from any device on any network.
#
# The whole project layout is preserved inside the image because
# app/core/config.py derives every path from PROJECT_ROOT (three levels above
# itself). The backend reads data/ and models/ from the project root, so
# flattening the tree would break both.

# ---------------------------------------------------------------- frontend
FROM node:20-slim AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# Relative /api, because the API is served from this same origin. A build-time
# absolute URL is what previously tied the Android app to one laptop's address.
ENV VITE_API_BASE_URL=/api
ENV VITE_MEDIA_BASE_URL=
RUN npm run build

# ----------------------------------------------------------------- runtime
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is required by scikit-learn's and TensorFlow's OpenMP code, and
# libglib2.0-0 by opencv. Installed before the app layer so they stay cached.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so code changes do not invalidate the pip layer. This
# installs TensorFlow, which dominates both image size and build time - and is
# what makes the three trained image models work rather than silently
# degrading to demo heuristics.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --upgrade pip && pip install -r backend/requirements.txt

# Application code, knowledge base and trained models.
COPY backend/ ./backend/
COPY data/ ./data/
COPY ml/ ./ml/
COPY models/ ./models/

# The built interface, from the stage above.
COPY --from=frontend /build/dist ./frontend/dist

# The irrigation model is committed now, but train one if a build context ever
# lacks it, so the container never quietly falls back to the rule engine.
ARG TRAIN_IRRIGATION_MODEL=1
RUN if [ "$TRAIN_IRRIGATION_MODEL" = "1" ] && [ ! -f models/irrigation_model.joblib ]; then \
        echo "No trained irrigation model in the build context - training one now..." && \
        python ml/irrigation/train.py --rows 8000 ; \
    else \
        echo "Using the committed irrigation model." ; \
    fi

# Uploads and the SQLite fallback need somewhere writable. Hugging Face Spaces
# runs the container as a non-root user, so these are created and opened up at
# build time rather than left for the app to create at runtime.
RUN mkdir -p /app/uploads/plants /app/uploads/soil \
    && chmod -R 777 /app/uploads

WORKDIR /app/backend

# Hugging Face Spaces expects 7860; other hosts inject their own $PORT.
ENV PORT=7860
EXPOSE 7860

# start-period is generous: importing TensorFlow and loading three models takes
# well over a minute on a free CPU tier, and a tighter window would have the
# platform kill a container that was starting normally.
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT:-7860}/api/system/health || exit 1

# Shell form so ${PORT} is expanded at runtime. Bind 0.0.0.0 so the platform's
# proxy can reach the process.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
