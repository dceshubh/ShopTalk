# ShopTalk — multi-stage image (docs/ShopTalk_Plan.md §11 / Rubric #3 "Dockerized").
#
# One image runs BOTH processes the project runs locally as two separate ones — the API
# (`uvicorn src.api.main:app`) and the UI (`streamlit run src/ui/app.py`) — selected per
# service via `command:` in docker-compose.yml. "Dev looks like prod" stays true: the same
# artifact that boots on a laptop is what would run on an AWS g4dn.xlarge box.
#
# Model artifacts (Chroma index, HF model cache, Piper voices, the feedback DB, Redis data)
# are deliberately NOT baked into the image — see docker-compose.yml's volume mounts. That
# keeps this image small, cacheable, and rebuildable independently of the data layer: a
# fresh `products_enriched.parquet` from a Kaggle run never requires a new image.

# ---------------------------------------------------------------------------------------
# Stage 1 — builder: resolve and install the pinned dependency set into a venv
# ---------------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# build-essential is defensive insurance for any pinned wheel that lacks a prebuilt
# manylinux/cp312 binary on this platform and falls back to a source build (none are known
# to as of this pin-set, but the alternative — a build that fails three layers deep inside
# `pip install` on a box with no compiler — costs far more than the ~120MB this stage pays
# and discards). It never reaches the runtime stage below.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------------------
# Stage 2 — runtime: copy the venv + source only (no compilers, no pip cache, no .git)
# ---------------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1"

WORKDIR /app

# Only what the running service needs — `tests/`, `notebooks/`, `docs/` stay out of the
# image (they're dev/eval-time artifacts, not serving-time ones).
COPY src/ src/
COPY configs/ configs/
COPY pyproject.toml ./

# `configs/config.yaml: paths` resolves relative to PROJECT_ROOT (src/common/config.py) —
# these are the mount points docker-compose binds the host's `data/`/`weights/`/`.env` to.
RUN mkdir -p data weights

EXPOSE 8000 8501

# A sensible default so `docker run shoptalk` alone does something useful (boots the API);
# docker-compose overrides this per-service — see `command:` on the `ui` service there.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
