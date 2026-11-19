# syntax=docker/dockerfile:1
# vectraudit demo for Hugging Face Spaces (Docker SDK).
#
# In a Space repo (just this Dockerfile + README.md) the source is cloned from the public
# GitHub repository at VECTRAUDIT_REF.
# From a vectraudit checkout the local source is used instead:
#   docker build -f deploy/hf-space/Dockerfile -t vectraudit-space .
#   docker run --rm -p 7860:7860 vectraudit-space

ARG VECTRAUDIT_REPO=https://github.com/rmnvg/Vectraudit.git
ARG VECTRAUDIT_REF=main

# --- 1. source + built UI -----------------------------------------------------------------
FROM node:22-slim AS ui
ARG VECTRAUDIT_REPO
ARG VECTRAUDIT_REF
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY . /context
RUN if [ -f /context/pyproject.toml ]; then cp -r /context /src; \
    else git clone --depth 1 --branch "$VECTRAUDIT_REF" "$VECTRAUDIT_REPO" /src \
        && rm -rf /src/.git; fi
WORKDIR /src/frontend
RUN npm ci --no-audit --no-fund && npm run build

# --- 2. runtime -------------------------------------------------------------------------
FROM python:3.12-slim
# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/user/.cache/huggingface \
    VECTRAUDIT_CACHE_DIR=/home/user/.cache/vectraudit \
    TOKENIZERS_PARALLELISM=false
WORKDIR /home/user

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build.
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --from=ui --chown=user /src /home/user/vectraudit-src
RUN pip install --user "/home/user/vectraudit-src[all]"

# Bake the demo: data, embedding model and projections (PCA 2D/3D, UMAP 2D).
RUN python -c "from pathlib import Path; from vectraudit.demo import build_demo; build_demo(Path('/home/user/demo'))" \
    && python /home/user/vectraudit-src/deploy/hf-space/warm_cache.py /home/user/demo \
    && rm -rf /home/user/demo/raw

EXPOSE 7860
CMD ["vectraudit", "serve", "/home/user/demo/ivf_pq.index", \
     "--vectors", "/home/user/demo/vectors.npy", \
     "--meta", "/home/user/demo/chunks.jsonl", \
     "--queries", "/home/user/demo/queries.npy", \
     "--embedder", "all-MiniLM-L6-v2", "--normalize-text", \
     "--host", "0.0.0.0", "--port", "7860", "--no-browser", "--demo-mode"]
