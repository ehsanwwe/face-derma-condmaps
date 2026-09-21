# syntax=docker/dockerfile:1

# ---------- builder: compile insightface + install python deps into a venv ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# server.py forces CPU, so the small CPU-only torch wheels are enough
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.docker.txt .
RUN pip install -r requirements.docker.txt


# ---------- runtime ----------
FROM python:3.11-slim

# libgomp1: OpenMP runtime needed by torch/onnxruntime; libglib2.0-0: needed by opencv
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app

WORKDIR /app
COPY --chown=app:app config ./config
COPY --chown=app:app detectors ./detectors
COPY --chown=app:app src ./src
COPY --chown=app:app third_party ./third_party
COPY --chown=app:app static ./static
COPY --chown=app:app server.py ./

# fail early (with a hint) if the git submodule was not checked out
RUN test -f third_party/face-parsing.PyTorch/model.py \
    || (echo "ERROR: third_party/face-parsing.PyTorch is empty. Run: git submodule update --init --recursive" && exit 1)

USER app

# Bake the downloadable weights into the image so the container works without
# internet at runtime: insightface buffalo_l (detector) + torchvision resnet18
# (BiSeNet backbone init, fetched by face-parsing.PyTorch/resnet.py).
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])" \
    && python -c "import torch.utils.model_zoo as m; m.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')"

# BiSeNet checkpoint (79999_iter.pth) is NOT in the image: mount ./models at /app/models
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
