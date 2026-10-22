# syntax=docker/dockerfile:1.4
# Enable BuildKit features for better caching and parallel builds

# Start from Drake's official image (includes pydrake, solvers, etc.)
FROM robotlocomotion/drake:latest

# Set a dedicated workspace directory
WORKDIR /workspace

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---- System utilities & Python tooling ----
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    python3-venv \
    python3-dev \
    libgl1 \
    libgl1-mesa-dri \
    libegl1 \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxxf86vm1 \
    libasound2t64 \
    libpulse0 \
    vim \
    && rm -rf /var/lib/apt/lists/*

# ---- Install uv (fast Python package installer) ----
# Install uv to /usr/local/bin for system-wide availability
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# ---- Create and use a virtual environment ----
ENV VIRTUAL_ENV=/opt/venv
RUN /usr/local/bin/uv venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY pyproject.toml README.md ./
COPY src ./src

# Install the project and all development tools from pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --editable ".[dev]"

# Make sure Jupyter kernel is available in notebooks
RUN python -m ipykernel install --name=mpc_spacecraft --display-name "Python (mpc_spacecraft)"

# ---- Locale & time zone setup ----
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Default command
CMD [ "bash" ]
