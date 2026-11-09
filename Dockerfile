
# Stage 1: Install project dependencies
FROM python:3.13-slim AS builder

WORKDIR /app

# Copy uv from its official Docker image
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_DOWNLOADS=never
ENV UV_LINK_MODE=copy

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install locked dependencies
RUN uv sync --frozen --no-dev --no-install-project


# Stage 2: Run the FastAPI application
FROM python:3.13-slim AS runtime

WORKDIR /app

# Scientific Python runtime dependency
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
