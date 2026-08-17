FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies globally in the container
RUN uv sync --locked --no-dev

# Put the virtual environment in PATH so commands like 'uvicorn' and 'celery' work automatically
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY . .

# Expose port (if run as API)
EXPOSE 8000

# Default command (FastAPI server)
# Can be overridden by docker-compose for celery workers
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
