FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy app source
COPY main.py skill_registry.py ./
COPY skills/ ./skills/

# Non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# data/ is mounted at runtime
VOLUME ["/app/data"]

CMD ["/app/.venv/bin/python", "main.py"]
