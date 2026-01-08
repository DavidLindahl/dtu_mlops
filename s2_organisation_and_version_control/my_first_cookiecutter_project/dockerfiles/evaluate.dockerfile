# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY data/ data/
COPY LICENSE LICENSE
COPY models/ models/
RUN uv sync --locked --no-cache --no-install-project

# Run the train script
ENTRYPOINT ["uv", "run", "src/cookiecutter_project/evaluate.py"]


ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv
