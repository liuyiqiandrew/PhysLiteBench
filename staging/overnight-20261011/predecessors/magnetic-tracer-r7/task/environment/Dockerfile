FROM python:3.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache numpy==2.3.3 scipy==1.16.3 pytest==8.4.2
WORKDIR /app
COPY README.md model.py test_public.py ./
COPY data/ ./data/
