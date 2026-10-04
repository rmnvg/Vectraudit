# syntax=docker/dockerfile:1
# Vectraudit hosted control plane: one image for `migrate`, `serve` and admin commands.
#   docker build -f deploy/cloud/Dockerfile -t vectraudit-cloud .
# Provider-neutral: needs only a PostgreSQL URL and a secret key (see .env.example).

FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv build --wheel -o /dist

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir "$(ls /tmp/*.whl)[cloud]" && rm /tmp/*.whl
USER app
ENV PYTHONUNBUFFERED=1
EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8800/healthz', timeout=3)"
ENTRYPOINT ["vectraudit-cloud"]
CMD ["serve", "--port", "8800"]
