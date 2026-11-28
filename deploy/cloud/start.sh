#!/bin/sh
# Start (or upgrade) Vectraudit Cloud with Docker Compose, in one command:
#   ./start.sh                      # first run writes .env with random secrets
#   VECTRAUDIT_CLOUD_PUBLIC_URL=https://vectraudit.example.com ./start.sh
# Then create the first workspace with the command it prints. Re-running it after pulling
# new code rebuilds the image, applies migrations and restarts the API (take a backup
# first: see docs/cloud-operations.md#upgrades).
set -eu
cd "$(dirname "$0")"

command -v docker >/dev/null || { echo "Docker is required: https://docs.docker.com/get-docker/" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Docker Compose v2 is required (docker compose)." >&2; exit 1; }

random() {  # URL-safe random text; the password goes into a database URL unescaped
  if command -v openssl >/dev/null; then
    openssl rand -hex "$1"
  else
    python3 -c "import secrets; print(secrets.token_hex($1))"
  fi
}

if [ ! -f .env ]; then
  umask 077
  public_url="${VECTRAUDIT_CLOUD_PUBLIC_URL:-http://localhost:8800}"
  sed -e "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$(random 24)|" \
      -e "s|^VECTRAUDIT_CLOUD_SECRET_KEY=.*|VECTRAUDIT_CLOUD_SECRET_KEY=$(random 32)|" \
      -e "s|^VECTRAUDIT_CLOUD_PUBLIC_URL=.*|VECTRAUDIT_CLOUD_PUBLIC_URL=$public_url|" \
      .env.example > .env
  echo "Wrote .env with new random secrets (public URL: $public_url). Keep it out of git."
else
  echo "Using the existing .env."
fi

docker compose up -d --build --wait

for _ in $(seq 1 60); do
  if body=$(curl -fsS http://127.0.0.1:8800/readyz 2>/dev/null); then
    echo "Vectraudit Cloud is ready: $body"
    cat <<'EOF'

Next:
  docker compose run --rm api create-workspace acme --name "Acme" --owner-email you@example.com
  # prints the owner's token once; sign in at the public URL with it.
Put a TLS-terminating reverse proxy in front of 127.0.0.1:8800 before anyone else uses it,
and schedule ./backup.sh (see docs/cloud-operations.md).
EOF
    exit 0
  fi
  sleep 2
done
echo "The API did not become ready; see: docker compose logs migrate api" >&2
exit 1
