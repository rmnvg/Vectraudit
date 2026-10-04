#!/bin/sh
# Restore a dump made by backup.sh (replaces the current data). Stop the API first:
#   docker compose stop api && ./restore.sh backups/<file>.dump && docker compose start api
set -eu
cd "$(dirname "$0")"
[ -f "${1:-}" ] || { echo "usage: $0 backups/<file>.dump" >&2; exit 1; }
docker compose exec -T db pg_restore -U vectraudit -d vectraudit --clean --if-exists < "$1"
echo "restored $1"
