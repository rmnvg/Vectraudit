#!/bin/sh
# Restore a dump made by backup.sh, replacing everything in the database. Stop the API first:
#   docker compose stop api webhooks && ./restore.sh backups/<file>.dump && docker compose start api webhooks
# The schema is dropped and the dump replayed in one transaction, so tables a newer version
# added (after an upgrade you are rolling back) are removed too, and a restore that fails
# part-way leaves the database as it was.
set -eu
cd "$(dirname "$0")"
[ -f "${1:-}" ] || { echo "usage: $0 backups/<file>.dump" >&2; exit 1; }
if [ -n "$(docker compose ps --status running -q api webhooks 2>/dev/null)" ]; then
  echo "Stop the API and the webhook worker first: docker compose stop api webhooks" >&2
  exit 1
fi
# Decode the whole dump first: a damaged file fails here, before anything is dropped.
sql=$(mktemp)
trap 'rm -f "$sql"' EXIT
docker compose exec -T db pg_restore --no-owner -f - < "$1" > "$sql"
{
  echo "SET client_min_messages = warning; DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
  cat "$sql"
} | docker compose exec -T db psql -q -v ON_ERROR_STOP=1 --single-transaction -U vectraudit -d vectraudit >/dev/null
echo "restored $1"
