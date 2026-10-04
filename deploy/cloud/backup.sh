#!/bin/sh
# Nightly backup: a compressed custom-format dump next to this file (copy it off the host).
#   ./backup.sh            -> backups/vectraudit-YYYYmmdd-HHMMSS.dump
# Writes to a temporary file and renames it only after a successful, non-empty dump, so a
# failed run never leaves a file that looks like a backup.
set -eu
cd "$(dirname "$0")"
mkdir -p backups
out="backups/vectraudit-$(date -u +%Y%m%d-%H%M%S).dump"
trap 'rm -f "$out.part"' EXIT
docker compose exec -T db pg_dump -U vectraudit -d vectraudit -Fc > "$out.part"
[ -s "$out.part" ] || { echo "backup failed: empty dump" >&2; exit 1; }
mv "$out.part" "$out"
echo "$out"
