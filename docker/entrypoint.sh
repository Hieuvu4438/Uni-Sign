#!/usr/bin/env sh
set -eu

mkdir -p "${TEMP_DIR:-/tmp/unisign}"
exec "$@"
