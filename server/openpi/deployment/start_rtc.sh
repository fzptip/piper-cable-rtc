#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${RTC_ENV_FILE:-$REPO/../../configs/server.env}"
if [[ -n "${RTC_ENV_FILE:-}" || -f "$ENV_FILE" ]]; then
  source "$ENV_FILE"
fi
PYTHON="${RTC_PYTHON:-/root/miniconda3/envs/pi0.5_wq/bin/python}"
# Deliberately replace the legacy external source override with this checkout.
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$REPO/scripts"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$REPO"
exec "$PYTHON" "$REPO/scripts/serve_rtc.py" "$@"
