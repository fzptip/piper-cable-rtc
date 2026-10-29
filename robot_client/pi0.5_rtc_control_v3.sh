#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Current v3 preset. Arguments at the end may override these defaults.
exec bash "$HERE/pi0.5_control.sh" \
  --url ws://127.0.0.1:18000/ \
  --rtc-mode guided \
  --chunk-steps 15 \
  --rtc-prefix-attention-horizon 20 \
  --rtc-delay 5 \
  --rtc-transition-steps 6 \
  --rtc-smoothing-alpha 1.0 \
  --boundary-smooth-steps 0 \
  --action-rate 30 \
  --publish-rate 90 \
  --action-interpolation linear \
  "$@"
