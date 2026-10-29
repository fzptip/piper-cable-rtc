#!/usr/bin/env bash
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${PIPER_ENV_FILE:-$HERE/../configs/robot.env}"
if [[ -n "${PIPER_ENV_FILE:-}" || -f "$ENV_FILE" ]]; then
  source "$ENV_FILE"
fi

source "${PIPER_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${PIPER_CONDA_ENV:-wwq_aloha}"
source "${PIPER_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
if [[ -n "${PIPER_WORKSPACE_SETUP:-}" ]]; then
  source "$PIPER_WORKSPACE_SETUP"
fi

cd "$HERE"
exec "${PIPER_PYTHON:-python}" "$HERE/pi05_ws_client.py" "$@"
