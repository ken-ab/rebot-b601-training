#!/usr/bin/env bash
# System-disk deployment. RTC GPU warm-up completes before /healthz opens.
set -euo pipefail
root=/root/b601-deploy
bundle="$root/B601_PI05_DEPLOYMENT"
export HF_HOME="$root/.cache/huggingface"
export UV_CACHE_DIR="$root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$root/python"
export PIP_CACHE_DIR="$root/.cache/pip"
export OPENPI_DATA_HOME="$bundle/cache/openpi"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$root/logs"
service_alive() {
    "$bundle/openpi/.venv/bin/python" - "$1" "$2" <<'PY'
import os, pathlib, sys
try:
    pid = int(pathlib.Path(sys.argv[1]).read_text().strip())
    command = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
    os.kill(pid, 0)
    valid = sys.argv[2].encode() in command
except (OSError, ValueError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}
model_pid="$root/logs/rtc-service.pid"
relay_pid="$root/logs/jpeg-relay.pid"
if ! curl -fsS --max-time 3 http://127.0.0.1:8002/healthz >/dev/null 2>&1 \
    && ! service_alive "$model_pid" "$root/b601_rtc_server.py"; then
    nohup "$bundle/openpi/.venv/bin/python" -u "$root/b601_rtc_server.py" \
        --bundle "$bundle" --port 8002 > "$root/logs/rtc-service.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$root/logs/rtc-service.pid"
fi
if ! curl -fsS --max-time 3 http://127.0.0.1:8001/healthz >/dev/null 2>&1 \
    && ! service_alive "$relay_pid" "$root/b601_jpeg_transport.py"; then
    nohup "$bundle/openpi/.venv/bin/python" "$root/b601_jpeg_transport.py" \
        --port 8001 --upstream-port 8002 > "$root/logs/jpeg-relay.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$root/logs/jpeg-relay.pid"
fi
printf 'B601_RTC_LOADING_ON_GPU\n'
while ! curl -fsS --max-time 3 http://127.0.0.1:8002/healthz >/dev/null 2>&1; do
    if ! service_alive "$model_pid" "$root/b601_rtc_server.py"; then
        tail -n 30 "$root/logs/rtc-service.log" >&2
        exit 1
    fi
    sleep 1
done
while ! curl -fsS --max-time 3 http://127.0.0.1:8001/healthz >/dev/null 2>&1; do
    if ! service_alive "$relay_pid" "$root/b601_jpeg_transport.py"; then
        tail -n 30 "$root/logs/jpeg-relay.log" >&2
        exit 1
    fi
    sleep 1
done
printf 'B601_RTC_INFERENCE_SERVICES_STARTED\n'
