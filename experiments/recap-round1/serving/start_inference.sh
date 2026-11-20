#!/usr/bin/env bash
# GPU warm-up completes before /healthz opens.
set -euo pipefail
root=/root/b601-deploy
bundle="$root/B601_PI05_DEPLOYMENT"
model_dir="$bundle"
if [[ -e "$root/active-policy" || -L "$root/active-policy" ]]; then
    model_dir="$root/active-policy"
fi
export HF_HOME="$root/.cache/huggingface"
export UV_CACHE_DIR="$root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$root/python"
export PIP_CACHE_DIR="$root/.cache/pip"
export OPENPI_DATA_HOME="$bundle/cache/openpi"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
denoise_steps="${B601_DENOISE_STEPS:-10}"
case "$denoise_steps" in 5|10) ;; *) printf 'B601_DENOISE_STEPS must be 5 or 10\n' >&2; exit 1 ;; esac
sampler_flags=()
case "${B601_ENABLE_RTC:-0}" in
    0) sampler_flags+=(--no-rtc) ;;
    1) ;;
    *) printf 'B601_ENABLE_RTC must be 0 or 1\n' >&2; exit 1 ;;
esac
if [[ "${B601_ENABLE_RTC:-0}" == 0 && -s "$root/reference_motor_atol.txt" ]]; then
    read -r reference_atol < "$root/reference_motor_atol.txt"
    sampler_flags+=(--golden-motor-atol "$reference_atol")
fi
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
        --bundle "$bundle" --checkpoint "$model_dir/checkpoint" --reference "$model_dir/example_observation.npz" \
        --port 8002 --denoise-steps "$denoise_steps" "${sampler_flags[@]}" \
        > "$root/logs/rtc-service.log" 2>&1 < /dev/null &
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
