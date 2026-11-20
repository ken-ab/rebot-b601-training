#!/usr/bin/env bash
# Proposed server launcher. It never patches source or starts a second stage automatically.
set -euo pipefail

if [[ ${1:-} != check && ${1:-} != smoke && ${1:-} != train ]]; then
    echo 'Usage: bash run_on_server.sh {check|smoke|train} [training arguments]'
    exit 2
fi

repo='/home/yujiegao/桌面/zzk/rebot-b601-training'
storage="${B601_STORAGE_ROOT:-/data2/yujiegao/b601-pi05}"
gpus="${B601_GPUS:-GPU-a2df82e1-ad48-ea47-fa69-5ee02ec32b0e,GPU-8d91782a-1f41-2aa0-ef48-585ded17365d}"
fsdp="${B601_FSDP_DEVICES:-2}"
mode="$1"
shift

export PATH="/home/yujiegao/.local/bin:$PATH"
command -v uv >/dev/null || { echo 'uv is missing; follow SERVER_PLAN_CN.md first.' >&2; exit 2; }
if ! rg -q 'action_sequence_keys=\("action",\)' "$repo/openpi_b601/train_pi05.py" ||
   ! rg -q '"--fsdp-devices"' "$repo/openpi_b601/train_pi05.py"; then
    echo 'The reviewed b601_server_adaptation.patch must be applied first.' >&2
    exit 2
fi

mkdir -p "$storage/cache/openpi" "$storage/cache/uv" "$storage/cache/huggingface"
cd "$repo"
case "$mode" in
    check) defaults=() ;;
    smoke) defaults=(--steps 20 --batch-size 2 --workers 0) ;;
    train) defaults=(--steps 10000 --batch-size 4 --workers 2) ;;
esac

# Only this child process gets the project identity and selected devices.
exec env -u GH_TOKEN -u GITHUB_TOKEN \
    GH_CONFIG_DIR='/home/yujiegao/.config/gh-zzk' \
    GIT_CONFIG_GLOBAL='/home/yujiegao/.config/gh-zzk/gitconfig' \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=http.version \
    GIT_CONFIG_VALUE_0=HTTP/1.1 \
    UV_PYTHON='/home/yujiegao/.local/bin/python3.11' \
    UV_CACHE_DIR="$storage/cache/uv" \
    UV_LINK_MODE=copy \
    HF_HOME="$storage/cache/huggingface" \
    OPENPI_DATA_HOME="$storage/cache/openpi" \
    CUDA_VISIBLE_DEVICES="$gpus" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${B601_MEM_FRACTION:-0.9}" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    bash openpi_b601/run_pi05.sh "$mode" --fsdp-devices "$fsdp" "${defaults[@]}" "$@"
