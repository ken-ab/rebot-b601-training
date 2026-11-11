#!/usr/bin/env bash
set -euo pipefail
base="$(cd -- "$(dirname -- "$0")" && pwd -P)"
python3 - "$base" <<'PY'
import hashlib,json,pathlib,sys
base=pathlib.Path(sys.argv[1])
manifest=json.loads((base/'portable-manifest.json').read_text())
for record in manifest['files']:
    p=(base/record['path']).resolve()
    if not p.is_relative_to(base) or not p.is_file() or p.stat().st_size!=record['bytes']:
        raise SystemExit('Missing/size mismatch: '+record['path'])
    digest=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(8*1024*1024):digest.update(block)
    if digest.hexdigest()!=record['sha256']:raise SystemExit('SHA256 mismatch: '+record['path'])
print('PORTABLE_FILES_VERIFIED',len(manifest['files']))
PY
if [[ "${1:-}" == "--verify-only" ]]; then exit 0; fi
if [[ "$EUID" != 0 ]]; then echo 'Run restore as root (sudo bash restore.sh).' >&2; exit 2; fi
nvidia-smi
mkdir -p /root/b601-deploy/B601_PI05_DEPLOYMENT "$base/tools"
# The HF snapshot contains weights/statistics/tokenizer and its original code.
# Apply the actual cloud snapshot afterwards so the latest code wins.
cp -a "$base/hf-model/." /root/b601-deploy/B601_PI05_DEPLOYMENT/
tar -xpf "$base/archives/b601-code-assets.tar.gz" -C /root
tar -xpf "$base/archives/b601-python311.tar.gz" -C /root
tar -xpf "$base/archives/portable-zstd.tar.gz" -C "$base/tools"
bundle=/root/b601-deploy/B601_PI05_DEPLOYMENT
if [[ -f "$base/archives/b601-linux-runtime.tar.zst" ]]; then
    LD_LIBRARY_PATH="$base/tools/zstd/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$base/tools/zstd/bin/zstd" -dc "$base/archives/b601-linux-runtime.tar.zst" | tar -xpf - -C /root
else
    # Rebuild from the cached Linux files when the original VM expires before
    # its full environment archive transfers. No Windows wheels are installed.
    command -v gcc >/dev/null || { echo 'Wheel rebuild needs GCC: install build-essential linux-libc-dev.' >&2; exit 2; }
    py=/root/b601-deploy/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11
    "$py" -m venv "$bundle/openpi/.venv"
    py="$bundle/openpi/.venv/bin/python"
    "$py" -m pip install --no-index --find-links "$base/linux-wheelhouse" --no-deps \
        -r "$base/linux-wheelhouse/requirements-wheels.txt"
    "$py" -m pip install --no-index --find-links "$base/linux-wheelhouse" --no-deps --no-build-isolation \
        "$base"/linux-wheelhouse/antlr4-python3-runtime-*.tar.gz "$base"/linux-wheelhouse/evdev-*.tar.gz
    mkdir -p /root/b601-deploy/sources
    tar -xpf "$base/sources/lerobot-0cf864870cf29f4738d3ade893e6fd13fbd7cdb5.tar.gz" -C /root/b601-deploy/sources
    "$py" -m pip install --no-index --no-deps --no-build-isolation \
        /root/b601-deploy/sources/lerobot-0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
    "$py" -m pip install --no-index --no-deps --no-build-isolation \
        -e "$bundle/openpi" -e "$bundle/openpi/packages/openpi-client"
fi
export OPENPI_DATA_HOME="$bundle/cache/openpi"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"$bundle/openpi/.venv/bin/python" - <<'PY'
import jax
devices=jax.devices()
print('JAX_DEVICES',devices)
if not any(device.platform=='gpu' for device in devices):
    raise SystemExit('No JAX GPU. Check the new image/driver before starting service.')
PY
bash /root/b601-deploy/start_inference.sh
echo 'RTC ready: upstream 8002, JPEG 8001. Forward local 8766 to remote 8001.'
