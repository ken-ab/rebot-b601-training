#!/bin/sh
set -eu

case "${1:-}" in
    check|smoke|train) ;;
    *) echo "Usage: bash run_pi05.sh {check|smoke|train} [--exp-name NAME] [--base-params /path/to/params] [--batch-size N] [--steps N] [--workers N] [--resume]"; exit 2 ;;
esac

case "$0" in
    */*) script_dir="${0%/*}" ;;
    *) script_dir=. ;;
esac
kit="$(CDPATH= cd -- "${script_dir:-/}" && pwd -P)"
repo_dir="$(dirname -- "$kit")"
revision=215abfb217dbac7d5f1273282331b9b1866c0479
upstream=https://github.com/Physical-Intelligence/openpi.git
openpi_dir="${B601_OPENPI_ROOT:-$kit/.openpi}"

command -v uv >/dev/null || { echo "Install uv in the server environment first (https://docs.astral.sh/uv/getting-started/installation/)." >&2; exit 2; }
command -v flock >/dev/null || { echo "Install flock (util-linux) in the server environment first." >&2; exit 2; }
export GIT_LFS_SKIP_SMUDGE=1

# Serialize setup within this checkout. Training does not hold the setup lock.
(
    flock 9
    if [ ! -e "$openpi_dir" ]; then
        git init "$openpi_dir"
    fi
    if [ ! -e "$openpi_dir/.git" ]; then
        echo "OpenPI path is not a Git checkout: $openpi_dir" >&2
        exit 2
    fi
    head="$(git -C "$openpi_dir" rev-parse --verify HEAD 2>/dev/null || true)"
    if [ -z "$head" ]; then
        git -C "$openpi_dir" fetch --depth 1 "$upstream" "$revision"
        git -C "$openpi_dir" checkout --detach "$revision"
    elif [ "$head" != "$revision" ]; then
        echo "The B601 config expects OpenPI $revision. Use a new checkout or unset B601_OPENPI_ROOT." >&2
        exit 2
    fi
    git -C "$openpi_dir" submodule update --init --recursive
    cd -- "$openpi_dir"
    uv sync --frozen

    if [ ! -f "$kit/prepared/manifest.json" ]; then
        if [ -e "$kit/prepared" ]; then
            echo "Move the incomplete directory $kit/prepared aside, then rerun." >&2
            exit 2
        fi
        dataset=blue-tissue-center-20261109-final-60
        archive="$dataset.tar.gz"
        cache="$kit/.data"
        mkdir -p -- "$cache"
        stage="$(mktemp -d "$cache/stage.XXXXXX")"
        trap 'rm -rf -- "$stage"' EXIT
        if [ ! -f "$cache/$archive" ]; then
            command -v gh >/dev/null || { echo "Install and authenticate gh for the private final-60 Release." >&2; exit 2; }
            gh release download final-60 --repo ken-ab/rebot-b601-training --pattern "$archive" --dir "$stage"
            (cd -- "$stage" && sha256sum -c "$repo_dir/DATASET.sha256")
            mv -- "$stage/$archive" "$cache/$archive"
        else
            (cd -- "$cache" && sha256sum -c "$repo_dir/DATASET.sha256") || {
                echo "Move the invalid cached archive $cache/$archive aside, then rerun." >&2
                exit 2
            }
        fi
        tar -xzf "$cache/$archive" -C "$stage"
        uv run --frozen --no-sync python "$kit/prepare_dataset.py" \
            "$stage/$dataset" "$stage/prepared" --splits "$repo_dir/splits.json"
        mv -- "$stage/prepared" "$kit/prepared"
    fi
) 9>"$kit/.setup.lock"

cd -- "$openpi_dir"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
exec uv run --frozen --no-sync python "$kit/train_pi05.py" "$@"
