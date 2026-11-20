#!/usr/bin/env bash
# Independently back up the selected offline training artifacts as each becomes available.
set -euo pipefail
root=${B601_EVO_ROOT:-/data/b601-recap-round1/evo}
cd "$root"
umask 077
mkdir -p delivery
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$root/sources/Evo-RL/src"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1

wait_for() {
    local path=$1
    until test -s "$path"; do sleep 15; done
}

pack_parts() {
    local name=$1
    shift
    local path="$root/delivery/$name-parts"
    mkdir -p "$path"
    test ! -s "$path/CHECKSUMS" || return 0
    tar -cf - "$@" | split --bytes=256M --numeric-suffixes --suffix-length=3 - "$path/part-"
    (cd "$path" && sha256sum part-[0-9][0-9][0-9] > CHECKSUMS.partial && mv CHECKSUMS.partial CHECKSUMS)
}

stage=$1
exec 9>"delivery/$stage.lock"
flock -n 9
case "$stage" in
    value)
        wait_for value/metrics.json
        bundle="$root/delivery/value-model"
        venv/bin/python code/portable/evo_bundle.py export --bundle "$bundle"
        cp code/portable/{evo_bundle.py,evo_value.py,VALUE_README.md} "$bundle/"
        cp sources/evorl.tar.gz "$bundle/evorl-source.tar.gz"
        cp sources/Evo-RL/LICENSE "$bundle/Evo-RL-LICENSE"
        cp value/config.json "$bundle/transfer-config.json"
        nice -n 10 venv/bin/python code/portable/evo_bundle.py verify --bundle "$bundle" --device cpu \
            --cache "$root/delivery-cache/value" --report "$bundle/readback.json"
        cd delivery
        pack_parts value value-model
        ;;
    labels)
        wait_for label_summary.json
        mkdir -p delivery/tagged-data
        cp -R labels delivery/tagged-data/
        cp -L manifest.json label_summary.json contract.json delivery/tagged-data/
        cp value/predictions/complete.json delivery/tagged-data/prediction-summary.json
        cd delivery
        pack_parts labels tagged-data
        ;;
    policy)
        wait_for policy/result.json
        venv/bin/python code/evo_checkpoint.py fingerprint --checkpoint policy/checkpoint \
            --output delivery/cloud-checkpoint-arrays.json
        wait_for policy/complete
        cd policy
        pack_parts policy-info result.json config.json baseline.json best.json training.jsonl \
            parent-action-readback.json round1-action-readback.json checkpoint/conditioning.json
        ;;
    *)
        exit 2
        ;;
esac
date --iso-8601=seconds > "$root/delivery/$stage.ready"
