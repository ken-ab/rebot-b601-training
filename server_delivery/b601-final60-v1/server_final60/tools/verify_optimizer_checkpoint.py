"""Read every saved optimizer array on CPU and bind counts to its checkpoint."""
import argparse
import datetime
import json
from pathlib import Path
import time

import jax
import numpy as np
import orbax.checkpoint as ocp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert jax.default_backend() == 'cpu', 'Run with JAX_PLATFORMS=cpu and CUDA_VISIBLE_DEVICES empty'
    checkpoint = args.checkpoint.resolve()
    bound = json.loads((checkpoint / 'B601_ACTIVE_TIME.json').read_text())
    assert bound['validation_status'] == 'passed'
    started = time.monotonic()
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(checkpoint / 'train_state')
        state = checkpointer.restore(
            checkpoint / 'train_state',
            ocp.args.PyTreeRestore(
                item=metadata,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), metadata),
            ),
        )
    step = int(np.asarray(state['step']))
    assert step == bound['optimizer_step'] == bound['step_label'] + 1
    arrays = []
    counts = {}
    moments = {'mu': 0, 'nu': 0}
    for keys, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        name = jax.tree_util.keystr(keys)
        array = np.asarray(value)
        assert np.isfinite(array).all(), name
        arrays.append(array)
        if name.endswith("['count']"):
            counts[name] = int(array)
            assert counts[name] == step, name
        for label in moments:
            if "['" + label + "']" in name:
                moments[label] += 1
                if label == 'nu':
                    assert (array >= 0).all(), name
    assert len(counts) == 2 and moments['mu'] > 0 and moments['nu'] == moments['mu']
    result = {
        'status': 'passed', 'at': datetime.datetime.now().astimezone().isoformat(),
        'checkpoint': str(checkpoint), 'restored_step': step,
        'optimizer_and_schedule_counts': counts,
        'restored_arrays': len(arrays), 'restored_bytes': sum(a.nbytes for a in arrays),
        'moment_arrays': moments, 'all_arrays_finite': True,
        'second_moments_nonnegative': True, 'checkpoint_counter_matches': True,
        'elapsed_seconds': time.monotonic() - started,
        'scope': 'Saved optimizer/model step readback; not an exact DataLoader-position resume claim',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), args.output
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
