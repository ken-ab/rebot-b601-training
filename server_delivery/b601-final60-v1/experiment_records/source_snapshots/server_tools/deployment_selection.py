"""Bounded action-validation selection; never changes the training protocol."""
import hashlib
import json
from pathlib import Path

import numpy as np


def select_deployment(root, monitor, training_stats):
    root, monitor = Path(root), Path(monitor)
    expected_norm_hash = json.loads((monitor / 'RUN_PARAMETERS.json').read_text())['norm_stats_sha256']
    training_stats = Path(training_stats)
    assert hashlib.sha256(training_stats.read_bytes()).hexdigest() == expected_norm_hash
    expected_norm = json.loads(training_stats.read_text())
    assert set(expected_norm['norm_stats']) == {'state', 'actions'}
    for stats in expected_norm['norm_stats'].values():
        assert set(stats) == {'mean', 'std', 'q01', 'q99'}
        for values in stats.values():
            values = np.asarray(values, dtype=np.float64)
            assert values.shape == (7,) and np.isfinite(values).all()
    canonical_norm_hash = hashlib.sha256(json.dumps(expected_norm, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    reference = np.load(root / 'evaluation/base/predictions.npz')
    sources = [('checkpoint10000', root / 'evaluation/checkpoint10000'),
               ('flow_best', root / 'evaluation/best'), ('last', root / 'evaluation/last'),
               ('interim5000', root / 'evaluation/interim5000')]
    records, identities = [], set()
    for label, directory in sources:
        metrics = json.loads((directory / 'METRICS.json').read_text())
        assert metrics['full_protocol'] and metrics['evaluated_action_rows'] == 5118
        assert (metrics['seed'], metrics['denoise_steps'], metrics['stride']) == (20261109, 10, 25)
        assert metrics['inference_xla_flags'] == '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true'
        assert (metrics['jax_version'], metrics['jaxlib_version']) == ('0.5.3', '0.5.3')
        checkpoint = Path(metrics['checkpoint'])
        state = json.loads((checkpoint / 'B601_ACTIVE_TIME.json').read_text())
        assert state['validation_status'] == 'passed'
        identity = state['optimizer_step']
        stats = checkpoint / 'assets/rebot-b601-train/norm_stats.json'
        # OpenPI save() omits the prepared file's final newline. Require exact
        # parsed values/structure as well as the original prepared file's hash.
        checkpoint_norm = json.loads(stats.read_text())
        assert hashlib.sha256(json.dumps(checkpoint_norm, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest() == canonical_norm_hash
        data = np.load(directory / 'predictions.npz')
        assert data['prediction'].shape == (211, 25, 7)
        assert np.isfinite(data['prediction']).all() and np.isfinite(data['target']).all()
        for key in ('target', 'valid_mask', 'episode_source_frame'):
            np.testing.assert_array_equal(data[key], reference[key])
        errors = (data['prediction'].astype(np.float64) - data['target'])[data['valid_mask']]
        assert len(errors) == 5118
        mae = np.abs(errors).mean(axis=0)
        rmse = np.sqrt(np.square(errors).mean(axis=0))
        np.testing.assert_allclose(mae, metrics['per_axis_mae'], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(rmse, metrics['per_axis_rmse'], atol=2e-5, rtol=2e-5)
        episode_mae, episode_rmse = [], []
        for episode in np.unique(data['episode_source_frame'][:, 0]):
            rows = data['episode_source_frame'][:, 0] == episode
            episode_errors = (data['prediction'][rows].astype(np.float64)-data['target'][rows])[data['valid_mask'][rows]]
            episode_mae.append(np.abs(episode_errors).mean(axis=0))
            episode_rmse.append(np.sqrt(np.square(episode_errors).mean(axis=0)))
        assert len(episode_mae) == 10
        macro_mae, macro_rmse = np.mean(episode_mae, axis=0), np.mean(episode_rmse, axis=0)
        np.testing.assert_allclose(macro_mae, metrics['macro_episode_per_axis_mae'], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(macro_rmse, metrics['macro_episode_per_axis_rmse'], atol=2e-5, rtol=2e-5)
        if identity in identities:
            continue
        identities.add(identity)
        records.append({'label': label, 'optimizer_step': identity, 'checkpoint': str(checkpoint),
            'evaluation_directory': str(directory), 'per_axis_mae_float64': mae.tolist(),
            'norm_stats_raw_sha256': hashlib.sha256(stats.read_bytes()).hexdigest(),
            'per_axis_rmse_float64': rmse.tolist(), 'mean_joint_1to6_mae': float(mae[:6].mean()),
            'gripper_mae': float(mae[6]), 'mean_joint_1to6_axis_rmse': float(rmse[:6].mean()),
            'gripper_rmse': float(rmse[6]),
            'macro_episode_per_axis_mae': macro_mae.tolist(),
            'macro_episode_per_axis_rmse': macro_rmse.tolist()})
    assert 1 <= len(records) <= 4

    def dominates(a, b):
        av = np.asarray([a['mean_joint_1to6_mae'], a['gripper_mae']])
        bv = np.asarray([b['mean_joint_1to6_mae'], b['gripper_mae']])
        return bool(np.all(av <= bv) and np.any(av < bv))

    frontier = [r for r in records if not any(dominates(other, r) for other in records)]
    # Fixed before final results: retain the verified 10k incumbent if nondominated;
    # otherwise prefer final flow-best, then last, then the interim 5k audit model.
    primary = frontier[0]
    result = {'protocol': str(root / 'ACTION_SELECTION_PROTOCOL.json'), 'candidates': records,
        'primary': primary, 'alternatives': frontier[1:], 'pareto_labels': [r['label'] for r in frontier],
        'selection_rule': 'Pareto on separate six-joint mean MAE and gripper MAE; fixed preference checkpoint10000,flow_best,last,interim5000 among nondominated candidates',
        'original_flow_best_unchanged': json.loads((monitor / 'BEST.json').read_text()),
        'targets_masks_sources_norm_verified': True, 'metrics_recomputed_float64': True,
        'prepared_norm_stats_raw_sha256': expected_norm_hash,
        'canonical_norm_stats_sha256': canonical_norm_hash,
        'qualification': 'At most four trained checkpoints; validation reused for selection, not an untouched test or physical robot success'}
    (root / 'ACTION_SELECTION_RESULT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result
