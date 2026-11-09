"""Local monitoring hooks around the unchanged pinned OpenPI training loop.

The wall clock starts after the first successful training dispatch, excluding
setup and initial JIT compilation. Saving and periodic validation count as
active experiment time. Checkpoint receipts preserve that definition on resume.
"""
from contextlib import contextmanager
import dataclasses
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from b601_data import ACTION_HORIZON, TRAIN_ID, VAL_ID
from openpi.models import model as model_lib
from openpi.training import checkpoints, data_loader, sharding


class LongRunComplete(Exception):
    """Requested active wall time reached after a committed checkpoint."""


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def timestamp():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def run_monitored(train_main, config, args, prepared):
    import wandb
    root = Path(config.checkpoint_dir)
    monitor = root.parent / (root.name + '_monitor')
    monitor.mkdir(parents=True, exist_ok=True)
    target_seconds = float(args.long_run_hours) * 3600
    original_mesh = sharding.set_mesh
    original_save = checkpoints.save_state
    original_restore = checkpoints.restore_state
    original_log = wandb.log
    globals_dict = train_main.__globals__
    original_init = globals_dict['init_train_state']
    original_init_wandb = globals_dict['init_wandb']
    log_delegate = {'function': original_log}
    runtime = {'started_monotonic': None, 'previous_active_seconds': 0.0,
               'dispatches_this_session': 0, 'evaluation_seconds_this_session': 0.0,
               'checkpoint_seconds_this_session': 0.0, 'manager': None}
    best_index = monitor / 'BEST.json'
    best_score = json.loads(best_index.read_text())['validation_flow_loss'] if best_index.exists() else float('inf')

    data_config = config.data.create(config.assets_dirs, config.model)
    validation_config = dataclasses.replace(data_config, repo_id=VAL_ID)
    raw = data_loader.create_torch_dataset(validation_config, ACTION_HORIZON, config.model)
    dataset = data_loader.transform_dataset(raw, validation_config)
    episodes = [json.loads(line) for line in
                (prepared / 'datasets' / VAL_ID / 'meta/episodes.jsonl').read_text().splitlines()]
    split_manifest = json.loads((prepared / 'manifest.json').read_text())
    selected, offset = [], 0
    for episode in episodes:
        length = episode['length']
        for local_frame in np.linspace(0, length - 1, 8, dtype=int).tolist():
            selected.append({'index': offset + local_frame, 'local_episode': episode['episode_index'],
                             'source_episode': split_manifest['val']['source_episode_indices'][episode['episode_index']],
                             'frame': local_frame})
        offset += length
    assert len(selected) == 80 and offset == split_manifest['val']['frames']
    atomic_json(monitor / 'EVAL_PROTOCOL.json', {'split': 'validation; used for model selection',
        'source_episode_indices': split_manifest['val']['source_episode_indices'], 'observations': selected,
        'normalization_source': TRAIN_ID, 'rng_seed': 20261109, 'train': False,
        'metric': 'normalized padded-dimension flow matching loss; not joint-angle error',
        'action_horizon': ACTION_HORIZON, 'batch_size': config.batch_size})
    cpu_batches = []
    for first in range(0, len(selected), config.batch_size):
        points = selected[first:first + config.batch_size]
        if len(points) != config.batch_size:
            raise ValueError('Validation observations must divide into complete global batches')
        samples = [dataset[point['index']] for point in points]
        batch = jax.tree.map(lambda *values: np.stack(values), *samples)
        observation = model_lib.Observation.from_dict(batch)
        cpu_batches.append((observation, batch['actions']))
    del dataset, raw
    compiled_eval = None
    eval_mesh = None
    data_sharding = None

    def elapsed():
        current = 0 if runtime['started_monotonic'] is None else time.monotonic() - runtime['started_monotonic']
        return runtime['previous_active_seconds'] + current

    def evaluate(state):
        started = time.monotonic()
        values = []
        with original_mesh(eval_mesh):
            for number, (obs, actions) in enumerate(cpu_batches):
                obs, actions = jax.device_put((obs, actions), data_sharding)
                key = jax.random.fold_in(jax.random.key(20261109), number)
                result = float(jax.device_get(compiled_eval(state.params, obs, actions, key)))
                if not np.isfinite(result):
                    raise FloatingPointError(f'Nonfinite validation loss in batch {number}')
                values.append(result)
        duration = time.monotonic() - started
        runtime['evaluation_seconds_this_session'] += duration
        return float(np.mean(values)), duration

    def initialized(cfg, init_rng, mesh, *, resume):
        nonlocal compiled_eval, eval_mesh, data_sharding
        state, state_sharding = original_init(cfg, init_rng, mesh, resume=resume)
        eval_mesh = mesh
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        graphdef = state.model_def

        def loss(params, observation, actions, key):
            model = nnx.merge(graphdef, params)
            model.eval()
            return jnp.mean(model.compute_loss(key, observation, actions, train=False))

        compiled_eval = jax.jit(loss, in_shardings=(state_sharding.params, data_sharding, data_sharding, replicated),
                                out_shardings=replicated)
        if not resume:
            baseline, duration = evaluate(state)
            atomic_json(monitor / 'BASELINE_FLOW.json', {'recorded_at': timestamp(), 'validation_flow_loss': baseline,
                        'evaluation_seconds': duration, 'optimizer_step': 0, 'base': str(cfg.weight_loader),
                        'baseline_kind': 'base weights plus randomly initialized LoRA A and B; not a pure-base control'})
            logging.info('Fixed validation initial-model flow loss %.7f (%.1fs)', baseline, duration)
        return state, state_sharding

    @contextmanager
    def timed_mesh(mesh):
        with original_mesh(mesh):
            yield
        if runtime['started_monotonic'] is None:
            runtime['started_monotonic'] = time.monotonic()
            runtime['first_dispatch_completed_at'] = timestamp()
            atomic_json(monitor / 'SESSION_START.json', {'first_dispatch_completed_at': runtime['first_dispatch_completed_at'],
                'pid': os.getpid(), 'previous_active_seconds': runtime['previous_active_seconds'],
                'target_active_seconds': target_seconds, 'definition': __doc__})
        runtime['dispatches_this_session'] += 1

    def log_metrics(info, *positional, **keywords):
        if 'loss' in info:
            metrics = {key: float(np.asarray(value)) for key, value in info.items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                atomic_json(monitor / 'NONFINITE_FAILURE.json', {'at': timestamp(), 'step': keywords.get('step'),
                            'metrics': {key: str(value) for key, value in metrics.items()}})
                raise FloatingPointError('Nonfinite training loss or norm; preserved latest committed checkpoint')
            row = {'at': timestamp(), 'step_label': keywords.get('step'), 'active_seconds': elapsed(), **metrics}
            with (monitor / 'TRAIN_METRICS.jsonl').open('a') as output:
                output.write(json.dumps(row, allow_nan=False) + '\n')
        return log_delegate['function'](info, *positional, **keywords)

    def initialized_wandb(*positional, **keywords):
        result = original_init_wandb(*positional, **keywords)
        # wandb.init rebinds the module-level log, including in disabled mode.
        log_delegate['function'] = wandb.log
        wandb.log = log_metrics
        return result

    def save_evaluate(manager, state, loader, step):
        nonlocal best_score
        runtime['manager'] = manager
        started = time.monotonic()
        original_save(manager, state, loader, step)
        manager.wait_until_finished()
        runtime['checkpoint_seconds_this_session'] += time.monotonic() - started
        checkpoint = root / str(step)
        provisional = {'at': timestamp(), 'step_label': step, 'optimizer_step': int(jax.device_get(state.step)),
                       'active_seconds': elapsed(), 'validation_status': 'pending'}
        atomic_json(checkpoint / 'B601_ACTIVE_TIME.json', provisional)
        try:
            score, duration = evaluate(state)
        except Exception as exc:
            atomic_json(checkpoint / 'B601_ACTIVE_TIME.json', {**provisional, 'active_seconds': elapsed(),
                        'validation_status': 'failed', 'validation_error': repr(exc)})
            raise
        row = {'at': timestamp(), 'step_label': step, 'optimizer_step': int(jax.device_get(state.step)),
            'checkpoint': str(checkpoint), 'validation_flow_loss': score, 'evaluation_seconds': duration,
            'validation_status': 'passed',
            'active_seconds': elapsed(), 'target_active_seconds': target_seconds,
            'dispatches_this_session': runtime['dispatches_this_session'],
            'evaluation_seconds_this_session': runtime['evaluation_seconds_this_session'],
            'checkpoint_seconds_this_session': runtime['checkpoint_seconds_this_session']}
        atomic_json(checkpoint / 'B601_ACTIVE_TIME.json', row)
        with (monitor / 'VALIDATION_METRICS.jsonl').open('a') as output:
            output.write(json.dumps(row, allow_nan=False) + '\n')
        if score < best_score:
            best_score = score
            preserved = monitor / 'best_candidates' / f'step_{step:09d}'
            preserved.parent.mkdir(parents=True, exist_ok=True)
            if not preserved.exists():
                shutil.copytree(checkpoint, preserved, copy_function=os.link)
            manifest = {str(path.relative_to(preserved)): {'bytes': path.stat().st_size}
                        for path in preserved.rglob('*') if path.is_file()}
            atomic_json(preserved / 'B601_FILE_SIZES.json', manifest)
            atomic_json(best_index, {**row, 'preserved_checkpoint': str(preserved),
                        'selection_metric': 'fixed-80-observation validation flow loss'})
        atomic_json(monitor / 'LATEST.json', row)
        logging.info('Validation at update %d: %.7f; active %.3fh / %.3fh; best %.7f',
                     row['optimizer_step'], score, row['active_seconds']/3600, target_seconds/3600, best_score)
        if row['active_seconds'] >= target_seconds:
            atomic_json(monitor / 'LONGRUN_COMPLETE.json', row)
            raise LongRunComplete()

    def restore_timed(manager, state, loader, step=None):
        # Bind elapsed time to the checkpoint actually committed and restored.
        committed = manager.latest_step() if step is None else step
        runtime['manager'] = manager
        restored = original_restore(manager, state, loader, step=committed)
        progress = root / str(committed) / 'B601_ACTIVE_TIME.json'
        if progress.exists():
            receipt = json.loads(progress.read_text())
            if receipt['optimizer_step'] != int(jax.device_get(restored.step)):
                raise ValueError('Elapsed-time receipt does not match restored optimizer step')
            runtime['previous_active_seconds'] = receipt['active_seconds']
            if receipt.get('validation_status') != 'passed':
                score, duration = evaluate(restored)
                atomic_json(progress, {**receipt, 'validation_status': 'passed',
                    'validation_flow_loss': score, 'revalidated_at': timestamp(), 'revalidation_seconds': duration})
        atomic_json(monitor / f'RESTORE_{os.getpid()}.json', {'recorded_at': timestamp(),
            'committed_step_label': committed, 'optimizer_step': int(jax.device_get(restored.step)),
            'previous_active_seconds': runtime['previous_active_seconds']})
        return restored

    globals_dict['init_train_state'] = initialized
    globals_dict['init_wandb'] = initialized_wandb
    parameters = {'recorded_at': timestamp(), 'arguments': vars(args),
        'model': dataclasses.asdict(config.model), 'lr_schedule': dataclasses.asdict(config.lr_schedule),
        'optimizer': dataclasses.asdict(config.optimizer), 'global_batch_size': config.batch_size,
        'fsdp_devices': config.fsdp_devices, 'max_steps': config.num_train_steps,
        'save_interval': config.save_interval, 'target_active_seconds': target_seconds,
        'split_manifest_sha256': hashlib.sha256((prepared / 'manifest.json').read_bytes()).hexdigest(),
        'norm_stats_sha256': hashlib.sha256((prepared / 'assets' / TRAIN_ID / 'norm_stats.json').read_bytes()).hexdigest(),
        'resume_limitation': 'optimizer/model/step restored; upstream DataLoader position is not restored'}
    initial_parameters = monitor / 'RUN_PARAMETERS.json'
    if initial_parameters.exists():
        initial = json.loads(initial_parameters.read_text())
        for key in ('model', 'lr_schedule', 'optimizer', 'global_batch_size', 'fsdp_devices',
                    'split_manifest_sha256', 'norm_stats_sha256'):
            if initial[key] != parameters[key]:
                raise ValueError(f'Resume would change recorded training protocol: {key}')
    else:
        atomic_json(initial_parameters, parameters)
    atomic_json(monitor / 'sessions' / f'{time.time_ns()}_{os.getpid()}.json', parameters)
    try:
        with patch.object(sharding, 'set_mesh', timed_mesh), patch.object(checkpoints, 'save_state', save_evaluate), \
                patch.object(checkpoints, 'restore_state', restore_timed), \
                patch.object(wandb, 'log', log_metrics):
            train_main(config)
        if elapsed() < target_seconds:
            raise RuntimeError('Maximum step budget exhausted before required active wall time; resume with same LR schedule')
    except LongRunComplete:
        logging.info('Long run completed after a committed, evaluated checkpoint')
    finally:
        globals_dict['init_train_state'] = original_init
        globals_dict['init_wandb'] = original_init_wandb
        wandb.log = log_delegate['function']
        if runtime['manager'] is not None:
            runtime['manager'].wait_until_finished()
            runtime['manager'].close()
