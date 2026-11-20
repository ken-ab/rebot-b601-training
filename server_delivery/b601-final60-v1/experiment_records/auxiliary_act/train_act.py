"""A separate fixed ACT arm for a Windows8GB deployment candidate."""
import argparse
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from act_common import CACHE, ROOT, CachedDataset, evaluate, make_policy, move_batch


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    output = Path('/data2/yujiegao/b601-act-final60/runs') / args.name
    output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    policy = make_policy(pretrained=True)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=1e-4)
    parameters = {'at': datetime.datetime.now().astimezone().isoformat(), 'arguments': vars(args),
        'config': dataclasses.asdict(policy.config), 'trainable_parameters': sum(p.numel() for p in policy.parameters() if p.requires_grad),
        'network_parameters': sum(p.numel() for p in policy.model.parameters()), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'data_cache_acceptance_sha256': hashlib.sha256((CACHE/'CACHE_ACCEPTANCE.json').read_bytes()).hexdigest(),
        'optimizer': {'name': 'AdamW', 'lr': 1e-5, 'backbone_lr': 1e-5, 'weight_decay': 1e-4,
                      'betas': [.9, .999], 'eps': 1e-8, 'clip_grad_norm': 1.0},
        'selection': 'Minimum full5118-row deployment normalizedL1; target-freezero-latent, no VAEposterior',
        'precision': 'float32,TF32disabled, noquantization', 'seed': 42,
        'scope': 'IndependentACTarm; originalpi05recipe/run preserved'}
    atomic_json(output/'RUN_PARAMETERS.json', parameters)
    baseline = evaluate(policy)
    atomic_json(output/'BASELINE.json', baseline)
    generator = torch.Generator().manual_seed(42)
    loader = torch.utils.data.DataLoader(CachedDataset('train'), batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0, pin_memory=True, generator=generator)
    iterator = iter(loader)
    best = float('inf')
    started = time.monotonic()
    interval = []
    for step in range(1, args.steps+1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, 'cuda')
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss, components = policy(batch)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'NonfiniteACTtrainingloss at{step}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'loss': loss.item(), 'grad_norm_before_clipping': grad.item(),
               **components, 'elapsed_seconds': time.monotonic()-started}
        assert all(np.isfinite(value) for key,value in row.items() if key != 'step')
        interval.append(row)
        if args.steps == 20 or step % 100 == 0:
            mean = {key: float(np.mean([r[key] for r in interval])) for key in row if key != 'step'}
            record = {'step': step, 'interval_updates': len(interval), **mean}
            with (output/'TRAIN_METRICS.jsonl').open('a') as target:
                target.write(json.dumps(record, allow_nan=False)+'\n')
            print(json.dumps(record), flush=True)
            interval.clear()
        if step % 1000 == 0 or step == args.steps:
            checkpoint = output / f'step_{step:06d}'
            checkpoint.mkdir()
            policy.save_pretrained(checkpoint/'policy')
            torch.save({'step': step, 'optimizer': optimizer.state_dict(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, checkpoint/'training_state.pt')
            metrics = evaluate(policy, checkpoint/'evaluation')
            receipt = {'step': step, 'checkpoint': str(checkpoint), 'elapsed_seconds': time.monotonic()-started,
                'normalized_valid_l1': metrics['normalized_valid_l1'],
                'mean_joint_1to6_mae': metrics['mean_joint_1to6_mae'], 'gripper_mae': metrics['gripper_mae'],
                'cuda_max_memory_allocated_mib': torch.cuda.max_memory_allocated()/1048576,
                'cuda_max_memory_reserved_mib': torch.cuda.max_memory_reserved()/1048576}
            atomic_json(output/'LATEST.json', receipt)
            if receipt['normalized_valid_l1'] < best:
                best = receipt['normalized_valid_l1']
                atomic_json(output/'BEST.json', receipt)
            print(json.dumps({'validation': receipt}), flush=True)
    completion = {**receipt, 'status': 'complete', 'at': datetime.datetime.now().astimezone().isoformat()}
    atomic_json(output/'COMPLETE.json', completion)
    print(json.dumps(completion), flush=True)


if __name__ == '__main__':
    main()
