"""Audit actual long-run completion, models, evaluations and archive contents."""
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
KIT = ROOT.parent / 'rebot-b601-training/openpi_b601'
MONITOR = KIT / 'runs/pi05_b601_lora/b601-final60-v1_monitor'
BUNDLE = ROOT / 'B601_PI05_DEPLOYMENT'


def read(path):
    return json.loads(Path(path).read_text())


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stage_result(label):
    receipt = read(ROOT / 'receipts' / (label + '.json'))
    assert receipt['status'] == 'passed' and receipt['returncode'] == 0, label
    return receipt


def last_json(log):
    for line in reversed(Path(log).read_text().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    raise ValueError('No JSON result in ' + str(log))


completion = read(MONITOR / 'LONGRUN_COMPLETE.json')
session = read(MONITOR / 'SESSION_START.json')
assert completion['active_seconds'] >= completion['target_active_seconds'] >= 44100
assert completion['optimizer_step'] == completion['step_label'] + 1
assert completion == read(MONITOR / 'LATEST.json')
assert completion == read(Path(completion['checkpoint']) / 'B601_ACTIVE_TIME.json')
wall_delta = (datetime.datetime.fromisoformat(completion['at']) -
              datetime.datetime.fromisoformat(session['first_dispatch_completed_at'])).total_seconds()
assert abs(session['previous_active_seconds'] + wall_delta - completion['active_seconds']) < 30
assert read(ROOT / 'PIPELINE_STATUS.json')['stage'] == 'complete'
stage_result('formal-longrun-v1')
stage_result('deployment-export-v1')
summary = read(BUNDLE / 'TRAINING_SUMMARY.json')
selection = read(BUNDLE / 'DEPLOYMENT_SELECTION.json')
assert summary['training'] == completion
assert summary['selected'] == selection['primary']
receipt = read(ROOT / 'DELIVERY_RECEIPT.json')
assert receipt['selected'] == summary['selected']
assert receipt['delivered_model_labels'] == list(selection['delivered_models'])

canonical = np.load(ROOT / 'evaluation/checkpoint10000/predictions.npz')
evaluations = {}
directories = [(name, BUNDLE / 'evaluation' / name) for name in ('base', 'best', 'last')]
directories += [('candidate_' + row['label'], BUNDLE / 'evaluation/candidates' / row['label'])
                for row in selection['candidates']]
for label, directory in directories:
    metrics = read(directory / 'METRICS.json')
    data = np.load(directory / 'predictions.npz')
    assert metrics['full_protocol'] and metrics['evaluated_action_rows'] == 5118
    assert (metrics['seed'], metrics['stride'], metrics['denoise_steps']) == (20261109, 25, 10)
    assert metrics['inference_xla_flags'] == '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true'
    assert (metrics['jax_version'], metrics['jaxlib_version']) == ('0.5.3', '0.5.3')
    assert data['prediction'].shape == (211, 25, 7) and np.isfinite(data['prediction']).all()
    for key in ('target', 'valid_mask', 'episode_source_frame'):
        np.testing.assert_array_equal(data[key], canonical[key])
    errors = (data['prediction'].astype(np.float64) - data['target'])[data['valid_mask']]
    assert len(errors) == 5118
    mae, rmse = np.abs(errors).mean(0), np.sqrt(np.square(errors).mean(0))
    np.testing.assert_allclose(mae, metrics['per_axis_mae'], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(rmse, metrics['per_axis_rmse'], atol=2e-5, rtol=2e-5)
    macro = []
    for episode in np.unique(data['episode_source_frame'][:, 0]):
        rows = data['episode_source_frame'][:, 0] == episode
        error = (data['prediction'][rows].astype(np.float64) - data['target'][rows])[data['valid_mask'][rows]]
        macro.append(np.abs(error).mean(0))
    assert len(macro) == 10
    np.testing.assert_allclose(np.mean(macro, 0), metrics['macro_episode_per_axis_mae'], atol=2e-5, rtol=2e-5)
    assert metrics['base_model'] == (label == 'base')
    evaluations[label] = {'mean_joint_1to6_mae': float(mae[:6].mean()),
                          'gripper_mae': float(mae[6]), 'per_axis_mae': mae.tolist(),
                          'per_axis_rmse': rmse.tolist()}

model_checks = {}
checkpoints = {completion['optimizer_step']: Path(completion['checkpoint'])}
for label, model in selection['delivered_models'].items():
    checkpoints[model['optimizer_step']] = Path(model['checkpoint'])
    test_label = ('deployment-inference-readback-v1' if label == 'primary' else
                  'deployment-inference-' + label + '-readback-v1')
    check = last_json(stage_result(test_label)['log'])
    assert check['model'] == label and check['optimizer_step'] == model['optimizer_step']
    assert check['golden_matches'] and check['all_finite'] and check['shape'] == [25, 7]
    assert (check['atol'], check['rtol']) == (1e-4, 1e-5)
    norm = read(BUNDLE / model['relative_checkpoint'] / 'assets/rebot-b601-train/norm_stats.json')
    canonical_norm = hashlib.sha256(json.dumps(norm, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert canonical_norm == selection['canonical_norm_stats_sha256']
    model_checks[label] = check
for step, checkpoint in checkpoints.items():
    stage_result('final-params-' + str(step) + '-readback-v1')
    check = read(ROOT / 'diagnostics' / ('final_params_' + str(step) + '.json'))
    assert check['checkpoint'] == str(checkpoint.resolve()) and check['all_parameters_finite']
    assert check['parameter_arrays'] == 71 and check['lora_arrays'] == 20
stage_result('final-optimizer-readback-v1')
optimizer = read(ROOT / 'diagnostics/final_optimizer.json')
assert optimizer['status'] == 'passed' and optimizer['restored_step'] == completion['optimizer_step']
assert Path(optimizer['checkpoint']).resolve() == Path(completion['checkpoint']).resolve()
stage_result('websocket-final-v1')
websocket = read(ROOT / 'diagnostics/websocket_final_v1/REPORT.json')
assert websocket['status'] == 'passed' and websocket['all_finite']
assert websocket['shape'] == [3, 25, 7] and websocket['own_test_server_stopped']
assert websocket['client_preprocessed_rgb224_uint8'] and websocket['wire_image_bytes'] == 301056
assert websocket['provided_client_smoke_cli_passed']
assert Path(websocket['bundle']).resolve() == BUNDLE.resolve()
assert not websocket['robot_connected_or_actuated']
compact_proof = read(BUNDLE / 'receipts/CLIENT_PREPROCESS_EQUIVALENCE.json')
assert compact_proof['status'] == 'passed' and compact_proof['observations'] == 211
assert compact_proof['effective_inputs_elementwise_identical'] and compact_proof['wire_roundtrip_checked']
assert digest_file(BUNDLE / 'client/b601_observation.py') == compact_proof['client_source_sha256']
assert digest_file(BUNDLE / 'client/openpi_client_src/openpi_client/image_tools.py') == compact_proof['upstream_image_tools_sha256']
transport_proof = read(BUNDLE / 'receipts/CLIENT_TRANSPORT_ACCEPTANCE.json')
assert transport_proof['status'] == 'passed' and transport_proof['connection_discarded_after_timeout']
assert transport_proof['second_request_rejected_locally'] and transport_proof['automatic_inference_retries'] == 0
assert digest_file(BUNDLE / 'client/b601_websocket_client.py') == transport_proof['source_sha256']
guard_proof = read(BUNDLE / 'receipts/ACTION_GUARD_ACCEPTANCE.json')
assert guard_proof['status'] == 'passed' and guard_proof['case_count'] >= 36
assert guard_proof['default_template_cannot_arm'] and not guard_proof['real_hardware_tested']
assert not guard_proof['guard_can_stop_already_moving_hardware']
assert digest_file(BUNDLE / 'client/action_guard.py') == guard_proof['guard_source_sha256']
assert digest_file(BUNDLE / 'client/robot_deployment_template.json') == guard_proof['template_sha256']
robot_config = read(BUNDLE / 'client/robot_deployment_template.json')
assert all(value is None for values in robot_config['limits'].values() for value in values)
assert all(value is False for value in robot_config['local_checks'].values())
assert robot_config['calibration_reference'] is None and robot_config['max_observation_age_seconds'] is None
stage_result('final-low-memory-profile-v1')
memory = read(ROOT / 'diagnostics/inference_platform_final_v1/REPORT.json')
assert memory['status'] == 'passed' and memory['delivery_scope'] == 'final'
assert memory['selected_training_model'] == summary['selected'] and memory['allocator'] == 'platform'
assert Path(memory['bundle']).resolve() == BUNDLE.resolve()

manifest = read(BUNDLE / 'MANIFEST.json')
notice_text = (BUNDLE / 'NOTICE').read_text()
assert 'ai.google.dev/gemma/terms' in notice_text and 'MODIFICATION NOTICE' in notice_text
assert (BUNDLE / 'LICENSE').is_file() and (BUNDLE / 'LICENSE_GEMMA.txt').is_file()
for model in selection['delivered_models'].values():
    modification = (BUNDLE / model['relative_checkpoint'] / 'B601_MODIFICATIONS.md').read_text()
    assert str(model['optimizer_step']) in modification and 'Section 3.2' in modification
for relative, expected in manifest.items():
    path = (BUNDLE / relative).resolve()
    assert path.is_relative_to(BUNDLE.resolve()) and path.is_file(), relative
    assert path.stat().st_size == expected['bytes'] and digest_file(path) == expected['sha256'], relative
assert len(manifest) == receipt['manifest_files']
archive = Path(receipt['archive'])
assert archive.stat().st_size == receipt['archive_bytes']
assert digest_file(archive) == receipt['archive_sha256']
archive_files, archive_links, seen = 0, 0, {}
with tarfile.open(archive, 'r|gz') as stream:
    for member in stream:
        if member.isdir():
            continue
        relative = str(PurePosixPath(member.name).relative_to(BUNDLE.name))
        assert '..' not in PurePosixPath(relative).parts
        expected = ({'bytes': (BUNDLE / 'MANIFEST.json').stat().st_size,
                     'sha256': digest_file(BUNDLE / 'MANIFEST.json')}
                    if relative == 'MANIFEST.json' else manifest[relative])
        if member.islnk():
            observed = seen[member.linkname]
            archive_links += 1
        else:
            assert member.isfile(), member.name
            digest, size = hashlib.sha256(), 0
            source = stream.extractfile(member)
            while block := source.read(8 * 1024 * 1024):
                digest.update(block)
                size += len(block)
            observed = {'bytes': size, 'sha256': digest.hexdigest()}
            archive_files += 1
        assert observed == expected, relative
        assert member.name not in seen
        seen[member.name] = observed
assert len(seen) == len(manifest) + 1
client_zip = Path(receipt['windows_client_archive'])
assert client_zip.stat().st_size == receipt['windows_client_bytes']
assert digest_file(client_zip) == receipt['windows_client_sha256']
with zipfile.ZipFile(client_zip) as client_archive:
    client_manifest = json.loads(client_archive.read('B601_PI05_WINDOWS_CLIENT/MANIFEST.json'))
    assert len(client_manifest) == receipt['windows_client_manifest_files']
    assert len(client_archive.namelist()) == len(client_manifest) + 1
    for relative, expected in client_manifest.items():
        assert expected == manifest[relative]
        content = client_archive.read('B601_PI05_WINDOWS_CLIENT/' + relative)
        assert len(content) == expected['bytes']
        assert hashlib.sha256(content).hexdigest() == expected['sha256']
report = {'status': 'passed', 'at': datetime.datetime.now().astimezone().isoformat(),
          'completion': completion, 'wall_clock_crosscheck_seconds': wall_delta,
          'selected': summary['selected'], 'evaluations_recomputed_float64': evaluations,
          'model_checks': model_checks, 'final_optimizer': optimizer,
          'manifest_files': len(manifest), 'archive_regular_files': archive_files,
          'archive_hardlinks_verified': archive_links, 'archive_sha256': receipt['archive_sha256'],
          'windows_client_zip_files_verified': len(client_manifest),
          'windows_client_zip_sha256': receipt['windows_client_sha256'],
          'final_websocket': websocket, 'final_low_memory': memory,
          'actual_windows_and_robot_execution': False,
          'scope': 'Actual12.25h server training, offline evaluation and portableparameter delivery; not physicalrobot efficacy'}
output = ROOT / 'INDEPENDENT_FINAL_AUDIT.json'
assert not output.exists(), output
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'output': str(output), 'completion': completion,
                  'selected': summary['selected'], 'archive_sha256': receipt['archive_sha256']}), flush=True)
