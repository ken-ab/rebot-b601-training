"""Independently read the actual ACT archive and recompute delivery evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'B601_ACT_DEPLOYMENT'
receipt = json.loads((ROOT / 'DELIVERY_RECEIPT.json').read_text())
run = Path(receipt['selected']['checkpoint']).parent
assert receipt['status'] == 'complete'
assert receipt['selected'] == json.loads((run / 'BEST.json').read_text())
assert receipt['training_completion'] == json.loads((run / 'COMPLETE.json').read_text())
assert receipt['training_completion']['step'] == 20000
archive = Path(receipt['archive'])
assert archive.stat().st_size == receipt['archive_bytes']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['archive_sha256']
manifest = json.loads((BUNDLE / 'MANIFEST.json').read_text())
for relative, expected in manifest.items():
    path = BUNDLE / relative
    assert path.stat().st_size == expected['bytes'], relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], relative
members = 0
with tarfile.open(archive, 'r:gz') as stream:
    for member in stream:
        if not member.isfile():
            continue
        relative = str(Path(member.name).relative_to(BUNDLE.name))
        raw = stream.extractfile(member).read()
        if relative == 'MANIFEST.json':
            path = BUNDLE / relative
            expected = {'bytes': path.stat().st_size,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        else:
            expected = manifest[relative]
        assert len(raw) == expected['bytes'], relative
        assert hashlib.sha256(raw).hexdigest() == expected['sha256'], relative
        members += 1
assert members == len(manifest) + 1
prediction = np.load(BUNDLE / 'evaluation/predictions.npz')
errors = (prediction['prediction'].astype(np.float64) -
          prediction['target'])[prediction['valid_mask']]
metrics = json.loads((BUNDLE / 'evaluation/METRICS.json').read_text())
assert len(errors) == 5118
np.testing.assert_allclose(np.abs(errors).mean(0), metrics['per_axis_mae'],
                           atol=1e-9, rtol=1e-9)
np.testing.assert_allclose(np.sqrt((errors**2).mean(0)), metrics['per_axis_rmse'],
                           atol=1e-9, rtol=1e-9)
selftests = {}
for device in ('cpu', 'cuda'):
    full = json.loads((ROOT / 'diagnostics' / ('runtime_final_' + device + '_v1') /
                       'REPORT.json').read_text())
    assert full['status'] == 'passed' and full['valid_action_rows'] == 5118
    check = json.loads((BUNDLE / 'verification' /
                       ('act-final-selftest-' + device + '-v1-result.json')).read_text())
    assert check['status'] == 'passed' and check['shape'] == [25, 7]
    selftests[device] = {'self_test': check, 'full_readback': full}
report = {'status': 'passed', 'at': datetime.datetime.now().astimezone().isoformat(),
          'archive_sha256': receipt['archive_sha256'], 'archive_bytes': receipt['archive_bytes'],
          'manifest_files': len(manifest), 'archive_regular_files': members,
          'selected_step': receipt['selected']['step'],
          'actual_final_step': receipt['training_completion']['step'],
          'all_archive_member_hashes_match_manifest': True,
          'full_metrics_recomputed_float64': True, 'selected_metrics': metrics,
          'selftests': selftests, 'actual_windows_and_robot_execution': False,
          'main_pi05_status': 'Separate12.25h objective still requires its own completion'}
(ROOT / 'INDEPENDENT_DELIVERY_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'archive_mib': archive.stat().st_size / 1048576,
                  'selected': receipt['selected'], 'selftests': selftests}, indent=2))
