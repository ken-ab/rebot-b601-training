"""Report configuration completeness; no connection to a model or robot."""
import argparse
import json
from pathlib import Path
import platform

from action_guard import configuration_issues

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path,
                    default=Path(__file__).resolve().parent/'robot_deployment_template.json')
args = parser.parse_args()
issues = configuration_issues(json.loads(args.config.read_text()))
print(json.dumps({'status': 'needs_local_setup' if issues else 'configuration_complete',
    'system': platform.system(), 'issues': issues, 'hardware_verified_by_this_script': False,
    'robot_commands_sent': False,
    'scope': 'Configuration/attestation check; field completion alone does not prove physical safety'}, ensure_ascii=False))
raise SystemExit(2 if issues else 0)
