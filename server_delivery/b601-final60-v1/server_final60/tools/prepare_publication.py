"""Stage reviewed code/model files only after actual final acceptance; never uploads."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent/'rebot-b601-training'
BUNDLE = ROOT/'B601_PI05_DEPLOYMENT'
FORBIDDEN = {'.git', '.ssh', '.env', '.venv', '__pycache__', 'stored_tokens', 'token', 'credentials'}
TEXT_SUFFIXES = {'.py', '.sh', '.md', '.json', '.jsonl', '.txt', '.toml', '.yaml', '.yml', '.csv'}
PATTERNS = {
    'huggingface_token': re.compile(r'\bhf_[A-Za-z0-9]{20,}\b'),
    'github_token': re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b'),
    'github_fine_grained_token': re.compile(r'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    'private_key': re.compile(r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----'),
    'credential_url': re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'),
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(8*1024*1024):
            result.update(block)
    return result.hexdigest()


def copy_file(source, destination):
    source, destination = Path(source), Path(destination)
    assert not any(part in FORBIDDEN for part in destination.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_size > 32*1024*1024 and 'checkpoint' in source.parts:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def scan(folder):
    findings, text_files = [], 0
    for path in folder.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(folder)
        assert not any(part in FORBIDDEN for part in relative.parts)
        if path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith('LICENSE') or path.name == 'NOTICE':
            assert path.stat().st_size < 32*1024*1024, relative
            value = path.read_text(errors='strict')
            text_files += 1
            for name, pattern in PATTERNS.items():
                if pattern.search(value):
                    findings.append({'file': str(relative), 'pattern': name})
    return {'text_files_scanned': text_files, 'findings': findings,
            'scope': 'Known credential patterns and forbidden paths; no secret values are logged'}


def model_card(audit):
    completion, selected = audit['completion'], audit['selected']
    metrics = audit['evaluations_recomputed_float64']
    rows = '\n'.join(f"| {name} | {value['mean_joint_1to6_mae']:.6f} | {value['gripper_mae']:.6f} |"
                     for name,value in metrics.items())
    return f'''---
tags:
- robotics
- openpi
- jax
- pi05
- b601
language:
- zh
license: other
license_name: gemma-terms-of-use
license_link: https://ai.google.dev/gemma/terms
---
# B601-RS blue tissue pick-and-place — fine-tuned π₀.₅

This release contains the **actual fine-tuned OpenPI π₀.₅** for the fixed instruction “拿起蓝色餐巾纸，把它放进黑色框里”. It is a custom OpenPI JAX/Orbax deployment, with complete parameters, training-only normalization, tokenizer, B601 transforms and inference entrypoints. It is not an ACT conversion or a Transformers AutoModel checkpoint.

Training completed {completion['active_seconds']/3600:.4f} active hours and {completion['optimizer_step']} optimizer updates. The activity clock excludes setup/initial compilation and includes normal checkpoint/validation time. Final checkpoint label is {completion['step_label']}; primary deployment candidate is **{selected['label']}**, after {selected['optimizer_step']} updates. Selection is bounded and validation-reused, not an untouched test.

## Data and method

User-provided private final-60 dataset, with a fixed split of 50 training and 10 validation episodes: 28,729/5,118 frames, 25 fps, overhead and wrist RGB, seven-axis state/actions and a 25-step horizon. The raw dataset remains in its original private repository. One recorded observation is included for the deployment self-test; evaluation files also contain recorded states, action targets and predictions.

Base: `gs://openpi-assets/checkpoints/pi05_base`; OpenPI revision `215abfb217dbac7d5f1273282331b9b1866c0479`. JAX 0.5.3, global batch 4, FSDP across two GPUs, seed 42, 500 warmup steps, peak learning rate 2.5e-5, and floor 2.5e-6 after 10,000 steps. Native LoRA language/action branches and dense vision training follow the recorded freeze filter; this is why the release includes complete model parameters. Exact configuration and executed source history are under `receipts/` and `executed_training_source_versions/`.

Hardware: two RTX A4500 GPUs with 20 GB each on a server with two Xeon Gold 6530 processors and approximately 503 GiB of Linux-visible system memory. Separate 20-step real-data smokes passed finite loss/gradient and checkpoint readback before the formal run. Formal console losses are means over 100 updates, rounded to four decimals; they are not full-precision records of every step.

## Offline validation

211 windows cover all 5,118 valid actions in 10 validation episodes. Identical targets, masks, source frames and train-only normalization are used across candidates. Values below are independently recomputed in float64 in original absolute-adapter-input units; the gripper is reported separately from the six-joint mean.

| Model | Six-joint mean MAE | Gripper MAE |
| --- | ---: | ---: |
{rows}

Full per-axis, per-episode and macro MAE/RMSE metrics and predictions are in `evaluation/`. The pure base model was separately evaluated; the initial random-LoRA flow proxy is not substituted for that control. Offline action agreement is **not physical grasp success**.

## Deployment

- Server GPU inference with observation capture and USB execution on native Windows.
- Local Windows host with π₀.₅ GPU inference inside WSL2 and USB execution on Windows.

Both paths use the same fine-tuned parameters. See `PI05_INFERENCE_ARCHITECTURE_CN.md` and `WINDOWS_WSL2_DEPLOYMENT_CN.md`. The client sends two 224×224 RGB uint8 images and raw seven-axis feedback; output is 25×7 `absolute_adapter_input`. The original B601 adapter performs joint direction and gripper scale conversion.

The standalone Windows client ZIP is a separate delivery artifact; no model weights are needed on Windows for the server route. The included client performs recorded-sample RPC. Integration with the actual cameras, original USB adapter and local control loop still requires the user's machine.

## Checks and limitations

Actual server checks include complete parameter/optimizer readback, strict real-sample goldens, agreement between network and direct inference, pixel/wire equivalence on 211 observations, bounded transport with connection invalidation, 36 synthetic action-guard cases and complete archive/file hashes. See `INDEPENDENT_FINAL_AUDIT.json` and `DEPLOYMENT_READINESS_CN.md`.

The default robot configuration cannot arm. Actual hardware limits, calibration, stop behavior, cameras, Windows 4070/WSL operation, WAN latency, collision/contact/torque/position feedback and closed-loop success still require local validation. Disarming the software guard does not physically stop hardware already in motion. Templates and manual declarations are not physical test evidence. Do not infer hardware limits from demonstration minimum/maximum values.

## Licenses and versions

OpenPI code uses Apache License 2.0 (`LICENSE`). Gemma components and derivatives are governed by the accompanying `LICENSE_GEMMA.txt`, including its Section 3.2 use restrictions. Keep `NOTICE`, each checkpoint's `B601_MODIFICATIONS.md`, component notices and upstream source provenance. The full weights must not be described as merely Apache-licensed code. This card does not replace the copied upstream terms. Model files are version-bound by `MANIFEST.json`; this publication snapshot additionally has `PUBLICATION_MANIFEST.json`.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'PUBLICATION_READY')
    args=parser.parse_args()
    audit=read(ROOT/'INDEPENDENT_FINAL_AUDIT.json')
    assert audit['status']=='passed' and audit['completion']['active_seconds']>=44100
    assert read(ROOT/'FINAL_ACCEPTANCE_STATUS.json')['stage']=='complete_pending_root_completion_audit'
    assert not args.output.exists(),args.output
    pending=args.output.with_name(args.output.name+'.pending')
    pending.mkdir(parents=True,exist_ok=False,mode=0o700)
    model=pending/'huggingface_model';code=pending/'github_code'
    manifest=read(BUNDLE/'MANIFEST.json')
    for relative,expected in manifest.items():
        source=(BUNDLE/relative).resolve()
        assert source.is_relative_to(BUNDLE.resolve()) and source.is_file()
        assert source.stat().st_size==expected['bytes'] and digest(source)==expected['sha256']
        copy_file(source,model/relative)
    copy_file(BUNDLE/'MANIFEST.json',model/'MANIFEST.json')
    delivery=read(ROOT/'DELIVERY_RECEIPT.json')
    client_zip=Path(delivery['windows_client_archive'])
    assert client_zip.name=='B601_PI05_WINDOWS_CLIENT.zip'
    assert digest(client_zip)==audit['windows_client_zip_sha256']==delivery['windows_client_sha256']
    assert client_zip.stat().st_size==delivery['windows_client_bytes']
    copy_file(client_zip,model/client_zip.name)
    (model/'B601_PI05_WINDOWS_CLIENT.sha256').write_text(
        delivery['windows_client_sha256']+'  '+client_zip.name+'\n')
    for name in ('INDEPENDENT_FINAL_AUDIT.json','FINAL_REPORT_CN.md'):
        copy_file(ROOT/name,model/name)
    (model/'README.md').write_text(model_card(audit))
    tracked=subprocess.check_output(['git','-C',str(REPO),'ls-files','openpi_b601'],text=True).splitlines()
    extra=['openpi_b601/b601_policy.py','openpi_b601/evaluate_b601.py',
           'openpi_b601/serve_b601.py','openpi_b601/longrun_monitor.py']
    for relative in sorted(set(tracked+extra)):
        source=REPO/relative
        if source.is_file():
            assert source.resolve().is_relative_to(REPO.resolve()),relative
            assert source.stat().st_size<10*1024*1024,relative
            copy_file(source,code/relative)
    docs=['FINAL_REPORT_CN.md','DEPLOYMENT_READINESS_CN.md','PI05_INFERENCE_ARCHITECTURE_CN.md',
          'WINDOWS_WSL2_DEPLOYMENT_CN.md','PUBLICATION_PLAN_CN.md','INDEPENDENT_FINAL_AUDIT.json']
    for name in docs:copy_file(ROOT/name,code/'docs/server_final60'/name)
    for name in ('LICENSE','LICENSE_GEMMA.txt','NOTICE'):
        copy_file(BUNDLE/name,code/'server_final60/licenses'/name)
    for relative in manifest:
        if relative.startswith(('client/','evaluation/')):
            copy_file(BUNDLE/relative,code/'server_final60'/relative)
    for name in ('verify_client_preprocessing.py','verify_client_transport.py','verify_optimizer_checkpoint.py',
                 'deployment_selection.py','audit_final_delivery.py','export_delivery.py','finalize_acceptance.py',
                 'prepare_publication.py','publish_hf.py','publish_github.py'):
        copy_file(ROOT/name,code/'server_final60/tools'/name)
    copy_file(ROOT.parent/'B601_SERVER_ASSESSMENT_20261109/run_on_server.sh',
              code/'server_final60/tools/run_on_this_server.sh')
    (code/'server_final60/README.md').write_text('''# Server final-60 delivery

Primary model: fine-tuned π₀.₅. Read `../docs/server_final60/FINAL_REPORT_CN.md` and the deployment/readiness documents. The final publication's `../MODEL_DOWNLOAD.json` records actual weight hosting status and any verified download revision. Checkpoint weights and credential files do not belong in Git. Tools with absolute server paths document this run and require intentional path/device adaptation elsewhere. Physical Windows/USB validation remains pending.
''')
    scans={'huggingface_model':scan(model),'github_code':scan(code)}
    (pending/'CREDENTIAL_SCAN_REPORT.json').write_text(json.dumps(scans,indent=2)+'\n')
    assert not any(result['findings'] for result in scans.values()),'Credential-pattern findings; inspect redacted report before publishing'
    inventories={}
    for label,folder in [('huggingface_model',model),('github_code',code)]:
        inventory={str(p.relative_to(folder)):{'bytes':p.stat().st_size,'sha256':digest(p)}
                   for p in folder.rglob('*') if p.is_file()}
        (folder/'PUBLICATION_MANIFEST.json').write_text(json.dumps(inventory,indent=2)+'\n')
        inventories[label]={'files':len(inventory),'bytes':sum(v['bytes'] for v in inventory.values()),
                            'manifest_sha256':digest(folder/'PUBLICATION_MANIFEST.json')}
    receipt={'status':'prepared_not_uploaded','at':datetime.datetime.now().astimezone().isoformat(),
             'output':str(args.output),'source_final_audit_sha256':digest(ROOT/'INDEPENDENT_FINAL_AUDIT.json'),
             'inventories':inventories,'credential_scan':scans,'privacy_intended':'private',
             'real_hardware_verified':False}
    (pending/'PREPARATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    pending.replace(args.output)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
