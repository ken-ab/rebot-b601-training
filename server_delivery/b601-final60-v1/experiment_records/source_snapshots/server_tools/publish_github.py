"""Publish accepted code/evidence on an isolated branch and create a draft PR."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from prepare_publication import scan

ROOT = Path(__file__).resolve().parent
GH = ROOT.parent / 'github-zzk'
CHECKOUT = ROOT / 'PUBLICATION_PREP/github_checkout'
REPO = 'ken-ab/rebot-b601-training'
BRANCH = 'server/pi05-final60-results-20261110'
PREFIX = Path('server_delivery/b601-final60-v1')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_CONFIG_GLOBAL='/dev/null',
               GIT_CONFIG_NOSYSTEM='1')
    return subprocess.check_output(['git', '-C', str(CHECKOUT), *args], env=env, text=True).strip()


def github(*args):
    return subprocess.check_output([str(GH), *args], text=True).strip()


def main():
    assert read(ROOT/'INDEPENDENT_FINAL_AUDIT.json')['status']=='passed'
    assert read(ROOT/'FINAL_ACCEPTANCE_STATUS.json')['stage']=='complete_pending_root_completion_audit'
    prep = read(ROOT/'PUBLICATION_READY/PREPARATION_RECEIPT.json')
    assert prep['source_final_audit_sha256']==digest(ROOT/'INDEPENDENT_FINAL_AUDIT.json')
    assert prep['status']=='prepared_not_uploaded'
    code = ROOT/'PUBLICATION_READY/github_code'
    expected = prep['inventories']['github_code']
    manifest = code/'PUBLICATION_MANIFEST.json'
    assert digest(manifest)==expected['manifest_sha256']
    inventory = read(manifest)
    assert len(inventory)==expected['files']
    assert sum(row['bytes'] for row in inventory.values())==expected['bytes']
    for relative, row in inventory.items():
        path = (code/relative).resolve()
        assert path.is_relative_to(code.resolve()) and path.is_file()
        assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
        assert row['bytes'] < 95*1024*1024
    assert prep['credential_scan']['github_code']['findings']==[]
    assert not scan(code)['findings']
    assert github('api','user','--jq','.login')=='ken-ab'
    access = json.loads(github('api','repos/'+REPO))
    assert access['private'] is True and access['permissions']['push'] is True
    assert git('remote','get-url','origin')=='https://github.com/'+REPO+'.git'
    assert git('branch','--show-current')==BRANCH
    assert git('status','--porcelain')==''
    checkout_receipt = read(ROOT/'PUBLICATION_PREP/GITHUB_CHECKOUT_RECEIPT.json')
    assert git('rev-parse','HEAD')==checkout_receipt['base_revision']
    destination = CHECKOUT/PREFIX
    assert not destination.exists()
    shutil.copytree(code,destination)
    hf_path = ROOT/'HF_PUBLICATION_RECEIPT.json'
    if hf_path.exists():
        hf = read(hf_path)
        assert hf['status']=='published_and_verified' and hf['private'] is True
        download = {'status':'published_and_verified','repository':hf['repository'],
                    'revision':hf['revision'],'url':hf['url'],'private':True,
                    'publication_manifest_sha256':hf['publication_manifest_sha256']}
    else:
        delivery = read(ROOT/'DELIVERY_RECEIPT.json')
        download = {'status':'model_saved_on_server_huggingface_publication_pending',
                    'server_archive':delivery['archive'],
                    'server_archive_sha256':delivery['archive_sha256'],
                    'server_windows_client_zip':delivery['windows_client_archive'],
                    'note':'No Hugging Face upload or target URL is claimed until its remote verification receipt exists.'}
    (destination/'MODEL_DOWNLOAD.json').write_text(json.dumps(download,indent=2)+'\n')
    readme = CHECKOUT/'README.md'
    readme.write_text(readme.read_text()+'''

## 本服务器 final-60 微调结果与部署交接

本次 π₀.₅ 长跑、完整离线评估、实际硬件、方法与部署检查见
[服务器交付报告](server_delivery/b601-final60-v1/docs/server_final60/FINAL_REPORT_CN.md)。
权重下载与发布状态见 [MODEL_DOWNLOAD.json](server_delivery/b601-final60-v1/MODEL_DOWNLOAD.json)。
软件和离线验收已在报告中逐项区分；Windows/WSL 与现场机械臂闭环验证仍需在目标机执行。
''')
    assert not scan(destination)['findings']
    git('add','--','README.md',str(PREFIX))
    changed = git('diff','--cached','--name-only').splitlines()
    assert changed and all(name=='README.md' or name.startswith(str(PREFIX)+'/') for name in changed)
    git('diff','--cached','--check')
    review = {'at':datetime.datetime.now().astimezone().isoformat(),
              'files_staged':changed,'code_manifest_sha256':digest(manifest),
              'source_final_audit_sha256':prep['source_final_audit_sha256'],
              'model_download':download}
    (ROOT/'PUBLICATION_PREP/GITHUB_STAGED_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
    git('commit','-m','Document B601 final-60 pi05 training, evaluation and deployment')
    revision = git('rev-parse','HEAD')
    git('push','--set-upstream','origin',BRANCH)
    remote = git('ls-remote','--heads','origin','refs/heads/'+BRANCH).split()[0]
    assert remote==revision
    title = 'B601 final-60: server pi05 results and deployment handoff'
    body = '''This publishes the completed final-60 π₀.₅ training run, independently recomputed offline evaluation, hardware/method records and deployment sources in a dedicated delivery folder.

Validation includes the actual 12.25-hour training gate, complete model/optimizer readback, strict inference goldens, network/direct inference agreement and archive/client hashes. The report distinguishes these server checks from pending Windows/WSL and physical robot acceptance. Model hosting status is explicit in MODEL_DOWNLOAD.json; no checkpoint weights or credentials are committed here.
'''
    request = {'title':title,'head':BRANCH,'base':'main','body':body,'draft':True}
    request_path = ROOT/'PUBLICATION_PREP/GITHUB_PR_REQUEST.json'
    request_path.write_text(json.dumps(request,indent=2)+'\n')
    pull = json.loads(github('api','--method','POST','repos/'+REPO+'/pulls','--input',str(request_path)))
    assert pull['head']['sha']==revision and pull['draft'] is True
    result = {'status':'published_branch_and_draft_pr_verified',
              'at':datetime.datetime.now().astimezone().isoformat(), 'repository':REPO,
              'branch':BRANCH,'revision':revision,'pull_request':pull['html_url'],
              'code_manifest_sha256':digest(manifest),'main_not_modified':True,
              'training_checkout_not_modified':True,'model_download':download}
    (ROOT/'GITHUB_PUBLICATION_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
