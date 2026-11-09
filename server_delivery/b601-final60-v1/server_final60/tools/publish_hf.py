"""Publish an accepted, scanned model snapshot to the user's confirmed private repo."""
import hashlib
import json
from pathlib import Path
import re
import sys

from huggingface_hub import HfApi, hf_hub_download
from prepare_publication import scan

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(8*1024*1024):
            value.update(block)
    return value.hexdigest()


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def main():
    target = read(ROOT/'PUBLICATION_TARGET.json')
    assert target.get('repository_confirmed_by_user') is True, 'Need the actual repository URL from the user'
    repo = target.get('huggingface_repository')
    owner = target.get('huggingface_expected_account')
    assert repo and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo)
    assert owner and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', owner)
    profile = Path(target['project_hf_home']).resolve()
    assert profile == Path('/home/yujiegao/.cache/huggingface-zzk')/owner
    token_file = profile/'token'
    assert token_file.is_file(), 'Project upload authorization is still missing'
    token = token_file.read_text().strip()
    assert token, 'Project token is empty; do not fall back to a default identity'
    try:
        api = HfApi(endpoint='https://huggingface.co', token=token)
        user = api.whoami()
        assert user['name'] == owner, 'Project token belongs to a different account'
        prep = read(ROOT/'PUBLICATION_READY/PREPARATION_RECEIPT.json')
        assert prep['status'] == 'prepared_not_uploaded'
        assert prep['source_final_audit_sha256'] == digest(ROOT/'INDEPENDENT_FINAL_AUDIT.json')
        model = ROOT/'PUBLICATION_READY/huggingface_model'
        prepared = prep['inventories']['huggingface_model']
        assert digest(model/'PUBLICATION_MANIFEST.json') == prepared['manifest_sha256'], 'Publication manifest changed after preparation'
        assert prep['credential_scan']['huggingface_model']['findings'] == []
        inventory = read(model/'PUBLICATION_MANIFEST.json')
        assert len(inventory) == prepared['files']
        assert sum(row['bytes'] for row in inventory.values()) == prepared['bytes']
        for relative, expected in inventory.items():
            path = (model/relative).resolve()
            assert path.is_relative_to(model.resolve()) and path.is_file(), relative
            assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], relative
        assert not scan(model)['findings'], 'Credential-pattern scan rejected the current payload'
        info = api.repo_info(repo, repo_type='model', files_metadata=True)
        assert info.private is True, 'Target repository must be private before uploading this delivery'
        expected_paths = list(inventory)+['PUBLICATION_MANIFEST.json']
        existing = {item.rfilename for item in info.siblings}
        assert existing <= set(expected_paths)|{'.gitattributes'}, 'Unexpected existing files require review; no deletion or forced replacement'
        revision = info.sha
        # Bounded commits avoid an unbounded retry loop and keep progress reviewable.
        ordered = sorted(inventory)
        ordered.append('PUBLICATION_MANIFEST.json')
        commits = []
        for first in range(0,len(ordered),40):
            batch = ordered[first:first+40]
            print(json.dumps({'stage':'upload_batch','repository':repo,'first':first,
                              'files':len(batch),'total_files':len(ordered)}),flush=True)
            commit = api.upload_folder(repo_id=repo, repo_type='model', folder_path=str(model),
                allow_patterns=batch, ignore_patterns=['.cache/**'], parent_commit=revision,
                commit_message=f'B601 final60 accepted release files {first+1}-{first+len(batch)}')
            revision = commit.oid
            commits.append(revision)
        remote = api.repo_info(repo,repo_type='model',revision=revision,files_metadata=True)
        assert remote.private is True and remote.sha == revision
        siblings = {item.rfilename:item for item in remote.siblings}
        verified = []
        all_expected = {**inventory,'PUBLICATION_MANIFEST.json':{
            'bytes':(model/'PUBLICATION_MANIFEST.json').stat().st_size,
            'sha256':digest(model/'PUBLICATION_MANIFEST.json')}}
        for relative, expected in all_expected.items():
            item = siblings[relative]
            lfs = field(item,'lfs')
            remote_sha = field(lfs,'sha256')
            remote_size = field(item,'size',None) or field(lfs,'size',None)
            if remote_size is not None:
                assert remote_size == expected['bytes'], relative
            if remote_sha:
                assert remote_sha == expected['sha256'],relative
                method = 'remote_content_addressed_sha256'
            else:
                path = hf_hub_download(repo,filename=relative,repo_type='model',revision=revision,
                    cache_dir=str(profile/'verification-cache'),token=token,endpoint='https://huggingface.co')
                assert digest(path) == expected['sha256'],relative
                method = 'downloaded_and_sha256_checked'
            verified.append({'file':relative,'bytes':expected['bytes'],'method':method})
        result = {'status':'published_and_verified','repository':repo,'revision':revision,
            'url':f'https://huggingface.co/{repo}/tree/{revision}','private':True,
            'account':owner,'commits':commits,'files_verified':verified,
            'publication_manifest_sha256':digest(model/'PUBLICATION_MANIFEST.json'),
            'real_robot_or_windows_execution':False}
        (ROOT/'HF_PUBLICATION_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
        target.update({'upload_executed':True,'verified_model_revision':revision})
        (ROOT/'PUBLICATION_TARGET.json').write_text(json.dumps(target,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='files_verified'}),flush=True)
    except Exception as error:
        # Preserve useful error information without echoing credentials.
        message = str(error).replace(token,'[REDACTED]')
        message = re.sub(r'\bhf_[A-Za-z0-9]{20,}\b','[REDACTED]',message)
        print(json.dumps({'status':'failed','error_type':type(error).__name__,'message':message}),flush=True)
        raise SystemExit(1) from None


if __name__=='__main__':
    main()
