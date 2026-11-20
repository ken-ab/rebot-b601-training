"""Download public pi05_base params with generation-pinned size/MD5 checks."""
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request

import google_crc32c

ROOT = Path('/data2/yujiegao/b601-pi05/models/pi05_base')
PREFIX = 'checkpoints/pi05_base/params/'
DEST = ROOT / 'params'
STAGE = ROOT / '.params.download'
LOCK = threading.Lock()
PROGRESS = {}
ROOT.mkdir(parents=True, exist_ok=True)
STAGE.mkdir(exist_ok=True)

def request_json(url):
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30))

items, page = [], ''
while True:
    query = {'prefix': PREFIX, 'fields': 'items(name,size,md5Hash,crc32c,generation),nextPageToken'}
    if page:
        query['pageToken'] = page
    data = request_json('https://storage.googleapis.com/storage/v1/b/openpi-assets/o?' + urllib.parse.urlencode(query))
    items.extend(data.get('items', []))
    page = data.get('nextPageToken', '')
    if not page:
        break
assert items and all(x['name'].startswith(PREFIX) and 'crc32c' in x for x in items)
(ROOT / 'OBJECT_INVENTORY.json').write_text(json.dumps(items, indent=2) + '\n')

def digest(path, algorithm):
    value = google_crc32c.Checksum() if algorithm == 'crc32c' else hashlib.new(algorithm)
    with path.open('rb') as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.digest()

def fetch(item):
    relative = item['name'][len(PREFIX):]
    assert relative and not Path(relative).is_absolute() and '..' not in Path(relative).parts
    target = STAGE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    expected = int(item['size'])
    expected_crc = base64.b64decode(item['crc32c'])
    expected_md5 = base64.b64decode(item['md5Hash']) if item.get('md5Hash') else None
    if (target.exists() and target.stat().st_size == expected and digest(target, 'crc32c') == expected_crc
            and (expected_md5 is None or digest(target, 'md5') == expected_md5)):
        return {'name': relative, 'bytes': expected, 'sha256': digest(target, 'sha256').hex(), 'reused': True}
    url = ('https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/' +
           urllib.parse.quote(item['name'], safe='') + '?alt=media&generation=' + item['generation'])
    for attempt in range(10):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > expected:
                part.rename(part.with_name(part.name + f'.invalid-{int(time.time())}'))
                offset = 0
            if expected == 0:
                part.touch(exist_ok=True)
            if offset < expected:
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                    mode = 'ab' if offset and response.status == 206 else 'wb'
                    with part.open(mode) as output:
                        while chunk := response.read(8 * 1024 * 1024):
                            output.write(chunk)
                            with LOCK:
                                PROGRESS[relative] = output.tell()
                                temporary = ROOT / 'DOWNLOAD_PROGRESS.tmp'
                                temporary.write_text(json.dumps(PROGRESS, indent=2) + '\n')
                                temporary.replace(ROOT / 'DOWNLOAD_PROGRESS.json')
            if part.stat().st_size != expected:
                raise ValueError(f'Length mismatch: {relative}')
            if digest(part, 'crc32c') != expected_crc or (expected_md5 is not None and digest(part, 'md5') != expected_md5):
                part.rename(part.with_name(part.name + f'.checksum-mismatch-{int(time.time())}'))
                raise ValueError(f'Checksum mismatch: {relative}')
            part.replace(target)
            result = {'name': relative, 'bytes': expected, 'sha256': digest(target, 'sha256').hex(),
                      'generation': item['generation'], 'crc32c_verified': True, 'md5_verified': expected_md5 is not None}
            print(json.dumps(result), flush=True)
            return result
        except Exception as exc:
            print(f'{relative}: attempt {attempt + 1}: {type(exc).__name__}', flush=True)
            if attempt == 9:
                raise
            time.sleep(min(2 ** attempt, 30))

if DEST.exists():
    raise SystemExit('Complete destination already exists; verify its receipt before reusing.')
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(fetch, items))
assert sum(row['bytes'] for row in results) == sum(int(row['size']) for row in items)
STAGE.rename(DEST)
(ROOT / 'VERIFIED_DOWNLOAD.json').write_text(json.dumps({'path': str(DEST), 'files': results,
    'total_bytes': sum(row['bytes'] for row in results), 'finished_unix': time.time()}, indent=2) + '\n')
print('VERIFIED BASE PARAMS:', DEST, flush=True)
