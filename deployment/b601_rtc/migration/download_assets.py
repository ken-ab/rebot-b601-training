"""Cache B601 model, pinned sources and Linux cp311 wheels on any host OS.

Downloads files only. Does not install Linux packages into Windows.
Private model access uses the normal Hugging Face credential cache.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

MODEL = "dahjgfa/b601-pi05-final60"
MODEL_REVISION = "8c11eb8751d5cc6f688fa51db523da5bd8f2518a"
SOURCE_REVISIONS = {
    "openpi": ("Physical-Intelligence/openpi", "215abfb217dbac7d5f1273282331b9b1866c0479"),
    "lerobot": ("huggingface/lerobot", "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(url, path, expected_sha=None, expected_bytes=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and (expected_bytes is None or path.stat().st_size == expected_bytes):
        if expected_sha is None or sha256(path) == expected_sha:
            return path
    partial = path.with_name(path.name + ".part")
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "b601-portable-cache/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                append = offset > 0 and response.status == 206
                with partial.open("ab" if append else "wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            if expected_bytes is not None and partial.stat().st_size != expected_bytes:
                raise ValueError("Downloaded size differs from source manifest")
            if expected_sha is not None and sha256(partial) != expected_sha:
                partial.rename(partial.with_name(partial.name + f".bad-{time.time_ns()}"))
                raise ValueError("Downloaded SHA-256 differs from source manifest")
            partial.replace(path)
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def linux_tags():
    from packaging.tags import compatible_tags, cpython_tags
    # Ubuntu 22.04+ x86_64; do not accidentally fetch Windows wheels.
    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(35, 4, -1)]
    platforms += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    tags = list(cpython_tags((3, 11), abis=["cp311"], platforms=platforms))
    tags += list(compatible_tags((3, 11), interpreter="cp311", platforms=platforms))
    return {tag: rank for rank, tag in enumerate(tags)}


def wheel_record(name, version, ranks):
    from packaging.utils import parse_wheel_filename
    with urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
        release = json.load(response)
    candidates = []
    for file in release["urls"]:
        if file.get("yanked") or file["packagetype"] != "bdist_wheel":
            continue
        try:
            tags = parse_wheel_filename(file["filename"])[3]
        except ValueError:
            continue
        matching = [ranks[tag] for tag in tags if tag in ranks]
        if matching:
            candidates.append((min(matching), file))
    if candidates:
        file = min(candidates, key=lambda pair: pair[0])[1]
        kind = "wheel"
    else:
        file = next((f for f in release["urls"] if f["packagetype"] == "sdist"), None)
        if file is None:
            raise ValueError(f"No Linux cp311 wheel/source for {name}=={version}")
        kind = "sdist"
    return {"name": name, "version": version, "kind": kind, "filename": file["filename"],
            "url": file["url"], "sha256": file["digests"]["sha256"], "bytes": file["size"]}


def cache_wheels(versions, destination):
    ranks = linux_tags()
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        tasks = {pool.submit(wheel_record, name, version, ranks): name
                 for name, version in versions.items() if name not in ("openpi", "openpi-client", "lerobot")}
        for future in as_completed(tasks):
            try:
                records.append(future.result())
            except Exception as error:
                failures.append({"package": tasks[future], "error": type(error).__name__})
    # Build backends are installed in uv's isolated build environment on the
    # original server, so they do not all appear in its runtime package list.
    build_versions = {"hatchling": "1.27.0", "poetry-core": "2.1.3",
                      "pathspec": "0.12.1", "editables": "0.5"}
    try:
        with urlopen("https://pypi.org/pypi/trove-classifiers/json", timeout=30) as response:
            build_versions["trove-classifiers"] = json.load(response)["info"]["version"]
        for name, version in build_versions.items():
            if name not in versions:
                record = wheel_record(name, version, ranks)
                record["purpose"] = "offline_build_backend"
                records.append(record)
    except Exception as error:
        failures.append({"package": "build-backends", "error": type(error).__name__})
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "download-manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (destination / "requirements-wheels.txt").write_text("".join(
        f"{r['name']}=={r['version']}\n" for r in sorted(records, key=lambda r: r["name"]) if r["kind"] == "wheel"))
    source_only = [r["name"] for r in records if r["kind"] == "sdist"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(fetch_file, r["url"], destination / r["filename"], r["sha256"], r["bytes"]): r
                 for r in sorted(records, key=lambda r: -r["bytes"])}
        for future in as_completed(tasks):
            record = tasks[future]
            try:
                future.result()
                print("CACHED", record["filename"], flush=True)
            except Exception as error:
                failures.append({"package": record["name"], "error": type(error).__name__})
    result = {"packages": len(records), "source_only_packages": source_only,
              "expected_bytes": sum(r["bytes"] for r in records), "failures": failures}
    (destination / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_model:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=MODEL_REVISION, local_dir=root / "hf-model", max_workers=4)
        manifest = json.loads((root / "hf-model/MANIFEST.json").read_text())
        for relative, expected in manifest.items():
            path = root / "hf-model" / relative
            if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"Model verification failed: {relative}")
        print("MODEL_VERIFIED", len(manifest), flush=True)
    if not args.model_only:
        for name, (repository, revision) in SOURCE_REVISIONS.items():
            fetch_file(f"https://codeload.github.com/{repository}/tar.gz/{revision}",
                       root / "sources" / f"{name}-{revision}.tar.gz")
        versions = json.loads(Path(__file__).with_name("python-packages.json").read_text())
        result = cache_wheels(versions, root / "linux-wheelhouse")
        print(json.dumps(result), flush=True)
        if result["failures"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
