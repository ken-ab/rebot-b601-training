"""Build a verified B601 ZIP, preferring installed runtime over wheel rebuild."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    model = root / "hf-model"
    original = json.loads((model / "MANIFEST.json").read_text())
    archives = {r["file"]: r for r in json.loads((root / "ARCHIVES.json").read_text())}
    files = []
    runtime = root / "b601-linux-runtime.tar.zst"
    runtime_record = archives["b601-linux-runtime.tar.zst"]
    full_runtime = (runtime.is_file() and runtime.stat().st_size == runtime_record["bytes"]
                    and sha256(runtime) == runtime_record["sha256"])
    required_archives = ["b601-code-assets.tar.gz"]
    if full_runtime:
        required_archives.append("b601-linux-runtime.tar.zst")
    for name in required_archives:
        path = root / name
        expected = archives[name]
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise SystemExit(f"Archive not yet complete/verified: {path}")
        files.append((path, "archives/" + name, expected["sha256"]))
    for name in ("b601-python311.tar.gz", "portable-zstd.tar.gz"):
        path = root / name
        digest = (root / (name + ".sha256")).read_text().split()[0]
        if sha256(path) != digest:
            raise SystemExit(f"Archive SHA mismatch: {path}")
        files.append((path, "archives/" + name, digest))
    if not full_runtime:
        wheelhouse = root / "linux-wheelhouse"
        status = json.loads((wheelhouse / "status.json").read_text())
        if status["failures"]:
            raise SystemExit("Linux wheel cache has unresolved downloads")
        for record in json.loads((wheelhouse / "download-manifest.json").read_text()):
            path = wheelhouse / record["filename"]
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise SystemExit(f"Dependency cache SHA mismatch: {path}")
        for path in sorted(wheelhouse.iterdir()):
            if path.is_file() and not path.name.endswith(".part"):
                files.append((path, "linux-wheelhouse/" + path.name, None))
        for path in sorted((root / "sources").glob("*.tar.gz")):
            files.append((path, "sources/" + path.name, None))
    for relative, expected in original.items():
        path = model / relative
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise SystemExit(f"Model SHA mismatch: {relative}")
    for path in sorted(model.rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(model).parts:
            files.append((path, "hf-model/" + path.relative_to(model).as_posix(), None))
    here = Path(__file__).resolve().parent
    for name in ("restore.sh", "README.md", "python-packages.json", "deployment-info.json", "installed-sources.json"):
        files.append((here / name, name, None))
    manifest = {"environment_mode": "installed_runtime" if full_runtime else "linux_wheel_cache",
                "model_repository": "dahjgfa/b601-pi05-final60",
                "model_revision": "8c11eb8751d5cc6f688fa51db523da5bd8f2518a", "files": []}
    destination = root / "B601-RTC-portable.zip"
    partial = root / "B601-RTC-portable.zip.partial"
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path, relative, digest in files:
            archive.write(path, relative)
            manifest["files"].append({"path": relative, "bytes": path.stat().st_size,
                                      "sha256": digest or sha256(path)})
        archive.writestr("portable-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    partial.replace(destination)
    (root / "B601-RTC-portable.zip.sha256").write_text(sha256(destination) + "  " + destination.name + "\n")
    (root / "portable-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"archive": str(destination), "bytes": destination.stat().st_size,
                      "files": len(files), "environment_mode": manifest["environment_mode"], "verified": True}))


if __name__ == "__main__":
    main()
