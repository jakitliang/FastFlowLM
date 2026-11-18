"""Assemble a separate test bundle, without installing drivers or copying user models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

BASELINE = "838cfca5d9370adb7cc5ba19746bed21e14ecd01"
OFFICIAL_URL = "https://github.com/ROCm/FastFlowLM/releases/download/v1.0.13/fastflowlm_1.0.13_windows_amd64.zip"
BUNDLE_SCRIPTS = ("README.zh-CN.md", "build-windows.cmd", "package.py", "serve-test.ps1", "verify-http.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_zip", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-log", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.check_output(["git", "-C", str(root), *a])
    if git("merge-base", BASELINE, "HEAD").decode().strip() != BASELINE:
        raise RuntimeError("HEAD does not descend from the audited stable baseline")
    source_files = git("diff", "--name-only", "--no-renames", "-z", BASELINE).decode().split("\0")[:-1]
    patch = git("diff", "--binary", BASELINE)
    if args.output.exists() or Path(str(args.output) + ".zip").exists():
        raise RuntimeError("Output already exists; refusing to overwrite")
    args.output.mkdir(parents=True)
    with zipfile.ZipFile(args.official_zip) as archive:
        for entry in archive.infolist():
            target = (args.output / entry.filename).resolve()
            if not target.is_relative_to(args.output.resolve()):
                raise RuntimeError("Unsafe ZIP member")
        archive.extractall(args.output)
    for engine in (root / "src/lib/xrt").glob("*.dll"):
        if sha(engine) != sha(args.output / engine.name):
            raise RuntimeError(f"Engine differs from official stable ZIP: {engine.name}")
    shutil.copy2(root / "src/build/flm.exe", args.output / "flm.exe")
    patch_dir = args.output / "chatbox-patch"
    patch_dir.mkdir()
    for name in BUNDLE_SCRIPTS:
        shutil.copy2(root / "chatbox-patch" / name, patch_dir / name)
    for name in source_files:
        src = root / name
        if src.is_file():
            dst = patch_dir / "source" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (patch_dir / "source.patch").write_bytes(patch)
    if args.build_log:
        shutil.copy2(args.build_log, patch_dir / "build.log")
    for suite in ("http_guard", "model_metadata"):
        log = root / "src/build-tests" / suite / "Testing/Temporary/LastTest.log"
        shutil.copy2(log, patch_dir / f"{suite}-windows.log")
    manifest = {
        "baseline": BASELINE, "tag": "v1.0.13", "modified_test_build": True,
        "platform": "windows-x64", "backend": "XRT / FLM NPU", "npu_inference_tested": False,
        "official_zip_url": OFFICIAL_URL, "official_zip_sha256": sha(args.official_zip),
        "files": {p.relative_to(args.output).as_posix(): sha(p)
                  for p in sorted(args.output.rglob("*")) if p.is_file()},
    }
    (patch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    zip_path = Path(shutil.make_archive(str(args.output), "zip", root_dir=args.output.parent, base_dir=args.output.name))
    checksum = zip_path.with_suffix(".zip.sha256")
    checksum.write_text(f"{sha(zip_path)}  {zip_path.name}\n", encoding="ascii")
    print(json.dumps({"directory": str(args.output), "zip": str(zip_path), "bytes": zip_path.stat().st_size,
                      "sha256": sha(zip_path)}, indent=2))


if __name__ == "__main__":
    main()