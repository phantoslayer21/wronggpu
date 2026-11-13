# SPDX-License-Identifier: MIT
"""Prepare a private native Model M bundle with no HIP code objects or HIP runtime DLLs.

The runtime still offers explicit HIP/hybrid backends when reference assets are supplied
separately. Backend 3 resolves route capabilities from the DXIL manifest alone.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    source, out = args.build.resolve(), args.out.resolve()
    manifest_path = source / "dx12" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = [Path("wronggpu_runtime.dll"), Path("variants.json"), Path("preset-M.v1.map.json"),
             Path("dx12/manifest.json")]
    for kernel in manifest["kernels"]:
        name = kernel["file"]
        if Path(name).name != name or not name.endswith(".dxil"):
            raise RuntimeError("invalid manifest shader filename")
        rel = Path("dx12") / name
        if digest(source / rel) != kernel["sha256"]:
            raise RuntimeError("shader hash mismatch: " + name)
        files.append(rel)
    hashes = {str(rel): digest(source / rel) for rel in files}
    out.mkdir(parents=True, exist_ok=False)
    for rel in files:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, out / rel)
        if digest(out / rel) != hashes[str(rel)]:
            raise RuntimeError("copied artifact hash mismatch")
    (out / "native-bundle.json").write_text(json.dumps({"source": str(source), "backend": 3,
        "runtime": "d3d12", "hip_code_objects": [], "sha256": hashes,
        "scope": "Private Model M D3D12 package; supplied-device qualification recorded separately; no installed-game qualification"}, indent=2))
    print(str(out))


if __name__ == "__main__":
    main()
