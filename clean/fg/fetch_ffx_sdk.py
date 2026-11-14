# SPDX-License-Identifier: MIT
"""Fetches the few files of AMD's public FSR SDK that clean/fg/ffx_swapchain_test.cpp needs, at one pinned commit, into
a folder OUTSIDE this repository. Nothing fetched here is committed, vendored or shipped: the test loads AMD's signed
DLLs from that folder and builds against its headers.

    python clean/fg/fetch_ffx_sdk.py <destination folder>

Source: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, tag v2.4.0 = commit
7539ae8b6568ad29ed78a753404351daf3b5cca7 (licence: Kits/FidelityFX/docs/license.md, fetched with the rest).
Every file is checked against the git blob hash that GitHub's tree for that commit lists; a mismatch stops the script.
A MANIFEST.txt with each file's size and SHA-256 is written next to the files. Python standard library only.
"""
import hashlib
import json
import os
import sys
import urllib.request

REPO = 'GPUOpen-LibrariesAndSDKs/FidelityFX-SDK'
COMMIT = '7539ae8b6568ad29ed78a753404351daf3b5cca7'
FILES = [
    'Kits/FidelityFX/docs/license.md',
    'Kits/FidelityFX/api/include/ffx_api.h',
    'Kits/FidelityFX/api/include/ffx_api_types.h',
    'Kits/FidelityFX/api/include/ffx_api_loader.h',
    'Kits/FidelityFX/api/include/dx12/ffx_api_dx12.h',
    'Kits/FidelityFX/framegeneration/include/ffx_framegeneration.h',
    'Kits/FidelityFX/framegeneration/include/ffx_framegeneration_api_types.h',
    'Kits/FidelityFX/framegeneration/include/dx12/ffx_api_framegeneration_dx12.h',
    'Kits/FidelityFX/signedbin/amd_fidelityfx_loader_dx12.dll',
    'Kits/FidelityFX/signedbin/amd_fidelityfx_framegeneration_dx12.dll',
]


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'wronggpu-fetch-ffx-sdk'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def git_blob_sha(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    dest = os.path.abspath(sys.argv[1])
    tree = json.loads(get('https://api.github.com/repos/%s/git/trees/%s?recursive=1' % (REPO, COMMIT)))
    blobs = {e['path']: e['sha'] for e in tree.get('tree', []) if e.get('type') == 'blob'}
    lines = ['source https://github.com/%s commit %s' % (REPO, COMMIT)]
    for path in FILES:
        if path not in blobs:
            print('not in the tree at %s: %s' % (COMMIT, path))
            return 2
        data = get('https://raw.githubusercontent.com/%s/%s/%s' % (REPO, COMMIT, path))
        if git_blob_sha(data) != blobs[path]:
            print('git blob hash mismatch for %s' % path)
            return 3
        out = os.path.join(dest, *path.split('/'))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f:
            f.write(data)
        lines.append('%s %d %s' % (hashlib.sha256(data).hexdigest(), len(data), path))
        print('ok %9d %s' % (len(data), path))
    with open(os.path.join(dest, 'MANIFEST.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
