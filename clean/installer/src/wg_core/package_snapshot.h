// SPDX-License-Identifier: MIT
// The player's OptiScaler package as the elevated worker installs it: bound to the launcher's request by a digest and
// installed from the worker's own copy.
//
// The launcher computes the package's digest before the permission prompt (PackageDigest) and puts it in the job
// (optisha256=). The worker copies the package into a new folder only administrators can write (MakePrivateFolder,
// SnapshotPackage), computes the digest of that copy and installs from the copy only when it is the job's digest.
//
// The digest is the SHA-256 of a manifest listing every file with its SHA-256: for an archive (.7z / .zip) the archive
// file itself; for a folder, or the OptiScaler.dll in a folder, every file in that folder and its subfolders (folders and
// files that are links are left out on both sides).
#pragma once

#include "util.h"

namespace wg {

// The digest of the package `chosen` (an archive, a folder, or OptiScaler.dll in a folder). False with *why when it cannot
// be read or is larger than a package can be.
bool PackageDigest(const wstring& chosen, wstring& digest, wstring& why);

// Copies the package `chosen` into `intoDir` (an existing private folder), reading each file once; true when the digest of
// the copy is `expected`. `copy`: the package to install from (the archive's copy, or the folder's copy).
bool SnapshotPackage(const wstring& chosen, const wstring& expected, const wstring& intoDir, wstring& copy, wstring& why);

// A new folder with a random name in `parent` (empty: the Windows folder's Temp) that only administrators and SYSTEM can
// open (and this account, when this process is not elevated: the tests' stand-in for the elevated worker). It must not exist
// before: a name that is taken is never reused.
bool MakePrivateFolder(const wstring& parent, wstring& path, wstring& why);

// 16 random bytes as 32 hex digits (the system's random number generator).
wstring RandomHex();

}  // namespace wg
