# SPDX-License-Identifier: MIT
"""The download of WrongGPU's edition of OptiScaler, for run_launcher_tests.py (which calls run() in its place in the suite;
this file keeps that suite below the 256 KB source limit, as launcher_ignore_cases.py does). The section is moved unchanged:
run()'s arguments are the suite's own helpers it used as module names. Written for this project."""
import os
import shutil

from run_installer_tests import read, sha, snapshot, diff, write


def run(t, opti, game, edition_package, OPTI_INI):
    """WrongGPU's edition of OptiScaler, downloaded by the launcher (optiedition.h): pinned size + SHA-256, a mismatch refused,
    the cache reused, offline falls back to the player's own download. A local test server (127.0.0.1) stands in for the
    release page; the test build never downloads without WRONGGPU_TEST_EDITION."""
    print("\n[edition] the launcher downloads WrongGPU's edition of OptiScaler: pinned hash, mismatch refused, cache, offline fallback")
    import http.server
    import threading
    import zipfile
    ed_pkg, dll = edition_package(t, "edition-src")
    srv_dir = os.path.join(t.work, "edition-server")
    shutil.rmtree(srv_dir, ignore_errors=True)
    os.makedirs(srv_dir)
    zpath = os.path.join(srv_dir, "edition.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("OptiScaler.dll", dll)
        z.writestr("OptiScaler.ini", OPTI_INI)
        z.writestr("LICENSE-OptiScaler.txt", "GPL-3.0 (test stand-in)")
    zbytes = read(zpath)
    hits = []

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=srv_dir, **k)

        def do_GET(self):
            hits.append(self.path)
            return super().do_GET()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cache = os.path.join(t.data, "optiscaler", "edition-test.zip")
    try:
        pin = lambda url, s=None: "%s|%d|%s|%s" % (url, len(zbytes), s or sha(zbytes), sha(dll))
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:%d/edition.zip" % port)
        if os.path.exists(cache):
            os.remove(cache)
        root, b, exe = game(t, "Edition Game", validated=True)
        before = snapshot(root)
        rc = t.run("install", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        proxy = [n for n in ("dxgi.dll", "winmm.dll", "version.dll") if os.path.isfile(os.path.join(b, n))]
        t.check("no OptiScaler chosen: the launcher downloads its edition (one GET), checks it (sha256), caches it in its data folder and installs it",
                rc == 0 and len(hits) == 1 and read(cache) == zbytes and "downloaded and checked" in log and proxy and read(os.path.join(b, proxy[0])) == dll and
                "(WrongGPU edition, plugin interface 5)" in log, (rc, t.j.get("message"), hits, proxy, t.logtail()))
        rc = t.run("repair", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        t.check("the next install / repair reuses the cached, checked copy (no second download)",
                rc == 0 and len(hits) == 1 and "from the launcher's cache" in log, (rc, hits, t.logtail()))
        # the cached copy changed on disk: not used, downloaded again
        write(cache, zbytes[:-1] + b"X")
        rc = t.run("repair", exe, "--noscan", "--force")
        t.check("a cached copy that doesn't match is never used: downloaded again and checked", rc == 0 and len(hits) == 2 and read(cache) == zbytes, (rc, hits))
        # a download that isn't the pinned file: refused, deleted, never used; the player's own download (Settings) instead
        os.remove(cache)
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:%d/edition.zip" % port, "0" * 64)
        rc = t.run("repair", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        t.check("hash mismatch: refused with the plain words, nothing cached, no OptiScaler without the player's own download (exit 2)",
                rc == 2 and "download refused" in log and not os.path.exists(cache) and not os.path.exists(cache + ".part"), (rc, t.j.get("message"), t.logtail()))
        rc = t.run("set", "--global", "optiscaler=" + opti)
        rc = t.run("repair", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        t.check("hash mismatch with the player's own OptiScaler download set: that one is used", rc == 0 and "mock OptiScaler" in read(os.path.join(b, proxy[0])).decode("latin-1") if proxy else False,
                (rc, t.j.get("message"), t.logtail()))
        # offline (nothing answers): the player's own download
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:9/edition.zip")
        rc = t.run("repair", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        t.check("offline: the player's own OptiScaler download is used (fallback), nothing cached", rc == 0 and "download failed" in log and not os.path.exists(cache),
                (rc, t.j.get("message"), t.logtail()))
        t.run("set", "--global", "optiscaler=")
        rc = t.run("repair", exe, "--noscan", "--force")
        t.check("offline without the player's own download: refused (exit 2), said in words", rc == 2 and "OptiScaler" in t.j.get("message", ""), (rc, t.j.get("message")))
        # only https:// outside 127.0.0.1 (a test build allows the local test server only)
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://example.invalid/edition.zip")
        rc = t.run("repair", exe, "--noscan", "--force")
        t.check("plain http:// to another host is never used", rc == 2 and "only https://" in read(t.log).decode("utf-8", "replace"), (rc, t.logtail()))
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:%d/edition.zip" % port)
        rc = t.run("uninstall", exe)
        t.check("uninstall: the game folder back byte for byte",
                rc == 0 and snapshot(root) == before, (rc, diff(before, snapshot(root))))
        shutil.rmtree(root, ignore_errors=True)
    finally:
        httpd.shutdown()
        t.env.pop("WRONGGPU_TEST_EDITION", None)
