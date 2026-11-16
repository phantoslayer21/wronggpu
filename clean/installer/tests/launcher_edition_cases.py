# SPDX-License-Identifier: MIT
"""The download of WrongGPU's edition of OptiScaler, for run_launcher_tests.py (which calls run() in its place in the suite;
this file keeps that suite below the 256 KB source limit, as launcher_ignore_cases.py does). run()'s arguments are the
suite's own helpers it used as module names. Written for this project."""
import os
import shutil

from run_installer_tests import read, sha, snapshot, diff, write, has_direct_route


def run(t, opti, game, edition_package, OPTI_INI, no_direct=None):
    """WrongGPU's edition of OptiScaler, downloaded by the launcher (optiedition.h): pinned size + SHA-256, a mismatch refused,
    the cache reused, offline falls back to the player's own download. A local test server (127.0.0.1) stands in for the
    release page; the test build never downloads without WRONGGPU_TEST_EDITION.

    The section always runs with the launcher under test (--launcher). The download, cache and address rules are checked
    on both routes; only how a repair without any OptiScaler ends depends on the route: a launcher without the direct route
    (-DWRONGGPU_DIRECT_ROUTE=OFF, as releases) has nothing left to repair with (exit 2, said in words), one with it repairs
    through it (exit 0). When the launcher under test has the direct route, no_direct (--no-direct-launcher, a testing build
    without it) runs the section again in a pass of its own, for the exit-2 refusals."""
    print("\n[edition] the launcher downloads WrongGPU's edition of OptiScaler: pinned hash, mismatch refused, cache, offline fallback")
    passes = [(t.exe, "")]
    if has_direct_route(t.exe):
        if no_direct:
            if has_direct_route(no_direct):
                raise SystemExit("--no-direct-launcher %s has the direct route: build it with -DWRONGGPU_DIRECT_ROUTE=OFF" % no_direct)
            passes.append((no_direct, "without the direct route: "))
        else:
            print("  note: this launcher has the direct route, so a repair without any OptiScaler goes on through it (exit 0); the "
                  "exit-2 refusals need a testing launcher built with -DWRONGGPU_DIRECT_ROUTE=OFF, as releases: give one with "
                  "--no-direct-launcher")
    saved = t.exe
    try:
        for exe, tag in passes:
            t.exe = exe
            if tag:
                print("  (again, in a pass of its own, with the launcher without the direct route: %s)" % exe)
            _section(t, opti, game, edition_package, OPTI_INI, tag, not has_direct_route(exe))
    finally:
        t.exe = saved


def _section(t, opti, game, edition_package, OPTI_INI, tag, refuses):
    import http.server
    import threading
    import zipfile

    def check(name, ok, detail=""):
        t.check(tag + name, ok, detail)

    def ends(name, rc, rules, words, detail):
        """A repair without any OptiScaler: the download, cache and address rules (rules) hold on both routes; without the
        direct route it is refused (exit 2) with the words (words), with it the repair goes on through the direct route."""
        if refuses:
            check(name + "; nothing else to repair with: refused (exit 2), said in words", rc == 2 and rules and words, detail)
        else:
            check(name + "; the repair goes on through the direct route (exit 0)", rc == 0 and rules, detail)

    def repair():
        """A repair; returns its exit code and the log lines it wrote (this run's only)."""
        start = os.path.getsize(t.log) if os.path.isfile(t.log) else 0
        rc = t.run("repair", exe, "--noscan", "--force")
        return rc, read(t.log)[start:].decode("utf-8", "replace")

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
        t.run("set", "--global", "optiscaler=")  # no player's own download (an earlier pass may have set one)
        root, b, exe = game(t, "Edition Game", validated=True)
        before = snapshot(root)
        rc = t.run("install", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        proxy = [n for n in ("dxgi.dll", "winmm.dll", "version.dll") if os.path.isfile(os.path.join(b, n))]
        check("no OptiScaler chosen: the launcher downloads its edition (one GET), checks it (sha256), caches it in its data folder and installs it",
              rc == 0 and len(hits) == 1 and read(cache) == zbytes and "downloaded and checked" in log and proxy and read(os.path.join(b, proxy[0])) == dll and
              "(WrongGPU edition, plugin interface 5)" in log, (rc, t.j.get("message"), hits, proxy, t.logtail()))
        rc, seg = repair()
        check("the next install / repair reuses the cached, checked copy (no second download)",
              rc == 0 and len(hits) == 1 and "from the launcher's cache" in seg, (rc, hits, seg[-1500:]))
        # the cached copy changed on disk: not used, downloaded again
        write(cache, zbytes[:-1] + b"X")
        rc, seg = repair()
        check("a cached copy that doesn't match is never used: downloaded again and checked", rc == 0 and len(hits) == 2 and read(cache) == zbytes, (rc, hits))
        # a download that isn't the pinned file: refused, deleted, never used; the player's own download (Settings) instead
        os.remove(cache)
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:%d/edition.zip" % port, "0" * 64)
        rc, seg = repair()
        ends("hash mismatch: the download refused with the plain words, nothing cached, no partial file", rc,
             "download refused" in seg and len(hits) == 3 and not os.path.exists(cache) and not os.path.exists(cache + ".part"),
             True, (rc, t.j.get("message"), hits, seg[-1500:]))
        t.run("set", "--global", "optiscaler=" + opti)
        rc, seg = repair()
        check("hash mismatch with the player's own OptiScaler download set: that one is used, nothing cached",
              rc == 0 and bool(proxy) and "mock OptiScaler" in read(os.path.join(b, proxy[0])).decode("latin-1") and not os.path.exists(cache),
              (rc, t.j.get("message"), seg[-1500:]))
        # offline (nothing answers): the player's own download
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:9/edition.zip")
        rc, seg = repair()
        check("offline: the player's own OptiScaler download is used (fallback), nothing cached", rc == 0 and "download failed" in seg and not os.path.exists(cache),
              (rc, t.j.get("message"), seg[-1500:]))
        t.run("set", "--global", "optiscaler=")
        rc, seg = repair()
        ends("offline without the player's own download: the download failed, nothing cached", rc,
             "download failed" in seg and not os.path.exists(cache), "OptiScaler" in t.j.get("message", ""), (rc, t.j.get("message"), seg[-1500:]))
        # only https:// outside 127.0.0.1 (a test build allows the local test server only)
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://example.invalid/edition.zip")
        rc, seg = repair()
        ends("plain http:// to another host is never used (only https://), nothing cached", rc,
             "only https://" in seg and not os.path.exists(cache), True, (rc, seg[-1500:]))
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:%d/edition.zip" % port)
        rc = t.run("uninstall", exe)
        check("uninstall: the game folder back byte for byte",
              rc == 0 and snapshot(root) == before, (rc, diff(before, snapshot(root))))
        shutil.rmtree(root, ignore_errors=True)
    finally:
        httpd.shutdown()
        t.env.pop("WRONGGPU_TEST_EDITION", None)
