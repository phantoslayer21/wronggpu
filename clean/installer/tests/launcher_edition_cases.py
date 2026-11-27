# SPDX-License-Identifier: GPL-3.0-only
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
        start = os.path.getsize(t.log) if os.path.isfile(t.log) else 0
        rc = t.run("install", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        unpack = [l.split(" in ", 1)[1].strip() for l in read(t.log)[start:].decode("utf-8", "replace").splitlines()
                  if "OptiScaler: unpacked edition-test.zip in " in l]
        check("the edition archive is unpacked in the run's own temporary folder (TEMP in the work folder), nowhere else",
              len(unpack) == 1 and os.path.abspath(unpack[0]).lower().startswith(os.path.join(t.work, "tmp").lower() + os.sep) and
              t.env.get("TEMP") == os.path.join(t.work, "tmp"), (unpack, t.env.get("TEMP")))
        proxy = [n for n in ("dxgi.dll", "winmm.dll", "version.dll") if os.path.isfile(os.path.join(b, n))]
        check("no OptiScaler chosen: the launcher downloads its edition (one GET), checks it (sha256), caches it in its data folder and installs it",
              rc == 0 and len(hits) == 1 and read(cache) == zbytes and "downloaded and checked" in log and proxy and read(os.path.join(b, proxy[0])) == dll and
              "(WrongGPU edition, plugin interface 6)" in log, (rc, t.j.get("message"), hits, proxy, t.logtail()))
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

        # ---- v5 -> v6 upgrade, cache isolation, v6 kept, and needsHostRestore RR game with v6 ----
        import mockpe
        v5_dir = os.path.join(t.work, "edition-v5-src")
        shutil.rmtree(v5_dir, ignore_errors=True)
        v5_dll = mockpe.make_dll({"ProductName": "OptiScaler", "FileDescription": "OptiScaler WrongGPU edition v5 (test stand-in)"}, (10, 0, 0, 5),
                                 extra=b"OptiScaler WrongGPU edition v5 stand-in (synthetic, never loaded)", dll_name="OptiScaler.dll")
        write(os.path.join(v5_dir, "OptiScaler.dll"), v5_dll)
        write(os.path.join(v5_dir, "OptiScaler.ini"), OPTI_INI)
        t.env["WRONGGPU_TEST_PRIOR_EDITION_DLL"] = sha(v5_dll)

        # 1. v5 archive cached -> ignored, v6 fetched
        if os.path.exists(cache):
            os.remove(cache)
        v5_cache = os.path.join(t.data, "optiscaler", "OptiScaler_v10.0.0-pre1_20261030_WrongGPU-edition-v5.zip")
        with zipfile.ZipFile(v5_cache, "w", zipfile.ZIP_DEFLATED) as z5:
            z5.writestr("OptiScaler.dll", v5_dll)
            z5.writestr("OptiScaler.ini", OPTI_INI)
        hits_before = len(hits)
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:%d/edition.zip" % port)
        rootC, bC, exeC = game(t, "Edition CacheV5 Game", validated=True)
        rc = t.run("install", exeC, "--noscan", "--force")
        pC = [n for n in ("dxgi.dll", "winmm.dll", "version.dll") if os.path.isfile(os.path.join(bC, n))]
        check("v5 archive cached: ignored, v6 fetched and installed",
              rc == 0 and len(hits) == hits_before + 1 and os.path.isfile(cache) and pC and read(os.path.join(bC, pC[0])) == dll,
              (rc, len(hits) - hits_before, pC, t.j.get("message")))

        # 2. v6 installed -> kept, and cached v6 used with no network
        rc_s = t.run("status", exeC, "--noscan")
        gC = t.j.get("game", {})
        check("v6 installed: status is installed (not update-available), optiscaler_edition=true, outdated=false",
              rc_s == 0 and gC.get("status") == "installed" and gC.get("optiscaler_edition") is True and gC.get("optiscaler_edition_outdated") is False, gC)
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:9/edition.zip")
        rc_r, seg_r = (t.run("repair", exeC, "--noscan", "--force"), "")
        check("v6 archive cached: used with no network (unreachable port) and v6 kept",
              rc_r == 0 and len(hits) == hits_before + 1 and pC and read(os.path.join(bC, pC[0])) == dll,
              (rc_r, len(hits) - hits_before, t.j.get("message")))
        t.run("uninstall", exeC)
        shutil.rmtree(rootC, ignore_errors=True)
        if os.path.exists(v5_cache):
            os.remove(v5_cache)

        # 3. Managed v5 installed -> flagged update-available (older OptiScaler edition) and upgraded to v6 on update & repair
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:%d/edition.zip" % port)
        rootM, bM, exeM = game(t, "Edition ManagedV5 Game", validated=True)
        rc = t.run("install", exeM, "--opti=" + v5_dir, "--noscan", "--force")
        rc_s = t.run("status", exeM, "--noscan")
        gM = t.j.get("game", {})
        check("managed v5 installed: flagged as update-available (older OptiScaler edition), optiscaler_edition=false, outdated=true",
              rc == 0 and rc_s == 0 and gM.get("status") == "update-available" and "older OptiScaler edition" in gM.get("status_text", "") and
              gM.get("optiscaler_edition") is False and gM.get("optiscaler_edition_outdated") is True, gM)
        rc_u = t.run("update", exeM, "--noscan", "--force")
        rc_s2 = t.run("status", exeM, "--noscan")
        gM2 = t.j.get("game", {})
        pM = [n for n in ("dxgi.dll", "winmm.dll", "version.dll") if os.path.isfile(os.path.join(bM, n))]
        check("managed v5 update: upgraded to v6, status=installed, optiscaler_edition=true, outdated=false",
              rc_u == 0 and rc_s2 == 0 and gM2.get("status") == "installed" and gM2.get("optiscaler_edition") is True and
              gM2.get("optiscaler_edition_outdated") is False and pM and read(os.path.join(bM, pM[0])) == dll, (rc_u, gM2, pM))
        # Also verify repair upgrades a managed v5 install to v6
        rc_v5 = t.run("install", exeM, "--opti=" + v5_dir, "--noscan", "--force")
        rc_rep = t.run("repair", exeM, "--noscan", "--force")
        pM2 = [n for n in ("dxgi.dll", "winmm.dll", "version.dll") if os.path.isfile(os.path.join(bM, n))]
        check("managed v5 repair: upgraded to v6",
              rc_v5 == 0 and rc_rep == 0 and pM2 and read(os.path.join(bM, pM2[0])) == dll, (rc_v5, rc_rep, pM2))
        t.run("uninstall", exeM)
        shutil.rmtree(rootM, ignore_errors=True)

        # 4. Unmanaged v5 DLL in game folder -> treated as outdated edition (Migrate, NOT foreign OptiScaler), upgraded in place without --switch
        rootU, bU, exeU = game(t, "Edition UnmanagedV5 Game", validated=True)
        write(os.path.join(bU, "winmm.dll"), v5_dll)
        write(os.path.join(bU, "OptiScaler.ini"), OPTI_INI)
        beforeU = snapshot(rootU)
        rc_sU = t.run("status", exeU, "--noscan")
        gU = t.j.get("game", {})
        co_opti = [c for c in gU.get("coexistence", []) if c.get("tool") == "optiscaler"]
        check("unmanaged v5: status=eligible (not players-optiscaler/foreign-mod), coexistence decision=migrate, proxy reused (winmm.dll)",
              rc_sU == 0 and gU.get("status") == "eligible" and co_opti and co_opti[0].get("decision") == "migrate" and
              gU.get("proxy_plan", {}).get("name", "").lower() == "winmm.dll", gU)
        rc_iU = t.run("install", exeU, "--noscan", "--force")  # no --switch!
        rc_sU2 = t.run("status", exeU, "--noscan")
        gU2 = t.j.get("game", {})
        check("unmanaged v5 install (without --switch): upgraded in place at winmm.dll to v6, optiscaler_edition=true",
              rc_iU == 0 and rc_sU2 == 0 and read(os.path.join(bU, "winmm.dll")) == dll and not os.path.exists(os.path.join(bU, "dxgi.dll")) and
              gU2.get("status") == "installed" and gU2.get("optiscaler_edition") is True, (rc_iU, gU2, t.j.get("message")))
        rc_unU = t.run("uninstall", exeU)
        check("unmanaged v5 uninstall: original v5 files restored byte for byte",
              rc_unU == 0 and snapshot(rootU) == beforeU, (rc_unU, diff(beforeU, snapshot(rootU))))
        shutil.rmtree(rootU, ignore_errors=True)

        # 5. RR game that needs the edition (needsHostRestore: re9.exe) -> RR offered with v6
        rootR = os.path.join(t.work, "games", "Edition Re9 Game")
        shutil.rmtree(rootR, ignore_errors=True)
        exeR = os.path.join(rootR, "re9.exe")
        write(exeR, "mock RE9 game exe " * 200)
        import run_installer_tests as _rit
        write(os.path.join(rootR, "amd_fidelityfx_dx12.dll"), _rit.AMD_V1)
        write(os.path.join(rootR, "nvngx_dlss.dll"), t.model.file)
        write(os.path.join(rootR, "nvngx_dlssd.dll"), b"MZ mock dlss rr")
        write(os.path.join(rootR, "sl.dlss_d.dll"), mockpe.make_dll({"ProductName": "Game interface RR plugin"}, (2, 8, 0, 0), extra=b"synthetic"))
        write(os.path.join(rootR, "dinput8.dll"), b"MZ synthetic loader stand-in: REFramework v1.5 (test)")
        write(os.path.join(rootR, "reframework", "config.txt"), "test")
        beforeR = snapshot(rootR)
        startR = os.path.getsize(t.log) if os.path.isfile(t.log) else 0
        rc_R = t.run("install", exeR, "--noscan", "--force", "--experimental")
        logR = read(t.log)[startR:].decode("utf-8", "replace")
        rc_sR = t.run("status", exeR, "--noscan")
        gR = t.j.get("game", {})
        check("needsHostRestore RR game (re9.exe) with v6: installed with v6 DLL and RR plugin, no edition warning, plugin interface 6 logged",
              rc_R == 0 and rc_sR == 0 and read(os.path.join(rootR, "dxgi.dll")) == dll and
              os.path.isfile(os.path.join(rootR, "plugins", "WrongGPU-RR.asi")) and
              not any("no OptiScaler edition" in w for w in t.j.get("warnings", [])) and
              "plugin interface 6" in logR and gR.get("optiscaler_edition") is True and gR.get("rr_installed") is True,
              (rc_R, gR, t.j.get("warnings"), logR[-1200:]))
        t.run("uninstall", exeR)
        shutil.rmtree(rootR, ignore_errors=True)
    finally:
        httpd.shutdown()
        t.env.pop("WRONGGPU_TEST_EDITION", None)
        t.env.pop("WRONGGPU_TEST_PRIOR_EDITION_DLL", None)
