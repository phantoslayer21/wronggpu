# SPDX-License-Identifier: MIT
"""rr-prepare and rr-recheck (RR's kernels prepared for a game ahead of its start), for run_launcher_tests.py's RR section
(which calls run() in its place; this file keeps that suite below the 256 KB source limit, as launcher_edition_cases.py
does). Written for this project.

The default run stays inside its work folder and never prepares anything on the GPU:
- the no-card path runs with a fresh data folder (the helper's run folder not made yet) and a test build's driver answer
  "none" (WRONGGPU_TEST_RR_DRIVER);
- with the suite's stand-in runtime (a fresh clone's test payload: no kernels), the helper really runs, under a test driver
  version, and must fail to load that runtime: the expected result, with its diagnostic. The helper must keep the suite's
  redirected LOCALAPPDATA (a test build's rule, rrprep.h RrHelperEnvironment);
- the checks that need RR's kernels really prepared (a payload with the real runtime and an RDNA 4 card) fill the user's
  real graphics driver cache, so they run only with the opt-in --rr-prepare-gpu.
A test build starts no helper at all unless the test allows it (WRONGGPU_TEST_ALLOW_RR_HELPER=1, rrprep.cpp): only the
stand-in case and the opt-in pass do; a negative control shows a game the classification does not stop gets no helper."""
import os
import shutil

import launcher_build_facts as facts
from run_installer_tests import read, write


def _helper_gone(run_dir):
    """No helper left in the helper's run folder; an absent folder is empty (the no-card path never makes it)."""
    return not os.path.isdir(run_dir) or not os.listdir(run_dir)


def run(t, exe, b, gpu_opt_in=False):
    """rr-prepare --check, rr-recheck, rr-prepare's no-card path, the helper with the stand-in runtime and, with the opt-in, a
    real runtime and an RDNA 4 card, the prepared state across a driver or runtime change. exe: the RR game's exe, b: its folder."""
    # release 0.6.0 (owner: RR's kernels pre-compiled inside the launcher, rrprep.h): rr-prepare runs the launcher as a helper
    # under the game's program name, which loads the installed wronggpu_runtime.dll (the real runtime of this payload, RR
    # built in) and creates RR's pipelines on the GPU; the driver keeps them per program name, so the game finds them ready
    prep_dir = os.path.join(t.data, "rr-prepare")  # the launcher data folder (the suite runs with --data)
    rc = t.run("rr-prepare", exe, "--check")
    t.check("rr-prepare --check before: not prepared yet", rc == 0 and t.j.get("rr_prepare", {}).get("ready") is False and
            "not prepared yet" in t.j.get("rr_prepare", {}).get("not_ready_why", ""), t.j)
    # release 0.6.0 polish: the window's start prepares again only games prepared before (never prepared: Play's job)
    rc = t.run("rr-recheck", exe)
    rq = t.j.get("rr_recheck", {})
    t.check("rr-recheck before any prepare: not stale (state not-prepared), the start leaves it to Play",
            rc == 0 and rq.get("stale") is False and rq.get("state_before") == "not-prepared", t.j)
    glog = os.path.join(b, "wronggpu-rr.log")  # the RR log an earlier case wrote (Last run): prepare must not touch it
    glog_before = read(glog) if os.path.exists(glog) else None
    game_before = sorted(os.listdir(b))

    # no AMD graphics card, a fresh data folder: result 2, exit 0, no helper, no run folder made
    saved = t.data
    t.data = os.path.join(t.work, "data-rr-nocard")
    shutil.rmtree(t.data, ignore_errors=True)
    t.env["WRONGGPU_TEST_RR_DRIVER"] = "none"
    try:
        rc = t.run("rr-prepare", exe, "--passes=1")
    finally:
        t.env.pop("WRONGGPU_TEST_RR_DRIVER", None)
        nocard_dir = os.path.join(t.data, "rr-prepare")
        t.data = saved
    rp = t.j.get("rr_prepare", {})
    t.check("rr-prepare without an AMD graphics card (fresh data folder): result 2, exit 0, the reason in words, no helper run, "
            "nothing in the game's folder", rc == 0 and rp.get("result") == 2 and "no AMD graphics card" in rp.get("why", "") and
            _helper_gone(os.path.join(nocard_dir, "run")) and not os.path.isfile(os.path.join(nocard_dir, "prepared.txt")) and
            sorted(os.listdir(b)) == game_before, (rc, rp, os.listdir(b)))
    shutil.rmtree(os.path.join(t.work, "data-rr-nocard"), ignore_errors=True)

    envlog = os.path.join(t.work, "rr-helper-env.log")
    if os.path.exists(envlog):
        os.remove(envlog)
    # the negative control: a game the in-game classification does not stop, a driver, no permission: no helper starts
    t.env["WRONGGPU_TEST_RR_DRIVER"] = "99.0.0.1"
    t.env["WRONGGPU_TEST_HELPER_ENV_LOG"] = envlog
    try:
        rc = t.run("rr-prepare", exe, "--passes=1")
    finally:
        t.env.pop("WRONGGPU_TEST_RR_DRIVER", None)
        t.env.pop("WRONGGPU_TEST_HELPER_ENV_LOG", None)
    rp = t.j.get("rr_prepare", {})
    t.check("rr-prepare without the test's permission: no helper starts, whatever the classification (result 3, exit 1, said why)",
            rc == 1 and rp.get("result") == 3 and "WRONGGPU_TEST_ALLOW_RR_HELPER" in rp.get("why", "") and not os.path.exists(envlog) and
            _helper_gone(os.path.join(prep_dir, "run")), (rc, rp))

    if facts.stand_in_runtime(b):
        # the suite's stand-in runtime: the helper really runs (a test driver version, so on any PC) and cannot load it; it
        # must keep the suite's LOCALAPPDATA even when the caller's environment asks for the real one (the suite strips that)
        saved_env = t.env
        t.env = t.child_env(dict(os.environ, WRONGGPU_TEST_REAL_DRIVER_CACHE="1"))
        t.env.update({k: v for k, v in saved_env.items() if k in ("WRONGGPU_TEST_LIBRARIES", "WRONGGPU_TEST_ACCESS_LOG", "WRONGGPU_TEST_RUN_ID")})
        stripped = "WRONGGPU_TEST_REAL_DRIVER_CACHE" not in t.env
        t.env.update(WRONGGPU_TEST_RR_DRIVER="99.0.0.1", WRONGGPU_TEST_HELPER_ENV_LOG=envlog, WRONGGPU_TEST_ALLOW_RR_HELPER="1")
        try:
            rc = t.run("rr-prepare", exe, "--passes=1")
        finally:
            t.env = saved_env
        rp = t.j.get("rr_prepare", {})
        why = rp.get("why", "")
        t.check("rr-prepare with the suite's stand-in runtime: the helper ran and could not load it (result 3, exit 1, the helper's "
                "code and the log in the reason), nothing recorded as prepared",
                rc == 1 and rp.get("result") == 3 and ("the helper ended with 7" in why or "the helper ended with 8" in why) and
                "prepare.log" in why and rp.get("ready") is False and not os.path.isfile(os.path.join(prep_dir, "prepared.txt")), (rc, rp))
        seen = read(envlog).decode("utf-8", "replace").splitlines() if os.path.isfile(envlog) else []
        t.check("rr-prepare: the helper kept the suite's redirected LOCALAPPDATA, also when the caller's environment had "
                "WRONGGPU_TEST_REAL_DRIVER_CACHE=1 (stripped before any launcher call)",
                stripped and seen == ["LOCALAPPDATA=" + t.env["LOCALAPPDATA"]], (stripped, seen, t.env["LOCALAPPDATA"]))
        t.check("rr-prepare: the helper copy is gone, nothing new in the game's folder (the runtime log went to the data folder)",
                _helper_gone(os.path.join(prep_dir, "run")) and (read(glog) if os.path.exists(glog) else None) == glog_before and
                sorted(os.listdir(b)) == game_before, (os.listdir(prep_dir) if os.path.isdir(prep_dir) else None, os.listdir(b)))
        print("  skipped: rr-prepare's checks of RR's kernels really prepared (this payload's WrongGPU runtime is the suite's "
              "stand-in: they need a payload with the real runtime, an RDNA 4 card and --rr-prepare-gpu)")
        return
    if not gpu_opt_in:
        print("  skipped: rr-prepare's GPU checks (a real runtime prepares RR's kernels into the user's real graphics driver "
              "cache: only with the opt-in --rr-prepare-gpu)")
        return
    # opt-in: the helper allowed, with the real driver cache, as a player's launcher fills it
    t.env.update(WRONGGPU_TEST_REAL_DRIVER_CACHE="1", WRONGGPU_TEST_ALLOW_RR_HELPER="1")
    try:
        _gpu(t, exe, b, prep_dir, glog, glog_before)
    finally:
        t.env.pop("WRONGGPU_TEST_REAL_DRIVER_CACHE", None)
        t.env.pop("WRONGGPU_TEST_ALLOW_RR_HELPER", None)


def _gpu(t, exe, b, prep_dir, glog, glog_before):
    """--rr-prepare-gpu: a payload with the real runtime; an RDNA 4 card prepares, another PC answers result 2."""
    rc = t.run("rr-prepare", exe, "--passes=1")
    rp = t.j.get("rr_prepare", {})
    gpu = rp.get("result") != 2  # 2: no RDNA 4 graphics card on this PC (nothing to prepare)
    t.check("rr-prepare: the helper prepared the default mode's kernels (result 0, ready, every kernel reported), or no RDNA 4 card",
            rc == 0 and (not gpu or (rp.get("result") == 0 and rp.get("ready") is True and rp.get("kernels", 0) >= 50)), rp)
    t.check("rr-prepare: the helper copy is gone, nothing new in the game's folder (the runtime log went to the data folder)",
            _helper_gone(os.path.join(prep_dir, "run")) and (read(glog) if os.path.exists(glog) else None) == glog_before and
            (not gpu or os.path.isfile(os.path.join(prep_dir, "prepare.log"))), (os.listdir(prep_dir) if os.path.isdir(prep_dir) else None, os.listdir(b)))
    if gpu:
        ptxt = read(os.path.join(prep_dir, "prepared.txt")).decode("utf-8", "replace")
        t.check("rr-prepare: prepared.txt keeps the game's program name, the driver's version, the runtime's SHA-256, the passes",
                ptxt.lower().startswith(os.path.basename(exe).lower() + "\t" + rp.get("driver", "?").lower() + "\t") and ptxt.rstrip().endswith("\t1"), ptxt)
        plog = read(os.path.join(prep_dir, "prepare.log")).decode("utf-8", "replace")
        t.check("rr-prepare: the runtime ran as the game's program name (the driver's cache key)",
                ("as " + os.path.join(prep_dir, "run", os.path.basename(exe))).lower() in plog.lower() and "prepare: done" in plog, plog[-1500:])
        rc = t.run("rr-prepare", exe, "--passes=1")
        rp2 = t.j.get("rr_prepare", {})
        t.check("rr-prepare again: the driver had every kernel (quick: under 15 s, was %s ms)" % rp.get("ms"),
                rc == 0 and rp2.get("result") == 0 and rp2.get("ms", 99999) < 15000, rp2)
        rc = t.run("rr-prepare", exe, "--check", "--passes=2")
        t.check("rr-prepare --check for the other modes (pass 2): not prepared (default mode only)", rc == 0 and t.j.get("rr_prepare", {}).get("ready") is False, t.j)
        write(os.path.join(prep_dir, "prepared.txt"), ptxt.replace("\t" + rp.get("driver", "?") + "\t", "\t1.2.3.4\t", 1).encode("utf-8"))
        rc = t.run("rr-prepare", exe, "--check")
        t.check("rr-prepare --check after a driver change: not prepared, the reason in words", rc == 0 and t.j.get("rr_prepare", {}).get("ready") is False and
                "driver changed" in t.j.get("rr_prepare", {}).get("not_ready_why", ""), t.j)
        # release 0.6.0 polish (owner 2026-11-02): the launcher's start prepares again after a driver / WrongGPU update
        rc = t.run("rr-recheck", exe)
        rq = t.j.get("rr_recheck", {})
        t.check("rr-recheck after a driver change: stale (driver-changed), the reason names the old driver, nothing run without --run",
                rc == 0 and rq.get("stale") is True and rq.get("state_before") == "driver-changed" and "1.2.3.4" in rq.get("why", "") and rq.get("result") == -1, t.j)
        rc = t.run("rr-recheck", exe, "--run")
        rq = t.j.get("rr_recheck", {})
        p2 = read(os.path.join(prep_dir, "prepared.txt")).decode("utf-8", "replace")
        t.check("rr-recheck --run: prepared again (result 0), ready, prepared.txt names this driver again",
                rc == 0 and rq.get("result") == 0 and rq.get("state") == "ready" and
                p2.lower().startswith(os.path.basename(exe).lower() + "\t" + rp.get("driver", "?").lower() + "\t"), (t.j, p2))
        cols = p2.splitlines()[0].split("\t")
        if len(cols) >= 4:
            cols[2] = "0" * 64  # another WrongGPU runtime (an update): its SHA-256 differs
            write(os.path.join(prep_dir, "prepared.txt"), ("\t".join(cols) + "\r\n").encode("utf-8"))
        rc = t.run("rr-recheck", exe)
        rq = t.j.get("rr_recheck", {})
        t.check("rr-recheck after a WrongGPU runtime change: stale (runtime-changed)",
                rc == 0 and rq.get("stale") is True and rq.get("state_before") == "runtime-changed", t.j)
        write(os.path.join(prep_dir, "prepared.txt"), p2.encode("utf-8"))
        rc = t.run("rr-recheck", exe)
        rq = t.j.get("rr_recheck", {})
        t.check("rr-recheck with the same driver and runtime: not stale (ready), nothing run",
                rc == 0 and rq.get("stale") is False and rq.get("state_before") == "ready" and rq.get("result") == -1, t.j)
