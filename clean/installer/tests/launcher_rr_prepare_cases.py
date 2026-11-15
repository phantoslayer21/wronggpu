# SPDX-License-Identifier: MIT
"""rr-prepare and rr-recheck (RR's kernels prepared for a game ahead of its start), for run_launcher_tests.py's RR section
(which calls run() in its place; this file keeps that suite below the 256 KB source limit, as launcher_edition_cases.py
does). The section is moved unchanged, except that its GPU checks skip, with the reason, when the installed WrongGPU runtime
is the suite's own stand-in (a fresh clone's test payload), which cannot prepare anything. Written for this project."""
import os

import launcher_build_facts as facts
from run_installer_tests import read, write


def run(t, exe, b):
    """rr-prepare --check, rr-recheck, rr-prepare (the helper under the game's program name) and, with an RDNA 4 card and a
    real runtime, the prepared state across a driver or runtime change. exe: the RR game's exe, b: its folder."""
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
    rc = t.run("rr-prepare", exe, "--passes=1")
    rp = t.j.get("rr_prepare", {})
    gpu = rp.get("result") != 2  # 2: no RDNA 4 graphics card on this PC (nothing to prepare)
    if gpu and facts.stand_in_runtime(b):  # a fresh clone's suite payload: its runtime has no kernels
        print("  skipped: rr-prepare's GPU checks (this payload's WrongGPU runtime is the suite's stand-in, which cannot prepare "
              "RR's kernels: they need a payload with the real runtime); result %s" % rp.get("result"))
        gpu = None
    if gpu is not None:
        t.check("rr-prepare: the helper prepared the default mode's kernels (result 0, ready, every kernel reported), or no RDNA 4 card",
                rc == 0 and (not gpu or (rp.get("result") == 0 and rp.get("ready") is True and rp.get("kernels", 0) >= 50)), rp)
    t.check("rr-prepare: the helper copy is gone, nothing new in the game's folder (the runtime log went to the data folder)",
            not os.listdir(os.path.join(prep_dir, "run")) and (read(glog) if os.path.exists(glog) else None) == glog_before and
            (not gpu or os.path.isfile(os.path.join(prep_dir, "prepare.log"))), (os.listdir(prep_dir), os.listdir(b)))
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
