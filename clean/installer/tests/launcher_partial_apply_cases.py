# SPDX-License-Identifier: GPL-3.0-only
"""A partial apply, for run_launcher_tests.py's RR group, which calls run() in its place with a game that has RR installed
(this file keeps that suite below the 256 KB source limit, as launcher_state_cases.py does).

The RR write fails (the test seam WRONGGPU_TEST_FAIL_RR_STATE of a WRONGGPU_TESTING build) after the upscaler settings were
saved: the apply is refused, and its message says exactly what was applied (the upscaler settings) and what was not (the RR
settings), never "not applied" for what was. The RR settings stay as they were. The same apply without the fault succeeds.
Written for this project."""


def run(t, exe, b, rr_ini, state_kv):
    rr_before, sg_before = rr_ini(b), state_kv(b).get("sparkle_guard")
    flip = "off" if sg_before == "on" else "on"
    t.env["WRONGGPU_TEST_FAIL_RR_STATE"] = "1"
    try:
        rc = t.run("set", exe, "sparkle_guard=" + flip, "rr_mode=fast")
    finally:
        del t.env["WRONGGPU_TEST_FAIL_RR_STATE"]
    err = t.j.get("error") or t.j.get("message", "")  # (an operation's refusal is its message)
    t.check("a partial apply (the RR write fails after the upscaler settings were saved): refused (exit 2); the message lists the "
            "upscaler settings as applied and the RR settings as not applied; the RR settings unchanged",
            rc == 2 and err.startswith("Settings applied in part.") and "Applied: the upscaler settings" in err and
            "Not applied: the Ray Reconstruction settings" in err and "Settings not applied" not in err and
            state_kv(b).get("sparkle_guard") == flip and rr_ini(b) == rr_before, (rc, err, rr_ini(b)))
    rc = t.run("set", exe, "sparkle_guard=" + (sg_before or "off"), "rr_mode=" + rr_before.get("rr_mode", "accurate"))
    t.check("...the same apply without the fault: exit 0", rc == 0, (rc, t.j.get("error")))
