# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): the per-call configuration values that are settled, as a CPU reference.

Written for this project. None of these values changes the networks or their weights (one weight set serves every
configuration); they decide the intermediate time and which stages run.

- Multi-frame: N generated frames per real frame (1 by default, at most a cap), index i in 1..N. The intermediate
  time is t = i / (N + 1), one single-precision division. "Multi-frame active" is N > 1; "last index" is i equal to
  the maximum index.
- UI recomposition (FG preset A = off, B = on): set by the application; a profile value can force it.

OPEN (clean/fg/FORMATS.md): the cap on N; the one further mode that also accepts i = N + 1, and what it does; which
maximum "last index" compares with (N, or the cap); the private override that can turn UI recomposition off. Where
these matter, the caller passes them explicitly; nothing here picks them.
"""
import numpy as np

PRESET_A = "A"   # UI recomposition off
PRESET_B = "B"   # UI recomposition on

# Profile values that force UI recomposition (True = on, preset B; False = off, preset A). Any other value leaves the
# application's setting as it is.
PROFILE_FORCE = {0x1: False, 0x2: True, 0xFFFFFE: False, 0xFFFFFF: True}


def check_multi_frame(n, i, *, n_max, extra_index=False):
    """Validates a multi-frame count n and index i. n_max: the cap on n (OPEN, so required). extra_index: the caller
    states that the one further mode that also accepts i = n + 1 is in use (what that mode is, is OPEN)."""
    for name, v in (("n", n), ("i", i), ("n_max", n_max)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise TypeError("%s must be an integer" % name)
    if n_max < 1:
        raise ValueError("n_max must be at least 1")
    if not 1 <= n <= n_max:
        raise ValueError("multi-frame count %d outside 1..%d" % (n, n_max))
    top = n + 1 if extra_index else n
    if not 1 <= i <= top:
        raise ValueError("multi-frame index %d outside 1..%d" % (i, top))


def intermediate_time(n, i):
    """t = i / (n + 1) as one IEEE single-precision division of the two counts (each exact in float32 here).
    Returns a numpy float32."""
    if not (0 <= i < 1 << 24 and 0 <= n + 1 < 1 << 24):
        raise ValueError("counts must be exact in float32")
    return np.float32(i) / np.float32(n + 1)


def multi_frame_active(n):
    return n > 1


def last_index(i, maximum):
    """True when i equals the maximum index. Which maximum the original compares with (n or the cap) is OPEN, so the
    caller passes it."""
    return i == maximum


def ui_recomposition(app_setting, profile_value=None):
    """Whether UI recomposition is on (preset B) or off (preset A): the application's setting, unless a profile value in
    PROFILE_FORCE forces it. profile_value None: no profile value. (A private override that can turn it off is OPEN and
    not modelled.)"""
    if not isinstance(app_setting, bool):
        raise TypeError("app_setting must be True or False")
    if profile_value is not None and profile_value in PROFILE_FORCE:
        return PROFILE_FORCE[profile_value]
    return app_setting


def preset(ui_recomp_on):
    return PRESET_B if ui_recomp_on else PRESET_A
