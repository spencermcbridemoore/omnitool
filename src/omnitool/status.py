"""Per-member status codes (``docs/PHASE0_SPEC.md`` §6).

The numbers are defined once more in ``src/omnitool/status.h`` for the device
code; a test asserts the two agree. The NumPy reference carries its own
integer constants because it may import nothing but ``numpy`` and the
tableau module; a test asserts those agree with this enum as well.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["Status"]


class Status(IntEnum):
    NOT_RUN = -1
    """Initial value; never returned by a finished run."""
    SUCCESS = 0
    """Reached ``t_end``."""
    MAX_STEPS = 1
    """``n_accepted + n_rejected == max_steps``."""
    STEP_TOO_SMALL = 2
    """``|h|`` fell below ``16 * eps * max(|t|, 1)``."""
    NON_FINITE = 3
    """NaN or Inf in ``y`` or in the error norm."""
