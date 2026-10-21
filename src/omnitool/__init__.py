"""omnitool: a common GPU ensemble ODE core.

Integrates many independent members of an ODE ensemble with per-member
adaptive stepping, dense output at shared save times, FP32 and FP64, and
status codes. Physics-specific packages attach through optional adapters.

See ``docs/PHASE0_SPEC.md`` for the first vertical slice and ``docs/adr/``
for the design decisions.
"""

from omnitool.api import solve_batch
from omnitool.problem import BatchResult, Problem
from omnitool.status import Status

__version__ = "0.1.0"

__all__ = ["BatchResult", "Problem", "Status", "__version__", "solve_batch"]
