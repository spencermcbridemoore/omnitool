"""omnitool: a common GPU ensemble ODE core.

Integrates many independent members of an ODE ensemble with per-member
adaptive stepping, dense output at shared save times, FP32 and FP64, and
status codes. Physics-specific packages attach through optional adapters.

See ``docs/PHASE0_SPEC.md`` for the first vertical slice and ``docs/adr/``
for the design decisions.
"""

__version__ = "0.0.0"

__all__ = ["__version__"]
