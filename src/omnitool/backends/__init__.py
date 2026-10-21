"""Execution backends and their registry (spec §5, ADR 0004, ADR 0005).

:func:`available` lists the backends that are implemented in this checkout;
the test fixture and the public API consult it, so an unimplemented backend
is neither parametrised nor silently accepted. ``KNOWN`` is the full set of
names the spec reserves, implemented or not.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

__all__ = ["KNOWN", "available", "get"]

KNOWN: tuple[str, ...] = ("cpu", "cuda")
"""Backend names the spec reserves. ``cudasim`` is deferred by ADR 0005."""

# name -> (module, attribute) of the ``solve_batch_<name>`` entry point.
_ENTRY_POINTS: dict[str, tuple[str, str]] = {
    "cpu": ("omnitool.backends.cpu", "solve_batch_cpu"),
    # "cuda": ("omnitool.backends.cuda", "solve_batch_cuda"),  # registered with the kernel
}


def available() -> tuple[str, ...]:
    """Backends implemented in this checkout, in ``KNOWN`` order.

    A backend is available when its entry point imports; ``cuda`` will
    additionally require CuPy and a device at call time, which the test
    fixture checks separately.
    """
    out = []
    for name in KNOWN:
        if name in _ENTRY_POINTS:
            out.append(name)
    return tuple(out)


def get(name: str) -> Callable:
    """Return the ``solve_batch_<name>`` entry point or raise.

    ``ValueError`` for a name outside :data:`KNOWN`; ``NotImplementedError``
    for a known backend that is not implemented in this checkout.
    """
    if name not in KNOWN:
        raise ValueError(f"backend must be one of {KNOWN}, got {name!r}")
    if name not in _ENTRY_POINTS:
        raise NotImplementedError(
            f"backend {name!r} is not implemented yet; available: {available()}"
        )
    module, attr = _ENTRY_POINTS[name]
    return getattr(import_module(module), attr)
