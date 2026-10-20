"""Plain-NumPy reference integrators, written from the primary sources.

This package is the oracle of the verification layer (ADR 0003). It may
import ``numpy`` and ``omnitool.tableaus`` only; a test asserts that it never
imports ``omnitool.backends``.
"""
