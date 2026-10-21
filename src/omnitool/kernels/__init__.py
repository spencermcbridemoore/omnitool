"""CUDA C kernel templates (``.cu``) and the Python templating that inserts
the problem-specific right-hand side (ADR 0004). The device source is plain
C, compiled only through ``omnitool.backends.nvrtc``.
"""
