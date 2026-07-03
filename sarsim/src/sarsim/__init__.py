"""sarsim — SAR phase-history simulator (Willis et al. 2020 reimplementation).

Import side effect: on macOS, point Dr.Jit at Homebrew's libLLVM before any
mitsuba import (must happen before the JIT initializes its thread state).
Harmless when mitsuba is unused or the path is absent.
"""
import os
import sys

if sys.platform == 'darwin':
    os.environ.setdefault(
        'DRJIT_LIBLLVM_PATH', '/opt/homebrew/opt/llvm/lib/libLLVM.dylib'
    )

C0 = 299_792_458.0  # speed of light, m/s (float64 everywhere host-side)
