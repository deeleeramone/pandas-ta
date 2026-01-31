# -*- coding: utf-8 -*-
"""
Compatibility layer for optional Numba support.

This module provides a fallback njit decorator when Numba is not available,
allowing pandas-ta to work without Numba (with reduced performance).
"""

__all__ = ["njit", "HAS_NUMBA"]

try:
    from numba import njit as _njit

    HAS_NUMBA = True

    def njit(*args, **kwargs):
        """Wrapper around numba.njit that preserves all functionality."""
        # Handle both @njit and @njit(...) usage
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Called as @njit without arguments
            return _njit(args[0])
        else:
            # Called as @njit(...) with arguments
            return _njit(*args, **kwargs)

except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """
        No-op decorator when Numba is not available.
        Functions will run as regular Python/NumPy code.
        """
        # Handle both @njit and @njit(...) usage
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Called as @njit without arguments
            return args[0]
        else:
            # Called as @njit(...) with arguments - return decorator
            def decorator(func):
                return func

            return decorator
