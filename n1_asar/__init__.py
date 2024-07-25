"""Top-level package for xarray-safe-asar."""

from importlib.metadata import version

import n1_asar  # noqa: F401

try:
    __version__ = version("xarray-safe-asar")
except Exception:
    __version__ = "999"
