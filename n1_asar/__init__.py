"""Top-level package for xarray-n1-asar."""

from importlib.metadata import version

import n1_asar  # noqa: F401

try:
    __version__ = version("xarray-n1-asar")
except Exception:
    __version__ = "999"
