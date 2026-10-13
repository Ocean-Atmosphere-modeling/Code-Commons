"""romscommons — small, tested helpers for inspecting ROMS input files.

Modules: ``io`` (config, NetCDF, ROMS time, ocean.in), ``grid`` (C-grid), ``vertical``
(s-coordinate), ``diagnostics`` (Courant, stiffness, N², transports, bulk estimates),
``checks`` (PASS/WARN/FAIL/INFO), ``tides``, ``plotting``. ``plotting`` is imported on
demand so the rest works without a display backend.
"""
from . import checks, diagnostics, grid, io, tides, vertical  # noqa: F401
from .checks import THRESHOLDS, Check, Report, Status  # noqa: F401
from .io import load_config, open_nc  # noqa: F401

__version__ = "0.1.0"
