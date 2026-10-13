"""Each injected defect (BUILD_SPEC §6 table) -> expected status, via the same check functions
the notebooks call. Rows whose notebook-level check is built in a later phase are marked
``pending`` and turned on in that phase.
"""
import numpy as np
import pytest

from romscommons import checks as C
from romscommons import diagnostics as D
from romscommons import grid as G
from romscommons import io
from romscommons.checks import Status

RUN = dict(TIME_REF=20000101.0, DSTART=9497.0, NTIMES=4800, DT=180.0)


def pending(phase, what):
    return pytest.param(what, marks=pytest.mark.skip(reason=f"Phase {phase}: {what}"), id=what.split(" ->")[0])


# ---------------------------------------------------------------- good case: no defect fires
def test_good_case_generic_checks_pass(synthetic):
    g = G.load_grid(synthetic["grid"])
    ini = io.open_nc(synthetic["ini"]).isel(ocean_time=0)
    assert C.upper_bound("rx0", float(D.rx0(g).to_array().max()), "rx0_warn").status == Status.PASS
    assert C.finite_at_wet(ini.temp, g.mask_rho).status == Status.PASS
    frc = io.open_nc(synthetic["frc_native"])
    start, end = io.run_window(RUN)
    for t in ("wind_time", "pair_time", "qair_time", "rain_time", "srf_time", "lrf_time"):
        assert C.time_gaps(frc[t]).status == Status.PASS
        assert C.time_covers(frc[t], start, end).status == Status.PASS


# ---------------------------------------------------------------- defects detectable now
def test_grid_spike_rx0_warn(defects):
    g = G.load_grid(defects["grid"])
    c = C.upper_bound("rx0", float(D.rx0(g).to_array().max()), "rx0_warn")
    assert c.status == Status.WARN


def test_ini_nan_wet_points_fail(defects):
    g = G.load_grid(defects["grid"])
    ini = io.open_nc(defects["ini"]).isel(ocean_time=0)
    c = C.finite_at_wet(ini.temp, g.mask_rho)
    assert c.status == Status.FAIL and c.value == 3


def test_frc_12h_gap_warn(defects):
    frc = io.open_nc(defects["frc_native"])
    c = C.time_gaps(frc["wind_time"])
    assert c.status == Status.WARN and c.value == pytest.approx(12.0)  # one 12 h step


def test_frc_ends_early_fail(defects):
    frc = io.open_nc(defects["frc_native"])
    c = C.time_covers(frc["srf_time"], *io.run_window(RUN))
    assert c.status == Status.FAIL and "before run end" in c.message


def test_frc_coarse_lon_convention_differs(defects):
    """Raw ingredient for the nb02 FAIL: forcing is -180..180 while the ROMS grid is 0..360."""
    g = G.load_grid(defects["grid"])
    co = io.open_nc(defects["frc_coarse"])
    assert G.lon_convention(g.lon_rho) == "0-360"
    assert G.lon_convention(co.lon) == "-180-180"


# ---------------------------------------------------------------- defects checked in later phases
@pytest.mark.parametrize("row", [
    pending(1, "ini ubar = 1.5 x depth-avg u -> WARN ubar_rms"),
    pending(1, "ini THETA_S in file != ocean.in -> FAIL vertical params"),
    pending(1, "ini temperature inversion -> WARN N2"),
    pending(2, "frc Pair in Pa -> FAIL units"),
    pending(2, "frc Tair in K -> FAIL units"),
    pending(2, "frc_coarse longitudes -180..180 -> FAIL lon convention"),
    pending(3, "tide phases in radians -> WARN"),
    pending(3, "tide constituent with Cmin > Cmax -> FAIL"),
    pending(4, "nud coefficient 1/(0.001 day) -> FAIL coef*DT"),
    pending(5, "bry +0.2 m/s ubar west only -> WARN ssh_drift"),
    pending(5, "bry temp +2 degC vs ini at t0 -> WARN icbc_temp"),
    pending(5, "bry M2 signal in zeta -> WARN bry_tide_m2"),
])
def test_pending(row):  # pragma: no cover - placeholders
    pass


def test_defect_ingredients_present(defects):
    """The generator really wrote the later-phase defects (guards against silent regressions)."""
    ini = io.open_nc(defects["ini"])
    assert float(ini.theta_s) == 5.0
    frc = io.open_nc(defects["frc_native"])
    assert float(frc.Pair.median()) > 5000
    assert float(io.open_nc(defects["frc_coarse"]).Tair.median()) > 150
    tide = io.open_nc(defects["tide"])
    assert float(tide.tide_Ephase.max()) <= 2 * np.pi
    assert bool((abs(tide.tide_Cmin) > tide.tide_Cmax).any())
    nud = io.open_nc(defects["nud"])
    assert float(nud.temp_NudgeCoef.max()) * 180.0 / 86400 >= 1.0
