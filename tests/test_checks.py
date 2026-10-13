import numpy as np
import pandas as pd
import pytest
import xarray as xr

from romscommons import checks as C
from romscommons.checks import Check, Report, Status

from test_grid import uniform_grid


def test_thresholds_have_value_and_source():
    for key, entry in C.THRESHOLDS.items():
        assert set(entry) == {"value", "source"}, key
        assert entry["source"], key
    assert C.threshold("rx0_warn") == 0.2 and C.threshold("nud_coefdt_fail") == 1.0


def test_report_worst_summary_frame():
    r = Report("demo")
    assert r.worst() == Status.PASS
    r.add(Check("a", Status.PASS, 1, 2, "s"))
    r.add([Check("b", "INFO"), Check("c", Status.WARN, 3.0, 2.0, "s", "too big")])
    assert r.worst() == Status.WARN
    r.add(Check("d", Status.FAIL))
    assert r.worst() == Status.FAIL and r["c"].status == Status.WARN
    assert "1 FAIL" in r.summary() and "worst: FAIL" in r.summary()
    df = r.to_frame()
    assert list(df.columns) == ["check", "status", "value", "threshold", "source", "message"]
    assert len(df) == 4 and df.loc[2, "status"] == "WARN"
    assert r.display().to_html()  # Styler renders
    assert [c.name for c in r.find("^[ab]$")] == ["a", "b"]


def test_info_never_outranks_pass():
    r = Report()
    r.add([Check("a", Status.PASS), Check("b", Status.INFO)])
    assert r.worst() == Status.PASS


def test_required_vars():
    ds = xr.Dataset({"zeta": ("x", [0.0]), "temp": ("x", [1.0])})
    assert C.required_vars(ds, ["zeta", "temp"]).status == Status.PASS
    c = C.required_vars(ds, ["zeta", "salt"])
    assert c.status == Status.FAIL and c.value == ["salt"]


def test_dims_match_grid():
    g = uniform_grid(M=6, L=8)
    good = xr.DataArray(np.zeros((1, 3, 6, 7)), dims=("ocean_time", "s_rho", "eta_u", "xi_u"))
    assert C.dims_match_grid(good, g, "u", N=3, vertical="s_rho").status == Status.PASS
    bad = xr.DataArray(np.zeros((1, 3, 6, 8)), dims=("ocean_time", "s_rho", "eta_u", "xi_u"))
    assert C.dims_match_grid(bad, g, "u", N=3, vertical="s_rho").status == Status.FAIL
    wrong_n = xr.DataArray(np.zeros((1, 4, 6, 7)), dims=("ocean_time", "s_rho", "eta_u", "xi_u"))
    assert C.dims_match_grid(wrong_n, g, "u", N=3, vertical="s_rho").status == Status.FAIL


def test_finite_checks():
    m = xr.DataArray(np.array([[1, 1, 0]]), dims=("eta_rho", "xi_rho"))
    a = xr.DataArray(np.array([[1.0, 2.0, np.nan]]), dims=("eta_rho", "xi_rho"), name="t")
    assert C.finite_at_wet(a, m).status == Status.PASS
    assert C.finite_at_land(a, m).status == Status.WARN
    b = a.copy()
    b[0, 0] = np.inf
    c = C.finite_at_wet(b, m)
    assert c.status == Status.FAIL and c.value == 1


def test_in_range_and_max_abs():
    m = xr.DataArray(np.array([1, 1, 0]), dims="x")
    t = xr.DataArray(np.array([5.0, 20.0, 99.0]), dims="x", name="temp")
    assert C.in_range(t, "temp_range", m).status == Status.PASS  # 99 is on land
    assert C.in_range(t, "temp_range").status == Status.WARN
    assert C.in_range(t, (0.0, 10.0), m).status == Status.WARN
    z = xr.DataArray(np.array([-3.5, 0.0, 0.0]), dims="x", name="zeta")
    c = C.max_abs(z, "zeta_max", m)
    assert c.status == Status.WARN and c.value == 3.5


@pytest.mark.parametrize("units, expected, status", [
    ("meter second-1", "m/s", Status.PASS), ("m s-1", ["meter second-1"], Status.PASS),
    ("Pa", "millibar", Status.WARN), ("hPa", "millibar", Status.PASS), ("Celsius", "degC", Status.PASS),
    ("K", "Celsius", Status.WARN), ("percentage", "%", Status.PASS), ("day-1", "1/day", Status.PASS),
])
def test_units_match(units, expected, status):
    da = xr.DataArray([1.0], dims="x", name="v", attrs={"units": units})
    assert C.units_match(da, expected).status == status


def test_units_missing_is_info():
    assert C.units_match(xr.DataArray([1.0], dims="x"), "m").status == Status.INFO


def _times(*hours):
    return xr.DataArray(pd.Timestamp("2026-01-01").to_datetime64() + np.array(hours, dtype="timedelta64[h]"),
                        dims="t", name="t")


def test_time_monotonic():
    assert C.time_monotonic(_times(0, 1, 2)).status == Status.PASS
    c = C.time_monotonic(_times(0, 1, 1, 2))
    assert c.status == Status.FAIL and c.value["duplicates"] == 1
    assert C.time_monotonic(_times(0, 2, 1)).status == Status.FAIL


def test_time_gaps():
    assert C.time_gaps(_times(*range(10))).status == Status.PASS
    c = C.time_gaps(_times(0, 1, 2, 3, 15, 16, 17))
    assert c.status == Status.WARN and c.value == 12.0
    assert C.time_gaps(_times(0)).status == Status.INFO


def test_time_covers_edges():
    start, end = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01 10:00")
    assert C.time_covers(_times(*range(11)), start, end).status == Status.PASS  # exactly at both ends
    assert C.time_covers(_times(*range(10)), start, end).status == Status.FAIL  # ends 1 h early
    late = C.time_covers(_times(*range(1, 12)), start, end)
    assert late.status == Status.FAIL and "after run start" in late.message
    t = _times(0, 1)
    t.attrs["cycle_length"] = 365.25
    assert C.time_covers(t, start, end).status == Status.INFO


def test_upper_bound_and_compare():
    assert C.upper_bound("rx0", 0.1, "rx0_warn").status == Status.PASS
    c = C.upper_bound("rx0", 0.3, "rx0_warn")
    assert c.status == Status.WARN and c.threshold == 0.2 and "Beckmann" in c.source
    assert C.upper_bound("x", np.nan, "rx0_warn").status == Status.INFO
    assert C.compare("theta_s", 7.0, 7.0).status == Status.PASS
    assert C.compare("theta_s", 5.0, 7.0).status == Status.FAIL
    assert C.compare("Cs", np.zeros(3), np.full(3, 1e-7), atol=1e-6).status == Status.PASS
    assert C.compare("Cs", np.zeros(3), np.zeros(4)).status == Status.FAIL
    assert C.compare("name", "a", "b", bad_status=Status.WARN).status == Status.WARN
