import numpy as np
import pandas as pd
import pytest
import xarray as xr

from romscommons import grid as G
from romscommons import tides as T


def test_identify():
    assert T.identify([12.4206012, 12.0, 23.93447213, 25.81933871]) == ["M2", "S2", "K1", "O1"]
    assert T.identify([12.4206]) == ["M2"]
    assert T.identify([13.0]) == [None]


def test_ap2ep_ep2ap_round_trip():
    rng = np.random.default_rng(1)
    n = 500
    Au, Av = rng.uniform(0.01, 2, n), rng.uniform(0.01, 2, n)
    PHIu, PHIv = rng.uniform(0, 360, n), rng.uniform(0, 360, n)
    Cmax, Cmin, Cang, Cph = T.ap2ep(Au, PHIu, Av, PHIv)
    assert np.all(Cmax >= np.abs(Cmin)) and np.all((Cang >= 0) & (Cang < 180)) and np.all((Cph >= 0) & (Cph < 360))
    Au2, PHIu2, Av2, PHIv2 = T.ep2ap(Cmax, Cmin, Cang, Cph)
    np.testing.assert_allclose(Au2, Au, atol=1e-8)
    np.testing.assert_allclose(Av2, Av, atol=1e-8)
    dphi = lambda a, b: (a - b + 180) % 360 - 180  # noqa: E731
    np.testing.assert_allclose(dphi(PHIu2, PHIu), 0, atol=1e-8)
    np.testing.assert_allclose(dphi(PHIv2, PHIv), 0, atol=1e-8)


def test_ap2ep_known_cases():
    cmax, cmin, ang, ph = T.ap2ep(1.0, 0.0, 0.0, 0.0)  # rectilinear east-west
    assert (cmax, cmin, ang, ph) == pytest.approx((1, 0, 0, 0))
    cmax, cmin, ang, ph = T.ap2ep(1.0, 0.0, 1.0, 90.0)  # u = cos, v = sin: counter-clockwise circle
    assert cmax == pytest.approx(1) and cmin == pytest.approx(1)
    cmax, cmin, ang, ph = T.ap2ep(0.0, 0.0, 0.5, 30.0)  # rectilinear north-south
    assert ang == pytest.approx(90) and cmax == pytest.approx(0.5) and ph == pytest.approx(30)


def _da(x):
    return xr.DataArray(np.asarray(x, dtype=float), dims=("tide_period",))


def test_reconstruct_uv_matches_direct_and_roms_form():
    """Ellipses from ap2ep, reconstructed with the set_tides.F formula, give back u and v."""
    rng = np.random.default_rng(2)
    per = np.array([12.4206012, 12.0, 23.93447213])
    Au, Av = rng.uniform(0.1, 1, 3), rng.uniform(0.1, 1, 3)
    PHIu, PHIv = rng.uniform(0, 360, 3), rng.uniform(0, 360, 3)
    Cmax, Cmin, Cang, Cph = T.ap2ep(Au, PHIu, Av, PHIv)
    start = pd.Timestamp("2026-01-01")
    t = pd.date_range(start, periods=73, freq="1h").values
    u, v = T.reconstruct_uv(_da(Cmax), _da(Cmin), _da(Cang), _da(Cph), per, t, start)
    secs = (t - np.datetime64(start)).astype("timedelta64[s]").astype(float)
    w = 2 * np.pi / (per * 3600)
    u_direct = (Au[:, None] * np.cos(w[:, None] * secs - np.deg2rad(PHIu)[:, None])).sum(0)
    v_direct = (Av[:, None] * np.cos(w[:, None] * secs - np.deg2rad(PHIv)[:, None])).sum(0)
    np.testing.assert_allclose(u, u_direct, atol=1e-10)
    np.testing.assert_allclose(v, v_direct, atol=1e-10)
    ang = 0.3  # grid angle: ROMS returns grid-frame components
    ug, vg = T.reconstruct_uv(_da(Cmax), _da(Cmin), _da(Cang), _da(Cph), per, t, start, angle=ang)
    eg, ng = G.to_grid(u_direct, v_direct, ang)
    np.testing.assert_allclose(ug, eg, atol=1e-10)
    np.testing.assert_allclose(vg, ng, atol=1e-10)


def test_reconstruct_ssh_phase_convention():
    start = pd.Timestamp("2026-01-01")
    per = T.CONSTITUENTS["M2"]
    t = np.array([start, start + pd.Timedelta(hours=per / 4)], dtype="datetime64[ns]")
    z = T.reconstruct_ssh(_da([1.0]), _da([0.0]), [per], t, start)
    np.testing.assert_allclose(z, [1.0, 0.0], atol=1e-12)
    z = T.reconstruct_ssh(_da([1.0]), _da([90.0]), [per], t, start)  # high water a quarter period later
    np.testing.assert_allclose(z, [0.0, 1.0], atol=1e-12)


def test_reconstruct_on_synthetic_file(synthetic):
    from romscommons import io

    tide = io.open_nc(synthetic["tide"])
    start = pd.Timestamp("2026-01-01")
    t = pd.date_range(start, periods=25, freq="1h").values
    z = T.reconstruct_ssh(tide.tide_Eamp, tide.tide_Ephase, tide.tide_period, t, start)
    assert z.dims == ("time", "eta_rho", "xi_rho")
    assert float(abs(z).max()) <= float(tide.tide_Eamp.sum("tide_period").max()) + 1e-9
