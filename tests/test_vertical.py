import numpy as np
import pytest
import xarray as xr

from romscommons import vertical as V

# Parameter ranges per option (set_scoord.F comments): 1: 0 < theta_s <= 8, 0 <= theta_b <= 1;
# 3 (Geyer): exponents ~0.5-1 (theta_b = 0 makes Cbot identically 0); 2, 4, 5: theta_s <= 10, theta_b <= 3.
OPTIONS = ([(1, ts, tb) for ts, tb in ((7.0, 1.0), (3.0, 0.4), (0.5, 0.0))]
           + [(3, ts, tb) for ts, tb in ((0.65, 0.58), (1.0, 1.0))]
           + [(vs, ts, tb) for vs in (2, 4, 5) for ts, tb in ((7.0, 2.0), (3.0, 0.4), (0.5, 0.0))])


def test_s_coord_uniform():
    np.testing.assert_allclose(V.s_coord(4, "w"), [-1, -0.75, -0.5, -0.25, 0])
    np.testing.assert_allclose(V.s_coord(4, "rho"), [-0.875, -0.625, -0.375, -0.125])


def test_s_coord_vstretching5_matches_source_formula():
    s = V.s_coord(40, "w", 5)
    assert s[0] == -1 and s[-1] == 0
    assert s[1] == pytest.approx(-0.96)  # -(1-80+1+1600-40)/1560 - 0.01
    assert np.all(np.diff(s) > 0) and np.all(np.diff(V.s_coord(40, "rho", 5)) > 0)


@pytest.mark.parametrize("vs, ts, tb", OPTIONS)
def test_cs_curves_properties(vs, ts, tb):
    cs = V.cs_curves(12, vs, ts, tb)
    assert cs["Cs_w"][0] == -1 and cs["Cs_w"][-1] == pytest.approx(0, abs=1e-14)
    for c in (cs["Cs_w"], cs["Cs_r"]):
        assert np.all(np.diff(c) > 0)
        assert np.all((c >= -1 - 1e-12) & (c <= 1e-12))
    assert len(cs["Cs_r"]) == 12 and len(cs["Cs_w"]) == 13


def test_stretching_limits():
    s = V.s_coord(10, "rho")
    np.testing.assert_allclose(V.stretching(1, 0.0, 0.0, s), s)
    np.testing.assert_allclose(V.stretching(4, 0.0, 0.0, s), -s ** 2)
    with pytest.raises(ValueError):
        V.stretching(6, 1, 1, s)


def test_critical_depth():
    assert V.critical_depth(2, 50.0) == 50.0
    assert V.critical_depth(1, 50.0, hmin=10.0) == 10.0
    assert V.critical_depth(1, 5.0, hmin=10.0) == 5.0
    assert V.critical_depth(1, 50.0, hmin=-2.0, wet_dry=True) == 0.0


@pytest.mark.parametrize("vt", [1, 2])
@pytest.mark.parametrize("vs", [2, 4, 5])
def test_depth_identities(vt, vs):
    h = xr.DataArray(np.array([[10.0, 50.0, 400.0], [2000.0, 30.0, 5.0]]), dims=("eta_rho", "xi_rho"))
    zeta = xr.DataArray(np.array([[0.3, -0.2, 0.0], [1.0, -0.5, 0.1]]), dims=("eta_rho", "xi_rho"))
    p = dict(N=15, Vtransform=vt, Vstretching=vs, theta_s=6.0, theta_b=2.0, Tcline=20.0)
    d = V.depths(h, zeta, **p)
    np.testing.assert_allclose(d.z_w.isel(s_w=0), -h, atol=1e-10)
    np.testing.assert_allclose(d.z_w.isel(s_w=-1), zeta, atol=1e-10)
    assert float(d.Hz.min()) > 0
    np.testing.assert_allclose(d.Hz.sum("s_rho"), h + zeta, atol=1e-10)
    assert d.z_r.dims == ("s_rho", "eta_rho", "xi_rho")
    assert np.all((d.z_r.values > d.z_w.values[:-1]) & (d.z_r.values < d.z_w.values[1:]))


def test_depths_with_time_dim():
    h = xr.DataArray(np.full((2, 3), 100.0), dims=("eta_rho", "xi_rho"))
    zeta = xr.DataArray(np.zeros((4, 2, 3)), dims=("ocean_time", "eta_rho", "xi_rho"))
    z = V.z_w(h, zeta, N=5, Vtransform=2, Vstretching=4, theta_s=5, theta_b=1, Tcline=10)
    assert z.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")


def test_depth_average():
    h = xr.DataArray(np.full((2, 2), 100.0), dims=("eta_rho", "xi_rho"))
    d = V.depths(h, 0.0, N=10, Vtransform=2, Vstretching=4, theta_s=5, theta_b=1, Tcline=10)
    const = xr.ones_like(d.z_r) * 0.3
    np.testing.assert_allclose(V.depth_average(const, d.Hz), 0.3)
    zmid = 0.5 * (d.z_w.isel(s_w=slice(1, None)).values + d.z_w.isel(s_w=slice(None, -1)).values)
    zmid = xr.DataArray(zmid, dims=d.Hz.dims)
    np.testing.assert_allclose(V.depth_average(zmid, d.Hz), -50.0, atol=1e-10)  # exact for linear in z
    shifted = const.assign_coords(s_rho=d.z_r["s_rho"].values + 1e-9)  # labels differ: no silent drop
    np.testing.assert_allclose(V.depth_average(shifted, d.Hz), 0.3)


def test_vertical_params_from_dict_and_dataset(synthetic):
    p = V.vertical_params({"N": 12, "Vtransform": 2, "Vstretching": 4, "THETA_S": 7.0, "THETA_B": 2.0, "TCLINE": 50.0})
    assert p == {"N": 12, "Vtransform": 2, "Vstretching": 4, "theta_s": 7.0, "theta_b": 2.0, "Tcline": 50.0}
    ini = xr.open_dataset(synthetic["ini"])
    q = V.vertical_params(ini)
    assert q["N"] == 12 and q["hc"] == 50.0 and q["theta_s"] == 7.0


def test_synthetic_ini_cs_match_recomputed(synthetic):
    ini = xr.open_dataset(synthetic["ini"])
    cs = V.cs_curves(12, 4, 7.0, 2.0)
    np.testing.assert_allclose(ini["Cs_r"], cs["Cs_r"], atol=1e-6)
    np.testing.assert_allclose(ini["Cs_w"], cs["Cs_w"], atol=1e-6)


def test_moana_grid_cs_reproduced(moana_grid):
    """Real data: exactly one stretching option reproduces the Cs stored by pyroms."""
    g = xr.open_dataset(moana_grid)
    N = g.sizes["s_rho"]
    matches = []
    for vs in (1, 2, 3, 4, 5):
        cs = V.cs_curves(N, vs, float(g.theta_s), float(g.theta_b))
        if all(np.allclose(cs[k], g[k].values, atol=1e-6, rtol=0) for k in ("s_rho", "s_w", "Cs_r", "Cs_w")):
            matches.append(vs)
    assert matches == [5]
