import numpy as np
import pytest
import xarray as xr

from romscommons import grid as G


def uniform_grid(M=6, L=8, dx=1000.0, angle=0.0, h=100.0, lon0=10.0, lat0=-40.0):
    jj, ii = np.mgrid[0:M, 0:L].astype(float)
    dims = ("eta_rho", "xi_rho")
    return xr.Dataset({
        "h": (dims, np.full((M, L), h)), "pm": (dims, np.full((M, L), 1 / dx)), "pn": (dims, np.full((M, L), 1 / dx)),
        "angle": (dims, np.full((M, L), angle)), "f": (dims, np.full((M, L), -1e-4)),
        "mask_rho": (dims, np.ones((M, L))), "lon_rho": (dims, lon0 + 0.01 * ii), "lat_rho": (dims, lat0 + 0.01 * jj),
    })


@pytest.fixture(scope="module")
def sgrid(synthetic):
    return G.load_grid(synthetic["grid"])


def test_expected_dims(sgrid):
    assert G.expected_dims(sgrid, "rho") == {"eta_rho": 32, "xi_rho": 42}
    assert G.expected_dims(sgrid, "u") == {"eta_u": 32, "xi_u": 41}
    assert G.expected_dims(sgrid, "v") == {"eta_v": 31, "xi_v": 42}
    assert G.expected_dims(sgrid, "psi") == {"eta_psi": 31, "xi_psi": 41}
    for loc in ("u", "v", "psi"):
        assert dict(sgrid[f"mask_{loc}"].sizes) == G.expected_dims(sgrid, loc)


def test_load_grid_derives_masks_and_coords(tmp_path):
    g = uniform_grid()
    g["mask_rho"][2, 3] = 0
    g.to_netcdf(tmp_path / "g.nc")
    out = G.load_grid(tmp_path / "g.nc")
    assert out["mask_u"].shape == (6, 7) and out["mask_v"].shape == (5, 8) and out["mask_psi"].shape == (5, 7)
    assert out["mask_u"][2, 2] == 0 and out["mask_u"][2, 3] == 0 and out["mask_u"][2, 4] == 1
    assert out["mask_v"][1, 3] == 0 and out["mask_v"][2, 3] == 0
    assert out["mask_psi"].sum() == 35 - 4
    np.testing.assert_allclose(out["lon_u"].values, 10.005 + 0.01 * np.arange(7)[None].repeat(6, 0))


def test_load_grid_lon_dateline(tmp_path):
    g = uniform_grid(lon0=179.97)
    g["lon_rho"] = G.wrap_lon(g["lon_rho"], "-180-180")  # 179.97 .. -179.96
    g.to_netcdf(tmp_path / "g.nc")
    out = G.load_grid(tmp_path / "g.nc")
    assert np.all(np.abs(out["lon_u"].values) > 179.9)  # no averaging through 0


def test_rho2u_rho2v_linear_exact():
    g = uniform_grid()
    f = g["lon_rho"] * 0 + xr.DataArray(np.arange(8.0), dims="xi_rho") + 10 * xr.DataArray(np.arange(6.0), dims="eta_rho")
    f = f.transpose("eta_rho", "xi_rho")
    fu, fv = G.rho2u(f), G.rho2v(f)
    assert fu.dims == ("eta_u", "xi_u") and fv.dims == ("eta_v", "xi_v")
    np.testing.assert_allclose(fu.values, f.values[:, :-1] + 0.5)
    np.testing.assert_allclose(fv.values, f.values[:-1, :] + 5.0)
    back = G.u2rho(fu)
    np.testing.assert_allclose(back.values[:, 1:-1], f.values[:, 1:-1])
    assert G.v2rho(fv).dims == ("eta_rho", "xi_rho")


def test_rho2u_mask_aware_and_extra_dims():
    f = xr.DataArray(np.array([[1.0, np.nan, 3.0]]), dims=("eta_rho", "xi_rho"))
    m = xr.DataArray(np.array([[1.0, 0.0, 1.0]]), dims=("eta_rho", "xi_rho"))
    np.testing.assert_allclose(G.rho2u(f, m).values, [[1.0, 3.0]])
    f3 = xr.DataArray(np.ones((2, 3, 4, 5)), dims=("ocean_time", "s_rho", "eta_rho", "xi_rho"))
    assert G.rho2u(f3).dims == ("ocean_time", "s_rho", "eta_u", "xi_u")
    assert G.rho2v(f3).shape == (2, 3, 3, 5)


def test_rotation_round_trip_and_known_case():
    rng = np.random.default_rng(0)
    ue, vn, ang = rng.normal(size=50), rng.normal(size=50), rng.uniform(-np.pi, np.pi, 50)
    ug, vg = G.to_grid(ue, vn, ang)
    ue2, vn2 = G.to_earth(ug, vg, ang)
    np.testing.assert_allclose(ue2, ue, atol=1e-12)
    np.testing.assert_allclose(vn2, vn, atol=1e-12)
    # xi points north (angle = 90 deg): an eastward vector is along -eta
    ug, vg = G.to_grid(1.0, 0.0, np.pi / 2)
    assert ug == pytest.approx(0, abs=1e-12) and vg == pytest.approx(-1)


def test_metrics():
    g = uniform_grid(dx=2000.0)
    assert float(G.dx(g)[0, 0]) == 2000.0 and float(G.dy(g)[0, 0]) == 2000.0
    assert float(G.area(g)[0, 0]) == 4e6


def test_side_slices(sgrid):
    assert sgrid["h"].isel(G.side_slices("west")).dims == ("eta_rho",)
    assert sgrid["mask_u"].isel(G.side_slices("east", "u")).sizes["eta_u"] == 32
    assert sgrid["mask_v"].isel(G.side_slices("north", "v")).sizes["xi_v"] == 42
    assert G.side_dim("south", "v") == "xi_v" and G.normal_location("west") == "u"
    assert [G.inflow_sign(s) for s in ("west", "south", "east", "north")] == [1, 1, -1, -1]


def test_perimeter_order_and_length():
    M, L, dx = 6, 8, 1000.0
    p = G.perimeter(uniform_grid(M, L, dx))
    assert p.sizes["perim"] == 2 * L + 2 * M
    sides = list(dict.fromkeys(p["side"].values.tolist()))
    assert sides == ["south", "east", "north", "west"]
    assert (int(p["eta"][0]), int(p["xi"][0])) == (0, 0)  # starts at SW
    s = p.where(p["side"] == "south", drop=True)
    assert np.all(np.diff(s["xi"].values) == 1)  # south: west -> east
    n = p.where(p["side"] == "north", drop=True)
    assert np.all(np.diff(n["xi"].values) == -1)  # north: east -> west
    w = p.where(p["side"] == "west", drop=True)
    assert np.all(np.diff(w["eta"].values) == -1)  # west: north -> south
    c = p.attrs["corners"]
    assert c["SW"] == 0 and c["SE"] == pytest.approx((L - 1) * dx / 1e3)
    assert c["NE"] == pytest.approx(((L - 1) + (M - 1)) * dx / 1e3)
    assert c["SW_end"] == pytest.approx(2 * ((L - 1) + (M - 1)) * dx / 1e3)
    assert np.all(np.diff(p["dist"].values) >= 0)


def test_unroll_matches_perimeter_indices():
    g = uniform_grid()
    f = xr.DataArray(np.arange(48.0).reshape(6, 8), dims=("eta_rho", "xi_rho"))
    per_side = {s: f.isel(G.side_slices(s)) for s in ("west", "south", "east", "north")}
    flat = G.unroll(per_side)
    p = G.perimeter(g)
    np.testing.assert_array_equal(flat.values, f.values[p["eta"].values, p["xi"].values])


def test_open_sides():
    g = uniform_grid()
    lbc = {"zeta": {"west": "Cha", "south": "Clo", "east": "Per", "north": "Cha"}}
    assert G.open_sides(lbc, g) == ["west", "north"]
    g["mask_rho"][:, 0] = 0  # west side all land
    assert G.open_sides(lbc, g) == ["north"]
    assert G.open_sides(["South", "west"], g) == ["south"]


def test_lon_helpers():
    lon = np.array([178.0, 180.0, 182.0])
    assert G.lon_convention(lon) == "0-360"
    np.testing.assert_allclose(G.wrap_lon(lon, "-180-180"), [178, -180, -178])
    assert G.lon_convention(G.wrap_lon(lon, "-180-180")) == "-180-180"
    np.testing.assert_allclose(G.unwrap_lon(G.wrap_lon(lon, "-180-180")), lon)
    assert G.lon_convention([10.0, 20.0]) == "either"
    assert G.crosses_dateline(lon) and not G.crosses_dateline([10.0, 20.0])


def test_transect_indices(sgrid):
    t = G.transect_indices(sgrid, None, None)
    assert t.sizes["along"] == 42 and np.all(t["eta"] == 16)
    lon, lat = float(sgrid.lon_rho[5, 3]), float(sgrid.lat_rho[5, 3])
    lon2, lat2 = float(sgrid.lon_rho[25, 38]), float(sgrid.lat_rho[25, 38])
    t = G.transect_indices(sgrid, (lon, lat), (lon2 - 360, lat2))  # dateline-safe endpoint
    assert (int(t.eta[0]), int(t.xi[0])) == (5, 3) and (int(t.eta[-1]), int(t.xi[-1])) == (25, 38)
    assert np.all(np.abs(np.diff(t.eta)) <= 1) and np.all(np.abs(np.diff(t.xi)) <= 1)
    assert np.all(np.diff(t.dist) > 0)
