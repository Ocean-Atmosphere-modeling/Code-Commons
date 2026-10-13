import numpy as np
import pytest
import xarray as xr

from romscommons import diagnostics as D
from romscommons import grid as G
from romscommons import vertical as V

from test_grid import uniform_grid


def with_masks(g):
    m = g["mask_rho"]
    g["mask_u"] = (("eta_u", "xi_u"), (m.values[:, :-1] * m.values[:, 1:]))
    g["mask_v"] = (("eta_v", "xi_v"), (m.values[:-1, :] * m.values[1:, :]))
    return g


def test_barotropic_and_coriolis_courant_uniform():
    g = with_masks(uniform_grid(h=100.0, dx=1000.0))
    c = D.barotropic_courant(g, DT=60.0, NDTFAST=20)
    expected = 3.0 * np.sqrt(9.81 * 100.0 * 2) / 1000.0
    np.testing.assert_allclose(c, expected)
    np.testing.assert_allclose(D.coriolis_courant(g, 60.0), 60.0 * 1e-4)
    g["mask_rho"][0, 0] = 0
    assert np.isnan(D.barotropic_courant(g, 60.0, 20)[0, 0])


def test_advective_courant():
    g = with_masks(uniform_grid(dx=1000.0))
    u = xr.DataArray(np.full((3, 6, 7), 0.5), dims=("s_rho", "eta_u", "xi_u"))
    v = xr.DataArray(np.full((3, 5, 8), -0.25), dims=("s_rho", "eta_v", "xi_v"))
    c = D.advective_courant(u, v, g, DT=100.0)
    np.testing.assert_allclose(c["cadv_u"], 0.05)
    np.testing.assert_allclose(c["cadv_v"], 0.025)


def step_grid(N_interior_step=5):
    g = with_masks(uniform_grid(M=6, L=10, h=100.0))
    g["h"][:, N_interior_step:] = 300.0
    return g


def test_rx0_step():
    g = step_grid()
    r = D.rx0(g)
    assert float(r.rx0_u.max()) == pytest.approx(0.5)
    assert np.nanargmax(r.rx0_u.values[2]) == 4  # face between rho 4 and 5
    assert float(r.rx0_v.max()) == 0.0
    assert np.isnan(r.rx0_u.values[0]).all() and np.isnan(r.rx0_u.values[:, 0]).all()  # boundary ring excluded
    assert np.isfinite(D.rx0(g, interior=False).rx0_u.values[0]).all()


def test_rx0_ignores_dry_faces():
    g = step_grid()
    g["mask_rho"][:, 5] = 0
    g = with_masks(g)
    assert np.isnan(D.rx0(g).rx0_u.values[2, 4])


def test_rx1_sigma_coordinates_analytic():
    """Pure sigma (Vtransform 1, hc = 0, Cs = s): rx1 = rx0 (2N - 1) at the bottom layer."""
    g = step_grid()
    N = 10
    vp = dict(N=N, Vtransform=1, Vstretching=1, theta_s=0.0, theta_b=0.0, Tcline=0.0)
    r = D.rx1(g, vp)
    assert float(r.rx1_u.max()) == pytest.approx(0.5 * (2 * N - 1))
    flat = with_masks(uniform_grid())
    assert float(D.rx1(flat, vp).rx1_u.max()) == pytest.approx(0.0)


def test_hotspots():
    g = step_grid()
    hs = D.hotspots(D.rx0(g), g, n=3)
    assert len(hs) == 3 and hs["value"].iloc[0] == pytest.approx(0.5)
    assert set(hs.columns) >= {"field", "value", "location", "eta", "xi", "lon", "lat"}
    assert (hs["xi"] == 4).all()


def _column(T_top, T_bot, N=20, H=1000.0):
    h = xr.DataArray(np.full((1, 1), H), dims=("eta_rho", "xi_rho"))
    d = V.depths(h, 0.0, N=N, Vtransform=2, Vstretching=4, theta_s=0.1, theta_b=0.0, Tcline=10.0)
    z = d.z_r
    T = T_bot + (T_top - T_bot) * (z + H) / H
    S = xr.full_like(T, 35.0)
    return T, S, z


def test_n_squared_sign_and_magnitude():
    T, S, z = _column(20.0, 5.0)
    n2 = D.n_squared(T, S, z, lon=np.full((1, 1), 170.0), lat=np.full((1, 1), -40.0))
    assert n2.dims == ("s_w", "eta_rho", "xi_rho") and n2.sizes["s_w"] == 19
    assert float(n2.min()) > 0
    # g alpha dT/dz with alpha ~ 2e-4 1/K: order 3e-5 s-2
    assert 1e-5 < float(n2.mean()) < 1e-4
    Ti, Si, zi = _column(5.0, 20.0)
    assert float(D.n_squared(Ti, Si, zi, np.full((1, 1), 170.0), np.full((1, 1), -40.0)).max()) < 0


def test_first_mode_speed_constant_N():
    N, H = 20, 1000.0
    z = xr.DataArray(-(np.arange(N)[::-1] + 0.5)[:, None, None] * H / N, dims=("s_rho", "eta_rho", "xi_rho"))
    n2 = xr.DataArray(np.full((N - 1, 1, 1), 1e-5), dims=("s_w", "eta_rho", "xi_rho"))
    c1 = D.first_mode_speed(n2, z)
    expected = np.sqrt(1e-5) * (H - H / N) / np.pi  # integral spans rho levels
    assert c1.values.item() == pytest.approx(expected)


def test_boundary_transport_signs():
    g = with_masks(uniform_grid(M=6, L=8, h=100.0, dx=1000.0))
    ubar = xr.DataArray(np.full((3, 6), 0.1), dims=("v2d_time", "eta_u"))
    q_w = D.boundary_transport(ubar, g, "west")
    q_e = D.boundary_transport(ubar, g, "east")
    np.testing.assert_allclose(q_w, 0.1 * 100.0 * 1000.0 * 6)
    np.testing.assert_allclose(q_e, -q_w)
    zeta = xr.DataArray(np.full((3, 6), 1.0), dims=("zeta_time", "eta_rho"))
    np.testing.assert_allclose(D.boundary_transport(ubar, g, "west", zeta=zeta), 0.1 * 101.0 * 1000.0 * 6)
    vbar = xr.DataArray(np.full((3, 8), 0.1), dims=("v2d_time", "xi_v"))
    np.testing.assert_allclose(D.boundary_transport(vbar, g, "north"), -0.1 * 100 * 1000 * 8)


def test_synthetic_boundary_budget_closes(synthetic):
    from romscommons import io

    g = G.load_grid(synthetic["grid"])
    bry = io.open_nc(synthetic["bry"])
    total = sum(D.boundary_transport(bry[("ubar_" if s in ("west", "east") else "vbar_") + s], g, s,
                                     zeta=bry[f"zeta_{s}"]).values for s in ("west", "south", "east", "north"))
    assert np.abs(total).max() < 1.0  # m3/s, against ~3 Sv through each of south and north


def test_bulk_drag_and_stress():
    np.testing.assert_allclose(D.drag_coefficient(np.array([5.0, 15.0, 30.0])),
                               [1.2e-3, (0.49 + 0.065 * 15) * 1e-3, (0.49 + 0.065 * 25) * 1e-3])
    tx, ty = D.wind_stress(np.array([10.0]), np.array([0.0]))
    assert tx[0] == pytest.approx(1.22 * 1.2e-3 * 100) and ty[0] == 0


def test_ekman_transport_hemispheres():
    mx, my = D.ekman_transport(0.1, 0.0, f=1e-4)
    assert mx == 0 and my < 0  # northern hemisphere: to the right of an eastward wind
    mx, my = D.ekman_transport(0.1, 0.0, f=-1e-4)
    assert my > 0  # southern hemisphere: to the left


def test_curl_of_shear():
    g = uniform_grid(M=5, L=6, dx=1000.0)
    y = xr.DataArray(np.arange(5.0) * 1000.0, dims="eta_rho")
    tx = (0.001 * y * xr.ones_like(g["h"])).transpose("eta_rho", "xi_rho")
    c = D.curl(tx, xr.zeros_like(tx), g)
    np.testing.assert_allclose(c, -0.001)


def test_net_heat_flux_signs():
    q = D.net_heat_flux(swrad_net=0.0, lwrad_down=300.0, sst=15.0, tair=5.0, wind_speed=10.0, pair=1013.0,
                        rh_percent=70.0)
    assert float(q["sensible"]) > 0 and float(q["latent"]) > 0 and float(q["net"]) < 0
    assert D.saturation_vapor_pressure(0.0) == pytest.approx(6.112)
