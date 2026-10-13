"""Stability and consistency numbers for ROMS inputs.

Courant numbers mirror ``Utility/metrics.F``; rx0/rx1 mirror ``Utility/stiffness.F``
(VERIFY_LOG V14). Bulk-flux estimates are deliberately crude (not ROMS's COARE) and are
labelled as order-of-magnitude diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import grid as G
from .vertical import z_w

GRAV = 9.81  # m s-2, ROMS mod_scalars.F
RHO0 = 1025.0  # kg m-3, reference seawater density
RHO_AIR = 1.22  # kg m-3
CP_AIR = 1004.0  # J kg-1 K-1
LV = 2.5e6  # J kg-1
STEFAN_BOLTZMANN = 5.67e-8  # W m-2 K-4


# ---------------------------------------------------------------------------
# Courant numbers
# ---------------------------------------------------------------------------

def barotropic_courant(grid: xr.Dataset, DT: float, NDTFAST: int) -> xr.DataArray:
    """``(DT/NDTFAST)·sqrt(g·|h|·(pm² + pn²))`` at wet rho points (``metrics.F``).

    Uses ``h`` only (no ζ), exactly as ROMS prints it at initialization.
    """
    dtfast = float(DT) / float(NDTFAST)
    c = dtfast * np.sqrt(GRAV * abs(grid["h"]) * (grid["pm"] ** 2 + grid["pn"] ** 2))
    return c.where(grid["mask_rho"] > 0).rename("courant_barotropic").assign_attrs(
        units="1", long_name="barotropic (fast) Courant number")


def coriolis_courant(grid: xr.Dataset, DT: float) -> xr.DataArray:
    """``DT·|f|`` at wet rho points (``metrics.F``)."""
    c = float(DT) * abs(grid["f"])
    return c.where(grid["mask_rho"] > 0).rename("courant_coriolis").assign_attrs(
        units="1", long_name="Coriolis Courant number")


def advective_courant(u: xr.DataArray, v: xr.DataArray, grid: xr.Dataset, DT: float) -> xr.Dataset:
    """``|u|·DT·pm`` at u points and ``|v|·DT·pn`` at v points, max over s_rho if present."""
    pm_u, pn_v = G.rho2u(grid["pm"]), G.rho2v(grid["pn"])
    cu = abs(u) * float(DT) * pm_u.data
    cv = abs(v) * float(DT) * pn_v.data
    if "s_rho" in cu.dims:
        cu, cv = cu.max("s_rho"), cv.max("s_rho")
    cu = cu.where(grid["mask_u"].data > 0)
    cv = cv.where(grid["mask_v"].data > 0)
    return xr.Dataset({"cadv_u": cu.assign_attrs(units="1", long_name="advective Courant number (xi)"),
                       "cadv_v": cv.assign_attrs(units="1", long_name="advective Courant number (eta)")})


def first_mode_speed(N2: xr.DataArray, z_r: xr.DataArray) -> xr.DataArray:
    """WKB first baroclinic mode speed ``c1 ≈ (1/π)∫N dz`` (m s-1).

    ``N2`` is on the N−1 interior w levels (as returned by :func:`n_squared`) and ``z_r`` on
    the N rho levels; ``dz`` is the spacing between consecutive rho levels. Negative N² counts
    as zero.
    """
    zr = z_r.transpose(..., "s_rho", "eta_rho", "xi_rho")
    dz = zr.isel(s_rho=slice(1, None)).data - zr.isel(s_rho=slice(None, -1)).data
    n2 = N2.transpose(..., "s_w", "eta_rho", "xi_rho")
    c1 = (np.sqrt(n2.clip(min=0)) * dz).sum("s_w", skipna=True) / np.pi
    return c1.rename("c1").assign_attrs(units="m s-1", long_name="first-mode internal wave speed (WKB)")


def internal_courant(c1: xr.DataArray, grid: xr.Dataset, DT: float) -> xr.DataArray:
    """``c1·DT·sqrt(pm² + pn²)`` (internal-wave Courant number)."""
    c = c1 * float(DT) * np.sqrt(grid["pm"] ** 2 + grid["pn"] ** 2)
    return c.where(grid["mask_rho"] > 0).rename("courant_internal").assign_attrs(
        units="1", long_name="internal-wave Courant number")


# ---------------------------------------------------------------------------
# Grid stiffness
# ---------------------------------------------------------------------------

def _faces(zw: np.ndarray, axis_h: int):
    """Split z_w (k, j, i) into the two cells of each face along axis 1 (eta) or 2 (xi)."""
    if axis_h == 2:
        return zw[:, :, 1:], zw[:, :, :-1]
    return zw[:, 1:, :], zw[:, :-1, :]


def _stiffness(zw: np.ndarray, grid: xr.Dataset, interior: bool) -> dict[str, xr.DataArray]:
    out = {}
    for loc, axis in (("u", 2), ("v", 1)):
        A, B = _faces(zw, axis)
        r0 = np.abs((A[0] - B[0]) / (A[0] + B[0]))
        r1 = np.full_like(r0, np.nan)
        if zw.shape[0] > 1:
            num = A[1:] - B[1:] + A[:-1] - B[:-1]
            den = A[1:] + B[1:] - A[:-1] - B[:-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                r1 = np.nanmax(np.abs(num / den), axis=0)
        m = grid[f"mask_{loc}"].values > 0
        if interior:
            inner = np.zeros_like(m)
            inner[1:-1, 1:-1] = True
            m = m & inner
        dims = G.LOCATIONS[loc]
        out[f"rx0_{loc}"] = xr.DataArray(np.where(m, r0, np.nan), dims=dims)
        out[f"rx1_{loc}"] = xr.DataArray(np.where(m, r1, np.nan), dims=dims)
    return out


def rx0(grid: xr.Dataset, interior: bool = True) -> xr.Dataset:
    """Beckmann & Haidvogel ``|h₁ − h₂|/(h₁ + h₂)`` on u and v faces between wet cells.

    ``interior=True`` uses the same faces as ROMS (``stiffness.F``): both cells wet and not
    on the outer boundary ring.
    """
    h = grid["h"].values[None]
    st = _stiffness(-h, grid, interior)
    return xr.Dataset({"rx0_u": st["rx0_u"], "rx0_v": st["rx0_v"]},
                      attrs={"long_name": "Beckmann-Haidvogel stiffness ratio rx0"})


def rx1(grid: xr.Dataset, vparams: dict, zeta=None, interior: bool = True) -> xr.Dataset:
    """Haney ``rx1`` per face, max over k, with ζ = 0 (default) or the given ζ.

    ROMS computes it after ``set_depth`` with the initial ζ (VERIFY_LOG V14); pass the IC
    ζ to reproduce the logged value.
    """
    p = {k: vparams[k] for k in ("N", "Vtransform", "Vstretching", "theta_s", "theta_b", "Tcline")}
    p["hc"] = vparams.get("hc")
    zeta = 0.0 if zeta is None else zeta
    zw = z_w(grid["h"], zeta, **p).transpose("s_w", "eta_rho", "xi_rho").values
    st = _stiffness(zw, grid, interior)
    return xr.Dataset({"rx1_u": st["rx1_u"], "rx1_v": st["rx1_v"]},
                      attrs={"long_name": "Haney stiffness ratio rx1"})


def hotspots(fields: xr.Dataset | xr.DataArray, grid: xr.Dataset, n: int = 10) -> pd.DataFrame:
    """Top-``n`` values over all variables of ``fields`` with indices and lon/lat."""
    if isinstance(fields, xr.DataArray):
        fields = fields.to_dataset(name=fields.name or "value")
    rows = []
    coords = {}
    for loc, f in (("rho", lambda x: x), ("u", G.rho2u), ("v", G.rho2v)):
        if f"lon_{loc}" in grid:
            coords[loc] = (grid[f"lon_{loc}"].values, grid[f"lat_{loc}"].values)
        elif "lon_rho" in grid:  # derive from rho points when the grid lacks them
            coords[loc] = (f(G.unwrap_lon(grid["lon_rho"])).values, f(grid["lat_rho"]).values)
    for name, da in fields.data_vars.items():
        loc = G.location_of(da)
        eta, xi = G.LOCATIONS[loc]
        vals = da.transpose(eta, xi).values
        flat = np.where(np.isfinite(vals), vals, -np.inf).ravel()
        k = min(n, flat.size)
        for idx in np.argpartition(flat, -k)[-k:]:
            if not np.isfinite(flat[idx]):
                continue
            j, i = np.unravel_index(idx, vals.shape)
            row = {"field": name, "value": float(flat[idx]), "location": loc, eta: int(j), xi: int(i),
                   "eta": int(j), "xi": int(i)}
            if loc in coords:
                row["lon"] = float(coords[loc][0][j, i])
                row["lat"] = float(coords[loc][1][j, i])
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = ["field", "value", "location", "eta", "xi"] + [c for c in ("lon", "lat") if c in df]
    return df.sort_values("value", ascending=False).head(n)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def n_squared(temp: xr.DataArray, salt: xr.DataArray, z_r: xr.DataArray, lon, lat) -> xr.DataArray:
    """Buoyancy frequency squared N² (s-2) on the N−1 interior w levels, via TEOS-10 (gsw).

    ROMS ``temp`` is potential temperature (°C) and ``salt`` practical salinity; they are
    converted to Absolute Salinity and Conservative Temperature first. This approximates
    ROMS's own nonlinear equation of state (BUILD_SPEC A6).
    """
    import gsw

    temp, salt, z_r = xr.align(temp, salt, z_r, join="override")
    order = [d for d in temp.dims if d != "s_rho"]
    lead, horiz = order[:-2], order[-2:]
    dims = (*lead, "s_rho", *horiz)
    T = temp.transpose(*dims).values
    S = salt.transpose(*dims).values
    Z = np.broadcast_to(z_r.transpose(*[d for d in dims if d in z_r.dims]).values, T.shape)
    lat_a = np.broadcast_to(np.asarray(lat), T.shape)
    lon_a = np.broadcast_to(np.asarray(lon) % 360, T.shape)
    ax = len(lead)
    # gsw wants pressure increasing along the axis: flip to surface-first, then back.
    T, S, Z, lat_a, lon_a = (np.flip(a, axis=ax) for a in (T, S, Z, lat_a, lon_a))
    p = gsw.p_from_z(Z, lat_a)
    SA = gsw.SA_from_SP(S, p, lon_a, lat_a)
    CT = gsw.CT_from_pt(SA, T)
    N2, _ = gsw.Nsquared(SA, CT, p, lat_a, axis=ax)
    N2 = np.flip(N2, axis=ax)
    out_dims = (*lead, "s_w", *horiz)
    return xr.DataArray(N2, dims=out_dims, name="N2",
                        attrs={"units": "s-2", "long_name": "buoyancy frequency squared (TEOS-10 approximation)"})


# ---------------------------------------------------------------------------
# Boundary transports
# ---------------------------------------------------------------------------

def _face_geometry(grid: xr.Dataset, side: str):
    """Depth (h averaged to the boundary face), face width, and face mask along a side."""
    h, pm, pn = grid["h"].values, grid["pm"].values, grid["pn"].values
    if side in ("west", "east"):
        a, b = (0, 1) if side == "west" else (-2, -1)
        depth = 0.5 * (h[:, a] + h[:, b])
        width = 0.5 * (1 / pn[:, a] + 1 / pn[:, b])
        m = grid["mask_u"].values[:, 0 if side == "west" else -1]
    else:
        a, b = (0, 1) if side == "south" else (-2, -1)
        depth = 0.5 * (h[a, :] + h[b, :])
        width = 0.5 * (1 / pm[a, :] + 1 / pm[b, :])
        m = grid["mask_v"].values[0 if side == "south" else -1, :]
    return depth, width, m


def boundary_transport(vel_bar: xr.DataArray, grid: xr.Dataset, side: str, zeta=None) -> xr.DataArray:
    """Volume transport (m³ s⁻¹) through one side, **inflow positive**.

    ``vel_bar`` is the normal depth-mean velocity on that side (``ubar_west``/``ubar_east``
    or ``vbar_south``/``vbar_north``). Face depth = mean of the two adjacent ``h`` plus the
    boundary ζ at the rho point (a documented approximation). ``zeta`` usually has its own
    time axis (``zeta_time`` vs ``v2d_time``): it is interpolated onto the velocity times when
    both carry time coordinates, else the two axes must have the same length and are assumed
    identical.
    """
    depth, width, m = _face_geometry(grid, side)
    along = vel_bar.dims[-1]
    D = xr.DataArray(depth, dims=along)
    if zeta is not None:
        z = zeta.rename({zeta.dims[-1]: along})
        if z.ndim == 2 and vel_bar.ndim == 2 and z.dims[0] != vel_bar.dims[0]:
            zt, vt = z.dims[0], vel_bar.dims[0]
            if zt in z.coords and vt in vel_bar.coords:
                if not np.array_equal(z[zt].values, vel_bar[vt].values):
                    z = z.interp({zt: vel_bar[vt].values})
            elif z.sizes[zt] != vel_bar.sizes[vt]:
                raise ValueError(f"cannot match {zt} ({z.sizes[zt]}) to {vt} ({vel_bar.sizes[vt]}) without coordinates")
            z = xr.DataArray(z.data, dims=(vt, along))
        D = D + xr.DataArray(z.data, dims=z.dims)
    vb = xr.DataArray(vel_bar.data, dims=vel_bar.dims, coords={k: v for k, v in vel_bar.coords.items()
                                                                if along not in v.dims})
    q = (vb * D * xr.DataArray(width * m, dims=along)).sum(along) * G.inflow_sign(side)
    return q.rename(f"transport_{side}").assign_attrs(units="m3 s-1", long_name=f"inflow through {side} side")


# ---------------------------------------------------------------------------
# Bulk estimates (crude; order of magnitude only)
# ---------------------------------------------------------------------------

def drag_coefficient(U):
    """Large & Pond (1981) neutral drag coefficient: 1.2e-3 below 11 m/s,
    (0.49 + 0.065 U)·1e-3 for 11–25 m/s, capped at the 25 m/s value above."""
    U = np.clip(U, 0, 25)
    return xr.where(U < 11, 1.2e-3, (0.49 + 0.065 * U) * 1e-3) if isinstance(U, xr.DataArray) \
        else np.where(U < 11, 1.2e-3, (0.49 + 0.065 * U) * 1e-3)


def wind_stress(u10, v10, rho_air: float = RHO_AIR):
    """``τ = ρa·Cd·|U|·(u, v)`` (N m-2), same frame as the input components."""
    U = np.sqrt(u10 ** 2 + v10 ** 2)
    cd = drag_coefficient(U)
    return rho_air * cd * U * u10, rho_air * cd * U * v10


def curl(fx: xr.DataArray, fy: xr.DataArray, grid: xr.Dataset) -> xr.DataArray:
    """``∂fy/∂x − ∂fx/∂y`` at rho points; components in grid (xi, eta) frame.

    Centered differences with ``∂/∂x = pm·∂/∂ξ``; metric (curvature) terms are ignored.
    """
    ax_xi, ax_eta = fy.dims.index("xi_rho"), fx.dims.index("eta_rho")
    dfy = np.gradient(fy.values, axis=ax_xi) * grid["pm"].values
    dfx = np.gradient(fx.values, axis=ax_eta) * grid["pn"].values
    return xr.DataArray(dfy - dfx, dims=fy.dims, coords=fy.coords)


def ekman_transport(taux, tauy, f, rho0: float = RHO0):
    """Ekman transport (m² s⁻¹): ``(τy, −τx)/(ρ0 f)``. The sign of f puts it to the right of
    the wind in the north and to the left in the south."""
    return tauy / (rho0 * f), -taux / (rho0 * f)


def ekman_pumping(taux, tauy, grid: xr.Dataset, rho0: float = RHO0):
    """Ekman pumping ``w_E = curl(τ/(ρ0 f))`` (m s-1, positive up)."""
    f = grid["f"]
    return curl(taux / (rho0 * f), tauy / (rho0 * f), grid)


def saturation_vapor_pressure(T):
    """Saturation vapour pressure (mbar) over water: ``6.112·exp(17.67T/(T+243.5))``, T in °C."""
    return 6.112 * np.exp(17.67 * T / (T + 243.5))


def specific_humidity(e, P):
    """Specific humidity (kg/kg) from vapour pressure ``e`` and pressure ``P`` (both mbar)."""
    return 0.622 * e / (P - 0.378 * e)


def net_heat_flux(swrad_net, lwrad_down, sst, tair, wind_speed, pair, rh_percent,
                  Ch: float = 1.1e-3, Ce: float = 1.2e-3, emissivity: float = 0.97) -> xr.Dataset:
    """Crude net surface heat flux into the ocean (W m-2) and its parts.

    ``Q = SW_net + LW_down − εσT_s⁴ − Q_sens − Q_lat`` with ``Q_sens = ρa cp Ch U (T_s − T_a)``
    and ``Q_lat = ρa L Ce U (q_s − q_a)``; ``q_s = 0.98·q_sat(T_s)``. Not ROMS's COARE.
    """
    ts_k = sst + 273.15
    lw_up = emissivity * STEFAN_BOLTZMANN * ts_k ** 4
    q_s = 0.98 * specific_humidity(saturation_vapor_pressure(sst), pair)
    q_a = specific_humidity(rh_percent / 100.0 * saturation_vapor_pressure(tair), pair)
    sens = RHO_AIR * CP_AIR * Ch * wind_speed * (sst - tair)
    lat = RHO_AIR * LV * Ce * wind_speed * (q_s - q_a)
    net = swrad_net + lwrad_down - lw_up - sens - lat
    return xr.Dataset({"swrad": swrad_net, "lwrad_down": lwrad_down, "lw_up": lw_up,
                       "sensible": sens, "latent": lat, "net": net},
                      attrs={"units": "W m-2", "note": "positive into the ocean; sensible/latent are losses"})
