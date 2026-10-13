#!/usr/bin/env python
"""Write a small, structurally ROMS-conformant synthetic case (BUILD_SPEC §6).

Not a runnable ROMS case: the files have ROMS names, dims, units and staggering, so every
notebook can run end to end without real data.

    python examples/make_synthetic_case.py --out data/synthetic
    python examples/make_synthetic_case.py --out data/synthetic_defects --defects

``--defects`` injects one fault per row of the BUILD_SPEC §6 table; ``tests/test_defects.py``
asserts that each is detected.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from romscommons import grid as G
from romscommons import tides as T
from romscommons import vertical as V
from romscommons.io import time_ref

# ---------------------------------------------------------------------------
# Case constants
# ---------------------------------------------------------------------------
L, M = 42, 32  # xi_rho, eta_rho  (Lm = 40, Mm = 30)
DX = 7.0e3  # m, uniform cell size
ANGLE_DEG = 30.0  # xi axis, counter-clockwise from east
LON0, LAT0 = 180.0, -40.5  # domain centre (stored 0-360: crosses the dateline)
R_EARTH = 6371e3
OMEGA = 7.2921e-5
GRAV = 9.81
VPARAMS = dict(N=12, Vtransform=2, Vstretching=4, theta_s=7.0, theta_b=2.0, Tcline=50.0)
TIME_REF, DSTART, TIDE_START = 20000101.0, 9497.0, 9497.0  # 2026-01-01
DT, NDTFAST, NTIMES = 180.0, 30, 4800  # 10 days
RUN_DAYS = NTIMES * DT / 86400
TIME_UNITS_D = "days since 2000-01-01 00:00:00"
TIME_UNITS_S = "seconds since 2000-01-01 00:00:00"
H_SHELF, H_DEEP, X_BREAK, L_SLOPE = 30.0, 2500.0, -20e3, 55e3
ISLAND = (slice(13, 17), slice(4, 7))  # (eta, xi) rho cells that are land
JET_X, JET_W, JET_V = 25e3, 30e3, 0.2  # along-slope current: centre, width, speed

SIDES = ("west", "south", "east", "north")


def _write(ds: xr.Dataset, path: Path, float32: bool = False):
    enc = {}
    for name, var in ds.variables.items():
        e = {"_FillValue": None}
        if var.dtype.kind == "f":
            e.update(zlib=True, complevel=4)
            if float32 and var.ndim >= 2:
                e["dtype"] = "float32"
        enc[name] = e
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=enc)


def _days(t: pd.Timestamp) -> float:
    return (t - time_ref(TIME_REF)) / pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def _xy(ii, jj):
    """Grid-frame coordinates (m) of fractional rho indices."""
    return (ii - (L - 1) / 2) * DX, (jj - (M - 1) / 2) * DX


def _lonlat(ii, jj):
    x, y = _xy(ii, jj)
    th = np.deg2rad(ANGLE_DEG)
    east, north = x * np.cos(th) - y * np.sin(th), x * np.sin(th) + y * np.cos(th)
    lat = LAT0 + np.rad2deg(north / R_EARTH)
    lon = LON0 + np.rad2deg(east / (R_EARTH * np.cos(np.deg2rad(LAT0))))
    return lon % 360, lat


def bathymetry(ii, jj):
    """Shelf (30 m) -> slope -> 2500 m along xi; a gentle cos wiggle along eta that is
    symmetric between the south and north boundary rows (so the boundary volume budget closes)."""
    x, _ = _xy(ii, jj)
    xb = X_BREAK + 8e3 * np.cos(2 * np.pi * jj / (M - 1))
    logh = np.log(H_SHELF) + (np.log(H_DEEP) - np.log(H_SHELF)) * 0.5 * (1 + np.tanh((x - xb) / L_SLOPE))
    return np.exp(logh)


def make_grid() -> xr.Dataset:
    jj, ii = np.mgrid[0:M, 0:L].astype(float)
    lon, lat = _lonlat(ii, jj)
    mask = np.ones((M, L))
    mask[ISLAND] = 0.0
    g = xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), bathymetry(ii, jj), {"long_name": "bathymetry at RHO-points", "units": "meter"}),
            "f": (("eta_rho", "xi_rho"), 2 * OMEGA * np.sin(np.deg2rad(lat)),
                  {"long_name": "Coriolis parameter at RHO-points", "units": "second-1"}),
            "pm": (("eta_rho", "xi_rho"), np.full((M, L), 1 / DX), {"long_name": "curvilinear coordinate metric in XI", "units": "meter-1"}),
            "pn": (("eta_rho", "xi_rho"), np.full((M, L), 1 / DX), {"long_name": "curvilinear coordinate metric in ETA", "units": "meter-1"}),
            "angle": (("eta_rho", "xi_rho"), np.full((M, L), np.deg2rad(ANGLE_DEG)),
                      {"long_name": "angle between XI-axis and EAST", "units": "radians"}),
            "lon_rho": (("eta_rho", "xi_rho"), lon, {"long_name": "longitude of RHO-points", "units": "degree_east"}),
            "lat_rho": (("eta_rho", "xi_rho"), lat, {"long_name": "latitude of RHO-points", "units": "degree_north"}),
            "x_rho": (("eta_rho", "xi_rho"), _xy(ii, jj)[0], {"units": "meter"}),
            "y_rho": (("eta_rho", "xi_rho"), _xy(ii, jj)[1], {"units": "meter"}),
            "mask_rho": (("eta_rho", "xi_rho"), mask, {"long_name": "mask on RHO-points", "flag_values": [0.0, 1.0],
                                                      "flag_meanings": "land water"}),
            "spherical": ((), np.array(b"T"), {"long_name": "grid type logical switch"}),
            "xl": ((), L * DX, {"units": "meter"}),
            "el": ((), M * DX, {"units": "meter"}),
        },
        attrs={"type": "ROMS grid file", "title": "code-commons synthetic grid (not a runnable case)"},
    )
    for loc, (di, dj, ni, nj) in {"u": (0.5, 0, L - 1, M), "v": (0, 0.5, L, M - 1), "psi": (0.5, 0.5, L - 1, M - 1)}.items():
        jj2, ii2 = np.mgrid[0:nj, 0:ni].astype(float)
        lo, la = _lonlat(ii2 + di, jj2 + dj)
        dims = G.LOCATIONS[loc]
        g[f"lon_{loc}"] = (dims, lo, {"units": "degree_east"})
        g[f"lat_{loc}"] = (dims, la, {"units": "degree_north"})
    g["mask_u"] = (("eta_u", "xi_u"), mask[:, :-1] * mask[:, 1:])
    g["mask_v"] = (("eta_v", "xi_v"), mask[:-1, :] * mask[1:, :])
    g["mask_psi"] = (("eta_psi", "xi_psi"), mask[:-1, :-1] * mask[1:, :-1] * mask[:-1, 1:] * mask[1:, 1:])
    return g


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def _zeta_geostrophic(x, f):
    """Sea level in geostrophic balance with the along-slope jet: f v = g dζ/dx."""
    from math import erf, pi, sqrt

    erfv = np.vectorize(erf)
    prof = JET_W * sqrt(pi) / 2 * (1 + erfv((x - JET_X) / JET_W))
    z = f * JET_V / GRAV * prof
    return z - z.mean()


def _temp(z, y):
    return 8.0 + 10.0 * np.exp(z / 150.0) + 0.5 * (y / 150e3)


def _salt(z):
    return 34.2 + 0.5 * (1 - np.exp(z / 300.0))


def make_ini(g: xr.Dataset, vparams: dict | None = None) -> xr.Dataset:
    vp = dict(VPARAMS, **(vparams or {}))
    h = g["h"]
    zeta = xr.DataArray(_zeta_geostrophic(g["x_rho"].values, float(g["f"].mean())), dims=h.dims)
    d = V.depths(h, zeta, **vp)
    zr, Hz = d["z_r"], d["Hz"]
    y = g["y_rho"]
    temp = _temp(zr, y).transpose("s_rho", "eta_rho", "xi_rho")
    salt = _salt(zr).transpose("s_rho", "eta_rho", "xi_rho")
    # along-slope (eta-direction, grid frame) jet at v points; no cross-slope flow
    x_v = G.rho2v(g["x_rho"])
    z_v = G.rho2v(zr)
    v = (JET_V * np.exp(-((x_v - JET_X) / JET_W) ** 2) * (0.4 + 0.6 * np.exp(z_v / 300.0)) * g["mask_v"])
    v = v.transpose("s_rho", "eta_v", "xi_v")
    u = xr.zeros_like(G.rho2u(zr)).transpose("s_rho", "eta_u", "xi_u")
    Hz_u, Hz_v = G.rho2u(Hz), G.rho2v(Hz)
    ubar, vbar = V.depth_average(u, Hz_u), V.depth_average(v, Hz_v)
    cs = V.cs_curves(vp["N"], vp["Vstretching"], vp["theta_s"], vp["theta_b"])
    t0 = DSTART * 86400.0

    def rec(da):
        return da.expand_dims(ocean_time=1).data

    ds = xr.Dataset(
        {
            "ocean_time": (("ocean_time",), [t0], {"long_name": "time since initialization", "units": TIME_UNITS_S}),
            "zeta": (("ocean_time", "eta_rho", "xi_rho"), rec(zeta), {"long_name": "free-surface", "units": "meter",
                                                                     "time": "ocean_time"}),
            "ubar": (("ocean_time", "eta_u", "xi_u"), rec(ubar), {"long_name": "vertically integrated u-momentum component",
                                                                 "units": "meter second-1", "time": "ocean_time"}),
            "vbar": (("ocean_time", "eta_v", "xi_v"), rec(vbar), {"long_name": "vertically integrated v-momentum component",
                                                                 "units": "meter second-1", "time": "ocean_time"}),
            "u": (("ocean_time", "s_rho", "eta_u", "xi_u"), rec(u), {"long_name": "u-momentum component",
                                                                    "units": "meter second-1", "time": "ocean_time"}),
            "v": (("ocean_time", "s_rho", "eta_v", "xi_v"), rec(v), {"long_name": "v-momentum component",
                                                                    "units": "meter second-1", "time": "ocean_time"}),
            "temp": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), rec(temp), {"long_name": "potential temperature",
                                                                              "units": "Celsius", "time": "ocean_time"}),
            "salt": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), rec(salt), {"long_name": "salinity",
                                                                              "units": "nondimensional", "time": "ocean_time"}),
            "Vtransform": ((), np.int32(vp["Vtransform"]), {"long_name": "vertical terrain-following transformation equation"}),
            "Vstretching": ((), np.int32(vp["Vstretching"]), {"long_name": "vertical terrain-following stretching function"}),
            "theta_s": ((), float(vp["theta_s"]), {"long_name": "S-coordinate surface control parameter"}),
            "theta_b": ((), float(vp["theta_b"]), {"long_name": "S-coordinate bottom control parameter"}),
            "Tcline": ((), float(vp["Tcline"]), {"long_name": "S-coordinate surface/bottom layer width", "units": "meter"}),
            "hc": ((), V.critical_depth(vp["Vtransform"], vp["Tcline"], float(h.min())),
                   {"long_name": "S-coordinate parameter, critical depth", "units": "meter"}),
            "s_rho": (("s_rho",), cs["s_rho"], {"long_name": "S-coordinate at RHO-points"}),
            "s_w": (("s_w",), cs["s_w"], {"long_name": "S-coordinate at W-points"}),
            "Cs_r": (("s_rho",), cs["Cs_r"], {"long_name": "S-coordinate stretching curves at RHO-points"}),
            "Cs_w": (("s_w",), cs["Cs_w"], {"long_name": "S-coordinate stretching curves at W-points"}),
            "h": (("eta_rho", "xi_rho"), h.values, {"long_name": "bathymetry at RHO-points", "units": "meter"}),
            "spherical": ((), np.array(b"T"), {"long_name": "grid type logical switch"}),
        },
        attrs={"type": "ROMS initial conditions file", "title": "code-commons synthetic IC (not a runnable case)"},
    )
    return ds


# ---------------------------------------------------------------------------
# Atmospheric forcing
# ---------------------------------------------------------------------------

def _run_times(step_h: float, pad_h: float) -> np.ndarray:
    t0 = DSTART - pad_h / 24
    n = int(round((RUN_DAYS + 2 * pad_h / 24) / (step_h / 24))) + 1
    return t0 + np.arange(n) * step_h / 24


def make_frc_native(g: xr.Dataset) -> xr.Dataset:
    t = _run_times(1.0, 1.0)
    tt = (t - DSTART)[:, None, None]
    y = g["y_rho"].values[None]
    lon = g["lon_rho"].values[None]
    speed = (8 + 4 * np.sin(2 * np.pi * tt / 4)) * (1 + 0.2 * y / 150e3)
    direction = 2 * np.pi * tt / 6
    ue, vn = speed * np.cos(direction), speed * np.sin(direction)
    ug, vg = G.to_grid(ue, vn, g["angle"].values[None])  # native-grid winds are in the grid frame (VERIFY_LOG V4)
    local_hour = (24 * (t[:, None, None] % 1) + lon / 15) % 24
    sw = 0.94 * 900 * np.clip(np.cos(2 * np.pi * (local_hour - 12) / 24), 0, None)
    pair = 1013 + 8 * np.sin(2 * np.pi * tt / 5) + 2 * y / 150e3
    qair = 75 + 10 * np.sin(2 * np.pi * tt / 3) + 0 * y
    rain = 2e-5 * np.clip(np.sin(2 * np.pi * tt / 4), 0, None) + 0 * y
    lwd = 320 + 15 * np.sin(2 * np.pi * tt / 7) + 0 * y
    dims = ("eta_rho", "xi_rho")

    def var(tname, data, long_name, units, extra=None):
        return ((tname, *dims), data, {"long_name": long_name, "units": units, "time": tname, **(extra or {})})

    tvars = {k: ((k,), t, {"long_name": f"{lab} time", "units": TIME_UNITS_D})
             for k, lab in [("wind_time", "surface wind"), ("pair_time", "surface air pressure"),
                            ("qair_time", "surface relative humidity"), ("rain_time", "rain fall rate"),
                            ("srf_time", "solar shortwave radiation"), ("lrf_time", "longwave radiation")]}
    ds = xr.Dataset(
        {
            **tvars,
            "Uwind": var("wind_time", ug, "surface u-wind component (grid xi direction)", "meter second-1",
                         {"note": "grid frame: ROMS does not rotate winds already on the native grid"}),
            "Vwind": var("wind_time", vg, "surface v-wind component (grid eta direction)", "meter second-1",
                         {"note": "grid frame: ROMS does not rotate winds already on the native grid"}),
            "Pair": var("pair_time", pair, "surface air pressure", "millibar"),
            "Qair": var("qair_time", qair, "surface air relative humidity", "percentage"),
            "rain": var("rain_time", rain, "rain fall rate", "kilogram meter-2 second-1"),
            "swrad": var("srf_time", sw, "net solar shortwave radiation flux", "watt meter-2"),
            "lwrad_down": var("lrf_time", lwd, "downwelling longwave radiation flux", "watt meter-2"),
        },
        attrs={"type": "ROMS forcing file", "title": "code-commons synthetic forcing on the ROMS grid"},
    )
    return ds


def make_frc_coarse(g: xr.Dataset) -> xr.Dataset:
    lon_r = G.unwrap_lon(g["lon_rho"]).values
    lat_r = g["lat_rho"].values
    lon = np.arange(np.floor(lon_r.min() * 4) / 4 - 0.5, np.ceil(lon_r.max() * 4) / 4 + 0.5 + 1e-9, 0.25)
    lat = np.arange(np.floor(lat_r.min() * 4) / 4 - 0.5, np.ceil(lat_r.max() * 4) / 4 + 0.5 + 1e-9, 0.25)
    t = _run_times(1.0, 1.0)
    LON, LAT = np.meshgrid(lon, lat)
    base = 15.0 + 0.6 * (LAT - LAT0)  # warmer equatorward
    ilon = float(lon_r[ISLAND].mean())
    ilat = float(lat_r[ISLAND].mean())
    land = np.exp(-(((LON - ilon) * np.cos(np.deg2rad(LAT0))) ** 2 + (LAT - ilat) ** 2) / (2 * 0.12 ** 2))
    tair = base[None] + 1.0 * np.sin(2 * np.pi * (t[:, None, None] - DSTART)) - 5.0 * land[None]
    ds = xr.Dataset(
        {
            "tair_time": (("tair_time",), t, {"long_name": "surface air temperature time", "units": TIME_UNITS_D}),
            "lon": (("lon",), lon % 360, {"long_name": "longitude", "units": "degree_east"}),
            "lat": (("lat",), lat, {"long_name": "latitude", "units": "degree_north"}),
            "Tair": (("tair_time", "lat", "lon"), tair, {"long_name": "surface air temperature", "units": "Celsius",
                                                         "time": "tair_time", "coordinates": "lon lat"}),
        },
        attrs={"type": "ROMS forcing file",
               "title": "code-commons synthetic Tair on a 0.25 deg grid; ROMS interpolates it (coordinates attribute)"},
    )
    return ds


# ---------------------------------------------------------------------------
# Tides
# ---------------------------------------------------------------------------

TIDE_CONST = {  # name: (amplitude at the outer shelf [m], phase offset [deg], Cmin/Cmax)
    "M2": (0.60, 40.0, 0.10), "S2": (0.20, 75.0, 0.10), "K1": (0.15, 200.0, -0.05), "O1": (0.10, 180.0, -0.05)}


def make_tide(g: xr.Dataset) -> xr.Dataset:
    x, y = g["x_rho"].values, g["y_rho"].values
    h, mask = g["h"].values, g["mask_rho"].values
    names = list(TIDE_CONST)
    periods = np.array([T.CONSTITUENTS[n] for n in names])
    coast = 1 + 0.5 * np.exp(-(x - x.min()) / 80e3)  # Kelvin wave: larger toward the coast
    shape = (len(names), *h.shape)
    Eamp, Eph, Cmax, Cmin, Cang, Cph = (np.zeros(shape) for _ in range(6))
    for k, n in enumerate(names):
        a0, ph0, ecc = TIDE_CONST[n]
        Eamp[k] = a0 * coast
        Eph[k] = (ph0 + 20.0 * y / 150e3 + 5.0 * x / 150e3) % 360  # propagates along the shelf
        Cmax[k] = Eamp[k] * np.sqrt(GRAV / h)
        Cmin[k] = ecc * Cmax[k]
        Cang[k] = (ANGLE_DEG + 90.0) % 180  # along-shelf (eta) direction, measured from east
        Cph[k] = Eph[k]
    land = mask == 0
    for a in (Eamp, Eph, Cmax, Cmin, Cang, Cph):
        a[:, land] = 0.0
    dims = ("tide_period", "eta_rho", "xi_rho")
    start = time_ref(TIME_REF) + pd.Timedelta(days=TIDE_START)
    return xr.Dataset(
        {
            "tide_period": (("tide_period",), periods, {"long_name": "tide angular period", "units": "hours"}),
            "tide_Eamp": (dims, Eamp, {"long_name": "tidal elevation amplitude", "units": "meter"}),
            "tide_Ephase": (dims, Eph, {"long_name": "tidal elevation phase angle", "units": "degrees"}),
            "tide_Cmax": (dims, Cmax, {"long_name": "maximum tidal current, ellipse major axis", "units": "meter second-1"}),
            "tide_Cmin": (dims, Cmin, {"long_name": "minimum tidal current, ellipse minor axis", "units": "meter second-1"}),
            "tide_Cangle": (dims, Cang, {"long_name": "tidal current inclination angle (from east)", "units": "degrees"}),
            "tide_Cphase": (dims, Cph, {"long_name": "tidal current phase angle", "units": "degrees"}),
        },
        attrs={"type": "ROMS tidal forcing file", "components": " ".join(names),
               "phase_reference": f"{start:%Y-%m-%d %H:%M:%S} (TIDE_START = {TIDE_START} days since TIME_REF {TIME_REF})",
               "source": "synthetic Kelvin-wave-like constituents; nodal factors already included (f=1, u=0)",
               "title": "code-commons synthetic tide file"},
    )


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def make_bry(g: xr.Dataset, ini: xr.Dataset) -> xr.Dataset:
    t = _run_times(3.0, 3.0)
    tt = xr.DataArray(t - DSTART, dims="time")
    ds = xr.Dataset(attrs={"type": "ROMS boundary forcing file", "title": "code-commons synthetic boundary file"})
    for name in ("zeta_time", "v2d_time", "v3d_time", "temp_time", "salt_time"):
        ds[name] = ((name,), t, {"long_name": f"{name.replace('_time', '')} boundary time", "units": TIME_UNITS_D})
    st = ini.isel(ocean_time=0)
    slow_zeta = 0.02 * np.sin(2 * np.pi * tt / 5.0)  # subtidal sea-level signal (m)
    warming = 0.3 * tt / RUN_DAYS  # degC over the run
    tvar = {"zeta": "zeta_time", "ubar": "v2d_time", "vbar": "v2d_time", "u": "v3d_time", "v": "v3d_time",
            "temp": "temp_time", "salt": "salt_time"}
    units = {"zeta": "meter", "ubar": "meter second-1", "vbar": "meter second-1", "u": "meter second-1",
             "v": "meter second-1", "temp": "Celsius", "salt": "nondimensional"}
    for side in SIDES:
        for var in tvar:
            da = st[var]
            loc = G.location_of(da)
            edge = da.isel(G.side_slices(side, loc))
            edge = edge.drop_vars([c for c in edge.coords if c not in edge.dims], errors="ignore")
            series = edge.expand_dims(time=tt.size).copy()
            if var == "zeta":
                series = series + slow_zeta
            elif var == "temp":
                series = series + warming
            series = series.rename({"time": tvar[var]}).transpose(tvar[var], ...)
            ds[f"{var}_{side}"] = (series.dims, series.values,
                                   {"long_name": f"{var} {side}ern boundary condition", "units": units[var],
                                    "time": tvar[var]})
    return ds


# ---------------------------------------------------------------------------
# ocean.in (parser fixture) and nudging file (defects only)
# ---------------------------------------------------------------------------

OCEAN_IN = """\
! parser fixture — not a runnable ROMS input
! Written by examples/make_synthetic_case.py for the code-commons synthetic case.

       TITLE = code-commons synthetic case (parser fixture)
    MyAppCPP = SYNTHETIC
     VARNAME = External/varinfo.yaml

          Lm == 40            ! Number of I-direction INTERIOR RHO-points
          Mm == 30            ! Number of J-direction INTERIOR RHO-points
           N == 12            ! Number of vertical levels
         NAT =  2             ! Number of active tracers

! Lateral boundary conditions: West South East North

   LBC(isFsur) ==   Cha     Cha     Cha     Cha         ! free-surface
   LBC(isUbar) ==   Fla     Fla     Fla     Fla         ! 2D U-momentum
   LBC(isVbar) ==   Fla     Fla     Fla     Fla         ! 2D V-momentum
   LBC(isUvel) ==   RadNud  RadNud  RadNud  RadNud      ! 3D U-momentum
   LBC(isVvel) ==   RadNud  RadNud  RadNud  RadNud      ! 3D V-momentum
   LBC(isMtke) ==   Gra     Gra     Gra     Gra         ! mixing TKE

   LBC(isTvar) ==   RadNud  RadNud  RadNud  RadNud \\    ! temperature
                    RadNud  RadNud  RadNud  RadNud      ! salinity

   VolCons(west)  ==  F                            ! western  boundary
   VolCons(east)  ==  F                            ! eastern  boundary
   VolCons(south) ==  F                            ! southern boundary
   VolCons(north) ==  F                            ! northern boundary

      NTIMES == 4800
          DT == 180.0d0
     NDTFAST == 30
       NRREC == 0

  Vtransform == 2                          ! transformation equation
 Vstretching == 4                          ! stretching function
     THETA_S == 7.0d0                      ! surface stretching parameter
     THETA_B == 2.0d0                      ! bottom  stretching parameter
      TCLINE == 50.0d0                     ! critical depth (m)
       DCRIT == 0.10d0                     ! m

      DSTART =  9497.0d0                   ! days
  TIDE_START =  9497.0d0                   ! days
    TIME_REF =  20000101.0d0               ! yyyymmdd.dd

       TNUDG == 2*1.0d0                    ! days
       ZNUDG == 0.0d0                      ! days
      M2NUDG == 0.0d0                      ! days
      M3NUDG == 1.0d0                      ! days
      OBCFAC == 10.0d0                     ! nondimensional

 LnudgeM2CLM == F                          ! 2D momentum
 LnudgeM3CLM == T                          ! 3D momentum
  LnudgeTCLM == T T                        ! temperature, salinity

     GRDNAME == grid.nc
     ININAME == ini.nc
    TIDENAME == tide.nc
    NBCFILES == 1
     BRYNAME == bry.nc
   NCLMFILES == 1
     CLMNAME == clm.nc
     NUDNAME == nud.nc
     NFFILES == 2
     FRCNAME == frc_native.nc \\
                frc_coarse.nc
"""


def make_nud_defect(g: xr.Dataset) -> xr.Dataset:
    """Nudging coefficients 1/(0.001 day) in a 10-cell sponge: gamma*DT >> 1 (FAIL)."""
    jj, ii = np.mgrid[0:M, 0:L]
    d = np.minimum.reduce([ii, jj, L - 1 - ii, M - 1 - jj]).astype(float)
    taper = np.where(d < 10, 0.5 * (1 + np.cos(np.pi * d / 10)), 0.0) * g["mask_rho"].values
    coef = (1 / 0.001) * taper
    c3 = np.broadcast_to(coef, (VPARAMS["N"], M, L))
    dims = ("s_rho", "eta_rho", "xi_rho")
    attrs = {"units": "day-1"}
    return xr.Dataset({f"{n}_NudgeCoef": (dims, c3.copy(), {"long_name": f"{n} inverse nudging coefficients", **attrs})
                       for n in ("M3", "temp", "salt")},
                      attrs={"type": "ROMS nudging coefficients file", "title": "defect: 1/(0.001 day) coefficients"})


# ---------------------------------------------------------------------------
# Defects (BUILD_SPEC §6 table)
# ---------------------------------------------------------------------------

SPIKE = (16, 20)  # (eta, xi) of the bathymetry spike


def apply_ini_defects(ini: xr.Dataset, g: xr.Dataset) -> xr.Dataset:
    ini = ini.copy(deep=True)
    wet = np.argwhere(g["mask_rho"].values > 0)
    for j, i in wet[[100, 400, 900]]:  # three NaN wet points in temp
        ini["temp"][0, 5, j, i] = np.nan
    ini["ubar"] = ini["ubar"] * 1.5  # ubar/vbar 1.5 x depth average of u/v
    ini["vbar"] = ini["vbar"] * 1.5
    j, i = 8, 3  # surface temperature inversion on the shelf
    ini["temp"][0, -1, j, i] = float(ini["temp"][0, -2, j, i]) - 2.0
    return ini


def apply_frc_defects(native: xr.Dataset, coarse: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    native = native.copy(deep=True)
    native["Pair"] = native["Pair"] * 100.0  # Pa written, attribute still says millibar
    t = native["wind_time"].values  # 12 h gap in winds starting at day 4
    keep = ~((t > DSTART + 4.0) & (t < DSTART + 4.5 - 1e-9))
    native = native.isel(wind_time=keep)
    end = DSTART + RUN_DAYS - 1.0  # shortwave ends one day early
    native = native.isel(srf_time=native["srf_time"].values <= end + 1e-9)
    coarse = coarse.copy(deep=True)
    coarse["Tair"] = coarse["Tair"] + 273.15  # Kelvin, attribute still says Celsius
    coarse["lon"] = G.wrap_lon(coarse["lon"], "-180-180")  # -180..180 while the ROMS grid is 0..360
    return native, coarse


def apply_tide_defects(tide: xr.Dataset) -> xr.Dataset:
    tide = tide.copy(deep=True)
    for v in ("tide_Ephase", "tide_Cphase"):  # radians instead of degrees
        tide[v] = np.deg2rad(tide[v])
    k = 2  # K1: |Cmin| > Cmax
    cmax = tide["tide_Cmax"][k].copy()
    tide["tide_Cmax"][k] = abs(tide["tide_Cmin"][k])
    tide["tide_Cmin"][k] = -cmax
    return tide


def apply_bry_defects(bry: xr.Dataset) -> xr.Dataset:
    bry = bry.copy(deep=True)
    bry["ubar_west"] = bry["ubar_west"] + 0.2  # net inflow on the west side only
    bry["u_west"] = bry["u_west"] + 0.2
    for side in SIDES:
        bry[f"temp_{side}"] = bry[f"temp_{side}"] + 2.0  # +2 degC vs IC at t0
        t = bry["zeta_time"].values  # M2 tide in boundary zeta
        m2 = 0.3 * np.cos(2 * np.pi * (t - TIDE_START) * 24 / T.CONSTITUENTS["M2"])
        bry[f"zeta_{side}"] = bry[f"zeta_{side}"] + xr.DataArray(m2, dims="zeta_time")
    return bry


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_case(out, defects: bool = False) -> dict[str, Path]:
    """Write the case into ``out``; returns {role: path}."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g = make_grid()
    if defects:
        g["h"][SPIKE] = 4.0 * float(g["h"][SPIKE])
    ini_vp = {"theta_s": 5.0} if defects else None  # IC built with theta_s = 5, ocean.in says 7
    ini = make_ini(g, ini_vp)
    native, coarse = make_frc_native(g), make_frc_coarse(g)
    tide = make_tide(g)
    bry = make_bry(g, make_ini(g))  # boundaries consistent with the ocean.in vertical grid
    if defects:
        ini = apply_ini_defects(ini, g)
        native, coarse = apply_frc_defects(native, coarse)
        tide = apply_tide_defects(tide)
        bry = apply_bry_defects(bry)
    paths = {"grid": out / "grid.nc", "ini": out / "ini.nc", "frc_native": out / "frc_native.nc",
             "frc_coarse": out / "frc_coarse.nc", "tide": out / "tide.nc", "bry": out / "bry.nc",
             "ocean_in": out / "ocean_synthetic.in"}
    _write(g, paths["grid"])
    _write(ini, paths["ini"])
    _write(native, paths["frc_native"], float32=True)
    _write(coarse, paths["frc_coarse"], float32=True)
    _write(tide, paths["tide"])
    _write(bry, paths["bry"], float32=True)
    paths["ocean_in"].write_text(OCEAN_IN)
    if defects:
        paths["nud"] = out / "nud.nc"
        _write(make_nud_defect(g), paths["nud"])
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--defects", action="store_true", help="inject the BUILD_SPEC §6 defects")
    args = ap.parse_args(argv)
    paths = make_case(args.out, args.defects)
    total = sum(p.stat().st_size for p in paths.values()) / 1e6
    for role, p in paths.items():
        print(f"{role:11s} {p}  ({p.stat().st_size / 1e6:.2f} MB)")
    print(f"total {total:.1f} MB")


if __name__ == "__main__":
    main()
