"""Tidal constituents, ellipse conversions, and reconstruction as ROMS does it.

Reconstruction mirrors ``Nonlinear/set_tides.F`` (VERIFY_LOG V8):

* ``ζ(t) = Σ Eamp·cos(ω(t − TIDE_START) − Ephase)``
* ``u = Cmax·cosθ·cosψ − Cmin·sinθ·sinψ``, ``v = Cmax·sinθ·cosψ + Cmin·cosθ·sinψ``
  with ``θ = Cangle − angle`` (Cangle measured from east), ``ψ = ω(t − TIDE_START) − Cphase``.

ROMS applies no nodal corrections: the file's amplitudes and phases must already contain
them for TIDE_START.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

#: Constituent periods in hours.
CONSTITUENTS = {
    "M2": 12.4206012, "S2": 12.0, "N2": 12.65834751, "K2": 11.96723606,
    "K1": 23.93447213, "O1": 25.81933871, "P1": 24.06589, "Q1": 26.86835,
    "M4": 6.210300601, "MS4": 6.103339, "MN4": 6.269174,
    "Mf": 327.8599, "Mm": 661.3093,
}


def identify(periods, tol_h: float = 1e-3) -> list[str | None]:
    """Name of each period (hours) within ``tol_h`` of a known constituent, else None."""
    names = list(CONSTITUENTS)
    ref = np.array([CONSTITUENTS[n] for n in names])
    out = []
    for p in np.atleast_1d(np.asarray(periods, dtype=float)):
        k = int(np.argmin(np.abs(ref - p)))
        out.append(names[k] if abs(ref[k] - p) <= tol_h else None)
    return out


def ap2ep(Au, PHIu, Av, PHIv):
    """Amplitude/phase of u and v -> ellipse ``(Cmax, Cmin, Cangle, Cphase)`` (Xu 2002).

    Input phases in degrees with ``u(t) = Au·cos(ωt − PHIu)``. ``Cmin`` is signed (positive =
    counter-clockwise rotation); ``Cangle`` in [0, 180) degrees counter-clockwise from east;
    ``Cphase`` in [0, 360) degrees.
    """
    u = np.asarray(Au) * np.exp(-1j * np.deg2rad(PHIu))
    v = np.asarray(Av) * np.exp(-1j * np.deg2rad(PHIv))
    wp = (u + 1j * v) / 2  # counter-clockwise circle
    wm = np.conj(u - 1j * v) / 2  # clockwise circle
    Wp, Wm = np.abs(wp), np.abs(wm)
    THp, THm = np.angle(wp), np.angle(wm)
    Cmax = Wp + Wm
    Cmin = Wp - Wm
    pha = np.rad2deg((THm - THp) / 2)
    inc = np.rad2deg((THm + THp) / 2)
    pha = np.mod(pha + 360, 360)
    inc = np.mod(inc + 360, 360)
    flip = inc >= 180  # map inclination to [0, 180): rotate the axis by 180, shift phase by 180
    inc = np.where(flip, inc - 180, inc)
    pha = np.mod(np.where(flip, pha + 180, pha), 360)
    return Cmax, Cmin, inc, pha


def ep2ap(Cmax, Cmin, Cangle, Cphase):
    """Ellipse parameters -> ``(Au, PHIu, Av, PHIv)`` (inverse of :func:`ap2ep`, degrees)."""
    Cmax, Cmin = np.asarray(Cmax, dtype=float), np.asarray(Cmin, dtype=float)
    Wp, Wm = (Cmax + Cmin) / 2, (Cmax - Cmin) / 2
    THp = np.deg2rad(np.asarray(Cangle) - np.asarray(Cphase))
    THm = np.deg2rad(np.asarray(Cangle) + np.asarray(Cphase))
    wp, wm = Wp * np.exp(1j * THp), Wm * np.exp(1j * THm)
    cAu = wp + np.conj(wm)
    cAv = -1j * (wp - np.conj(wm))
    return (np.abs(cAu), np.mod(-np.rad2deg(np.angle(cAu)), 360),
            np.abs(cAv), np.mod(-np.rad2deg(np.angle(cAv)), 360))


def _elapsed_seconds(t, tide_start) -> xr.DataArray:
    if isinstance(t, xr.DataArray):
        tv, dim = t.values, t.dims[0]
    else:
        tv, dim = np.asarray(t), "time"
    t0 = np.datetime64(pd.Timestamp(tide_start), "ns")
    secs = (tv.astype("datetime64[ns]") - t0).astype("timedelta64[ns]").astype(np.int64) / 1e9
    return xr.DataArray(secs, dims=dim, coords={dim: tv})


def _per_constituent(x, like: xr.DataArray) -> xr.DataArray:
    if isinstance(x, xr.DataArray):
        return xr.DataArray(x.values, dims=like.dims[0])
    return xr.DataArray(np.asarray(x, dtype=float), dims=like.dims[0])


def _omega(period_h, like):
    return 2 * np.pi / (_per_constituent(period_h, like) * 3600.0)


def _strip(da: xr.DataArray) -> xr.DataArray:
    """Drop the constituent coordinate so summing over it never aligns on period labels."""
    return xr.DataArray(da.data, dims=da.dims, coords={k: v for k, v in da.coords.items()
                                                      if da.dims[0] not in v.dims})


def reconstruct_ssh(Eamp, Ephase, period_h, t, tide_start) -> xr.DataArray:
    """Tidal elevation as ROMS builds it: ``Σ Eamp·cos(ω(t − tide_start) − Ephase)``.

    ``Eamp``/``Ephase`` (m, degrees) have the constituent dim first; ``t`` is datetime64;
    ``tide_start`` is the date of TIDE_START.
    """
    Eamp, Ephase = _strip(Eamp), _strip(Ephase)
    dt = _elapsed_seconds(t, tide_start)
    arg = _omega(period_h, Eamp) * dt - np.deg2rad(Ephase)
    z = (Eamp * np.cos(arg)).sum(Eamp.dims[0]).transpose(dt.dims[0], ...)
    return z.rename("zeta_tide").assign_attrs(units="m")


def reconstruct_uv(Cmax, Cmin, Cangle, Cphase, period_h, t, tide_start, angle=0.0):
    """Tidal currents as ROMS builds them (``set_tides.F``) at rho points.

    ``angle = 0`` gives east/north components; pass the grid ``angle`` (radians) to get grid
    (xi, eta) components, which is what ROMS computes before averaging to u/v points.
    """
    Cmax, Cmin, Cangle, Cphase = map(_strip, (Cmax, Cmin, Cangle, Cphase))
    dt = _elapsed_seconds(t, tide_start)
    theta = np.deg2rad(Cangle) - angle
    psi = _omega(period_h, Cmax) * dt - np.deg2rad(Cphase)
    u = Cmax * np.cos(theta) * np.cos(psi) - Cmin * np.sin(theta) * np.sin(psi)
    v = Cmax * np.sin(theta) * np.cos(psi) + Cmin * np.cos(theta) * np.sin(psi)
    k, td = Cmax.dims[0], dt.dims[0]
    u = u.sum(k).transpose(td, ...).rename("u_tide").assign_attrs(units="m s-1")
    v = v.sum(k).transpose(td, ...).rename("v_tide").assign_attrs(units="m s-1")
    return u, v
