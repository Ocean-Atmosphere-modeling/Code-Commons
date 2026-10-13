"""ROMS terrain-following (s) coordinate: stretching curves and depths.

Mirrors ``Utility/set_scoord.F`` (s, Cs, hc) and ``Nonlinear/set_depth.F`` (z_w, z_r, Hz).
See VERIFY_LOG V16. Depths are negative downward; index k = 0 is the bottom.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def s_coord(N: int, kind: str = "rho", Vstretching: int = 4) -> np.ndarray:
    """Fractional vertical coordinate s in [-1, 0].

    Options 1–4: ``s_w = (k − N)/N`` (k = 0..N), ``s_rho = (k − N − 0.5)/N`` (k = 1..N).
    Option 5 (Souza et al. 2015) uses a quadratic Legendre polynomial in k instead.
    """
    if kind not in ("rho", "w"):
        raise ValueError("kind must be 'rho' or 'w'")
    if Vstretching == 5:
        rN = float(N)
        if kind == "w":
            rk = np.arange(0, N + 1, dtype=float)
        else:
            rk = np.arange(1, N + 1, dtype=float) - 0.5
        s = -(rk * rk - 2 * rk * rN + rk + rN * rN - rN) / (rN * rN - rN) \
            - 0.01 * (rk * rk - rk * rN) / (1.0 - rN)
        if kind == "w":
            s[0], s[-1] = -1.0, 0.0
        return s
    if kind == "w":
        return (np.arange(0, N + 1) - N) / N
    return (np.arange(1, N + 1) - N - 0.5) / N


def stretching(Vstretching: int, theta_s: float, theta_b: float, s) -> np.ndarray:
    """Stretching function C(s) for ``Vstretching`` 1–5 (``set_scoord.F``)."""
    s = np.asarray(s, dtype=float)
    ts, tb = float(theta_s), float(theta_b)
    if Vstretching == 1:  # Song and Haidvogel (1994)
        if ts == 0:
            return s.copy()
        return (1 - tb) * np.sinh(ts * s) / np.sinh(ts) + \
            tb * (0.5 * np.tanh(ts * (s + 0.5)) / np.tanh(0.5 * ts) - 0.5)
    if Vstretching == 2:  # Shchepetkin (2005)
        if ts <= 0:
            return s.copy()
        csur = (1 - np.cosh(ts * s)) / (np.cosh(ts) - 1)
        if tb <= 0:
            return csur
        aw = bw = 1.0
        cbot = np.sinh(tb * (s + 1)) / np.sinh(tb) - 1
        cw = (s + 1) ** aw * (1 + (aw / bw) * (1 - (s + 1) ** bw))
        return cw * csur + (1 - cw) * cbot
    if Vstretching == 3:  # Geyer: high bottom-boundary-layer resolution
        H = 3.0
        cbot = np.log(np.cosh(H * (s + 1) ** tb)) / np.log(np.cosh(H)) - 1
        csur = -np.log(np.cosh(H * np.abs(s) ** ts)) / np.log(np.cosh(H))
        cw = 0.5 * (1 - np.tanh(H * (s + 0.5)))
        return cw * cbot + (1 - cw) * csur
    if Vstretching in (4, 5):  # Shchepetkin (2010); 5 differs only in s
        csur = (1 - np.cosh(ts * s)) / (np.cosh(ts) - 1) if ts > 0 else -s ** 2
        if tb > 0:
            return (np.exp(tb * csur) - 1) / (1 - np.exp(-tb))
        return csur
    raise ValueError(f"Vstretching must be 1..5, got {Vstretching}")


def cs_curves(N: int, Vstretching: int, theta_s: float, theta_b: float) -> dict[str, np.ndarray]:
    """``s_rho, Cs_r, s_w, Cs_w`` exactly as ``set_scoord.F`` stores them.

    ROMS sets ``Cs_w(0) = −1`` and (options 2–5) ``Cs_w(N) = 0`` explicitly.
    """
    s_r, s_w = s_coord(N, "rho", Vstretching), s_coord(N, "w", Vstretching)
    Cs_r = stretching(Vstretching, theta_s, theta_b, s_r)
    Cs_w = stretching(Vstretching, theta_s, theta_b, s_w)
    Cs_w[0] = -1.0
    if Vstretching != 1:
        Cs_w[-1] = 0.0
    return {"s_rho": s_r, "Cs_r": Cs_r, "s_w": s_w, "Cs_w": Cs_w}


def critical_depth(Vtransform: int, Tcline: float, hmin: float | None = None, wet_dry: bool = False) -> float:
    """``hc``: Vtransform 2 → Tcline; Vtransform 1 → min(hmin, Tcline) (``max(hmin, 0)`` with WET_DRY).

    ROMS ``hmin`` is the minimum of ``h`` over the whole grid as read (land included).
    """
    if int(Vtransform) == 2:
        return float(Tcline)
    if int(Vtransform) == 1:
        if hmin is None:
            raise ValueError("Vtransform = 1 needs hmin")
        hm = max(float(hmin), 0.0) if wet_dry else float(hmin)
        return min(hm, float(Tcline))
    raise ValueError(f"Vtransform must be 1 or 2, got {Vtransform}")


_ALIASES = {"N": ("N",), "Vtransform": ("Vtransform",), "Vstretching": ("Vstretching",),
            "theta_s": ("theta_s", "THETA_S"), "theta_b": ("theta_b", "THETA_B"),
            "Tcline": ("Tcline", "TCLINE"), "hc": ("hc",)}


def vertical_params(src, N: int | None = None) -> dict:
    """Pull ``N, Vtransform, Vstretching, theta_s, theta_b, Tcline`` (and ``hc`` if present)
    from a params dict (ocean.in / config) or a dataset (IC or grid file)."""
    out = {}
    for key, names in _ALIASES.items():
        for n in names:
            v = None
            if isinstance(src, xr.Dataset):
                if n in src.variables:
                    v = src[n].values
                elif n in src.attrs:
                    v = src.attrs[n]
            elif isinstance(src, dict):
                v = next((src[k] for k in src if k.lower() == n.lower()), None)
            if v is not None:
                v = np.asarray(v).ravel()[0].item() if np.size(v) else None
                out[key] = v
                break
    if "N" not in out:
        if N is not None:
            out["N"] = N
        elif isinstance(src, xr.Dataset) and "s_rho" in src.sizes:
            out["N"] = src.sizes["s_rho"]
    for k in ("N", "Vtransform", "Vstretching"):
        if k in out and out[k] is not None:
            out[k] = int(out[k])
    return out


def _z(h, zeta, s, C, hc, Vtransform, sdim):
    s = xr.DataArray(s, dims=sdim)
    C = xr.DataArray(C, dims=sdim)
    if int(Vtransform) == 1:
        z0 = hc * s + (h - hc) * C
        z = z0 + zeta * (1 + z0 / h)
    elif int(Vtransform) == 2:
        z0 = (hc * s + h * C) / (hc + h)
        z = zeta + (zeta + h) * z0
    else:
        raise ValueError(f"Vtransform must be 1 or 2, got {Vtransform}")
    hdims = [d for d in h.dims] if isinstance(h, xr.DataArray) else []
    lead = [d for d in z.dims if d != sdim and d not in hdims]
    return z.transpose(*lead, sdim, *hdims)


def _setup(h, N, Vtransform, Vstretching, theta_s, theta_b, Tcline, hc):
    cs = cs_curves(int(N), int(Vstretching), theta_s, theta_b)
    if hc is None:
        hc = critical_depth(Vtransform, Tcline, hmin=float(np.nanmin(np.asarray(h))))
    return cs, float(hc)


def z_w(h, zeta=0.0, *, N, Vtransform, Vstretching, theta_s, theta_b, Tcline, hc=None) -> xr.DataArray:
    """Depth of w points (m, negative), dims (..., s_w, eta, xi). ``z_w[0] = −h``, ``z_w[N] = ζ``."""
    cs, hc = _setup(h, N, Vtransform, Vstretching, theta_s, theta_b, Tcline, hc)
    z = _z(h, zeta, cs["s_w"], cs["Cs_w"], hc, Vtransform, "s_w")
    return z.assign_coords(s_w=cs["s_w"]).rename("z_w").assign_attrs(units="m", long_name="depth of W-points")


def z_r(h, zeta=0.0, *, N, Vtransform, Vstretching, theta_s, theta_b, Tcline, hc=None) -> xr.DataArray:
    """Depth of rho points (m, negative), dims (..., s_rho, eta, xi)."""
    cs, hc = _setup(h, N, Vtransform, Vstretching, theta_s, theta_b, Tcline, hc)
    z = _z(h, zeta, cs["s_rho"], cs["Cs_r"], hc, Vtransform, "s_rho")
    return z.assign_coords(s_rho=cs["s_rho"]).rename("z_rho").assign_attrs(units="m", long_name="depth of RHO-points")


def hz(zw: xr.DataArray) -> xr.DataArray:
    """Layer thickness ``Hz(k) = z_w(k) − z_w(k−1)`` on s_rho."""
    top = zw.isel(s_w=slice(1, None)).variable
    bot = zw.isel(s_w=slice(None, -1)).variable
    d = top - bot
    dims = tuple("s_rho" if x == "s_w" else x for x in d.dims)
    return xr.DataArray(d.data, dims=dims, name="Hz", attrs={"units": "m", "long_name": "layer thickness"})


def depths(h, zeta=0.0, **params) -> xr.Dataset:
    """``z_r``, ``z_w``, ``Hz`` together. ``params`` as returned by :func:`vertical_params`."""
    p = {k: params[k] for k in ("N", "Vtransform", "Vstretching", "theta_s", "theta_b", "Tcline")}
    p["hc"] = params.get("hc")
    zw = z_w(h, zeta, **p)
    return xr.Dataset({"z_r": z_r(h, zeta, **p), "z_w": zw, "Hz": hz(zw)})


def depth_average(var: xr.DataArray, Hz: xr.DataArray) -> xr.DataArray:
    """Thickness-weighted vertical mean ``Σ var·Hz / Σ Hz`` over s_rho (Hz at var's location).

    Coordinate labels are ignored (``join="override"``): file and computed s_rho values may
    differ in the last bits, and an inner join would silently drop levels.
    """
    var, Hz = xr.align(var, Hz, join="override")
    return (var * Hz).sum("s_rho") / Hz.sum("s_rho")
