"""ROMS Arakawa-C grid helpers: staggering, rotation, metrics, boundaries, transects.

Index convention (0-based Python vs ROMS Fortran):

* rho point ``xi_rho = i`` is ROMS ``i`` (0 .. Lm+1); the outer ring is the boundary row.
* u point ``xi_u = i`` sits between rho ``i`` and ``i+1`` (ROMS u index ``i+1``).
* v point ``eta_v = j`` sits between rho ``j`` and ``j+1`` (ROMS v index ``j+1``).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .io import SIDES, open_nc

LOCATIONS = {"rho": ("eta_rho", "xi_rho"), "u": ("eta_u", "xi_u"), "v": ("eta_v", "xi_v"),
             "psi": ("eta_psi", "xi_psi")}

EARTH_RADIUS = 6371e3  # m


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _is_true(v) -> bool:
    v = np.asarray(v).ravel()[0] if np.size(v) else v
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode()
    return str(v).strip().upper() in ("T", "1", "TRUE", ".TRUE.")


def load_grid(path) -> xr.Dataset:
    """Open a ROMS grid file and fill in what ROMS would derive if it is missing.

    Adds ``mask_u/v/psi`` from ``mask_rho`` (product of neighbours, as ROMS does), lon/lat at
    u/v/psi points (averages of rho), and ``attrs["spherical"]``.
    """
    g = open_nc(path)
    if "mask_rho" not in g:
        g["mask_rho"] = xr.ones_like(g["h"])
    m = g["mask_rho"]
    if "mask_u" not in g:
        g["mask_u"] = _pair(m, "xi_rho", "xi_u", np.multiply)
    if "mask_v" not in g:
        g["mask_v"] = _pair(m, "eta_rho", "eta_v", np.multiply)
    if "mask_psi" not in g:
        g["mask_psi"] = _pair(g["mask_u"], "eta_u", "eta_psi", np.multiply).rename({"xi_u": "xi_psi"})
    sph = g["spherical"].values if "spherical" in g else ("lon_rho" in g)
    g.attrs["spherical"] = _is_true(sph) if "spherical" in g else bool(sph)
    for coord in ("lon", "lat"):
        if f"{coord}_rho" not in g:
            continue
        r = g[f"{coord}_rho"]
        if coord == "lon":  # average continuous longitudes, then restore the file's convention
            conv = lon_convention(r)
            back = (lambda x: wrap_lon(x, conv)) if conv in ("0-360", "-180-180") else (lambda x: x)
            r = unwrap_lon(r)
        else:
            back = lambda x: x  # noqa: E731
        if f"{coord}_u" not in g:
            g[f"{coord}_u"] = back(rho2u(r))
        if f"{coord}_v" not in g:
            g[f"{coord}_v"] = back(rho2v(r))
        if f"{coord}_psi" not in g:
            g[f"{coord}_psi"] = back(_psi_avg(r))
    if "angle" not in g:
        g["angle"] = xr.zeros_like(g["h"])
    return g


def _psi_avg(r: xr.DataArray) -> xr.DataArray:
    u = rho2u(r)  # (eta_u, xi_u) == (eta_rho, xi_rho - 1)
    p = 0.5 * (u.isel(eta_u=slice(None, -1)).variable + u.isel(eta_u=slice(1, None)).variable)
    return xr.DataArray(p.data, dims=("eta_psi", "xi_psi"), attrs=r.attrs)


def location_of(da: xr.DataArray) -> str:
    """'rho', 'u', 'v' or 'psi' from the horizontal dims of ``da``."""
    for loc, (eta, xi) in LOCATIONS.items():
        if eta in da.dims and xi in da.dims:
            return loc
    raise ValueError(f"cannot tell the C-grid location of dims {da.dims}")


def expected_dims(grid: xr.Dataset, location: str) -> dict[str, int]:
    """Expected (eta, xi) dims at a location: xi_rho = Lm+2, xi_u = Lm+1, eta_v = Mm+1, ..."""
    L, M = grid.sizes["xi_rho"], grid.sizes["eta_rho"]
    eta, xi = LOCATIONS[location]
    sizes = {"rho": (M, L), "u": (M, L - 1), "v": (M - 1, L), "psi": (M - 1, L - 1)}[location]
    return {eta: sizes[0], xi: sizes[1]}


def mask(grid: xr.Dataset, location: str) -> xr.DataArray:
    return grid[f"mask_{location}"]


def lonlat(grid: xr.Dataset, location: str):
    return grid[f"lon_{location}"], grid[f"lat_{location}"]


# ---------------------------------------------------------------------------
# Staggering
# ---------------------------------------------------------------------------

def _rename_dims(var: xr.Variable, old: str, new: str) -> tuple:
    return tuple(new if d == old else d for d in var.dims)


def _pair(da: xr.DataArray, dim: str, new_dim: str, op) -> xr.DataArray:
    a = da.variable.isel({dim: slice(None, -1)})
    b = da.variable.isel({dim: slice(1, None)})
    out = op(a, b)
    dims = _rename_dims(out, dim, new_dim)
    other = {"xi_rho": ("eta_rho", "eta_u"), "eta_rho": ("xi_rho", "xi_v")}.get(dim)
    if other:  # the other horizontal dim takes the new location's name too
        dims = tuple(other[1] if d == other[0] else d for d in dims)
    return xr.DataArray(out.data, dims=dims, attrs=da.attrs, name=da.name)


def _avg(da, dim, new_dim, mask=None):
    if mask is None:
        return _pair(da, dim, new_dim, lambda a, b: 0.5 * (a + b))
    m = mask.astype(float)
    num = _pair(da.fillna(0) * m, dim, new_dim, lambda a, b: a + b)
    den = _pair(m, dim, new_dim, lambda a, b: a + b)
    return (num / den).where(den > 0)


def rho2u(da: xr.DataArray, mask: xr.DataArray | None = None) -> xr.DataArray:
    """Average rho-point values to u points (2-point mean; mask-aware if ``mask`` given)."""
    return _avg(da, "xi_rho", "xi_u", mask).assign_attrs(da.attrs)


def rho2v(da: xr.DataArray, mask: xr.DataArray | None = None) -> xr.DataArray:
    """Average rho-point values to v points (2-point mean; mask-aware if ``mask`` given)."""
    return _avg(da, "eta_rho", "eta_v", mask).assign_attrs(da.attrs)


def _to_rho(da, dim, new_dim, other_old, other_new):
    first = da.isel({dim: slice(0, 1)})
    last = da.isel({dim: slice(-1, None)})
    padded = xr.concat([first, da, last], dim=dim)  # edge rho points take the nearest face value
    a = padded.variable.isel({dim: slice(None, -1)})
    b = padded.variable.isel({dim: slice(1, None)})
    dims = tuple(new_dim if d == dim else other_new if d == other_old else d for d in a.dims)
    return xr.DataArray((0.5 * (a + b)).data, dims=dims, attrs=da.attrs, name=da.name)


def u2rho(da: xr.DataArray) -> xr.DataArray:
    """u points -> rho points (interior: 2-point mean; edges: nearest u value)."""
    return _to_rho(da, "xi_u", "xi_rho", "eta_u", "eta_rho")


def v2rho(da: xr.DataArray) -> xr.DataArray:
    """v points -> rho points (interior: 2-point mean; edges: nearest v value)."""
    return _to_rho(da, "eta_v", "eta_rho", "xi_v", "xi_rho")


# ---------------------------------------------------------------------------
# Rotation (angle in radians at rho points: angle between xi and east)
# ---------------------------------------------------------------------------

def to_grid(u_e, v_n, angle):
    """East/north components -> grid (xi, eta) components (``set_data.F`` rotation)."""
    c, s = np.cos(angle), np.sin(angle)
    return u_e * c + v_n * s, v_n * c - u_e * s


def to_earth(u_g, v_g, angle):
    """Grid (xi, eta) components -> east/north components."""
    c, s = np.cos(angle), np.sin(angle)
    return u_g * c - v_g * s, u_g * s + v_g * c


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dx(grid):
    """Grid spacing in xi (m) = 1/pm."""
    return (1 / grid["pm"]).assign_attrs(units="m", long_name="grid spacing in xi")


def dy(grid):
    """Grid spacing in eta (m) = 1/pn."""
    return (1 / grid["pn"]).assign_attrs(units="m", long_name="grid spacing in eta")


def area(grid):
    """Cell area (m2) = 1/(pm pn)."""
    return (1 / (grid["pm"] * grid["pn"])).assign_attrs(units="m2", long_name="cell area")


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------

def side_slices(side: str, location: str = "rho") -> dict:
    """``isel`` indexers selecting the boundary row/column of a side at a location.

    Boundary files hold exactly these slices: ``temp_west = temp.isel(xi_rho=0)``,
    ``u_west = u.isel(xi_u=0)``, ``v_west = v.isel(xi_v=0)``, ...
    """
    eta, xi = LOCATIONS[location]
    return {"west": {xi: 0}, "east": {xi: -1}, "south": {eta: 0}, "north": {eta: -1}}[side]


def side_dim(side: str, location: str = "rho") -> str:
    """The dim that runs along a side (eta for west/east, xi for south/north)."""
    eta, xi = LOCATIONS[location]
    return eta if side in ("west", "east") else xi


def normal_location(side: str) -> str:
    """Location of the velocity normal to a side: u on west/east, v on south/north."""
    return "u" if side in ("west", "east") else "v"


_INFLOW_SIGN = {"west": 1, "south": 1, "east": -1, "north": -1}


def inflow_sign(side: str) -> int:
    """+1 if positive normal velocity on this side is inflow (W: +u, S: +v, E: -u, N: -v)."""
    return _INFLOW_SIGN[side]


_PERIM_ORDER = (  # side, reverse along-side order? (counter-clockwise walk starting at SW)
    ("south", False), ("east", False), ("north", True), ("west", True))
_CORNERS = {"south": "SW", "east": "SE", "north": "NE", "west": "NW"}


def perimeter(grid: xr.Dataset) -> xr.Dataset:
    """Counter-clockwise unrolled perimeter of rho points.

    Order: S (west→east), E (south→north), N (east→west), W (north→south). Corner points
    appear on both sides they belong to (zero distance between the copies). Returns a dataset
    on dim ``perim`` with ``side``, ``eta``, ``xi``, ``dist`` (km), ``lon``, ``lat``, ``mask``,
    and ``attrs["corners"]`` = {"SW": 0, "SE": d, "NE": d, "NW": d, "SW_end": total}.
    """
    M, L = grid.sizes["eta_rho"], grid.sizes["xi_rho"]
    sides, etas, xis, steps = [], [], [], []
    pm, pn = grid["pm"].values, grid["pn"].values
    corners = {}
    dist = 0.0
    dists = []
    for side, rev in _PERIM_ORDER:
        if side in ("south", "north"):
            jj = np.full(L, 0 if side == "south" else M - 1)
            ii = np.arange(L)
            spacing = 1 / pm[jj, ii]
        else:
            ii = np.full(M, L - 1 if side == "east" else 0)
            jj = np.arange(M)
            spacing = 1 / pn[jj, ii]
        if rev:
            jj, ii, spacing = jj[::-1], ii[::-1], spacing[::-1]
        corners[_CORNERS[side]] = dist / 1e3
        for n in range(len(jj)):
            if n > 0:
                dist += 0.5 * (spacing[n - 1] + spacing[n])
            dists.append(dist / 1e3)
        sides += [side] * len(jj)
        etas.append(jj)
        xis.append(ii)
    corners["SW_end"] = dist / 1e3
    eta, xi = np.concatenate(etas), np.concatenate(xis)
    ds = xr.Dataset({"side": ("perim", np.array(sides)), "eta": ("perim", eta), "xi": ("perim", xi),
                     "dist": ("perim", np.array(dists), {"units": "km", "long_name": "distance along perimeter"}),
                     "mask": ("perim", grid["mask_rho"].values[eta, xi])})
    if "lon_rho" in grid:
        ds["lon"] = ("perim", grid["lon_rho"].values[eta, xi])
        ds["lat"] = ("perim", grid["lat_rho"].values[eta, xi])
    ds.attrs["corners"] = corners
    return ds


def unroll(values: dict, along: str = "perim") -> xr.DataArray:
    """Concatenate per-side boundary arrays into perimeter order (see :func:`perimeter`).

    ``values`` maps side -> DataArray whose last dim runs along that side with rho-point
    length (e.g. ``temp_<side>``, ``zeta_<side>``, ``u_west``/``v_south`` normal velocity).
    North and west are reversed so the walk is counter-clockwise.
    """
    parts = []
    for side, rev in _PERIM_ORDER:
        da = values[side]
        dim = da.dims[-1]
        if rev:
            da = da.isel({dim: slice(None, None, -1)})
        parts.append(da.rename({dim: along}).drop_vars([c for c in da.coords if dim in da[c].dims], errors="ignore"))
    return xr.concat(parts, dim=along, coords="minimal", compat="override")


def open_sides(lbc, grid: xr.Dataset, var: str = "zeta") -> list[str]:
    """Sides that are open: LBC keyword not ``Clo``/``Per`` and at least one wet point.

    ``lbc`` is the parsed ocean.in LBC dict (uses entry ``var``), or a list of side names
    (config ``open_boundaries``) when no ocean.in is available.
    """
    if isinstance(lbc, (list, tuple)):
        candidates = [s.lower() for s in lbc]
    else:
        entry = lbc[var]
        candidates = [s for s in SIDES if str(entry[s]).upper() not in ("CLO", "PER")]
    return [s for s in candidates if float(grid["mask_rho"].isel(side_slices(s)).sum()) > 0]


# ---------------------------------------------------------------------------
# Longitudes and transects
# ---------------------------------------------------------------------------

def wrap_lon(lon, convention: str):
    """Wrap longitudes to ``"0-360"`` or ``"-180-180"``."""
    if convention == "0-360":
        return lon % 360
    if convention == "-180-180":
        return ((lon + 180) % 360) - 180
    raise ValueError(f"unknown convention {convention!r}")


def lon_convention(lon) -> str:
    """``"0-360"`` if any value > 180, ``"-180-180"`` if any value < 0, else ``"either"``."""
    lon = np.asarray(lon)
    lo, hi = np.nanmin(lon), np.nanmax(lon)
    if lo < 0 and hi > 180:
        return "mixed"
    if hi > 180:
        return "0-360"
    if lo < 0:
        return "-180-180"
    return "either"


def unwrap_lon(lon):
    """Longitudes made continuous for plotting (choose the wrap with the smaller span)."""
    a = wrap_lon(np.asarray(lon, dtype=float), "0-360")
    b = wrap_lon(np.asarray(lon, dtype=float), "-180-180")
    out = a if (np.nanmax(a) - np.nanmin(a)) <= (np.nanmax(b) - np.nanmin(b)) else b
    return xr.DataArray(out, dims=lon.dims, coords=lon.coords, attrs=lon.attrs) if isinstance(lon, xr.DataArray) else out


def crosses_dateline(lon) -> bool:
    """True if the domain spans the 180° meridian."""
    a = wrap_lon(np.asarray(lon, dtype=float), "0-360")
    return bool(np.nanmin(a) < 180 < np.nanmax(a)) and \
        (np.nanmax(a) - np.nanmin(a)) < 180


def nearest_index(grid: xr.Dataset, lon: float, lat: float, location: str = "rho") -> tuple[int, int]:
    """(eta, xi) of the grid point nearest to (lon, lat); dateline-safe."""
    glon, glat = grid[f"lon_{location}"].values, grid[f"lat_{location}"].values
    dlon = ((glon - lon + 180) % 360) - 180
    d2 = (dlon * np.cos(np.deg2rad(lat))) ** 2 + (glat - lat) ** 2
    j, i = np.unravel_index(np.nanargmin(d2), d2.shape)
    return int(j), int(i)


def _bresenham(j0, i0, j1, i1):
    pts = []
    di, dj = abs(i1 - i0), -abs(j1 - j0)
    si, sj = (1 if i0 < i1 else -1), (1 if j0 < j1 else -1)
    err = di + dj
    i, j = i0, j0
    while True:
        pts.append((j, i))
        if i == i1 and j == j1:
            break
        e2 = 2 * err
        if e2 >= dj:
            err += dj
            i += si
        if e2 <= di:
            err += di
            j += sj
    return pts


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a)) / 1e3


def transect_indices(grid: xr.Dataset, start=None, end=None) -> xr.Dataset:
    """Rho indices along a straight line in index space between the points nearest to
    ``start`` and ``end`` ((lon, lat) pairs). ``None`` -> the middle eta row.

    Returns a dataset on dim ``along`` with ``eta``, ``xi``, ``dist`` (km), ``lon``, ``lat``.
    Use ``da.isel(eta_rho=t.eta, xi_rho=t.xi)`` to extract a section.
    """
    if start is None or end is None:
        j = grid.sizes["eta_rho"] // 2
        pts = [(j, i) for i in range(grid.sizes["xi_rho"])]
    else:
        j0, i0 = nearest_index(grid, *start)
        j1, i1 = nearest_index(grid, *end)
        pts = list(dict.fromkeys(_bresenham(j0, i0, j1, i1)))
    eta = np.array([p[0] for p in pts])
    xi = np.array([p[1] for p in pts])
    lon = grid["lon_rho"].values[eta, xi]
    lat = grid["lat_rho"].values[eta, xi]
    step = np.r_[0.0, haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])]
    return xr.Dataset({"eta": ("along", eta), "xi": ("along", xi), "lon": ("along", lon), "lat": ("along", lat),
                       "dist": ("along", np.cumsum(step), {"units": "km", "long_name": "distance along transect"})})
