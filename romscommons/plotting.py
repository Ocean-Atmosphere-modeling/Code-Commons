"""Static matplotlib plots for ROMS inputs.

Conventions: units in every label, land gray, cmocean colormaps, dateline-safe longitudes,
cartopy optional (plain lon/lat axes when it is not installed).
"""
from __future__ import annotations

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from . import grid as G

try:  # cartopy is optional (BUILD_SPEC A4)
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:  # pragma: no cover - depends on environment
    ccrs = cfeature = None
    HAS_CARTOPY = False

LAND = "0.75"

_CMAPS = [  # (substring of variable name, colormap)
    ("phase", cmocean.cm.phase), ("Cangle", cmocean.cm.phase), ("temp", cmocean.cm.thermal),
    ("Tair", cmocean.cm.thermal), ("salt", cmocean.cm.haline), ("speed", cmocean.cm.speed),
    ("amp", cmocean.cm.amp), ("Cmax", cmocean.cm.speed), ("h", cmocean.cm.deep),
    ("zeta", cmocean.cm.balance), ("ubar", cmocean.cm.balance), ("vbar", cmocean.cm.balance),
    ("u", cmocean.cm.balance), ("v", cmocean.cm.balance),
]


def cmap_for(name: str | None, default=cmocean.cm.thermal):
    """cmocean colormap chosen from a variable name."""
    if not name:
        return default
    if name in ("h", "depth"):
        return cmocean.cm.deep
    for key, cm in _CMAPS:
        if key != "h" and key in name:
            return cm
    return default


def label(da: xr.DataArray, name: str | None = None) -> str:
    """``long name [units]`` for axes and colorbars."""
    text = name or da.attrs.get("long_name") or da.name or ""
    units = da.attrs.get("units", "")
    return f"{text} [{units}]" if units else f"{text} [-]"


def _projection(lon):
    return ccrs.PlateCarree(central_longitude=180 if np.nanmax(G.unwrap_lon(lon)) > 180 or G.crosses_dateline(lon) else 0)


def new_map_axes(grid: xr.Dataset, use_cartopy: bool | None = None, figsize=(7, 5)):
    """A figure + axes suited to the grid (GeoAxes when cartopy is available)."""
    use = HAS_CARTOPY if use_cartopy is None else (use_cartopy and HAS_CARTOPY)
    if use and "lon_rho" in grid:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=_projection(grid["lon_rho"].values))
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5")
        gl.top_labels = gl.right_labels = False
        return fig, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _is_geo(ax) -> bool:
    return HAS_CARTOPY and hasattr(ax, "projection")


def _xy(grid, loc):
    if f"lon_{loc}" in grid:
        return G.unwrap_lon(grid[f"lon_{loc}"]).values, grid[f"lat_{loc}"].values, True
    x, y = grid[f"x_{loc}"].values / 1e3, grid[f"y_{loc}"].values / 1e3
    return x, y, False


def _finish_axes(ax, grid, spherical):
    ax.set_facecolor(LAND)
    if _is_geo(ax):
        return
    if spherical:
        lat0 = float(np.nanmean(grid["lat_rho"]))
        ax.set_aspect(1 / np.cos(np.deg2rad(lat0)))
        ax.set_xlabel("Longitude [°E]")
        ax.set_ylabel("Latitude [°N]")
    else:
        ax.set_aspect("equal")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")


def map_field(da: xr.DataArray, grid: xr.Dataset, ax=None, cmap=None, title=None, vmin=None, vmax=None,
              use_cartopy: bool | None = None, add_colorbar: bool = True, cbar_label: str | None = None,
              mask_land: bool = True, **kw):
    """pcolormesh of a 2D field on its C-grid location with land gray. Returns the axes."""
    loc = G.location_of(da)
    if ax is None:
        _, ax = new_map_axes(grid, use_cartopy)
    x, y, spherical = _xy(grid, loc)
    data = da.where(grid[f"mask_{loc}"] > 0) if mask_land else da
    data = data.transpose(*G.LOCATIONS[loc]).values
    cmap = cmap or cmap_for(da.name)
    kw2 = dict(transform=ccrs.PlateCarree()) if _is_geo(ax) else {}
    pc = ax.pcolormesh(x, y, np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", **kw2, **kw)
    _finish_axes(ax, grid, spherical)
    if add_colorbar:
        plt.colorbar(pc, ax=ax, shrink=0.85, label=cbar_label or label(da))
    if title:
        ax.set_title(title)
    return ax


map = map_field  # noqa: A001 - name from BUILD_SPEC §5.7


def outline(grid: xr.Dataset, ax, color="k", lw=1.0, **kw):
    """Draw the ROMS domain boundary (rho perimeter) on ``ax``."""
    p = G.perimeter(grid)
    lon = G.unwrap_lon(p["lon"]).values
    kw2 = dict(transform=ccrs.PlateCarree()) if _is_geo(ax) else {}
    ax.plot(np.r_[lon, lon[0]], np.r_[p["lat"].values, p["lat"].values[0]], color=color, lw=lw, **kw2, **kw)
    return ax


def section(da3d: xr.DataArray, z: xr.DataArray, idx: xr.Dataset, ax=None, cmap=None, title=None,
            vmin=None, vmax=None, grid: xr.Dataset | None = None, add_colorbar=True):
    """Vertical section along transect indices ``idx`` (from :func:`grid.transect_indices`)."""
    sel = dict(eta_rho=xr.DataArray(idx["eta"].values, dims="along"), xi_rho=xr.DataArray(idx["xi"].values, dims="along"))
    v = da3d.isel(sel).transpose("s_rho", "along").values
    zz = z.isel(sel).transpose(..., "along").values
    zz = zz if zz.shape == v.shape else zz[-v.shape[0]:]
    dist = np.broadcast_to(idx["dist"].values, v.shape)
    if grid is not None:
        wet = grid["mask_rho"].isel(sel).values > 0
        v = np.where(wet[None, :], v, np.nan)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor(LAND)
    pc = ax.pcolormesh(dist, zz, np.ma.masked_invalid(v), cmap=cmap or cmap_for(da3d.name), vmin=vmin, vmax=vmax,
                       shading="gouraud")
    if add_colorbar:
        plt.colorbar(pc, ax=ax, label=label(da3d))
    ax.set_xlabel("Distance along transect [km]")
    ax.set_ylabel("z [m]")
    if title:
        ax.set_title(title)
    return ax


def profile(da3d: xr.DataArray, z: xr.DataArray, eta: int, xi: int, ax=None, name: str | None = None, **kw):
    """Vertical profile at rho point (eta, xi)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 4.5))
    ax.plot(da3d.isel(eta_rho=eta, xi_rho=xi).values, z.isel(eta_rho=eta, xi_rho=xi).values, marker=".", label=name, **kw)
    ax.set_xlabel(label(da3d))
    ax.set_ylabel("z [m]")
    return ax


def perimeter_section(values: xr.DataArray, z: xr.DataArray, perim: xr.Dataset, ax=None, cmap=None, title=None,
                      vmin=None, vmax=None, add_colorbar=True):
    """Unrolled perimeter section: values (s_rho, perim) vs distance and depth, corners marked."""
    v = values.transpose("s_rho", "perim").values
    zz = z.transpose("s_rho", "perim").values
    dist = np.broadcast_to(perim["dist"].values, v.shape)
    v = np.where(perim["mask"].values[None, :] > 0, v, np.nan)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor(LAND)
    pc = ax.pcolormesh(dist, zz, np.ma.masked_invalid(v), cmap=cmap or cmap_for(values.name), vmin=vmin, vmax=vmax,
                       shading="nearest")
    for name, d in perim.attrs["corners"].items():
        ax.axvline(d, color="k", lw=0.8, ls="--")
        ax.text(d, 1.01, name.replace("_end", ""), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    if add_colorbar:
        plt.colorbar(pc, ax=ax, label=label(values))
    ax.set_xlabel("Distance along perimeter, counter-clockwise from SW [km]")
    ax.set_ylabel("z [m]")
    if title:
        ax.set_title(title, pad=14)
    return ax


def quiver_earth(u_g: xr.DataArray, v_g: xr.DataArray, grid: xr.Dataset, stride: int = 2, ax=None,
                 scale=None, color="k", key: float | None = None, **kw):
    """Quiver of grid-frame velocity rotated to east/north at rho points."""
    u = G.u2rho(u_g) if "xi_u" in u_g.dims else u_g
    v = G.v2rho(v_g) if "eta_v" in v_g.dims else v_g
    ue, vn = G.to_earth(u.values, v.values, grid["angle"].values)
    wet = grid["mask_rho"].values > 0
    ue, vn = np.where(wet, ue, np.nan), np.where(wet, vn, np.nan)
    sl = (slice(None, None, stride), slice(None, None, stride))
    x, y, spherical = _xy(grid, "rho")
    if ax is None:
        _, ax = new_map_axes(grid)
    kw2 = dict(transform=ccrs.PlateCarree()) if _is_geo(ax) else {}
    q = ax.quiver(x[sl], y[sl], ue[sl], vn[sl], scale=scale, color=color, **kw2, **kw)
    if key:
        ax.quiverkey(q, 0.88, 1.03, key, f"{key} {u_g.attrs.get('units', 'm s-1')}", labelpos="E")
    _finish_axes(ax, grid, spherical)
    return ax


def ellipses(Cmax, Cmin, Cangle, grid: xr.Dataset, stride: int = 3, ax=None, scale: float | None = None,
             npts: int = 40):
    """Tidal current ellipses (east-referenced Cangle) at subsampled wet rho points.

    Red = counter-clockwise (Cmin > 0), blue = clockwise. ``scale`` is degrees of latitude per
    m s-1 (default: largest ellipse spans ~0.8 of the sampling spacing).
    """
    sl = (slice(None, None, stride), slice(None, None, stride))
    x, y, spherical = _xy(grid, "rho")
    cmax, cmin = np.asarray(Cmax)[sl], np.asarray(Cmin)[sl]
    th = np.deg2rad(np.asarray(Cangle)[sl])
    wet = (grid["mask_rho"].values > 0)[sl]
    xs, ys = x[sl], y[sl]
    lat0 = float(np.nanmean(y))
    if scale is None:
        spacing = np.nanmedian(np.abs(np.diff(ys, axis=0))) if ys.shape[0] > 1 else 0.1
        scale = 0.4 * spacing / max(float(np.nanmax(np.where(wet, cmax, np.nan))), 1e-6)
    t = np.linspace(0, 2 * np.pi, npts)
    segs, colors = [], []
    for j in range(cmax.shape[0]):
        for i in range(cmax.shape[1]):
            if not wet[j, i] or not np.isfinite(cmax[j, i]):
                continue
            e = cmax[j, i] * np.cos(t) * np.cos(th[j, i]) - cmin[j, i] * np.sin(t) * np.sin(th[j, i])
            n = cmax[j, i] * np.cos(t) * np.sin(th[j, i]) + cmin[j, i] * np.sin(t) * np.cos(th[j, i])
            fx = scale / np.cos(np.deg2rad(lat0)) if spherical else scale
            segs.append(np.c_[xs[j, i] + e * fx, ys[j, i] + n * scale])
            colors.append("tab:red" if cmin[j, i] > 0 else "tab:blue")
    if ax is None:
        _, ax = new_map_axes(grid)
    kw2 = dict(transform=ccrs.PlateCarree()) if _is_geo(ax) else {}
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=0.8, **kw2))
    if not _is_geo(ax):
        ax.autoscale_view()
    _finish_axes(ax, grid, spherical)
    ax.legend(handles=[Line2D([], [], color="tab:red", label="counter-clockwise"),
                       Line2D([], [], color="tab:blue", label="clockwise")], loc="lower right", fontsize=8)
    return ax


def coastal_patch(grid: xr.Dataset, n: int = 4) -> tuple[int, int]:
    """(i0, j0) of an n×n rho patch containing both land and water (first found)."""
    m = grid["mask_rho"].values
    M, L = m.shape
    for j in range(1, M - n):
        for i in range(1, L - n):
            w = m[j:j + n, i:i + n]
            if 0 < w.sum() < w.size:
                return i, j
    return 1, 1


def cgrid_zoom(grid: xr.Dataset, i0: int, j0: int, n: int = 4, ax=None):
    """Arakawa-C primer: rho/u/v/psi points and masks on an n×n patch (index space)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    mr, mu, mv, mp = (grid[f"mask_{k}"].values for k in ("rho", "u", "v", "psi"))
    for j in range(j0, j0 + n):
        for i in range(i0, i0 + n):
            if mr[j, i] == 0:
                ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=LAND, zorder=0))
    for k in range(n + 1):  # cell edges pass through psi points
        ax.plot([i0 - 0.5, i0 + n - 0.5], [j0 - 0.5 + k] * 2, color="0.6", lw=0.6, zorder=1)
        ax.plot([i0 - 0.5 + k] * 2, [j0 - 0.5, j0 + n - 0.5], color="0.6", lw=0.6, zorder=1)

    def pts(mask, di, dj, ni, nj, marker, color, name):
        xs, ys, wet = [], [], []
        for j in range(j0, j0 + nj):
            for i in range(i0, i0 + ni):
                if 0 <= j < mask.shape[0] and 0 <= i < mask.shape[1]:
                    xs.append(i + di), ys.append(j + dj), wet.append(mask[j, i] > 0)
        xs, ys, wet = np.asarray(xs), np.asarray(ys), np.asarray(wet)
        ax.scatter(xs[wet], ys[wet], marker=marker, c=color, s=70, zorder=3, label=f"{name} (water)")
        ax.scatter(xs[~wet], ys[~wet], marker=marker, facecolors="none", edgecolors=color, s=70, zorder=3,
                   label=f"{name} (land)")

    pts(mr, 0, 0, n, n, "o", "tab:blue", "rho")
    pts(mu, 0.5, 0, n - 1, n, ">", "tab:red", "u")
    pts(mv, 0, 0.5, n, n - 1, "^", "tab:green", "v")
    pts(mp, 0.5, 0.5, n - 1, n - 1, "P", "k", "psi")
    ax.set_xlim(i0 - 0.6, i0 + n - 0.4)
    ax.set_ylim(j0 - 0.6, j0 + n - 0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("xi index (rho) [-]")
    ax.set_ylabel("eta index (rho) [-]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.set_title("Arakawa C-grid: u between rho in xi, v between rho in eta")
    return ax
