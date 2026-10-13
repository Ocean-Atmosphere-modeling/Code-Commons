"""Smoke tests: every plotting helper runs on the synthetic case, with and without cartopy."""
import matplotlib.pyplot as plt
import pytest

from romscommons import grid as G
from romscommons import io
from romscommons import plotting as P
from romscommons import vertical as V


@pytest.fixture(scope="module")
def case(synthetic):
    g = G.load_grid(synthetic["grid"])
    ini = io.open_nc(synthetic["ini"]).isel(ocean_time=0)
    tide = io.open_nc(synthetic["tide"])
    z = V.depths(g.h, ini.zeta, **V.vertical_params(ini))
    return g, ini, tide, z


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("use_cartopy", [False, True])
def test_map_field(case, use_cartopy):
    g, ini, _, _ = case
    if use_cartopy and not P.HAS_CARTOPY:
        pytest.skip("cartopy not installed")
    ax = P.map(g.h, g, use_cartopy=use_cartopy, title="h")
    P.outline(g, ax)
    P.map_field(ini.ubar, g, use_cartopy=use_cartopy)  # u-point field
    P.quiver_earth(ini.ubar, ini.vbar, g, stride=3, ax=ax, key=0.1)


def test_cmap_and_label(case):
    g, ini, _, _ = case
    assert P.cmap_for("temp").name == "thermal" and P.cmap_for("h").name == "deep"
    assert P.cmap_for("tide_Ephase").name == "phase" and P.cmap_for("salt").name == "haline"
    assert P.label(ini.temp) == "potential temperature [Celsius]"


def test_section_profile_perimeter(case):
    g, ini, _, z = case
    t = G.transect_indices(g, None, None)
    P.section(ini.temp, z.z_r, t, grid=g)
    P.profile(ini.temp, z.z_r, 16, 30, name="slope")
    per = G.perimeter(g)
    temp = G.unroll({s: ini.temp.isel(G.side_slices(s)) for s in ("west", "south", "east", "north")})
    zp = G.unroll({s: z.z_r.isel(G.side_slices(s)) for s in ("west", "south", "east", "north")})
    P.perimeter_section(temp, zp, per, title="temp")


def test_ellipses_and_cgrid(case):
    g, _, tide, _ = case
    P.ellipses(tide.tide_Cmax[0], tide.tide_Cmin[0], tide.tide_Cangle[0], g, stride=4)
    i0, j0 = P.coastal_patch(g)
    m = g.mask_rho.values[j0:j0 + 4, i0:i0 + 4]
    assert 0 < m.sum() < 16
    P.cgrid_zoom(g, i0, j0)
