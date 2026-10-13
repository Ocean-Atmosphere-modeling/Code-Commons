import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from romscommons import io
from romscommons.checks import Status

from conftest import FIXTURES


# ---------------------------------------------------------------- Fortran values
@pytest.mark.parametrize("tok, expected", [
    ("300.0d0", 300.0), ("1.0d-3", 1e-3), ("1.0D+02", 100.0), ("5.0E-6", 5e-6), ("12", 12), ("-3", -3),
    ("T", True), ("F", False), (".TRUE.", True), ("grid.nc", "grid.nc"), ("Cha", "Cha"), (".5", 0.5),
])
def test_parse_fortran_value(tok, expected):
    assert io.parse_fortran_value(tok) == expected


# ---------------------------------------------------------------- ocean.in
@pytest.fixture(scope="module")
def oin():
    return io.parse_ocean_in(FIXTURES / "ocean_fixture.in")


def test_scalars_and_operators(oin):
    assert oin["Lm"] == 40 and oin["N"] == 12 and oin["NAT"] == 2
    assert oin["DT"] == 180.0 and oin["NDTFAST"] == 30
    assert oin["DSTART"] == 9497.5  # single '='
    assert oin["DCRIT"] == pytest.approx(0.1)
    assert oin["TITLE"] == "Fixture: hard cases for the ocean.in parser"


def test_repeat_and_lists(oin):
    assert oin["TNUDG"] == [10.0, 10.0]
    assert oin["Akt_bak"] == [5e-6, 1e-6]
    assert oin["LnudgeTCLM"] == [True, False]
    assert oin["Hout(idUvel)"] is True


def test_lbc_order_is_west_south_east_north(oin):
    lbc = oin["LBC"]
    assert lbc["zeta"] == {"west": "Cha", "south": "Clo", "east": "Che", "north": "Gra"}
    assert lbc["ubar"]["east"] == "Shc" and lbc["ubar"]["north"] == "Red"
    assert lbc["u"] == {"west": "RadNud", "south": "Clo", "east": "Rad", "north": "Mix"}


def test_lbc_tracer_continuation(oin):
    lbc = oin["LBC"]
    assert lbc["temp"] == {"west": "RadNud", "south": "Clo", "east": "Rad", "north": "Cla"}
    assert lbc["salt"] == {"west": "Gra", "south": "Clo", "east": "Per", "north": "Nes"}
    assert oin["ad_LBC"]["zeta"]["west"] == "Per"


def test_volcons(oin):
    assert oin["VolCons"] == {"west": True, "east": False, "south": False, "north": False}


def test_multifile_names(oin):
    assert oin["FRCNAME"] == ["winds_2017.nc", "winds_2018.nc", "heat_2017.nc", "heat_2018.nc", "tair.nc"]
    assert oin["GRDNAME"] == "../grid/my_grid.nc"


def test_get_ci(oin):
    assert io.get_ci(oin, "dcrit") == pytest.approx(0.1)
    assert io.get_ci(oin, "missing", 7) == 7


def test_parse_real_roms_in(roms_src):
    """The ROMS template has every hard case; parse it end to end."""
    oin = io.parse_ocean_in(roms_src / "User" / "External" / "roms.in")
    assert len(oin) > 100
    assert oin["Lm"] == 41 and oin["Mm"] == 80 and oin["N"] == 16
    assert oin["DT"] == 300.0 and oin["NTIMES"] == 1440
    assert oin["TIME_REF"] == 0.0
    assert oin["LBC"]["zeta"] == {"west": "Per", "south": "Clo", "east": "Per", "north": "Clo"}
    assert oin["LBC"]["temp"]["west"] == "Per" and oin["LBC"]["salt"]["north"] == "Clo"
    assert oin["VolCons"]["west"] is False
    assert oin["TNUDG"] == [0.0, 0.0]
    assert oin["FRCNAME"] == "roms_frc.nc"


# ---------------------------------------------------------------- time
def test_time_ref():
    assert io.time_ref(20000101.0) == pd.Timestamp("2000-01-01")
    assert io.time_ref(20020115.5) == pd.Timestamp("2002-01-15 12:00")


@pytest.mark.parametrize("special", [0, -1, -2])
def test_time_ref_special_values_raise(special):
    with pytest.raises(NotImplementedError, match="TIME_REF"):
        io.time_ref(special)


def test_run_window():
    p = {"TIME_REF": 20000101.0, "DSTART": 9497.0, "NTIMES": 4800, "DT": 180.0}
    start, end = io.run_window(p)
    assert start == pd.Timestamp("2026-01-01") and end == pd.Timestamp("2026-01-11")
    s2, e2 = io.run_window(p, ini_time="2026-01-02")
    assert s2 == pd.Timestamp("2026-01-02") and e2 == pd.Timestamp("2026-01-12")


def test_decode_time_values():
    t = io.decode_time_values([1.5], "days since 2000-01-01 00:00:00")
    assert t[0] == np.datetime64("2000-01-02T12:00")
    t = io.decode_time_values([3600.0], "seconds", TIME_REF=20000101.0)
    assert t[0] == np.datetime64("2000-01-01T01:00")
    t = io.decode_time_values([24.0], "hours since 1990-01-01 00:00:00 GMT")
    assert t[0] == np.datetime64("1990-01-02")
    with pytest.raises(ValueError):
        io.decode_time_values([1.0], "days")


def test_open_nc_decodes_bare_units_with_time_ref(tmp_path):
    ds = xr.Dataset({"ocean_time": ("ocean_time", [86400.0], {"units": "seconds"}),
                     "tide_period": ("tide_period", [12.42], {"units": "hours"}),
                     "x": ("ocean_time", [1.0])})
    ds.to_netcdf(tmp_path / "a.nc")
    out = io.open_nc(tmp_path / "a.nc", TIME_REF=20000101.0)
    assert out["ocean_time"].values[0] == np.datetime64("2000-01-02")
    assert out["tide_period"].dtype.kind == "f"  # periods are never decoded


def test_open_nc_cf_units(synthetic):
    ini = io.open_nc(synthetic["ini"])
    assert ini["ocean_time"].values[0] == np.datetime64("2026-01-01")
    frc = io.open_nc(synthetic["frc_native"])
    assert frc["wind_time"].values[-1] == np.datetime64("2026-01-11T01:00")  # rounded, no ns noise
    assert io.time_attr(frc["Uwind"]) == "wind_time"
    assert io.coordinates_attr(io.open_nc(synthetic["frc_coarse"])["Tair"]) == "lon lat"


# ---------------------------------------------------------------- ROMS log
LOG = """
 Metrics information for Grid 01:
 ===============================

 Minimum barotropic Courant Number =  1.23456789E-02
 Maximum barotropic Courant Number =  4.56789012E-01
 Maximum Coriolis   Courant Number =  1.75624191E-02

 Basin information for Grid 01:

 Maximum grid stiffness ratios:  rx0 =   1.396193E-01 (Beckmann and Haidvogel)
                                 rx1 =   1.249110-100 (Haney)
"""


def test_parse_roms_log(tmp_path):
    p = tmp_path / "roms.log"
    p.write_text(LOG)
    out = io.parse_roms_log(p)
    assert out["cbt_min"] == pytest.approx(1.23456789e-2)
    assert out["cbt_max"] == pytest.approx(0.456789012)
    assert out["ccor_max"] == pytest.approx(1.75624191e-2)
    assert out["rx0"] == pytest.approx(0.1396193)
    assert out["rx1"] == pytest.approx(1.249110e-100)  # Fortran drops the E for 3-digit exponents


# ---------------------------------------------------------------- config
def _write_cfg(tmp_path, synthetic, **over):
    data_dir = synthetic["grid"].parent
    cfg = {"name": "t", "ocean_in": str(data_dir / "ocean_synthetic.in"), "grid": None, "ini": None,
           "frc": None, "tide": None, "bry": None, "nud": None, "clm": None,
           "model": {"DT": 180.0, "N": 12}, "open_boundaries": ["west"], "output_dir": "out"}
    cfg.update(over)
    p = tmp_path / "cfg" / "c.yaml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_load_config_fills_from_ocean_in(tmp_path, synthetic):
    cfg = io.load_config(_write_cfg(tmp_path, synthetic))
    d = synthetic["grid"].parent
    assert cfg["grid"] == d / "grid.nc" and cfg["bry"] == d / "bry.nc"
    assert cfg["frc"] == [d / "frc_native.nc", d / "frc_coarse.nc"]
    assert cfg["nud"] is None  # NUDNAME is named but absent
    assert any(c.status == Status.INFO and "NUDNAME" in c.name for c in cfg["checks"])
    assert cfg["params"]["DSTART"] == 9497.0 and cfg["params"]["Dcrit"] == pytest.approx(0.1)
    assert cfg["lbc"]["zeta"]["west"] == "Cha"
    assert cfg["output_dir"] == tmp_path / "cfg" / "out"
    assert not [c for c in cfg["checks"] if c.status == Status.WARN]


def test_load_config_conflict_is_warn_and_ocean_in_wins(tmp_path, synthetic):
    cfg = io.load_config(_write_cfg(tmp_path, synthetic, model={"DT": 300.0}))
    assert cfg["params"]["DT"] == 180.0
    warns = [c for c in cfg["checks"] if c.status == Status.WARN]
    assert len(warns) == 1 and "DT" in warns[0].name


def test_load_config_relative_paths(tmp_path, synthetic):
    p = _write_cfg(tmp_path, synthetic, ocean_in=None, grid="../g.nc", frc="f.nc")
    cfg = io.load_config(p)
    assert cfg["grid"] == tmp_path / "g.nc"
    assert cfg["frc"] == [tmp_path / "cfg" / "f.nc"]
    assert cfg["lbc"] is None and cfg["open_boundaries"] == ["west"]
