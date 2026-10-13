"""Shared fixtures: synthetic cases generated once per session, optional real-data paths."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
MOANA_GRID = Path("/projects/f_omg_1/moa/moana/grd_moana5km.nc")


def _generator():
    spec = importlib.util.spec_from_file_location("make_synthetic_case", REPO / "examples" / "make_synthetic_case.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def gen():
    return _generator()


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory, gen):
    return gen.make_case(tmp_path_factory.mktemp("synthetic"), defects=False)


@pytest.fixture(scope="session")
def defects(tmp_path_factory, gen):
    return gen.make_case(tmp_path_factory.mktemp("synthetic_defects"), defects=True)


@pytest.fixture(scope="session")
def roms_src():
    p = Path(os.environ.get("ROMS_SRC", REPO.parent / "roms-reference"))
    if not (p / "ROMS").is_dir():
        pytest.skip(f"ROMS source not found at {p} (set ROMS_SRC)")
    return p


@pytest.fixture(scope="session")
def moana_grid():
    if not MOANA_GRID.exists():
        pytest.skip(f"{MOANA_GRID} not reachable")
    return MOANA_GRID
