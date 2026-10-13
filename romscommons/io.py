"""Read ROMS inputs: the config YAML, NetCDF files, ROMS time, ocean.in, and ROMS stdout."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .checks import Check, Status

#: ROMS order of the four values on an ``LBC(...)`` line (roms.in diagram: 1=W, 2=S, 3=E, 4=N).
SIDES = ("west", "south", "east", "north")

#: ocean.in file-name keyword for each config role.
FILE_KEYS = {"grid": "GRDNAME", "ini": "ININAME", "frc": "FRCNAME", "tide": "TIDENAME",
             "bry": "BRYNAME", "nud": "NUDNAME", "clm": "CLMNAME"}

#: ``LBC(isXxx)`` state-variable index -> short variable name used in boundary files.
LBC_NAMES = {"isFsur": "zeta", "isUbar": "ubar", "isVbar": "vbar", "isUvel": "u", "isVvel": "v",
             "isMtke": "tke", "isU2Sd": "u2sd", "isV2Sd": "v2sd", "isU3Sd": "u3sd", "isV3Sd": "v3sd"}

#: Tracer order in ROMS (active tracers first). Further tracers are named tracer3, tracer4, ...
TRACER_NAMES = ("temp", "salt")

#: Run parameters read from ocean.in (canonical spelling; matched case-insensitively).
MODEL_KEYS = ("DT", "NDTFAST", "NTIMES", "N", "Lm", "Mm", "Vtransform", "Vstretching", "THETA_S",
              "THETA_B", "TCLINE", "TIME_REF", "DSTART", "TIDE_START", "Dcrit", "NRREC")

_ROLE_KEYS = ("ocean_in", "roms_log", "grid", "ini", "frc", "tide", "bry", "nud", "clm")


# ---------------------------------------------------------------------------
# ocean.in
# ---------------------------------------------------------------------------

_KEY_RX = re.compile(r"^\s*([A-Za-z_][\w]*(?:\([^)]*\))?)\s*(==|=)\s*(.*)$")
_REPEAT_RX = re.compile(r"^(\d+)\*(.+)$")


def parse_fortran_value(tok: str):
    """Fortran-style token -> Python value: ``300.0d0`` -> 300.0, ``T`` -> True, else the string."""
    t = tok.strip()
    up = t.upper()
    if up in ("T", ".TRUE.", "TRUE"):
        return True
    if up in ("F", ".FALSE.", "FALSE"):
        return False
    if re.fullmatch(r"[+-]?\d+", t):
        return int(t)
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?", t):
        return float(re.sub(r"[dD]", "e", t))
    return t


def _expand(tokens: list[str]) -> list:
    out = []
    for tok in tokens:
        m = _REPEAT_RX.match(tok)
        if m and not tok.lower().endswith(".nc"):
            out.extend([parse_fortran_value(m.group(2))] * int(m.group(1)))
        else:
            out.append(parse_fortran_value(tok))
    return out


def _logical_lines(path: Path):
    """Yield (key, op, tokens) with comments stripped and continuation lines joined."""
    key = op = None
    tokens: list[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.split("!", 1)[0].rstrip()
            if key is None:
                m = _KEY_RX.match(line)
                if not m:
                    continue
                key, op, rest = m.group(1), m.group(2), m.group(3)
            else:
                rest = line
            rest = rest.strip()
            cont = rest.endswith("\\") or rest.endswith("|")
            tokens.extend(t for t in rest.split() if t not in ("\\", "|"))
            if tokens and tokens[-1].endswith(("\\", "|")):
                tokens[-1] = tokens[-1][:-1]
                if not tokens[-1]:
                    tokens.pop()
            if not cont:
                yield key, op, tokens
                key, op, tokens = None, None, []
    if key is not None:
        yield key, op, tokens


def parse_ocean_in(path) -> dict[str, Any]:
    """Parse a ROMS standard input file (``ocean.in`` / ``roms.in``).

    Handles ``KEY == value`` and ``KEY = value``, ``!`` comments, ``\\`` and ``|``
    continuation lines, Fortran reals (``300.0d0``), logicals ``T``/``F``, repeat syntax
    (``2*0.0d0``). Single values become scalars, several become lists.

    Special keys:

    * ``LBC(isXxx) == W S E N`` -> ``out["LBC"][name] = {"west":..., "south":..., "east":..., "north":...}``
      with ``name`` from :data:`LBC_NAMES`. The tracer entry (``isTvar``) spans one line per
      tracer; tracers are named ``temp``, ``salt``, ``tracer3``... Order is **W, S, E, N**
      (decoder: ``Utility/inp_decode.F``; VERIFY_LOG V12).
    * ``VolCons(side) == T/F`` -> ``out["VolCons"][side]`` (VERIFY_LOG V13).
    * ``ad_LBC(...)`` lines are kept under ``out["ad_LBC"]``.
    * ``TITLE`` is kept as one string.
    """
    path = Path(path)
    out: dict[str, Any] = {}
    for key, _op, tokens in _logical_lines(path):
        m = re.match(r"^(ad_)?LBC\((\w+)\)$", key)
        if m:
            target = out.setdefault("ad_LBC" if m.group(1) else "LBC", {})
            idx = m.group(2)
            groups = [tokens[i:i + 4] for i in range(0, len(tokens) - len(tokens) % 4, 4)]
            if idx == "isTvar":
                for n, g in enumerate(groups):
                    name = TRACER_NAMES[n] if n < len(TRACER_NAMES) else f"tracer{n + 1}"
                    target[name] = dict(zip(SIDES, g))
            elif groups:
                target[LBC_NAMES.get(idx, idx)] = dict(zip(SIDES, groups[0]))
            continue
        m = re.match(r"^(ad_)?VolCons\((\w+)\)$", key)
        if m:
            out.setdefault("ad_VolCons" if m.group(1) else "VolCons", {})[m.group(2).lower()] = \
                bool(_expand(tokens)[0]) if tokens else False
            continue
        if key.upper() == "TITLE":
            out[key] = " ".join(tokens)
            continue
        vals = _expand(tokens)
        out[key] = vals[0] if len(vals) == 1 else vals
    return out


def get_ci(d: dict, key: str, default=None):
    """Case-insensitive dict lookup."""
    if key in d:
        return d[key]
    low = key.lower()
    for k, v in d.items():
        if k.lower() == low:
            return v
    return default


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def time_ref(TIME_REF) -> pd.Timestamp:
    """ROMS ``TIME_REF`` (``yyyymmdd.f``) -> timestamp.

    ``TIME_REF > 0`` is a date, e.g. 20020115.5 -> 2002-01-15 12:00. The special values
    (VERIFY_LOG V17: −2 modified Julian days since 1968-05-23, −1 360-day calendar,
    0 365.25-day calendar since 0001-01-01) are not supported here.
    """
    v = float(TIME_REF)
    if v <= 0:
        raise NotImplementedError(
            f"TIME_REF = {TIME_REF}: special calendars are not supported. ROMS meaning "
            "(roms.in glossary): -2 = modified Julian days since 1968-05-23, -1 = 360-day "
            "calendar, 0 = 365.25-day calendar since 0001-01-01. Use TIME_REF = yyyymmdd.f.")
    ymd = int(v)
    y, m, d = ymd // 10000, (ymd // 100) % 100, ymd % 100
    return pd.Timestamp(year=y, month=m, day=d) + pd.Timedelta(seconds=round((v - ymd) * 86400))


def run_window(params: dict, ini_time=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Model run window (start, end).

    start = TIME_REF + DSTART days; end = start + NTIMES·DT seconds. ROMS actually sets its
    clock from the initial-conditions record (``get_state.F``; VERIFY_LOG V2) — pass
    ``ini_time`` to use that instead of DSTART.
    """
    start = pd.Timestamp(ini_time) if ini_time is not None else \
        time_ref(get_ci(params, "TIME_REF")) + pd.Timedelta(days=float(get_ci(params, "DSTART")))
    end = start + pd.Timedelta(seconds=float(get_ci(params, "NTIMES")) * float(get_ci(params, "DT")))
    return start, end


_TUNITS_RX = re.compile(r"^\s*(second|seconds|sec|s|minute|minutes|min|hour|hours|h|day|days|d)"
                        r"(?:\s+since\s+(.+?))?\s*$", re.IGNORECASE)
_FACTORS = {"s": 1, "sec": 1, "second": 1, "seconds": 1, "min": 60, "minute": 60, "minutes": 60,
            "h": 3600, "hour": 3600, "hours": 3600, "d": 86400, "day": 86400, "days": 86400}


def decode_time_values(values, units: str, TIME_REF=None) -> np.ndarray:
    """Numeric time values + units -> ``datetime64[ns]``.

    ``"<unit> since <date>"`` uses that date; a bare ``"seconds"``/``"days"`` is relative to
    ``TIME_REF`` (as ROMS does). Raises ValueError if neither is available.
    """
    m = _TUNITS_RX.match(units or "")
    if not m:
        raise ValueError(f"cannot decode time units {units!r}")
    factor = _FACTORS[m.group(1).lower()]
    if m.group(2):
        ref = pd.Timestamp(re.sub(r"\s*(GMT|UTC|Z)\s*$", "", m.group(2).strip()))
    elif TIME_REF is not None:
        ref = time_ref(TIME_REF)
    else:
        raise ValueError(f"time units {units!r} have no reference date and TIME_REF is not given")
    secs = np.asarray(values, dtype="float64") * factor
    return (np.datetime64(ref.to_datetime64(), "ns") + np.round(secs * 1e3).astype("timedelta64[ms]")) \
        .astype("datetime64[ns]")


def _round_times(ds: xr.Dataset) -> xr.Dataset:
    """Round decoded times to 1 ms: times stored as float days decode with ns noise."""
    for name in list(ds.variables):
        var = ds[name]
        if var.dtype.kind == "M" and var.ndim == 1:
            rounded = xr.Variable(var.dims, pd.DatetimeIndex(var.values).round("ms").values, var.attrs, var.encoding)
            ds = ds.assign_coords({name: rounded}) if name in ds.coords else ds.assign({name: rounded})
    return ds


def _decode_remaining(ds: xr.Dataset, TIME_REF) -> xr.Dataset:
    """Decode numeric variables whose units look like time (what xarray left alone)."""
    for name in list(ds.variables):
        var = ds[name]
        units = var.attrs.get("units")
        if not isinstance(units, str) or var.dtype.kind not in "fiu" or not _TUNITS_RX.match(units):
            continue
        if name.startswith("tide_period"):
            continue  # a period in hours, not a time axis
        is_time = name.endswith("time") or name in ds.dims or " since " in units.lower()
        if not is_time:
            continue
        try:
            decoded = decode_time_values(var.values, units, TIME_REF)
        except ValueError:
            continue
        attrs = {k: v for k, v in var.attrs.items() if k != "units"}
        new = xr.Variable(var.dims, decoded, attrs, encoding={**var.encoding, "units": units})
        ds = ds.assign_coords({name: new}) if name in ds.coords else ds.assign({name: new})
    return ds


def open_nc(path, chunks=None, TIME_REF=None) -> xr.Dataset:
    """Open a NetCDF file lazily with times decoded.

    Tries xarray's CF decoding first; on failure reopens undecoded and decodes from each
    variable's ``units`` (or ``TIME_REF`` when units carry no reference date). Timedeltas
    (e.g. ``tide_period`` in hours) are never decoded.
    """
    kw = dict(chunks=chunks, decode_timedelta=False)
    try:
        ds = xr.open_dataset(path, decode_times=True, **kw)
    except Exception:
        ds = xr.open_dataset(path, decode_times=False, **kw)
    return _round_times(_decode_remaining(ds, TIME_REF))


def time_attr(da: xr.DataArray) -> str | None:
    """The variable's ``time`` attribute — the time variable ROMS will use."""
    return da.attrs.get("time", da.encoding.get("time"))


def coordinates_attr(da: xr.DataArray) -> str | None:
    """The ``coordinates`` attribute (xarray moves it into ``encoding`` on read)."""
    return da.attrs.get("coordinates", da.encoding.get("coordinates"))


# ---------------------------------------------------------------------------
# ROMS stdout
# ---------------------------------------------------------------------------

_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+|[-+]\d{3})?)"
_LOG_PATTERNS = {
    "cbt_min": rf"Minimum\s+barotropic\s+Courant\s+Number\s*=\s*{_NUM}",
    "cbt_max": rf"Maximum\s+barotropic\s+Courant\s+Number\s*=\s*{_NUM}",
    "ccor_max": rf"Maximum\s+Coriolis\s+Courant\s+Number\s*=\s*{_NUM}",
    "rx0": rf"rx0\s*=\s*{_NUM}",
    "rx1": rf"rx1\s*=\s*{_NUM}",
}


def _fortran_float(s: str) -> float:
    s = s.replace("D", "E").replace("d", "e")
    m = re.match(r"^([-+]?\d*\.?\d*)([-+]\d{3})$", s)  # 1.2345-100 (no E for 3-digit exponents)
    if m:
        s = f"{m.group(1)}E{m.group(2)}"
    return float(s)


def parse_roms_log(path) -> dict[str, float]:
    """Courant numbers (``metrics.F``) and rx0/rx1 (``stiffness.F``) from ROMS stdout.

    Returns the values for the first grid; keys present only if found:
    ``cbt_min, cbt_max, ccor_max, rx0, rx1`` (VERIFY_LOG V14).
    """
    text = Path(path).read_text(errors="replace")
    out = {}
    for key, rx in _LOG_PATTERNS.items():
        m = re.search(rx, text)
        if m:
            out[key] = _fortran_float(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _resolve(p, base: Path):
    """Absolute, normalised path; symlinks are kept so paths read as the user wrote them."""
    if p is None or p == "":
        return None
    p = Path(str(p)).expanduser()
    return Path(os.path.normpath(p if p.is_absolute() else base / p))


def _same(a, b) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12))
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def load_config(path) -> dict[str, Any]:
    """Load a configuration YAML (BUILD_SPEC §4).

    Relative paths resolve against the YAML file's directory. If ``ocean_in`` is given it is
    the source of truth: its run parameters override ``model`` (each conflict is reported as
    a WARN in ``cfg["checks"]``), its LBC block fills ``cfg["lbc"]``, and file roles left
    ``null`` are filled from GRDNAME, ININAME, ... (relative to the ocean.in directory) when
    those files exist.
    """
    path = Path(os.path.abspath(Path(path).expanduser()))
    raw = yaml.safe_load(path.read_text()) or {}
    base = path.parent
    cfg: dict[str, Any] = {k: v for k, v in raw.items()}
    cfg["config_path"] = path
    cfg["config_dir"] = base
    cfg.setdefault("name", path.stem)
    checks: list[Check] = []

    for role in _ROLE_KEYS:
        val = raw.get(role)
        if role == "frc":
            vals = [] if val is None else ([val] if isinstance(val, str) else list(val))
            cfg[role] = [_resolve(v, base) for v in vals]
        else:
            cfg[role] = _resolve(val, base)
    cfg["output_dir"] = _resolve(raw.get("output_dir", f"outputs/{cfg['name']}"), base)
    cfg["ini_record"] = int(raw.get("ini_record", 0) or 0)

    params = dict(raw.get("model") or {})
    oin: dict[str, Any] = {}
    if cfg["ocean_in"] is not None:
        oin = parse_ocean_in(cfg["ocean_in"])
        oin_dir = cfg["ocean_in"].parent
        for key in set(MODEL_KEYS) | set(params):
            o_val = get_ci(oin, key)
            if o_val is None:
                continue
            y_key = next((k for k in params if k.lower() == key.lower()), key)
            if y_key in params and not _same(params[y_key], o_val):
                checks.append(Check(f"config vs ocean.in: {y_key}", Status.WARN, o_val, params[y_key],
                                    "ocean.in is the source of truth", f"YAML model value {params[y_key]} "
                                    f"overridden by ocean.in {o_val}"))
            params[y_key] = o_val
        for role, okey in FILE_KEYS.items():
            if (role == "frc" and cfg[role]) or (role != "frc" and cfg[role] is not None):
                continue
            names = get_ci(oin, okey)
            if names is None:
                continue
            names = names if isinstance(names, list) else [names]
            files = [_resolve(n, oin_dir) for n in names]
            exist = [f for f in files if f.exists()]
            missing = [f for f in files if not f.exists()]
            if missing:
                checks.append(Check(f"ocean.in {okey}", Status.INFO, [m.name for m in missing], "exists",
                                    "ocean.in file names", "named in ocean.in but not found; role left empty"))
            if role == "frc":
                cfg[role] = exist
            elif exist:
                cfg[role] = exist[0]
    cfg["params"] = params
    cfg["ocean_in_values"] = oin
    cfg["lbc"] = oin.get("LBC")
    cfg["volcons"] = oin.get("VolCons")
    cfg["open_boundaries"] = list(raw.get("open_boundaries") or [])
    cfg["points"] = list(raw.get("points") or [])
    cfg["checks"] = checks
    return cfg
