"""Explicit PASS / WARN / FAIL / INFO checks and the report that collects them.

Rules (BUILD_SPEC §5.5, CLAUDE.md rule 5):

* FAIL only for things that will break or invalidate a ROMS run.
* WARN for judgment calls (stiffness, Courant margins, plausibility ranges).
* Every check records the value, the threshold, and where the threshold comes from.
* Thresholds live only in :data:`THRESHOLDS`.
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr


class Status(str, enum.Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    INFO = "INFO"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# Severity used by Report.worst(); INFO never outranks PASS.
_SEVERITY = {Status.INFO: 0, Status.PASS: 1, Status.WARN: 2, Status.FAIL: 3}


def _t(value, source):
    return {"value": value, "source": source}


#: Every threshold used by a check. Change values here, nowhere else.
THRESHOLDS: dict[str, dict[str, Any]] = {
    # grid stiffness
    "rx0_warn": _t(0.2, "Beckmann & Haidvogel 1993; Haidvogel et al. 2000"),
    "rx1_warn": _t(6.0, "Haney 1991 (literature range 3-8)"),
    # Courant numbers
    "cbt_warn": _t(0.8, "rule of thumb for the split-explicit barotropic step"),
    "cadv_warn": _t(0.5, "advective CFL margin"),
    "cint_warn": _t(0.5, "internal-wave CFL margin"),
    # stratification and plausibility
    "n2_min": _t(-1e-5, "static stability (N^2 in s^-2)"),
    "temp_range": _t((-2.5, 35.0), "plausibility (degC)"),
    "salt_range": _t((2.0, 42.0), "plausibility (practical salinity)"),
    "speed_max": _t(3.0, "plausibility (m s-1)"),
    "zeta_max": _t(3.0, "plausibility (m)"),
    "ubar_rms": _t(0.01, "ubar/vbar vs depth average of u/v (m s-1)"),
    # nudging: gamma*DT, both in consistent units
    "nud_coefdt_warn": _t(0.1, "explicit relaxation stability"),
    "nud_coefdt_fail": _t(1.0, "explicit relaxation stability (gamma*DT >= 1 overshoots)"),
    # initial vs boundary mismatch at t0
    "icbc_temp": _t(0.5, "shock at start (degC)"),
    "icbc_salt": _t(0.1, "shock at start"),
    "icbc_zeta": _t(0.1, "shock at start (m)"),
    "icbc_vel": _t(0.1, "shock at start (m s-1)"),
    # boundary volume budget
    "ssh_drift_rate": _t(0.01, "boundary volume imbalance (m day-1 domain-mean)"),
    "ssh_drift_total": _t(0.1, "boundary volume imbalance (m over the run)"),
    "bry_tide_m2": _t(0.05, "tides double-counted when also added via ADD_FSOBC (m)"),
    # time axes
    "gap_factor": _t(1.5, "time step > 1.5 x median step counts as a gap"),
    # file-vs-config consistency
    "cs_atol": _t(1e-6, "recomputed Cs_r/Cs_w vs file (set_scoord.F)"),
    "h_atol": _t(1e-3, "h in file vs grid h (m)"),
    "angle_info_deg": _t(1.0, "grid rotation large enough that east/north != xi/eta"),
}


def threshold(key: str):
    """Return the value of a threshold (convenience)."""
    return THRESHOLDS[key]["value"]


def source(key: str) -> str:
    return THRESHOLDS[key]["source"]


@dataclass
class Check:
    """One check result."""

    name: str
    status: Status
    value: Any = None
    threshold: Any = None
    source: str = ""
    message: str = ""
    details: Any = field(default=None, repr=False)  # e.g. a hotspot table

    def __post_init__(self):
        self.status = Status(self.status)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        if v == 0 or (1e-3 <= abs(v) < 1e5):
            return f"{v:.4g}"
        return f"{v:.3e}"
    if isinstance(v, (np.datetime64, pd.Timestamp)):
        return str(pd.Timestamp(v))
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


_COLORS = {
    Status.PASS: "background-color: #d4edda",
    Status.WARN: "background-color: #fff3cd",
    Status.FAIL: "background-color: #f8d7da",
    Status.INFO: "background-color: #e2e3e5",
}


class Report:
    """An ordered collection of :class:`Check` results."""

    def __init__(self, title: str = "report"):
        self.title = title
        self.checks: list[Check] = []

    def add(self, check: Check | Iterable[Check]) -> Check | list[Check]:
        if isinstance(check, Check):
            self.checks.append(check)
        else:
            check = list(check)
            self.checks.extend(check)
        return check

    def __iter__(self):
        return iter(self.checks)

    def __len__(self):
        return len(self.checks)

    def __getitem__(self, name: str) -> Check:
        """Last check with this name."""
        for c in reversed(self.checks):
            if c.name == name:
                return c
        raise KeyError(name)

    def find(self, pattern: str) -> list[Check]:
        rx = re.compile(pattern)
        return [c for c in self.checks if rx.search(c.name)]

    def worst(self) -> Status:
        if not self.checks:
            return Status.PASS
        return max((c.status for c in self.checks), key=_SEVERITY.__getitem__)

    def counts(self) -> dict[Status, int]:
        return {s: sum(c.status == s for c in self.checks) for s in Status}

    def summary(self) -> str:
        n = self.counts()
        return (f"{self.title}: {len(self)} checks — {n[Status.PASS]} PASS, {n[Status.WARN]} WARN, "
                f"{n[Status.FAIL]} FAIL, {n[Status.INFO]} INFO → worst: {self.worst().value}")

    def to_frame(self) -> pd.DataFrame:
        rows = [{"check": c.name, "status": c.status.value, "value": _fmt(c.value),
                 "threshold": _fmt(c.threshold), "source": c.source, "message": c.message}
                for c in self.checks]
        return pd.DataFrame(rows, columns=["check", "status", "value", "threshold", "source", "message"])

    def display(self):
        """Colored table (pandas Styler); the last expression of a notebook cell renders it."""
        df = self.to_frame()
        return (df.style.map(lambda s: _COLORS.get(Status(s), ""), subset=["status"])
                .set_caption(self.summary()).hide(axis="index"))

    def _repr_html_(self):  # pragma: no cover - notebook only
        return self.display().to_html()

    def __repr__(self):
        return self.summary()


# ---------------------------------------------------------------------------
# Unit strings
# ---------------------------------------------------------------------------

_UNIT_ALIASES = {
    "m": ["m", "meter", "meters", "metre", "metres"],
    "m s-1": ["m s-1", "m/s", "meter second-1", "meters second-1", "metre second-1", "m s^-1", "ms-1",
              "m.s-1", "meter/second", "meters/second", "m sec-1"],
    "degC": ["celsius", "degc", "deg c", "degree_celsius", "degrees_celsius", "°c", "c", "degrees c",
             "degree celsius", "degrees celsius", "deg_c"],
    "K": ["k", "kelvin", "degk", "degrees_kelvin"],
    "psu": ["psu", "pss", "pss-78", "nondimensional", "1", "practical salinity", "practical_salinity_unit",
            "0.001", "1e-3"],
    "mbar": ["millibar", "millibars", "mbar", "hpa", "mb"],
    "Pa": ["pa", "pascal", "pascals"],
    "percent": ["percentage", "percent", "%"],
    "kg m-2 s-1": ["kilogram meter-2 second-1", "kg m-2 s-1", "kg/m2/s", "kg m^-2 s^-1", "kg/m^2/s"],
    "W m-2": ["watt meter-2", "watts meter-2", "w m-2", "w/m2", "w m^-2", "w/m^2"],
    "N m-2": ["newton meter-2", "n m-2", "n/m2", "n m^-2", "pa"],
    "day-1": ["day-1", "1/day", "day^-1", "days-1", "d-1", "1/d", "per day"],
    "s-1": ["second-1", "s-1", "1/s", "s^-1", "seconds-1"],
    "hours": ["hour", "hours", "h", "hr", "hrs"],
    "degrees": ["degree", "degrees", "deg"],
    "radians": ["radian", "radians", "rad"],
}
_UNIT_LOOKUP: dict[str, list[str]] = {}
for _canon, _names in _UNIT_ALIASES.items():
    for _n in _names:
        _UNIT_LOOKUP.setdefault(_n, []).append(_canon)


def canonical_units(units: str | None) -> list[str]:
    """Canonical names a units string may stand for ([] if unknown or missing)."""
    if units is None:
        return []
    key = re.sub(r"\s+", " ", str(units).strip().lower())
    return list(_UNIT_LOOKUP.get(key, []))


# ---------------------------------------------------------------------------
# Generic checks
# ---------------------------------------------------------------------------

def required_vars(ds: xr.Dataset, names: Sequence[str], *, label: str = "required variables",
                  source: str = "ROMS varinfo.yaml", missing_status: Status = Status.FAIL) -> Check:
    missing = [n for n in names if n not in ds.variables]
    if missing:
        return Check(label, missing_status, missing, "present", source,
                     f"missing: {', '.join(missing)}")
    return Check(label, Status.PASS, f"{len(names)} present", "present", source, "all present")


def dims_match(da: xr.DataArray, expected: dict[str, int], *, label: str | None = None,
               source: str = "ROMS C-grid staggering") -> Check:
    """The trailing dims of ``da`` must equal ``expected`` (ordered name -> size)."""
    label = label or f"dims {da.name}"
    names = list(expected)
    got = list(da.dims[-len(names):]) if len(names) else []
    got_sizes = [da.sizes[d] for d in got]
    exp_sizes = list(expected.values())
    if got == names and got_sizes == exp_sizes:
        return Check(label, Status.PASS, dict(zip(got, got_sizes)), dict(expected), source, "dims match")
    return Check(label, Status.FAIL, dict(zip(da.dims, da.shape)), dict(expected), source,
                 "dimension names/sizes do not match the grid")


def dims_match_grid(da: xr.DataArray, grid: xr.Dataset, location: str, N: int | None = None,
                    vertical: str | None = None, label: str | None = None) -> Check:
    """Check ``da`` against the grid's expected dims at ``location`` (rho/u/v/psi)."""
    from .grid import expected_dims

    exp = expected_dims(grid, location)
    if vertical is not None:
        exp = {vertical: (N if vertical == "s_rho" else N + 1), **exp}
    return dims_match(da, exp, label=label)


def _count(mask_da) -> int:
    return int(mask_da.sum().values)


def finite_at_wet(da: xr.DataArray, mask: xr.DataArray, *, label: str | None = None) -> Check:
    """FAIL if any NaN/Inf at wet points. ROMS does not catch NaN on read (VERIFY_LOG V3)."""
    label = label or f"finite at wet points: {da.name}"
    bad = _count((~np.isfinite(da)) & (mask > 0))
    src = "NaN passes the ROMS reader unchanged (nf_fread*.F; VERIFY_LOG V3)"
    if bad:
        return Check(label, Status.FAIL, bad, 0, src, f"{bad} non-finite values at wet points")
    return Check(label, Status.PASS, 0, 0, src, "all finite")


def finite_at_land(da: xr.DataArray, mask: xr.DataArray, *, label: str | None = None) -> Check:
    """WARN if NaN/Inf at land points: NaN * mask is still NaN in ROMS (VERIFY_LOG V3)."""
    label = label or f"finite at land points: {da.name}"
    bad = _count((~np.isfinite(da)) & (mask == 0))
    src = "NaN x 0 = NaN; land NaN can leak through stencils (VERIFY_LOG V3)"
    if bad:
        return Check(label, Status.WARN, bad, 0, src, f"{bad} non-finite values on land; fill with a finite value")
    return Check(label, Status.PASS, 0, 0, src, "all finite")


def in_range(da: xr.DataArray, key: str | tuple, mask: xr.DataArray | None = None, *,
             label: str | None = None) -> Check:
    """WARN if wet values fall outside a plausibility range (THRESHOLDS key or (lo, hi))."""
    lo, hi = THRESHOLDS[key]["value"] if isinstance(key, str) else key
    src = THRESHOLDS[key]["source"] if isinstance(key, str) else "user range"
    x = da.where(mask > 0) if mask is not None else da
    vmin, vmax = float(x.min(skipna=True).values), float(x.max(skipna=True).values)
    label = label or f"range {da.name}"
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return Check(label, Status.INFO, None, (lo, hi), src, "no finite wet values")
    if vmin < lo or vmax > hi:
        return Check(label, Status.WARN, (vmin, vmax), (lo, hi), src, "values outside the plausible range")
    return Check(label, Status.PASS, (vmin, vmax), (lo, hi), src, "within range")


def max_abs(da: xr.DataArray, key: str, mask: xr.DataArray | None = None, *, label: str | None = None) -> Check:
    """WARN if max |value| at wet points exceeds THRESHOLDS[key]."""
    lim = threshold(key)
    x = abs(da.where(mask > 0)) if mask is not None else abs(da)
    vmax = float(x.max(skipna=True).values)
    label = label or f"max |{da.name}|"
    status = Status.WARN if vmax > lim else Status.PASS
    return Check(label, status, vmax, lim, source(key), "exceeds plausible maximum" if status == Status.WARN else "ok")


def units_match(da: xr.DataArray, expected: str | Sequence[str], *, label: str | None = None,
                mismatch_status: Status = Status.WARN,
                source: str = "ROMS varinfo.yaml") -> Check:
    """Compare the ``units`` attribute with the units ROMS assumes.

    ROMS ignores the attribute for most fields (it applies the varinfo.yaml scale), so a
    mismatch is a WARN by default: the attribute is a clue, the values decide.
    """
    expected = [expected] if isinstance(expected, str) else list(expected)
    label = label or f"units {da.name}"
    units = da.attrs.get("units", da.encoding.get("units"))
    if units is None:
        return Check(label, Status.INFO, None, expected, source, "no units attribute")
    want = {c for e in expected for c in (canonical_units(e) or [e])}
    have = set(canonical_units(units)) or {units}
    if want & have:
        return Check(label, Status.PASS, units, expected, source, "units match")
    return Check(label, mismatch_status, units, expected, source, "units attribute differs from what ROMS assumes")


def _as_seconds(t) -> np.ndarray:
    v = np.asarray(t.values if isinstance(t, xr.DataArray) else t)
    if np.issubdtype(v.dtype, np.datetime64):
        return (v - v[0]).astype("timedelta64[ns]").astype(np.int64) / 1e9 if v.size else v.astype(float)
    return v.astype(float)


def time_monotonic(t: xr.DataArray, *, label: str | None = None) -> Check:
    """FAIL if the time axis is not strictly increasing (duplicates included)."""
    name = getattr(t, "name", "time")
    label = label or f"time monotonic: {name}"
    s = _as_seconds(t)
    d = np.diff(s)
    src = "ROMS needs increasing records to bracket model time"
    if d.size and (d <= 0).any():
        n_dup, n_back = int((d == 0).sum()), int((d < 0).sum())
        return Check(label, Status.FAIL, {"duplicates": n_dup, "decreasing": n_back}, "strictly increasing", src,
                     "time is not strictly increasing")
    return Check(label, Status.PASS, f"{s.size} records", "strictly increasing", src, "ok")


def time_gaps(t: xr.DataArray, *, label: str | None = None) -> Check:
    """WARN if any step exceeds gap_factor x the median step."""
    name = getattr(t, "name", "time")
    label = label or f"time gaps: {name}"
    s = _as_seconds(t)
    d = np.diff(s)
    fac = threshold("gap_factor")
    if d.size == 0:
        return Check(label, Status.INFO, "1 record", None, source("gap_factor"), "single record")
    med = float(np.median(d))
    big = np.nonzero(d > fac * med)[0]
    lim_h = fac * med / 3600
    if big.size:
        tv = np.asarray(t.values)
        worst = big[np.argmax(d[big])]
        msg = (f"{big.size} gap(s); largest {d[worst] / 3600:.3g} h after {_fmt(tv[worst])}"
               f" (median step {med / 3600:.3g} h)")
        return Check(label, Status.WARN, float(d.max() / 3600), lim_h, source("gap_factor"), msg)
    return Check(label, Status.PASS, float(d.max() / 3600), lim_h, source("gap_factor"),
                 f"no gaps (median step {med / 3600:.3g} h)")


def time_covers(t: xr.DataArray, run_start, run_end, *, cycle_length=None, label: str | None = None) -> Check:
    """FAIL unless a record exists at/before run start and at/after run end.

    Skipped (INFO) for climatologies, which carry a ``cycle_length`` attribute.
    """
    name = getattr(t, "name", "time")
    label = label or f"time covers run: {name}"
    src = "ROMS interpolates between two records bracketing model time"
    if cycle_length is None and isinstance(t, xr.DataArray):
        cycle_length = t.attrs.get("cycle_length")
    if cycle_length is not None:
        return Check(label, Status.INFO, f"cycle_length={cycle_length}", None, src,
                     "climatology (cyclic) — coverage not required")
    tv = np.asarray(t.values)
    first, last = tv.min(), tv.max()
    start, end = np.datetime64(pd.Timestamp(run_start)), np.datetime64(pd.Timestamp(run_end))
    ok_start, ok_end = first <= start, last >= end
    value = (first, last)
    if ok_start and ok_end:
        return Check(label, Status.PASS, value, (start, end), src, "covers the run window")
    parts = []
    if not ok_start:
        parts.append(f"starts {pd.Timestamp(first) - pd.Timestamp(start)} after run start")
    if not ok_end:
        parts.append(f"ends {pd.Timestamp(end) - pd.Timestamp(last)} before run end")
    return Check(label, Status.FAIL, value, (start, end), src, "; ".join(parts))


def upper_bound(label: str, value: float, key: str, *, bad_status: Status = Status.WARN,
                message: str = "") -> Check:
    """``value`` must not exceed ``THRESHOLDS[key]`` (e.g. rx0, Courant numbers)."""
    lim = threshold(key)
    value = float(value)
    if not np.isfinite(value):
        return Check(label, Status.INFO, None, lim, source(key), "no finite value")
    if value > lim:
        return Check(label, bad_status, value, lim, source(key), message or "above threshold")
    return Check(label, Status.PASS, value, lim, source(key), "ok")


def compare(label: str, value, expected, *, atol: float = 0.0, bad_status: Status = Status.FAIL,
            source: str = "", ok_msg: str = "match", bad_msg: str = "differs") -> Check:
    """Generic equality/tolerance check for scalars or arrays (reports max |difference|)."""
    a, b = np.asarray(value), np.asarray(expected)
    if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
        if a.shape != b.shape:
            return Check(label, bad_status, f"shape {a.shape}", f"shape {b.shape}", source, "shape differs")
        diff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        if diff <= atol:
            return Check(label, Status.PASS, value if a.size == 1 else diff, expected if b.size == 1 else atol,
                         source, ok_msg)
        return Check(label, bad_status, value if a.size == 1 else diff, expected if b.size == 1 else atol,
                     source, f"{bad_msg} (max |diff| = {diff:.3g})")
    ok = bool(np.all(a == b))
    return Check(label, Status.PASS if ok else bad_status, value, expected, source, ok_msg if ok else bad_msg)
