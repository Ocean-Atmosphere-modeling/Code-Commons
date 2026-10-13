# code-commons — Build Spec: ROMS Input Inspection Notebooks

Spec v1.1 · 2026-10-12 (v1.1: source-based fixes from VERIFY_LOG — V7, V12, V14, V16) · Owner: Suman · Builder: Claude Code

---

## 0. Purpose

Five Jupyter notebooks that let any ROMS user open the input files of a realistic
configuration, **see** what is inside them, **check** them against what ROMS expects, and
**compute the numbers that decide whether the run will be stable**. Equal goal: teaching.
A reader new to ROMS should finish each notebook knowing what the file is for, which
`ocean.in` keywords and CPP options consume it, and what goes wrong when it is bad.

**In scope:** inspection, visualization, diagnostics for IC, atmospheric forcing, tides,
nudging coefficients, boundary conditions; building a tide file (demo) and a nudging file.

**Out of scope:** grid generation; producing IC/BC/forcing from parent models; running
ROMS; CROCO (different file conventions).

---

## 1. Assumptions (defaults — change them here, not in code)

| # | Assumption | Default |
|---|---|---|
| A1 | ROMS flavor | Rutgers ROMS 4.x (`github.com/myroms/roms`). COAWST's ROMS should mostly work. |
| A2 | Notebook format | `.ipynb` paired with jupytext percent `.py` |
| A3 | Data | Every notebook runs end-to-end on a generated synthetic case; real cases via config YAML |
| A4 | Maps | cartopy if installed, plain lon/lat axes otherwise |
| A5 | Interactivity | Static matplotlib only (no ipywidgets) |
| A6 | Density for diagnostics | TEOS-10 via `gsw` — an approximation of ROMS's nonlinear EOS, labelled as such |
| A7 | Python | 3.11+ |

---

## 2. Repository layout

```
code-commons/
├── CLAUDE.md                 # standing rules for Claude Code
├── BUILD_SPEC.md             # this file
├── VERIFY_LOG.md             # answers to [VERIFY] items, with source refs
├── README.md                 # written in the last phase
├── environment.yml
├── pyproject.toml            # installs romscommons (editable)
├── configs/
│   ├── synthetic.yaml
│   ├── synthetic_defects.yaml
│   ├── moana.yaml            # real data: /projects/f_omg_1/moa/moana/Input/ (see §4.1)
│   └── TEMPLATE.yaml         # commented template for real configurations
├── romscommons/
│   ├── __init__.py
│   ├── io.py                 # open files, ROMS time, ocean.in parser, config loader
│   ├── grid.py               # staggering, rotation, metrics, boundaries, transects
│   ├── vertical.py           # s-coordinate stretching, z_r / z_w / Hz
│   ├── diagnostics.py        # Courant numbers, rx0/rx1, N², transports, bulk estimates
│   ├── checks.py             # Status, Check, Report, THRESHOLDS, generic checks
│   ├── tides.py              # constituent table, ap2ep/ep2ap, reconstruction
│   └── plotting.py           # maps, sections, perimeter plots, quivers, ellipses
├── notebooks/
│   ├── 01_initial_conditions.{ipynb,py}
│   ├── 02_atmospheric_forcing.{ipynb,py}
│   ├── 03_tides.{ipynb,py}
│   ├── 04_nudging.{ipynb,py}
│   └── 05_boundary_conditions.{ipynb,py}
├── examples/
│   └── make_synthetic_case.py
├── data/                     # generated, git-ignored
├── outputs/                  # files written by nb03/nb04, git-ignored
└── tests/
    ├── test_io.py  test_grid.py  test_vertical.py  test_diagnostics.py
    ├── test_tides.py  test_checks.py
    └── test_defects.py       # each injected defect → expected status
```

---

## 3. Environment (`environment.yml`)

conda-forge: `python>=3.11, numpy, scipy, pandas, xarray, netcdf4, dask, matplotlib,
cmocean, gsw, pyyaml, cartopy, pytmd, jupyterlab, jupytext, pytest, nbmake`.

`pyTMD` is used only in nb03's demo. `cartopy` is imported inside a `try` so plotting
falls back cleanly when it is absent.

---

## 4. Configuration (`configs/*.yaml`)

One YAML per configuration. Notebooks load it with `romscommons.io.load_config`.

```yaml
name: synthetic
ocean_in: ../data/synthetic/ocean_synthetic.in   # optional; if given, it is the source of truth
roms_log: null            # optional ROMS stdout, used to cross-check Courant / rx0 / rx1
grid: ../data/synthetic/grid.nc
ini: ../data/synthetic/ini.nc
ini_record: 0
frc: [../data/synthetic/frc_native.nc, ../data/synthetic/frc_coarse.nc]
tide: ../data/synthetic/tide.nc
bry: ../data/synthetic/bry.nc
nud: null                 # existing nudging file to inspect (nb04 part B)
clm: null
model:                    # used only for keys missing from ocean_in (or when ocean_in is null)
  DT: 180.0
  NDTFAST: 30
  NTIMES: 4800
  N: 12
  Vtransform: 2
  Vstretching: 4
  THETA_S: 7.0
  THETA_B: 2.0
  TCLINE: 50.0
  TIME_REF: 20000101.0
  DSTART: 9497.0
  TIDE_START: 9497.0
  Dcrit: 0.10
open_boundaries: [west, south, east, north]   # used only if LBC not available from ocean_in
points:                   # named stations for profiles / time series
  - {name: shelf, lon: 178.91, lat: -40.51}
  - {name: slope, lon: 180.23, lat: -40.36}
transect: {start: [178.58, -41.09], end: [181.38, -39.86]}   # null → middle eta row
nudging:                  # nb04 builder parameters
  width_cells: 10
  boundary_timescale_days: 1.0
  shape: cosine           # cosine | linear
  tracers: true
  momentum_3d: true
  momentum_2d: false
  interior_timescale_days: null
output_dir: ../outputs/synthetic
```

Rules: relative paths resolve against the YAML file's directory. If `ocean_in` is given,
file paths left `null` in the YAML may be filled from `GRDNAME`, `ININAME`, `FRCNAME`,
`TIDENAME`, `BRYNAME`, `NUDNAME`, `CLMNAME` (resolved relative to the ocean.in
directory). Any conflict between YAML `model` values and ocean.in is reported as WARN;
ocean.in wins.

### 4.1 Real configuration: Moana (`configs/moana.yaml`)

The Moana ROMS input files live in **`/projects/f_omg_1/moa/moana/Input/`** on the HPC
system. To build `configs/moana.yaml`:

1. List that directory (read-only — never write, rename or delete anything there).
2. Map files to roles: `ocean_in`, `grid`, `ini`, `frc` (list), `tide`, `bry`, `nud`, `clm`.
   Use ocean.in's file-name keywords first if an ocean.in is present; otherwise match by
   name and confirm by opening the file's dims/variables.
3. Use absolute paths in this YAML.
4. If a role has several candidates (e.g. multiple ini or bry years) or none, ask the user
   which to use — do not guess. Leave unresolved roles `null`.
5. If the path is not reachable from where Claude Code is running, still create
   `configs/moana.yaml` with the base path in a comment and `null` placeholders, and tell
   the user.

---

## 5. Phase 0 — shared package `romscommons`

Keep functions small, pure, xarray-in/xarray-out, with docstrings that cite the ROMS
routine they mirror. Signatures below are guidance, not contract.

### 5.1 `io.py`
- `load_config(path) -> dict`
- `open_nc(path, chunks=None) -> xr.Dataset` — `decode_times=True`; on failure, reopen
  undecoded and decode manually using the units attribute or TIME_REF.
- `parse_ocean_in(path) -> dict` — handles `KEY == value` and `KEY = value`, `!` comments,
  `\` continuation, Fortran reals (`300.0d0`), logicals `T/F`, repeat syntax (`2*0.0d0`),
  and `LBC(isXxx) == ...` lines → `{var: {"west":…, "south":…, "east":…, "north":…}}`.
  **ROMS order in LBC lines is West, South, East, North.** Tracer LBC continues over
  several lines (one per tracer).
- `time_ref(TIME_REF) -> datetime` — supports `yyyymmdd.f`; other values (0, −1, −2) raise
  `NotImplementedError` with a clear message. `[VERIFY V17]`
- `run_window(params) -> (start, end)` — start = TIME_REF + DSTART days;
  end = start + NTIMES·DT seconds.
- `parse_roms_log(path) -> dict` — regex for the barotropic/Coriolis Courant numbers and
  rx0/rx1 lines in ROMS stdout; tolerant of format differences. `[VERIFY V14]`

### 5.2 `grid.py`
- Expected dims from grid: `xi_rho = Lm+2, eta_rho = Mm+2, xi_u = Lm+1, eta_v = Mm+1,
  xi_psi = Lm+1, eta_psi = Mm+1`. `expected_dims(grid, location)` for rho/u/v/psi.
- Staggering: `rho2u`, `rho2v`, `u2rho`, `v2rho` (simple 2-point averages, mask-aware).
- Rotation (`angle` is in radians at rho points):
  `u_grid = u_e·cosθ + v_n·sinθ`, `v_grid = v_n·cosθ − u_e·sinθ`, and the inverse.
  Functions `to_grid(u_e, v_n, angle)`, `to_earth(u_g, v_g, angle)`.
- Metrics: `dx = 1/pm`, `dy = 1/pn`, `area = 1/(pm·pn)`.
- Boundaries: `side_slices(side, location)` → index slices for rho/u/v points on W/S/E/N.
  `perimeter(grid)` → counterclockwise unrolled perimeter (S: west→east, E: south→north,
  N: east→west, W: north→south), distance along it in km, and corner breakpoints.
  `open_sides(lbc, grid)` → a side is open if its LBC is not `Clo` and it has wet points.
- `transect_indices(grid, start, end)` → nearest (eta, xi) to each endpoint, straight line
  in index space (Bresenham), dropping duplicates. `None` → middle eta row.
- `wrap_lon(lon, convention)` and `lon_convention(lon)` — handle 0–360 vs −180–180 and
  domains crossing the dateline.

### 5.3 `vertical.py`
- `s_coord(N, kind, Vstretching)` — options 1–4: `s_rho = (k − N − 0.5)/N, k=1..N`;
  `s_w = (k − N)/N, k=0..N`. Option 5 uses a non-uniform (Legendre) `s` (VERIFY_LOG V16).
- `stretching(Vstretching, theta_s, theta_b, s, ...)` — port options 1–5 exactly from
  `set_scoord.F`. `[VERIFY V16]`
- `hc` rule: Vtransform 2 → `hc = Tcline`; Vtransform 1 → `hc = min(hmin, Tcline)`.
- Depths (mirror `set_depth.F`):
  - Vtransform 1: `z0 = hc·s + (h − hc)·C`;  `z = z0 + ζ·(1 + z0/h)`
  - Vtransform 2: `z0 = (hc·s + h·C)/(hc + h)`;  `z = ζ + (ζ + h)·z0`
- `z_r`, `z_w`, `Hz = diff(z_w)`, `depth_average(var, Hz)`.
- **Tests:** recomputed `Cs_r`/`Cs_w` match those stored in a file when present
  (atol 1e-6); `z_w[0] = −h`, `z_w[N] = ζ`; `Hz > 0`; `ΣHz = h + ζ`.

### 5.4 `diagnostics.py`
All match what ROMS prints at initialization (Courant: `metrics.F`; rx0/rx1: `stiffness.F`)
where an equivalent exists.
- `barotropic_courant = (DT/NDTFAST)·sqrt(g·h)·sqrt(pm² + pn²)`, g = 9.81. `[VERIFY V14]`
- `coriolis_courant = DT·|f|`
- `advective_courant` — `|u|·DT·pm` at u-points, `|v|·DT·pn` at v-points, max over k.
- `internal_courant = c1·DT·sqrt(pm² + pn²)`, with WKB first-mode speed `c1 ≈ (1/π)∫N dz`.
- `rx0` (Beckmann & Haidvogel) on u- and v-faces between wet cells:
  `|h₁ − h₂| / (h₁ + h₂)`.
- `rx1` (Haney), per face, max over k; ζ = 0 by default, optional IC ζ (ROMS computes it
  with the IC ζ — VERIFY_LOG V14); faces between interior wet rho points only, as in ROMS:
  `|z_w(k,i) − z_w(k,i−1) + z_w(k−1,i) − z_w(k−1,i−1)| / (z_w(k,i) + z_w(k,i−1) − z_w(k−1,i) − z_w(k−1,i−1))`.
- `n_squared(temp, salt, z, lon, lat)` via gsw (`SA_from_SP`, `CT_from_pt`, `Nsquared`).
  Docstring states ROMS temp = potential temperature, salt = practical salinity, and that
  this approximates ROMS's EOS.
- `boundary_transport(ubar|vbar, h, zeta, grid, side)` → volume transport (m³ s⁻¹),
  sign convention **inflow positive** (W: +ubar, S: +vbar, E: −ubar, N: −vbar);
  depth at u/v points = average of adjacent `h` + boundary ζ (documented approximation).
- Bulk estimates (crude, labelled as such, not ROMS's COARE):
  - Wind stress: Large & Pond (1981) Cd·10³ = 1.2 for U < 11, 0.49 + 0.065U for 11 ≤ U ≤ 25
    (cap above 25); ρₐ = 1.22 kg m⁻³. Curl, Ekman transport τ/(ρ₀ f) (sign via f, so the
    Southern Hemisphere is handled), Ekman pumping curl(τ/(ρ₀ f)).
  - Net heat flux into the ocean: `SW_net + LW_down − 0.97σT_s⁴ − Q_sens − Q_lat`,
    `Q_sens = ρₐ c_p C_h U (T_s − T_a)`, `Q_lat = ρₐ L C_e U (q_s − q_a)`, with
    c_p = 1004, L = 2.5e6, C_h = 1.1e-3, C_e = 1.2e-3; `e_sat = 6.112·exp(17.67T/(T+243.5))` mb,
    `q = 0.622e/(P − 0.378e)`, `q_s = 0.98·q_sat(T_s)`.

### 5.5 `checks.py`
- `Status` enum: PASS, WARN, FAIL, INFO.
- `Check(name, status, value, threshold, source, message)` dataclass.
- `Report` — `add()`, `to_frame()`, `display()` (colored pandas Styler), `summary()` one-liner,
  `worst()`.
- `THRESHOLDS` — single dict; each entry `{"value": ..., "source": "..."}`:

| Key | Default | Status if exceeded | Source to cite |
|---|---|---|---|
| `rx0_warn` | 0.2 | WARN | Beckmann & Haidvogel 1993; Haidvogel et al. 2000 |
| `rx1_warn` | 6.0 | WARN | Haney 1991 (literature range 3–8) |
| `cbt_warn` | 0.8 | WARN | rule of thumb for split-explicit barotropic step |
| `cadv_warn` | 0.5 | WARN | advective CFL margin |
| `cint_warn` | 0.5 | WARN | internal-wave CFL margin |
| `n2_min` | −1e-5 s⁻² | WARN (any wet column below) | static stability |
| `temp_range` | [−2.5, 35] °C | WARN | plausibility |
| `salt_range` | [2, 42] | WARN | plausibility |
| `speed_max` | 3 m s⁻¹ | WARN | plausibility |
| `zeta_max` | 3 m | WARN | plausibility |
| `ubar_rms` | 0.01 m s⁻¹ | WARN | ubar vs depth-averaged u |
| `nud_coefdt_warn` / `fail` | 0.1 / 1.0 | WARN / FAIL | explicit relaxation stability |
| `icbc_temp` / `salt` / `zeta` / `vel` | 0.5 °C / 0.1 / 0.1 m / 0.1 m s⁻¹ | WARN | shock at start |
| `ssh_drift` | 1 cm day⁻¹ mean or 0.1 m over run | WARN | boundary volume imbalance |
| `bry_tide_m2` | 0.05 m | WARN | tides double-counted |

- Generic checks reused by all notebooks: `required_vars`, `dims_match_grid`,
  `finite_at_wet`, `finite_at_land`, `in_range`, `units_match`, `time_monotonic`,
  `time_gaps` (> 1.5 × median step), `time_covers(run_start, run_end)` (needs a record
  at/before start and at/after end; skipped as INFO when `cycle_length` is present).

### 5.6 `tides.py`
- `CONSTITUENTS` — name → period (h): M2 12.4206012, S2 12.0, N2 12.65834751,
  K2 11.96723606, K1 23.93447213, O1 25.81933871, P1 24.06589, Q1 26.86835,
  M4 6.210300601, MS4 6.103339, MN4 6.269174, Mf 327.8599, Mm 661.3093.
  `identify(periods, tol_h=1e-3)`.
- `ap2ep(Au, PHIu, Av, PHIv)` and `ep2ap(...)` after Xu (2002) → (Cmax, Cmin, Cangle,
  Cphase), degrees. Round-trip test to 1e-8.
- `reconstruct_ssh(Eamp, Ephase, period_h, t, tide_start)` and `reconstruct_uv(...)`
  mirroring `set_tides.F`. Expected form `ζ(t) = Σ Eamp·cos(ω(t − tide_start) − Ephase)`.
  `[VERIFY V8]`

### 5.7 `plotting.py`
- `map(da, grid, ax=None, cmap=None, title=None)` — pcolormesh on lon/lat, land gray,
  units from attrs, dateline-safe (cartopy `central_longitude=180` when needed).
- `section(da3d, z, idx)`, `profile(...)`, `perimeter_section(values, z, perim)` with
  corner lines labelled SW/SE/NE/NW, `quiver_earth(u_g, v_g, grid, stride)`,
  `ellipses(Cmax, Cmin, Cangle, grid, stride)`, `cgrid_zoom(grid, i0, j0, n=4)` (shows
  rho/u/v/psi points and masks for a 4×4 patch).

### 5.8 Phase 0 tests
Unit tests for every function above with analytic inputs: rotation round trip; rx0 on a
step bathymetry with known answer; Courant on uniform grid; Cs and z identities;
ap2ep/ep2ap round trip; ocean.in parser on a fixture with continuation lines, repeat
syntax and LBC lines; perimeter ordering and lengths; time-coverage edge cases.

---

## 6. Synthetic case — `examples/make_synthetic_case.py`

Small, structurally ROMS-conformant (not a runnable ROMS case). Total < 20 MB.

- **Grid:** spherical, `xi_rho = 42, eta_rho = 32`, ~8 km cells, rotated 30°, longitudes
  178°E–182°E stored 0–360 (crosses the dateline), latitude ≈ −40° (f < 0). Shelf
  30 m → slope → 2500 m (smoothed so rx0 < 0.2), one small island for masking.
- **Vertical:** N = 12, Vtransform 2, Vstretching 4, θs 7, θb 2, Tcline 50; write
  `Cs_r`, `Cs_w`, `s_rho`, `s_w` and the parameters into the IC file.
- **ini.nc:** one record at run start; statically stable T/S; along-slope current
  ~0.2 m s⁻¹ in grid coordinates; `ubar` consistent with depth-averaged `u`.
- **frc_native.nc:** on the ROMS grid, hourly, start − 1 h to end + 1 h: Uwind, Vwind,
  Pair (mbar), Qair (%), rain, swrad (diurnal), lwrad_down, each with its own time variable.
- **frc_coarse.nc:** Tair on a 0.25° lon/lat grid (0–360) with a `coordinates`
  attribute, colder land values over the island region (land-contamination example).
- **tide.nc:** M2, S2, K1, O1; Kelvin-wave-like amplitude/phase; currents ≈ A·sqrt(g/h).
- **bry.nc:** all four sides, 3-hourly, subtidal, consistent with ini at t0.
- **ocean_synthetic.in:** only the keys the parser needs (DT, NDTFAST, NTIMES, N,
  Vtransform, Vstretching, THETA_S, THETA_B, TCLINE, TIME_REF, DSTART, TIDE_START,
  file names, LBC block including a multi-line tracer entry, nudging keys). Header comment:
  "parser fixture — not a runnable ROMS input".
- 10-day run window.

**`--defects`** writes the same case with these injected faults (each has a test in
`tests/test_defects.py` asserting the expected status):

| File | Defect | Expected |
|---|---|---|
| grid | single-cell bathymetry spike | rx0 WARN |
| ini | 3 NaN wet points in temp | FAIL finite_at_wet |
| ini | ubar = 1.5 × depth-avg u | WARN ubar_rms |
| ini | THETA_S in file ≠ ocean.in | FAIL vertical params |
| ini | one temperature inversion | WARN N² |
| frc | Pair in Pa | FAIL units |
| frc | Tair in K | FAIL units |
| frc | 12 h gap | WARN time_gaps |
| frc | ends 1 day before run end | FAIL time_covers |
| frc_coarse | longitudes −180–180 | FAIL lon convention |
| tide | phases in radians | WARN phase units |
| tide | one constituent with Cmin > Cmax | FAIL |
| bry | +0.2 m s⁻¹ ubar on west only | WARN ssh_drift |
| bry | temp +2 °C vs ini at t0 | WARN icbc_temp |
| bry | M2 signal in zeta | WARN bry_tide_m2 |
| nud | coefficient 1/(0.001 day) | FAIL coef·DT |

---

## 7. Notebook conventions (all five)

Section skeleton — same headings in every notebook:

0. **Title + "What this file is for"** — one paragraph.
1. **Where it enters ROMS** — `ocean.in` keyword(s), relevant CPP options, when ROMS
   reads it, how ROMS interpolates it in time/space.
2. **Inputs** — the one editable cell (`CONFIG = ...`), then load config, grid, files;
   print run window.
3. **Anatomy** — ncdump-style summary, then a table: variable, dims, expected dims,
   location (rho/u/v), units, expected units, time variable, status.
4. **Visual tour** — the file's contents, plotted.
5. **Diagnostics** — calculations that matter for stability or correctness.
6. **Report** — `report.display()` + one-line summary.
7. **Troubleshooting** — table: symptom/FAIL → usual cause → fix.
8. **References.**

Writing rules: markdown before every code cell; "Why it matters" callouts where a number
connects to a failure mode (blow-up, drift, shock). No hidden state between cells; a cell
re-run in order gives the same result.

---

## 8. Notebook specs

### 8.1 `01_initial_conditions`

**Where it enters ROMS:** `ININAME`; record chosen with `NRREC`; start time `DSTART`
(days since `TIME_REF`). ROMS computes depths from the grid `h`, the file's `zeta`, and the
vertical parameters. `[VERIFY V1, V2]`

**Required variables**

| Variable | Dims | Point | Units |
|---|---|---|---|
| ocean_time | (ocean_time) | — | time units attribute |
| zeta | (ocean_time, eta_rho, xi_rho) | rho | m |
| ubar | (ocean_time, eta_u, xi_u) | u | m s⁻¹ (grid coords) |
| vbar | (ocean_time, eta_v, xi_v) | v | m s⁻¹ (grid coords) |
| u | (ocean_time, s_rho, eta_u, xi_u) | u | m s⁻¹ (grid coords) |
| v | (ocean_time, s_rho, eta_v, xi_v) | v | m s⁻¹ (grid coords) |
| temp | (ocean_time, s_rho, eta_rho, xi_rho) | rho | °C, potential temperature |
| salt | (ocean_time, s_rho, eta_rho, xi_rho) | rho | practical salinity |

Optional but checked if present: Vtransform, Vstretching, theta_s, theta_b, Tcline, hc,
s_rho, s_w, Cs_r, Cs_w, h, spherical.

**Visual tour**
1. C-grid primer: `cgrid_zoom` on a 4×4 patch at a coastline (rho/u/v/psi points, masks).
2. Grid maps: `h`, resolution `1/pm`, `1/pn` (km), `angle` (deg), masks.
3. Vertical grid: s-levels along the transect over bathymetry; `Cs_r` and `Cs_w` vs s;
   `Hz` profile at the shallowest and deepest wet points; map of minimum `Hz`.
4. Fields: surface and bottom maps of temp, salt, speed; `zeta`; quiver of `ubar/vbar`
   rotated to east/north; T and S sections along the transect; profiles at `points`;
   T–S diagram from a random sample of wet cells.

**Diagnostics → checks**

| Check | Rule |
|---|---|
| Required variables | FAIL if missing |
| Dims match grid / staggering | FAIL |
| `len(s_rho)` = N | FAIL |
| Vertical params in file = ocean.in | FAIL if any differ |
| Recomputed Cs vs file Cs | FAIL if > 1e-6 |
| `h` in file = grid `h` (if present) | FAIL if > 1e-3 m |
| Record time = DSTART | WARN if different |
| Finite at wet points (each var) | FAIL |
| Finite at land points | WARN (NaN × mask = NaN) `[VERIFY V3]` |
| `_FillValue` on state variables | WARN `[VERIFY V3]` |
| Ranges (temp, salt, speed, zeta) | WARN |
| Total depth `h + ζ` > Dcrit (or > 0) | FAIL |
| `Hz > 0` everywhere | FAIL |
| ubar/vbar vs depth-averaged u/v (RMS) | WARN |
| max \|angle\| | INFO ("rotation matters here" if > 1°) |
| Barotropic Courant (map + max) | WARN |
| Coriolis Courant | INFO |
| Advective Courant (3D max) | WARN |
| Internal-wave Courant (c1 map) | WARN |
| rx0, rx1 (maps + top-10 hotspot table with lon/lat) | WARN |
| N² inversions (map of inverted cells per column) | WARN |

Final figure: **"where would it blow up first"** — one map overlaying the top hotspots
from rx1, Courant, and N² inversions.

If `roms_log` is given: table comparing notebook vs ROMS-printed Courant/rx0/rx1
(should agree to ~1e-3). This is the main real-data validation of Phase 1.

---

### 8.2 `02_atmospheric_forcing`

**Where it enters ROMS:** `FRCNAME` (list of files); `BULK_FLUXES` (ROMS computes fluxes
from atmospheric state) vs prescribed fluxes; each variable's `time` attribute names its
time variable; `cycle_length` marks climatology; forcing on a non-ROMS grid is
interpolated by ROMS when the variable carries a `coordinates` attribute; ROMS needs
records bracketing each model time; `DIURNAL_SRFLUX` when shortwave is daily-mean.

**Expected variables** (detect mode from what is present)

| Mode | Variable | Usual time var | Units |
|---|---|---|---|
| Bulk | Uwind, Vwind | wind_time | m s⁻¹ (10 m) `[VERIFY V4]` |
| Bulk | Pair | pair_time | mbar |
| Bulk | Tair | tair_time | °C |
| Bulk | Qair | qair_time | % relative humidity `[VERIFY V5]` |
| Bulk | rain | rain_time | kg m⁻² s⁻¹ |
| Bulk | swrad | srf_time | W m⁻² (net vs downward `[VERIFY V6]`) |
| Bulk | lwrad_down / lwrad | lrf_time | W m⁻² `[VERIFY V6]` |
| Flux | sustr, svstr | sms_time | N m⁻² |
| Flux | shflux | shf_time | W m⁻², positive into ocean |
| Flux | swflux | swf_time | m s⁻¹, (E−P)/ρw, positive = evaporation (V7; older ROMS: cm day⁻¹) |

Time variable names are conventions; the `time` attribute is what ROMS follows.

**Visual tour**
1. Per-variable inventory table: file, time var, first/last time, step, records, grid type
   (native / own lon-lat), units vs expected.
2. Timeline chart: one bar per variable spanning its coverage, with the run window shaded.
3. Snapshot maps at run start for each variable, ROMS grid outline drawn over forcing grids.
4. Time series: domain mean and each `points` station; wind quivers; wind rose (polar
   histogram); swrad diurnal composite; monthly means if the record is long.
5. Land contamination: Tair minus IC SST at t0 near the coast; forcing land mask vs ROMS
   coastline when available.

**Diagnostics → checks**

| Check | Rule |
|---|---|
| `time` attribute points to an existing variable | FAIL |
| Time monotonic, no duplicates | FAIL |
| Gaps > 1.5 × median step | WARN |
| Coverage of run window (non-cyclic) | FAIL |
| Pair median > 5000 | FAIL ("looks like Pa") |
| Tair median > 150 | FAIL ("looks like K") |
| Qair max ≤ 1 → fraction; median < 30 → maybe specific humidity | WARN |
| swrad max > 1e4 | FAIL ("looks like accumulated J m⁻²") |
| swrad < −1 | WARN |
| No night-time zeros in swrad with step ≤ 3 h | INFO (daily means → DIURNAL_SRFLUX) |
| lwrad_down outside 50–550 | WARN |
| Implied annual rain > 5 m yr⁻¹ or < 0 | WARN |
| Wind speed > 60 m s⁻¹ | WARN |
| Forcing grid covers ROMS domain | FAIL |
| Longitude convention matches ROMS grid | FAIL |
| Land contamination indicator (coastal \|Tair − SST\| ≫ offshore) | WARN |

**Calculations:** wind stress, curl, Ekman transport and pumping maps (f < 0 handled);
crude net heat-flux map and domain-mean series over the run using IC SST held fixed,
labelled "flux the ocean would feel if SST stayed at its initial value — order of
magnitude only".

**Markdown box — common source-data gotchas (ERA5-style):** pressure Pa → mbar;
temperature K → °C; dewpoint → relative humidity (Magnus); hourly accumulated radiation
J m⁻² → W m⁻² (÷ 3600); precipitation m → kg m⁻² s⁻¹ (× 1000/3600); accumulations are
stamped at the end of their window (half-step shift).

---

### 8.3 `03_tides`

**Where it enters ROMS:** `TIDENAME`; `SSH_TIDES`, `UV_TIDES`; `ADD_FSOBC`, `ADD_M2OBC`
add tidal elevation/currents to the boundary values; `RAMP_TIDES`; `TIDE_START` (days
since TIME_REF) is the phase reference; optional `POT_TIDES` (equilibrium potential).
ROMS applies no nodal corrections — amplitudes must already include `f` and phases
`V0 + u` at TIDE_START. `[VERIFY V8, V9]`

**Required variables:** `tide_period` (h), `tide_Eamp` (m), `tide_Ephase` (deg),
`tide_Cmax`, `tide_Cmin` (m s⁻¹), `tide_Cangle`, `tide_Cphase` (deg); dims
`(tide_period, eta_rho, xi_rho)`.

**Part A — inspect**
1. Constituent table: period → name (via `identify`), domain max/mean amplitude.
2. Co-tidal charts per constituent (amplitude color, phase contours with `phase`
   colormap; amphidromes visible).
3. Ellipse maps (subsampled) for the main constituents.
4. Reconstructed ζ and currents at `points` over the run (≥ one spring–neap cycle).
5. Calculations:
   - Form factor `F = (K1 + O1)/(M2 + S2)` map with classes (< 0.25 semidiurnal,
     0.25–1.5 mixed mainly semidiurnal, 1.5–3 mixed mainly diurnal, > 3 diurnal).
   - Maximum range ≈ 2 Σ amplitudes; spring/neap ratio (M2+S2)/(M2−S2).
   - M2 tidal excursion `L = Cmax·T/π`.
   - Simpson–Hunter `log10(h/U³)` (SI) map — tidal-mixing fronts; threshold is
     region-dependent (often quoted near 2.7).
   - Minimum total depth at lowest tide `h − Σ Eamp` vs Dcrit.
   - Tidal + IC advective Courant `(Cmax + |u_IC|)·DT·pm`.

| Check | Rule |
|---|---|
| Required vars, dims | FAIL |
| Unknown periods | WARN |
| Finite everywhere | FAIL wet / WARN land |
| Amplitudes ≥ 0; Cmax ≥ \|Cmin\| | FAIL |
| Phases look like radians (max ≤ 2π) | WARN |
| Zero/NaN amplitude on open-boundary wet points | WARN |
| No record of phase reference date in file attributes | WARN ("cannot verify phases are referenced to TIDE_START") |
| Run extends > 1 yr past TIDE_START | INFO (nodal factors drift) |
| `h − Σ Eamp` ≤ Dcrit without wet/dry | FAIL; < 2 m WARN |
| Tidal advective Courant | WARN |

**Part B — demo: build a ROMS tide file**
1. Source: TPXO or FES files supplied by the user (both require registration — the
   notebook explains this) read with pyTMD at ROMS rho points. If none configured, use a
   synthetic constituent set and say so.
2. Land/coast mismatch: fill ROMS wet points that are land in the source by
   nearest-neighbour from source wet points; map the filled points.
3. Transports → velocities: divide by ROMS `h` (preserves transport) — explain the
   alternative (source depth preserves velocity) and why we choose this; clip at a
   minimum depth.
4. Nodal corrections at TIDE_START (pyTMD; check the installed version's API): apply
   `A·f`, `G − (V0 + u)` to elevation and to the u and v components *before* ellipse
   conversion. Sanity test: M2 `f` in [0.96, 1.04].
5. `ap2ep` → Cmax, Cmin, Cangle, Cphase. Cangle reference (east vs grid) `[VERIFY V8]`.
6. Write `outputs/<name>/tide_<start>.nc` with ROMS names/units, plus global attributes
   recording source model, constituents, and the reference date.
7. Close the loop: run Part A's checks on the new file; reconstruct ζ at a point for 15
   days and compare with pyTMD's direct prediction — max difference < 5 mm (real source
   only).

---

### 8.4 `04_nudging`

**Where it enters ROMS:** `NUDNAME` (coefficients), `CLMNAME` (target fields), and flags
`LnudgeM2CLM`, `LnudgeM3CLM`, `LnudgeTCLM` (per tracer). Tendency
`∂C/∂t = … + γ(C_clm − C)`, γ in day⁻¹. Distinguish clearly from: boundary nudging inside
the LBC scheme (`TNUDG`, `ZNUDG`, `M2NUDG`, `M3NUDG`, `OBCFAC`, used with `RadNud`) and
the viscosity/diffusion sponge. `[VERIFY V10, V11]`

**Expected variables** `[VERIFY V10]`: `M2_NudgeCoef (eta_rho, xi_rho)`,
`M3_NudgeCoef (s_rho, eta_rho, xi_rho)`, `temp_NudgeCoef`, `salt_NudgeCoef` (or
`tracer_NudgeCoef`) `(s_rho, eta_rho, xi_rho)`; units `day-1`.

**Part A — build**
1. Open sides from ocean.in LBC (else config).
2. Distance-from-open-boundary field in cells (min over open sides; corners handled
   naturally); map it.
3. Taper over `width_cells`: cosine `0.5(1 + cos(π d/W))` or linear `1 − d/W`;
   γ = taper / `boundary_timescale_days`; optional weak interior γ; zero on land;
   broadcast over s_rho. Which fields get γ follows `nudging.tracers / momentum_3d /
   momentum_2d` (default: M2 off — Flather already handles the barotropic boundary).
4. Plots: timescale map (days, log color) with sponge-edge contour; γ vs distance curve;
   overlay on bathymetry.
5. Toy demo: `C(t) = C_clm + (C0 − C_clm)·e^(−γt)` at boundary, mid-sponge, sponge edge
   over 30 days — "how fast does the model forget".
6. Write `outputs/<name>/nud.nc` (attributes record all parameters and grid file).

**Part B — inspect an existing file** (`nud:` in config; same plots + checks).

| Check | Rule |
|---|---|
| Variable names/dims/units | FAIL |
| γ ≥ 0, finite | FAIL |
| γ·DT (DT in days) | WARN > 0.1, FAIL ≥ 1 |
| γ = 0 on land | WARN |
| Non-zero γ on a side whose LBC is closed | WARN |
| ocean.in flags on but file absent (or vice versa) | WARN |
| Climatology file (if configured) dims + time coverage | FAIL/WARN as in §5.5 |

---

### 8.5 `05_boundary_conditions`

**Where it enters ROMS:** `BRYNAME`; `LBC` keywords per variable and side (order W, S,
E, N): `Cha, Che, Cla, Clo, Fla, Gra, Mix, Nes, Per, Rad, RadNud, Red, Shc` (V12) — the notebook shows
which ones need data and gives the common realistic combination (Chapman ζ, Flather
ubar/vbar, RadNud for 3D momentum and tracers). Boundary nudging time scales; optional
boundary volume conservation; with `ADD_FSOBC/ADD_M2OBC` boundary data must be subtidal.
`[VERIFY V12, V13, V15]`

**Expected variables:** `zeta_<side>`, `ubar_<side>`, `vbar_<side>`, `u_<side>`,
`v_<side>`, `temp_<side>`, `salt_<side>` for each open side. West/east use eta dims
(e.g. `temp_west (temp_time, s_rho, eta_rho)`, `u_west (v3d_time, s_rho, eta_u)`,
`v_west (v3d_time, s_rho, eta_v)`); south/north use xi dims. Usual time vars:
`zeta_time, v2d_time, v3d_time, temp_time, salt_time`.

**Visual tour**
1. Map of the domain with open sides highlighted (from LBC + mask) and the LBC table.
2. **Unrolled perimeter section** at a chosen time: temp, salt, and normal velocity
   (inflow positive) vs perimeter distance and depth, corners marked, land gray.
3. Hovmöller (perimeter distance × time): ζ and surface temp.
4. Per-side time series: mean temp/salt, normal transport.
5. Anatomy extra: extract the IC's boundary slices into the same layout to show "a bry
   file is the edge of a 3D field over time".

**Diagnostics → checks**

| Check | Rule |
|---|---|
| Required vars for each open side; dims | FAIL |
| Time vars, monotonic, coverage of run | FAIL |
| Finite at wet boundary points | FAIL |
| Ranges | WARN |
| `h + ζ_bry` > Dcrit | FAIL |
| ubar vs depth-average of u per side (RMS) | WARN |
| IC–BC mismatch at t0 (bry interpolated to IC time), per var and side | WARN |
| Net boundary volume flux → domain-mean dη/dt (cm day⁻¹) and cumulative drift | WARN |
| Tides in bry while tides are added (least-squares M2/S2/K1/O1 fit on ζ when step ≤ 3 h and ≥ 3 days) | WARN |

Show the volume budget as a figure: per-side transports (Sv), net, and implied
cumulative sea-level change over the run.

---

## 9. Verification

- **Unit tests** (Phase 0) — analytic cases, see §5.8.
- **Defect tests** — `tests/test_defects.py`, one assertion per row of the §6 table,
  calling the same `romscommons` check functions the notebooks call.
- **Notebook execution** — `pytest --nbmake notebooks/` on `configs/synthetic.yaml`:
  no errors, no FAIL, < 2 min each.
- **Real-data acceptance** (done with the user): run on `configs/moana.yaml`
  (data in `/projects/f_omg_1/moa/moana/Input/`, §4.1); nb01 Courant/rx0/rx1 agree with the
  ROMS log; nb03 demo matches pyTMD prediction within 5 mm.

---

## 10. Build order and checkpoints

| Phase | Deliverable | Verify | Then |
|---|---|---|---|
| 0 | env, `romscommons`, synthetic generator (good + defects), configs (synthetic, defects, moana per §4.1), unit tests | `pytest` green | **Stop — summarize to user** |
| 1 | nb01 + its defect tests | nbmake + defects green | **Stop — user reviews; nb01 sets the pattern for the rest** |
| 2 | nb02 | same | continue |
| 3 | nb03 (A then B) | same + B closed-loop check | continue |
| 4 | nb04 | same | continue |
| 5 | nb05 | same | continue |
| 6 | README (purpose, install, how to point at your own files, notebook index, VERIFY summary), `TEMPLATE.yaml` | fresh-clone install works | **Stop — final summary** |

Commit at the end of each phase. Do not start a phase while the previous one's checks fail.

---

## 11. `[VERIFY]` register

Resolve against ROMS source; log answers in `VERIFY_LOG.md`.

| ID | Question | Look in |
|---|---|---|
| V1 | Does ROMS abort if IC vertical params differ from ocean.in? | `get_state.F`, `checkvars.F` |
| V2 | How is the IC record chosen (NRREC = 0 vs > 0) and is DSTART enforced? | `get_state.F`, `initial.F` |
| V3 | How are NaN / `_FillValue` treated on read? | `nf_fread2d.F`, `nf_fread3d.F` |
| V4 | Uwind/Vwind: east/north or grid coords, native grid vs interpolated? | `set_data.F` |
| V5 | Qair units and the RH vs specific-humidity branch | `varinfo.yaml`, `bulk_flux.F` |
| V6 | swrad net vs downward; lwrad vs lwrad_down and the CPP options involved | `bulk_flux.F`, `cppdefs.h` |
| V7 | swflux units and sign | `varinfo.yaml` |
| V8 | Tide reconstruction formula; Cangle reference; rotation to grid | `set_tides.F` |
| V9 | Tide file variable names; which are required for which options | `varinfo.yaml`, `get_tides` routine |
| V10 | Nudging coefficient names, dims, units | `get_nudgcoef.F`, `varinfo.yaml` |
| V11 | Where sponge factors come from (grid file vs analytic) | `ana_sponge.h`, grid reading code |
| V12 | Full LBC keyword list and which require data | `roms.in` template, `read_phypar.F` |
| V13 | Boundary volume-conservation keyword | `roms.in` template |
| V14 | Exact Courant / rx0 / rx1 formulas and stdout format | `metrics.F`, `stiffness.F` |
| V15 | Boundary time variable names | `varinfo.yaml` |
| V16 | Vstretching 1–5 formulas | `set_scoord.F` |
| V17 | TIME_REF special values | `roms.in` template, `get_date` routines |

---

## 12. References

- Shchepetkin & McWilliams (2005), *Ocean Modelling* — ROMS split-explicit kernel.
- Haidvogel et al. (2008), *J. Comput. Phys.* — ROMS overview.
- Beckmann & Haidvogel (1993), *JPO*; Haney (1991), *JPO* — pressure-gradient error, rx0/rx1.
- Marchesiello, McWilliams & Shchepetkin (2001), *Ocean Modelling* — open boundaries.
- Flather (1976); Chapman (1985), *JPO* — barotropic OBCs.
- Large & Pond (1981), *JPO*; Fairall et al. (2003), *J. Climate* — bulk fluxes.
- Egbert & Erofeeva (2002), *JAOT* — TPXO.
- Xu (2002) — tidal ellipse conversion (ap2ep).
- Pawlowicz, Beardsley & Lentz (2002), *Comput. Geosci.* — nodal corrections (T_TIDE).
- Simpson & Hunter (1974), *Nature* — tidal mixing fronts.
- ROMS wiki: https://www.myroms.org/wiki/ ; source: https://github.com/myroms/roms
