# VERIFY log — answers to BUILD_SPEC `[VERIFY]` items

Source checked: `github.com/myroms/roms`, shallow clone at `../roms-reference`,
commit `31cabd3` (2026-09-28). Paths below are relative to that clone.
Where source and spec disagreed, source won and the spec was corrected (marked **SPEC FIX**).

---

## V1 — Does ROMS abort if IC vertical params differ from ocean.in?

**No.** `Utility/checkvars.F` calls `netcdf_check_dim` (`Modules/mod_netcdf.F` ~l.600–700),
which aborts (`exit_flag=2`) when `xi_rho`, `eta_rho`, `xi_u`, `eta_u`, `xi_v`, `eta_v`,
`xi_psi`, `eta_psi` or `s_rho` differ from the model. Nothing compares `theta_s`, `theta_b`,
`Tcline`, `hc`, `Vtransform`, `Vstretching` in the IC file with ocean.in. ROMS uses the
ocean.in values and silently places the IC data on different depths.
→ Notebook keeps this as **FAIL** (it invalidates the run even though ROMS does not stop).

## V2 — How is the IC record chosen; is DSTART enforced?

`Utility/get_state.F` l.369–391 and l.433–438:

- `NRREC = 0` → record 1. `NRREC > 0` → that record (abort if > number of records).
  `NRREC < 0` → the record with the largest time (`LastRec`).
- For the nonlinear model the clock is set from the **file's** time:
  `time(ng)=INPtime`; `ntstart = NINT((time − dstart·day2sec)/dt) + 1` (clipped to ≥ 1);
  `ntend = ntstart + ntimes − 1`. **DSTART is not enforced**; a record time ≠ DSTART shifts
  the run. → Notebook: WARN when record time ≠ DSTART (as specified).
- Time units: `Modules/mod_netcdf.F::netcdf_get_time_1d` (l.6561+) converts
  `"<units> since <date>"` to elapsed time since TIME_REF when the dates differ, so input
  files may use any reference date.

## V3 — NaN / `_FillValue` on read

`Utility/nf_fread3d.F` l.229–262 and l.334–338 (same in `nf_fread2d.F`):

```fortran
IF (ABS(wrk(i)).ge.ABS(Aspval)) THEN
  wrk(i)=0.0_r8                   ! masked with _FillValue
```

`Aspval` = the variable's `_FillValue`, else `spval_check = 1.0E+35` (`Modules/mod_scalars.F` l.861).
- Values with |x| ≥ |_FillValue| become **0**, not NaN.
- **NaN is not caught**: `ABS(NaN) >= x` is false, so NaN passes into the model state.
- Values are copied as read. Land points are set to 0 only when reading packed
  water-points-only data (`READ_WATER`, `nf_fread3d.F` l.872–886). Otherwise NaN on land stays
  NaN, and `NaN × 0 = NaN` lets it leak through stencils next to the coast.
- A `_FillValue` with small magnitude zeroes every value larger in magnitude.
→ Notebook: NaN at wet points FAIL; NaN at land WARN; `_FillValue` on state variables WARN
  (and report its magnitude).

## V4 — Uwind/Vwind: east/north or grid; native vs interpolated

`Nonlinear/set_data.F` l.352–370 (`CURVGRID` only):

```fortran
IF (.not.Linfo(1,idUair,ng).or.                                 &
    (Iinfo(5,idUair,ng).ne.Lm(ng)+2).or.                        &
    (Iinfo(6,idUair,ng).ne.Mm(ng)+2)) THEN
   ... Uwind = Uwind*CosAngler + Vwind*SinAngler  (rotate east/north -> grid)
```

`Linfo(1,*)` = "gridded data" (`Modules/mod_ncparam.F` l.33).
- Winds that are point data or **interpolated from a coarse grid** are treated as
  east/north and rotated to grid (ξ, η).
- Winds already on the **native ROMS rho grid** are **not rotated**: they must be supplied in
  grid (ξ, η) components when the grid is curvilinear. Same rule for `sustr/svstr`
  (l.587–600). Without `CURVGRID`, nothing is rotated.
- Height: varinfo says "surface u-wind component" (m s⁻¹); bulk_flux uses `blk_ZW` from
  ocean.in (usually 10 m).

## V5 — Qair units and RH vs specific-humidity branch

`External/varinfo.yaml` `Qair`: units `percentage`, `scale: 0.01d0  # 1/100 or 1.0`.
`Nonlinear/bulk_flux.F` l.703–710:

```fortran
IF (RH.lt.2.0_r8) THEN        !RH fraction
...
ELSE          !RH input was actually specific humidity in g/kg
  Q(i)=RH/1000.0_r8
```

- With default varinfo, Qair in **%** → ×0.01 → fraction. Correct.
- Qair supplied as a fraction (0–1) → ×0.01 → 0–0.01 → ROMS sees ~0 % RH. Silent error.
- Specific humidity in g/kg works only if the user changed the varinfo scale to 1.0.
→ Notebook: Qair max ≤ 1 → WARN (fraction); median < 30 → WARN (maybe specific humidity).

## V6 — swrad net vs downward; lwrad vs lwrad_down; CPP options

- `swrad` (`varinfo.yaml` l.2056): standard_name `net_downward_shortwave_flux_at_sea_water_surface`,
  "net solar shortwave radiation flux". The albedo line in `bulk_flux.F` l.1248 is commented
  out (`!!`), so **swrad must be net** (downward × (1 − albedo)). Daily means need
  `DIURNAL_SRFLUX` (varinfo header l.47–51).
- Longwave (`Include/cppdefs.h` l.110–120, `bulk_flux.F` l.616–654):
  - `LONGWAVE`: ROMS computes net longwave (Berliand) from Tair, RH, **cloud**, SST.
  - `LONGWAVE_OUT`: input is **`lwrad_down`** (downward only); ROMS subtracts εσT_s⁴.
  - neither (default): input is **`lwrad`** = net longwave.

## V7 — swflux units and sign

`External/varinfo.yaml` l.1219–1228:
`swflux`: long_name "data surface net freshwater flux, (E-P)/rhow", units **meter second-1**,
standard_name `surface_upward_water_flux_from_file`, scale 1.0 → **positive = net evaporation**.
**SPEC FIX:** spec §8.2 said cm day⁻¹; this ROMS version expects m s⁻¹.

## V8 — Tide reconstruction formula; Cangle reference; rotation to grid

`Nonlinear/set_tides.F`:
- Elevation, l.280–288: `omega = 2π(time − tide_start·day2sec)/Tperiod`;
  `Etide += ramp·SSH_Tamp·COS(omega − SSH_Tphase)`. Matches spec form
  `ζ = Σ A cos(ω(t − TIDE_START) − φ)`. `ramp = tanh((tdays − dstart)/1 day)` with `RAMP_TIDES`, else 1.
- Currents, l.437–452:
  ```fortran
  angle=UV_Tangle(i,j,itide)-angler(i,j)
  phase=omega-UV_Tphase(i,j,itide)
  Uwrk=UV_Tmajor*COS(angle)*COS(phase)-UV_Tminor*SIN(angle)*SIN(phase)
  Vwrk=UV_Tmajor*SIN(angle)*COS(phase)+UV_Tminor*COS(angle)*SIN(phase)
  ```
  **Cangle is measured from east** (counter-clockwise); ROMS subtracts the grid angle itself.
  The tide file must hold east/north-referenced ellipses — do not pre-rotate to grid.
  Uwrk/Vwrk are then averaged to u/v points.

## V9 — Tide file variable names; which are needed for which option

`External/varinfo.yaml` l.2552–2627; read in `Nonlinear/get_idata.F` l.120–239.

| Variable | Units in file | ROMS scale | Needed by |
|---|---|---|---|
| tide_period | hours | 3600 → s | SSH_TIDES or UV_TIDES |
| tide_Eamp | meter | 1 | SSH_TIDES |
| tide_Ephase | degrees | π/180 | SSH_TIDES |
| tide_Cmax | meter second-1 | 1 | UV_TIDES |
| tide_Cmin | meter second-1 | 1 | UV_TIDES |
| tide_Cangle | degrees | π/180 | UV_TIDES |
| tide_Cphase | degrees | π/180 | UV_TIDES |

All use `tide_period` as their "time" dimension.

## V10 — Nudging coefficient names, dims, units

`External/varinfo.yaml` + `Utility/get_nudgcoef.F` l.173–216, 236–262, 381–410;
tracer names built in `Utility/inp_par.F` l.894–900 as `TRIM(Vname(1,idTvar(i)))//'_NudgeCoef'`.

| Variable | Type | Read when |
|---|---|---|
| M2_NudgeCoef | r2dvar (eta_rho, xi_rho) | `LnudgeM2CLM` |
| M3_NudgeCoef | r3dvar (s_rho, eta_rho, xi_rho) | `LnudgeM3CLM` |
| temp_NudgeCoef, salt_NudgeCoef, … | r3dvar | `LnudgeTCLM(itrc)`; specific name preferred |
| tracer_NudgeCoef | r3dvar | fallback when the specific name is missing |

Units: if the `units` attribute starts with `day` → ×1/86400; starts with `second` → ×1;
otherwise (or missing) **default day⁻¹**. Missing variable with flag on → abort.

## V11 — Where sponge factors come from

`Utility/get_grid.F` l.287–300 and `Nonlinear/initial.F` l.285–300: unless `ANA_SPONGE` is
defined, `visc_factor` and `diff_factor` are read from the **grid file**; with `ANA_SPONGE`
they come from `Functionals/ana_sponge.h`. Switched on per grid by `LuvSponge` and
`LtracerSponge` in ocean.in (`User/External/roms.in` l.326–327).

## V12 — Full LBC keyword list and which require data

Decoder `Utility/inp_decode.F` l.1771–1820 (source of truth; the glossary in `roms.in`
l.1515–1530 lists `Nud` but the decoder rejects it, and omits `Mix`):

| Keyword | Meaning | Needs boundary data (`acquire`) |
|---|---|---|
| Cha / Che | Chapman implicit / explicit (ζ only) | no (ζ data acquired if ubar/vbar use Fla or Shc) |
| Cla | Clamped | **yes** |
| Clo | Closed | no |
| Fla | Flather (2D momentum) | **yes**, and ζ too |
| Gra | Gradient | no |
| Mix | Mixed | **yes** |
| Nes | Nested (refinement) | no |
| Per | Periodic | no |
| Rad | Radiation | no |
| RadNud | Radiation + nudging | **yes** |
| Red | Reduced physics (2D momentum) | ζ only if `FSOBC_REDUCED` |
| Shc | Shchepetkin (2D momentum) | **yes**, and ζ too |

Order on each line: **West South East North** (`roms.in` diagram l.1520–1526: 1=W, 2=S, 3=E, 4=N).
Tracer LBC: one line per tracer joined by `\`; tracers not listed inherit the last line
(`inp_decode.F` l.1824–1842). **SPEC FIX:** keyword list in §8.5 updated.

## V13 — Boundary volume-conservation keyword

`User/External/roms.in` l.219–222 and glossary l.1564–1576:
`VolCons(west) == F`, `VolCons(east)`, `VolCons(south)`, `VolCons(north)` — one line per side
(note: order W, E, S, N here, by name). Glossary: "should not be activated if tidal forcing is
enabled".

## V14 — Courant / rx0 / rx1 formulas and stdout format

**SPEC FIX:** rx0/rx1 are in `Utility/stiffness.F`, not `metrics.F`.

- Barotropic Courant, `Utility/metrics.F` l.857–885, at `rmask > 0`, all rho points:
  `Cg = dtfast·SQRT(g·|h|·(pm² + pn²))`, `dtfast = dt/ndtfast` (`inp_par.F` l.640),
  `g = 9.81` (`mod_scalars.F` l.470). Coriolis: `dt·|f|`.
- rx0 / rx1, `Utility/stiffness.F` l.155–188: over **u-faces with umask > 0** and **v-faces with
  vmask > 0**, **interior rho points only** (i = IstrU..Iend, j = Jstr..Jend):
  `rx0 = |z_w(0,i) − z_w(0,i−1)| / |z_w(0,i) + z_w(0,i−1)|` (= |h₁−h₂|/(h₁+h₂)),
  `rx1 = |z_w(k,i)−z_w(k,i−1)+z_w(k−1,i)−z_w(k−1,i−1)| / |z_w(k,i)+z_w(k,i−1)−z_w(k−1,i)−z_w(k−1,i−1)|`.
  `stiffness` is called after `set_depth` with the **initial ζ** (`Nonlinear/initial.F`
  l.565 then l.832), so ROMS's rx1 includes IC ζ; spec said ζ = 0. `diagnostics.rx1` takes
  an optional ζ so both can be computed.
- stdout formats:
  ```
  ' Minimum barotropic Courant Number = ', 1pe15.8
  ' Maximum barotropic Courant Number = ', 1pe15.8
  ' Maximum Coriolis   Courant Number = ', 1pe15.8
  ' Maximum grid stiffness ratios:  rx0 = ',1pe14.6,' (Beckmann and Haidvogel)'
  <t34>'rx1 = ',1pe14.6,' (Haney)'
  ```

## V15 — Boundary time variable names

`External/varinfo.yaml`: `zeta_<side>` → `zeta_time`; `ubar_/vbar_<side>` → `v2d_time`;
`u_/v_<side>` → `v3d_time`; `temp_<side>` → `temp_time`; `salt_<side>` → `salt_time`.
Matches spec. These are conventions; ROMS follows each variable's `time` attribute.

## V16 — Vstretching 1–5 formulas

`Utility/set_scoord.F` l.164–533. `hc`: Vtransform 1 → `MIN(hmin, Tcline)` (with `WET_DRY`:
`MIN(MAX(hmin,0), Tcline)`); Vtransform 2 → `Tcline`.
`s_w(k) = (k−N)/N`, `s_r(k) = (k−N−0.5)/N` for options 1–4. `Cs_w(0) = −1`, `Cs_w(N) = 0`.
1. Song & Haidvogel: `(1−θb)·sinh(θs s)/sinh(θs) + θb·(0.5·tanh(θs(s+0.5))/tanh(0.5θs) − 0.5)`; `Cs = s` if θs = 0.
2. Shchepetkin 2005: `Csur = (1−cosh(θs s))/(cosh θs − 1)`; if θb > 0 blend with
   `Cbot = sinh(θb(s+1))/sinh θb − 1` using `Cweight = (s+1)^α(1 + (α/β)(1 − (s+1)^β))`, α=β=1; `Cs = s` if θs ≤ 0.
3. Geyer: `Hscale=3`; `Cbot = log(cosh(H(s+1)^θb))/log(cosh H) − 1`;
   `Csur = −log(cosh(H|s|^θs))/log(cosh H)`; `Cw = 0.5(1 − tanh(H(s+0.5)))`; `Cs = Cw·Cbot + (1−Cw)·Csur`.
4. Shchepetkin 2010: `Csur` as in 2 (or `−s²` if θs ≤ 0); if θb > 0 `Cs = (exp(θb Csur) − 1)/(1 − exp(−θb))`.
5. Souza: same C(s) as 4, **but s itself is non-uniform**:
   `s_w(k) = −(k² − 2kN + k + N² − N)/(N² − N) − 0.01(k² − kN)/(1 − N)`, and `s_r` uses k−0.5.
   **SPEC FIX:** `s_coord` depends on Vstretching.

Depths, `Nonlinear/set_depth.F` l.150–222: matches spec formulas for Vtransform 1 and 2.

## V17 — TIME_REF special values

`User/External/roms.in` l.2348–2374:
- `−2`: model time/DSTART in (modified) Julian days, units "since 1968-05-23 00:00:00 GMT".
- `−1`: 360-day calendar, "since 0000-12-30 00:00:00".
- `0`: 365.25-day year calendar, "since 0001-01-01 00:00:00".
- `> 0`: `yyyymmdd.f`, e.g. 20020115.5 → "since 2002-01-15 12:00:00".
`io.time_ref` supports `> 0` and raises `NotImplementedError` for the others (as specified).
