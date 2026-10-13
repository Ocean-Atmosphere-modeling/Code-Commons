# code-commons — instructions for Claude Code

## What this folder is

Teaching + QA notebooks for the input files of a realistic ROMS configuration
(Rutgers ROMS 4.x). The audience is any ROMS user, not one project. Five notebooks:

1. `01_initial_conditions` — inspect, visualize, stability numbers
2. `02_atmospheric_forcing` — inspect, visualize, unit/coverage sanity
3. `03_tides` — inspect, visualize, tidal diagnostics, demo: build a ROMS tide file
4. `04_nudging` — build and inspect a nudging-coefficient file
5. `05_boundary_conditions` — inspect, visualize, volume budget, IC/BC consistency

plus a small shared, tested package `romscommons/`.

**The full build spec is `BUILD_SPEC.md`. Read it before any work.** Build in the phase
order of BUILD_SPEC §10 and stop at the checkpoints listed there.

## Ground rules

1. **ROMS conventions come from ROMS source, not memory.** Items tagged `[VERIFY]` in
   BUILD_SPEC.md must be confirmed in ROMS source: use `$ROMS_SRC` if set, otherwise
   shallow-clone `https://github.com/myroms/roms` into `../roms-reference` (outside this
   folder; never vendor it). Find files by name — paths differ between versions. Record
   each answer (file + line or short snippet) in `VERIFY_LOG.md`. If source and spec
   disagree, source wins: fix the spec and tell the user.
2. **Inputs are read-only.** Notebooks never modify input NetCDF files. Only nb03 (tide
   demo) and nb04 (nudging builder) write files, and only under `outputs/<config name>/`.
3. **One config switch.** All paths and run parameters come from a YAML file in
   `configs/`. The only cell a user edits is `CONFIG = "configs/synthetic.yaml"`. No paths
   hard-coded anywhere else. Nothing project-specific (e.g. Moana) in code.
4. **Notebooks teach.** Every code cell is preceded by a markdown cell (≤ 5 sentences):
   what ROMS does with this, why it matters, what to look for. Code cells stay short
   (~25 lines); reusable logic lives in `romscommons` and is unit-tested.
5. **Checks are explicit.** Every check returns PASS / WARN / FAIL / INFO with the value,
   the threshold, and where the threshold comes from. FAIL is reserved for things that
   will break or invalidate a run (missing variable, wrong dims, non-finite wet values,
   non-positive depth, time not covering the run, nudging coef·DT ≥ 1). Judgment calls
   (rx1, Courant margins, value ranges) are WARN. Thresholds live only in
   `romscommons/checks.py::THRESHOLDS`.
6. **Large files.** Open lazily (xarray + dask chunks), compute on selected records or
   subsets, never `.load()` a full 4D forcing or boundary time series.
7. **Plots.** Units in every label; land masked gray; cmocean colormaps (`thermal`,
   `haline`, `deep`, `balance`, `speed`, `phase` for cyclic phases); cartopy optional;
   longitudes dateline-safe.
8. **Dependencies.** Only what `environment.yml` lists. Ask before adding one. Do not use
   xroms — the few ROMS formulas are implemented explicitly so readers can see them.
9. **Notebook files.** Jupytext-paired (`.ipynb` + `.py` percent format). Edit the `.py`,
   then `jupytext --sync`. Commit with outputs cleared.

## Commands

```bash
conda env create -f environment.yml && conda activate code-commons
pip install -e .
python examples/make_synthetic_case.py --out data/synthetic
python examples/make_synthetic_case.py --out data/synthetic_defects --defects
pytest                                   # unit + defect tests
pytest --nbmake notebooks/               # execute notebooks on the synthetic case
jupytext --sync notebooks/*.py
```

## Definition of done (every phase)

- `pytest` green, including the defect tests for that phase.
- The phase's notebook runs top-to-bottom on `configs/synthetic.yaml` with no errors,
  no FAIL in its report, in under ~2 minutes.
- Every `[VERIFY]` item the phase touches is resolved in `VERIFY_LOG.md`.
- A short summary to the user: what was built, any deviation from the spec, open questions.
