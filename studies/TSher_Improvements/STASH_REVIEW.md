# Stash review

Reviewed on 2026-11-11 while preparing `tomsheridan_improvements`.

## Improvements stash

`40af9402148dfdb6f19cf9ef7ef443f60a084d15` (`stash@{0}` at review time), based on
`fd7aaae6680039d5e757cb32f0641bbd51ec2eed`:

| Original path/work | Decision |
| --- | --- |
| `aerosandbox/aerodynamics/aero_3D/lifting_line.py` | Restore the two broadcast-safe moment-reference subtractions. They allow symbolic N-by-3 coordinates without changing the aerodynamic formulation. |
| `aerosandbox/aerodynamics/aero_3D/vortex_lattice_method.py` | Restore the analogous two broadcast-safe subtractions. |
| `studies/TSher_Improvements/CTOL_scoring.ipynb` | Restore score/geometry/weight sensitivity work, including its saved outputs; relocate to `ctol/CTOL_scoring.ipynb` and update input paths. Its production aerodynamic solver selection is preserved. |
| `studies/TSher_Improvements/2_aerodynamics.ipynb` | Restore the aerodynamic sizing/teaching notebook under `aerodynamics/`. |
| `studies/TSher_Improvements/headless_chicken/prandtl_comparison.ipynb` | Restore under `ctol/headless_chicken/`. |

The CTOL diff adds zero-constrained dummy variables, dual-based score sensitivity tables,
and direct scoring sensitivities. This recovery preserves the author's study; the full
CTOL optimization has not been rerun as part of the normal-flow implementation checks.
Restored notebook cell sources are retained except for setup and relocated file paths.

## MX-11 stash

`a1f106a76ab0981a8e8b33b007166846fca3847d` (`stash@{1}` at review time), based on
`4b3a4a5ec260c0af06c1e976e1951f0fe2ea1db3`:

Contains root-level `CTOL_scoring.ipynb` and `TripLocation.ipynb` work, plus removal of two
Windows `desktop.ini` files in other study directories. This belongs to the separate
`MX-11` branch and was not applied to `tomsheridan_improvements`.

The root CTOL source switches from stability derivatives to `run()`, removes neutral-point
outputs from the helper and its callers, and adds a NeuralFoil installation diagnostic.
The TripLocation source changes transition inputs from n_crit=9/free transition to
n_crit=5 and forced transition at 0.1 chord. Those are separate performance/input choices,
not required for the new analysis class or the current CTOL sensitivity recovery.

Both stashes are retained as backups. No stash was popped, dropped, or rewritten.
