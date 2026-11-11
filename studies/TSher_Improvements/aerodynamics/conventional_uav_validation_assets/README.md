# Conventional UAV comparison assets

Used by [01 - Comparison and Validation - Conventional UAV.ipynb](../01%20-%20Comparison%20and%20Validation%20-%20Conventional%20UAV.ipynb).
Run all cells from the notebook directory or the AeroSandbox repository root. The bundled
sources make execution independent of network availability. Predictions are recomputed on
each run. The branch-specific VortexLatticeViscous comparison is conditional on that class
being available; the other four methods are required.

## Reference data and provenance

`FASER_Aero.mat` is the University of Minnesota archived FASER aerodynamic database.
`FASER_config.m` supplies normalization dimensions and identifies its NASA origins.
`FASER_Aero_Lib.mdl` records how the simulation interpolates the database and adds
control, power, and other increments. These three files were retrieved unchanged from the
[UMN course simulation archive](https://dept.aem.umn.edu/~mettler/Courses/AEM%205333%20%28spring%202013%29/Matlab/Ultrastick%20Simulation/trunk/Simulation/Libraries/).

The notebook extracts `C6_bas.data[:, beta == 0, :]`. The MAT file identifies its output
order as **CD, CY, CL, Cl, Cm, Cn**. It contains processed, wind-tunnel-derived simulation
table values, not raw balance readings. No experimental standard deviations or Reynolds
dimension are supplied. No digitization, smoothing, curve fitting, or aerodynamic offset is
applied. `results/faser_reference_beta0.csv` is a lossless decimal extraction of that slice.

Primary background:

- [Owens, Cox & Morelli, 2006: Development of a Low-Cost Sub-Scale Aircraft for Flight Research: The FASER Project](https://ntrs.nasa.gov/citations/20060022541).
- [Morelli & DeLoach, 2003: Wind Tunnel Database Development using Modern Experiment Design and Multivariate Orthogonal Functions](https://ntrs.nasa.gov/citations/20030014297).

The earlier report's 2 psf dynamic pressure supplies an explicit **assumption** for this
comparison, not confirmed metadata for this archive. That report describes an earlier
aircraft configuration. Consequently the notebook is not a matched-condition validation
of either paper's exact test article.

## Geometry and reference conventions

`UltraStick120.avl` comes from [UASLab/OpenFlightSim](https://github.com/UASLab/OpenFlightSim).
Its source revision is recorded in `geometry_commit.txt` and `sources.json`. Its MIT license
is retained as `OpenFlightSim_LICENSE.md`. The main-wing and horizontal-tail station
coordinates are transcribed into the notebook without aerodynamic fitting. The source
fin's span coordinate is mapped from y to z to make a vertical tail; the unspecified root
airfoil is assumed to match its NACA 0005 tip. Neutral controls are represented by the
undeformed airfoil sections.

This related-aircraft geometry is a **proxy**, not a verified reconstruction of the NASA
wind-tunnel model. Its span and area differ from the FASER coefficient references, and
its dihedral is zero. The notebook preserves the source geometry and uses the FASER
configuration's S, b, and c for coefficient normalization. It uses the AVL geometry's
moment reference; the archive's origin cannot be reconciled conclusively. Therefore Cm
is plotted as a diagnostic and excluded from discrepancy metrics.

All solvers model lifting surfaces only. Fuselage, landing gear, propeller, installation
effects, and uncertainty in the tested article are not represented. Drag discrepancies
include those scope differences. The notebook does not fit a drag correction to conceal them.

Original authorship/copyright notices on the archived UMN files are retained. The MIT
license for the geometry repository does not establish a license for the independent UMN
archive. Do not describe these third-party assets as AeroSandbox-authored or assume that
the AeroSandbox license applies to them.

## Generated results

The notebook saves the extracted reference, all solver predictions and failure messages,
pre-stall discrepancy metrics, sampled stall-window peaks, mesh sensitivity, figures, and
run metadata to `results/`. Pre-stall discrepancies use matching nodes from -3 to 12 degrees.
Peak comparisons use -3 to 18 degrees and explicitly distinguish sampled interior peaks
from monotonic or incomplete sweeps. The full reference through 42 degrees is retained.

`run_metadata.json` records the checkout, tracked modifications, package versions, settings,
and source hashes. `sources.json` verifies the bundled source bytes before execution.
This notebook and its accompanying explanations were created with Codex; human review
is outstanding.

## Observed limitations in the saved run

AeroBuildup, VLM, and LiftingLine returned finite answers at all 12 main-sweep nodes.
NonlinearLL had 10 accepted nodes: its 18-degree residual exceeded the study's 1e-5
threshold, and its 19.5-degree solve failed. Both refined-mesh NonlinearLL cases exceeded
the configured CPU budget, so this run does not establish nonlinear mesh convergence.

The branch's VortexLatticeViscous produced nonfinite full-aircraft answers. VLM's current
sectional CL uses projected-y strip width, which is zero for a vertical fin; the viscous
closure propagates the resulting NaN. A separate two-angle diagnostic removes the fin
and obtains finite VortexLatticeViscous answers. Those diagnostic results are saved to
`results/vertical_fin_diagnostic.csv` and are not substituted into the main comparison.
No AeroSandbox library files were changed to obtain these results.
