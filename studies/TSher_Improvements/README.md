# Sheridan improvement studies

The studies are grouped by subject. Start notebooks from their own folder or from the
repository root; their setup cells locate the checkout and adjacent helper modules.
External programs such as AVL and the XDSM LaTeX compiler remain beside the studies
that use them. Temporary kernel and solver directories are not source artifacts.

| Folder | Contents |
| --- | --- |
| [aerodynamics](aerodynamics/) | Flying-wing and conventional-UAV comparisons, lifting-line corrections, transition studies, native spanwise distributions, VLM/AVL and viscous-VLM validation, and their source data/results. |
| [bsplines](bsplines/) | Differentiable B-spline demonstration and span-loading/twist experiments. |
| [ctol](ctol/) | CTOL scoring, the B-spline variant, propeller input data, XDSM generation, and headless-chicken/Prandtl work. |

## Normal-flow lifting line

The accepted two-correction implementation is available through the normal package API:

```python
import aerosandbox as asb

analysis = asb.NormalFlowLiftingLine(
    airplane=airplane,
    op_point=op_point,
    spanwise_resolution=20,
    n_crit=5.0,
    xtr_upper=0.3,
    xtr_lower=0.3,
)
aero = analysis.run_with_stability_derivatives(
    alpha=True, beta=False, p=False, q=False, r=False,
)
```

Normal flow, a NeuralFoil attached-flow slope, two circulation corrections, and 0.8
relaxation are the new class's defaults. `asb.LiftingLine` retains its original aerodynamic
formulation. The study-only `CorrectedLiftingLine` subclass keeps the alternative slope
and legacy-flow cases needed to reproduce the comparison notebook.

Start with [the correction comparison](aerodynamics/02%20-%20LiftingLine%20Corrections%20-%20Flying%20Wing.ipynb)
and its [method/data notes](aerodynamics/README_lifting_line_corrections.md).
Stored experiment outputs are retained, including unsuccessful solver cases and
cross-case regressions. Their original metadata records historical source paths and
hashes. Re-running a notebook refreshes its numerical results and provenance for this
layout. Promotion checks compare the public implementation against the saved results.

## Restored work and review status

[STASH_REVIEW.md](STASH_REVIEW.md) records the two reviewed stashes and which work was
restored. The recovered CTOL sensitivity notebook and older teaching/Prandtl notebooks
are development studies; relocating them does not certify their complete optimizations.

The normal-flow implementation and recent comparison studies were developed with Codex.
Automated checks cover numeric/symbolic agreement, derivatives, output compatibility,
and sample optimizations. Human math/code/data review is still required before proposing
an upstream change. Third-party source notices remain with the bundled data.
