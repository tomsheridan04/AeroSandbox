# Flying-wing LiftingLine correction experiment

Open **02 - LiftingLine Corrections - Flying Wing.ipynb** and run all cells from its folder
or the repository root. It recomputes the sweeps, compares them to the original flying-wing
digitization, and cross-checks a separate isolated Clark Y wing through stall. Source data
are bundled; no network access is needed to run the study.

The accepted implementation is `asb.NormalFlowLiftingLine`.
`lifting_line_corrections.py` retains the study-only `CorrectedLiftingLine` ablations,
which now reuse the public class's mesher and force integration.
`lifting_line_correction_study.py` contains case construction and common comparison metrics.

```python
import aerosandbox as asb

analysis = asb.NormalFlowLiftingLine(
    airplane=airplane,
    op_point=op_point,
    spanwise_resolution=20,
    n_crit=5.0,
    xtr_upper=0.3,
    xtr_lower=0.3,
    circulation_corrections=2,
)
aero = analysis.run_with_stability_derivatives(
    alpha=True, beta=False, p=False, q=False, r=False,
)
CL, CD, Cm, x_np = [aero[k] for k in ["CL", "CD", "Cm", "x_np"]]
gamma = analysis.vortex_strengths
```

The public class is available through the ordinary AeroSandbox import from any script.
Its defaults are normal flow, a NeuralFoil attached-flow slope, two corrections, and 0.8
relaxation. The study-only ablation subclass keeps its historical legacy-flow default
so all previous variants can still be compared. No model is a universal accuracy
recommendation.

The method performs a fixed number of differentiable linear solves, preserves force and
spanwise output keys, and supports symbolic geometry/operating-point variables with
CasADi. It adds sectional consistency residuals, slopes, confidence, and a lifting-surface
drag breakdown. `CD_induced` and `CD_profile` describe the lifting surfaces; total `CD`
also includes the inherited fuselage contribution when fuselages are present.

The tests cover exact zero-correction reproduction of the original solver, consistency
improvement, symbolic/numeric agreement, finite-difference checks of design derivatives,
the CTOL stability-derivative interface, multiple surfaces including a fin, fuselage
aggregation, velocity queries, and geometry translation with angular rates.

The flying-wing CL(alpha) digitization does not include a resolved experimental maximum.
Its high-lift shortfall metric is a bound against the observed polar, not CLmax error.
The independent data folder documents the Clark Y yawed polar and separate zero-yaw
maximum. Drag RMSE on common CL nodes and polar coverage are both needed to assess a
candidate fairly. No coefficient is fitted to force an experimental maximum or drag floor.

All results and runtime/source metadata are saved in `lifting_line_correction_results`.
Original experiment metadata retains its historical paths and hashes after relocation;
the `promotion_checks` subfolder records checks of the public class and current sources.
This is research code generated with Codex; human math/code/data review is outstanding.
The study does not establish a generally superior replacement for production LiftingLine.

## Saved-run findings

The two-correction normal-flow variant reduced flying-wing lift RMSE by approximately
43% and drag RMSE on the common CL nodes by 36%, at about 2.3 times the original runtime.
Four corrections gave very little additional flying-wing accuracy at about 3 times the
runtime. Both normal-flow variants have substantial remaining sectional residuals near
stall, and the mesh checks show changes of several percent in lift.

On the independent yawed Clark Y case, two corrections gave a sampled peak of 1.197
against the measured sampled peak of 1.194. On the separate zero-yaw case, the predicted
maximum increased from 1.083 to 1.362 against the reported 1.240: a smaller absolute error,
but an overprediction remains. The yawed case's drag error over common CL nodes increased
by approximately 21%, although pre-stall polar coverage improved from 5/11 to 11/11 nodes.
These tradeoffs are retained in the notebook and do not support a universal accuracy claim.

The original eight integration/differentiability tests passed. The notebook also executed the constrained
Opti example with symbolic alpha, chord, and span, and retained the CTOL-style stability
derivative interface. The promotion additionally restores broadcast-safe coordinate
subtractions from the improvements stash; it retains the original solver's aerodynamic
formulation. See the study index and stash review for the complete verification scope.
