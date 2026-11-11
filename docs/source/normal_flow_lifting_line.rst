Normal-flow lifting line
========================

``asb.NormalFlowLiftingLine`` is an explicit alternative to ``asb.LiftingLine``.
It uses normal-plane sectional flow, a NeuralFoil attached-flow lift slope, and
two relaxed circulation corrections by default. The original analysis remains
available for direct comparison.

Usage in a design script
------------------------

.. code-block:: python

   analysis = asb.NormalFlowLiftingLine(
       airplane=airplane,
       op_point=asb.OperatingPoint(velocity=25, alpha=5),
       spanwise_resolution=20,
       n_crit=5.0,
       xtr_upper=0.3,
       xtr_lower=0.3,
   )
   aero = analysis.run_with_stability_derivatives(
       alpha=True, beta=False, p=False, q=False, r=False,
   )
   CL, CD, Cm, x_np = [aero[key] for key in ["CL", "CD", "Cm", "x_np"]]
   circulation = analysis.vortex_strengths

The constructor retains the ordinary geometry, operating-point, transition,
meshing, moment-reference, and wake options. Numeric operating points and CasADi
design variables are supported. Iterate over numeric operating points to form a
polar. The inherited stability-derivative method evaluates one additional analysis
for every requested derivative; disable derivatives a script does not use.

Algorithm and cost
------------------

For each section, the flow is resolved in the plane perpendicular to its bound
vortex. Normal chord equals strip area divided by bound-leg length. Local angle,
Reynolds number, Mach number, profile drag, and pitching moment use this convention.
The supplied airfoil shape itself is not transformed for sweep.

The predictor solves a linear circulation system using NeuralFoil lift at the
initial sectional angle. Its lift slope is sampled at -2 and +2 degrees and
expressed per radian. Each correction recomputes local flow and reduces the
difference between NeuralFoil sectional lift and the lift implied by circulation.
Variable lift slopes multiply influence-matrix rows. A smooth positive bound is
applied only to the approximate Jacobian slope, not to the airfoil lift coefficient.

``circulation_corrections=2`` gives three linear solves in total. Each increment
uses ``relaxation=0.8``. Both settings are fixed numerical choices, not design
variables. There is no inner nonlinear optimizer or convergence-based loop.
Dense-solve complexity remains cubic in the number of strips, with a fixed
multiplier for the correction count. The documented study timings are numeric
per-operating-point measurements, not full mission-optimization benchmarks.

Diagnostics and limits
----------------------

The usual force, coefficient, component, stability-derivative, and per-wing
spanwise keys are retained. Additional outputs include ``sectional_alpha``,
``sectional_reynolds``, ``sectional_mach``, ``sectional_cl_neuralfoil``,
``sectional_cl_circulation``, ``sectional_residual``, ``sectional_slope``, and
``sectional_confidence``. ``CD_induced`` and ``CD_profile`` split lifting-surface
drag; the inherited total ``CD`` also includes fuselage contributions.

Two corrections do not guarantee a converged nonlinear lifting-line solution.
Inspect sectional residuals and mesh sensitivity, especially near stall.
Projected-y spanwise lift coefficients remain undefined for vertical surfaces;
their integrated forces and normal-plane consistency are evaluated separately.
The method does not establish universally better drag or CLmax predictions.
The flying-wing digitization does not resolve a measured maximum, and independent
Clark Y comparisons expose remaining peak errors and drag regressions.

The comparison notebooks, source provenance, and study index are under
``studies/TSher_Improvements/aerodynamics``. Their original outputs are retained
as historical results through the study reorganization. Human review is still
required before an upstream contribution.
