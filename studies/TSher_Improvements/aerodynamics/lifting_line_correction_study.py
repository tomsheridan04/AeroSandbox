"""Reproducible cases and metrics for the inexpensive LiftingLine study."""

from pathlib import Path
import time

import aerosandbox as asb
import numpy as onp
import pandas as pd

if __package__:
    from .lifting_line_corrections import CorrectedLiftingLine
else:
    from lifting_line_corrections import CorrectedLiftingLine

STUDY = Path(__file__).resolve().parent
ROOT = next(p for p in STUDY.parents if (p / "aerosandbox/__init__.py").is_file())
ASSETS = (
    ROOT / "tutorial/06 - Aerodynamics/01 - AeroSandbox 3D Aerodynamics Tools/assets"
)
OUTPUT = STUDY / "lifting_line_correction_results"

# Declare the alternatives before evaluating the reference data. None uses
# experimental coefficients, a fitted drag offset, or a prescribed CL cap.
VARIANTS = {
    "Original LiftingLine": None,
    "0.8 x 2pi, no correction": dict(
        slope_model="two_pi", slope_scale=0.8, circulation_corrections=0
    ),
    "NF attached slope, no correction": dict(
        slope_model="attached", circulation_corrections=0
    ),
    "NF local slope, no correction": dict(
        slope_model="local", circulation_corrections=0
    ),
    "2pi + 2 corrections": dict(slope_model="two_pi", circulation_corrections=2),
    "NF attached + 2 corrections": dict(
        slope_model="attached", circulation_corrections=2
    ),
    "NF local + 2 corrections": dict(slope_model="local", circulation_corrections=2),
    "Normal flow, NF attached + 2": dict(
        kinematics="normal", slope_model="attached", circulation_corrections=2
    ),
    "Normal flow, NF attached + 4": dict(
        kinematics="normal", slope_model="attached", circulation_corrections=4
    ),
}


def flying_wing():
    airfoil = asb.Airfoil(name="n64_1_A612", coordinates=ASSETS / "n64_1_A612.dat")
    return asb.Airplane(
        name="NACA_RM_A50K27 Wing",
        xyz_ref=[0.585, 0, 0],
        c_ref=0.32,
        wings=[
            asb.Wing(
                name="Main Wing",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0], chord=0.412, twist=0, airfoil=airfoil
                    ),
                    asb.WingXSec(
                        xyz_le=[1.1362, 1.549, 0], chord=0.206, twist=0, airfoil=airfoil
                    ),
                ],
            )
        ],
    )


def flying_reference():
    file = ASSETS / "NACA-RM-A50K27.xlsx"
    return {
        key: pd.read_excel(file, sheet_name=sheet)
        for key, sheet in [("lift", "CL"), ("drag", "CD"), ("moment", "CM")]
    }


def solver(
    variant, airplane, op_point, spanwise_resolution=12, model_size="medium", **kwargs
):
    options = VARIANTS[variant]
    if options is None:
        cls, options = asb.LiftingLine, {}
    elif (
        options.get("kinematics") == "normal"
        and options.get("slope_model") == "attached"
    ):
        cls = asb.NormalFlowLiftingLine
        options = {
            key: value
            for key, value in options.items()
            if key not in {"kinematics", "slope_model"}
        }
    else:
        cls = CorrectedLiftingLine
    return cls(
        airplane=airplane,
        op_point=op_point,
        model_size=model_size,
        spanwise_resolution=spanwise_resolution,
        **options,
        **kwargs,
    )


def sweep(
    airplane,
    alphas,
    velocity,
    variants=None,
    spanwise_resolution=12,
    model_size="medium",
    beta=0,
):
    rows = []
    for variant in variants or VARIANTS:
        for alpha in alphas:
            started = time.perf_counter()
            analysis = solver(
                variant,
                airplane,
                asb.OperatingPoint(velocity=velocity, alpha=float(alpha), beta=beta),
                spanwise_resolution=spanwise_resolution,
                model_size=model_size,
            )
            aero = analysis.run()
            coefficients = {
                key: float(onp.asarray(aero[key]).squeeze())
                for key in ["CL", "CD", "Cm"]
            }
            if not onp.isfinite(list(coefficients.values())).all():
                raise ValueError(f"Nonfinite answer: {variant}, alpha={alpha}")
            residual = onp.asarray(
                aero.get("sectional_residual", [onp.nan]), dtype=float
            )
            rows.append(
                dict(
                    variant=variant,
                    alpha=float(alpha),
                    **coefficients,
                    CD_induced=float(aero.get("CD_induced", onp.nan)),
                    CD_profile=float(aero.get("CD_profile", onp.nan)),
                    max_sectional_residual=float(onp.max(onp.abs(residual))),
                    seconds=time.perf_counter() - started,
                )
            )
        print(f"{variant}: {len(alphas)} points complete", flush=True)
    return pd.DataFrame(rows)


def rising_branch(frame):
    """Only the increasing-alpha branch up to its first sampled global peak."""
    frame = frame.sort_values("alpha").reset_index(drop=True)
    peak = int(onp.argmax(frame.CL.to_numpy()))
    branch = frame.iloc[: peak + 1]
    # Return every available pre-peak point; fail an invertibility check rather
    # than silently sorting a multi-valued polar into a different lift branch.
    return branch


def metrics(predictions, reference):
    rows = []
    branches = {
        name: rising_branch(group)
        for name, group in predictions.groupby("variant", sort=False)
    }
    invertible_branches = [b for b in branches.values() if onp.all(onp.diff(b.CL) > 0)]
    common_low = max(b.CL.min() for b in invertible_branches)
    common_high = min(b.CL.max() for b in invertible_branches)
    for name, group in predictions.groupby("variant", sort=False):
        group = group.sort_values("alpha")
        ref_lift = reference["lift"]
        lift_mask = ref_lift.alpha.between(group.alpha.min(), group.alpha.max())
        predicted_lift = onp.interp(
            ref_lift.loc[lift_mask, "alpha"], group.alpha, group.CL
        )
        lift_error = predicted_lift - ref_lift.loc[lift_mask, "CL"].to_numpy()
        branch = rising_branch(group)
        invertible = bool(onp.all(onp.diff(branch.CL) > 0))
        ref_drag = reference["drag"]
        drag_mask = (
            ref_drag.CL.between(branch.CL.min(), branch.CL.max())
            if invertible
            else onp.zeros(len(ref_drag), dtype=bool)
        )
        drag_error = (
            onp.interp(ref_drag.loc[drag_mask, "CL"], branch.CL, branch.CD)
            - ref_drag.loc[drag_mask, "CD"].to_numpy()
        )
        common_mask = (
            ref_drag.CL.between(common_low, common_high)
            if invertible
            else onp.zeros(len(ref_drag), dtype=bool)
        )
        common_error = (
            onp.interp(ref_drag.loc[common_mask, "CL"], branch.CL, branch.CD)
            - ref_drag.loc[common_mask, "CD"].to_numpy()
        )
        ref_moment = reference.get("moment")
        moment_error = []
        if ref_moment is not None and invertible:
            moment_mask = ref_moment.CL.between(common_low, common_high)
            moment_error = (
                onp.interp(ref_moment.loc[moment_mask, "CL"], branch.CL, branch.Cm)
                - ref_moment.loc[moment_mask, "Cm"].to_numpy()
            )
        peak = group.loc[group.CL.idxmax()]
        interior = group.alpha.min() < peak.alpha < group.alpha.max()
        highest_observed = max(reference["lift"].CL.max(), reference["drag"].CL.max())
        rows.append(
            dict(
                variant=name,
                CL_alpha_RMSE=float(onp.sqrt(onp.mean(lift_error**2))),
                CD_at_CL_RMSE=(
                    float(onp.sqrt(onp.mean(drag_error**2)))
                    if len(drag_error)
                    else onp.nan
                ),
                CD_common_CL_RMSE=(
                    float(onp.sqrt(onp.mean(common_error**2)))
                    if len(common_error)
                    else onp.nan
                ),
                CD_common_nodes=int(onp.sum(common_mask)),
                Cm_common_CL_RMSE=(
                    float(onp.sqrt(onp.mean(onp.asarray(moment_error) ** 2)))
                    if len(moment_error)
                    else onp.nan
                ),
                CD_polar_coverage=f"{int(onp.sum(drag_mask))}/{len(ref_drag)}",
                CD_polar_nodes=int(onp.sum(drag_mask)),
                predicted_peak_CL=float(peak.CL),
                predicted_peak_alpha=float(peak.alpha),
                resolved_interior_peak=interior,
                observed_high_CL_shortfall=max(0.0, highest_observed - float(peak.CL)),
                seconds_per_point=float(group.seconds.median()),
                worst_sectional_residual=float(group.max_sectional_residual.max()),
            )
        )
    return pd.DataFrame(rows)


def clark_y_wing():
    ordinates = pd.read_csv(
        STUDY / "lifting_line_correction_data/clark_y_published_ordinates.csv"
    )
    upper = ordinates[["x_over_c", "y_upper_over_c"]].to_numpy()[::-1]
    lower = ordinates[["x_over_c", "y_lower_over_c"]].to_numpy()[1:]
    airfoil = asb.Airfoil(
        name="Clark Y, NACA TR 431 Table I", coordinates=onp.concatenate([upper, lower])
    )
    chord = 14.14 * 0.0254
    span = 6 * chord
    return asb.Airplane(
        name="Isolated rectangular Clark Y wing, AR 6",
        xyz_ref=[chord / 4, 0, 0],
        s_ref=span * chord,
        c_ref=chord,
        b_ref=span,
        wings=[
            asb.Wing(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, 0], chord=chord, airfoil=airfoil),
                    asb.WingXSec(xyz_le=[0, span / 2, 0], chord=chord, airfoil=airfoil),
                ],
            )
        ],
    )


def clark_reference():
    data = pd.read_csv(
        STUDY / "lifting_line_correction_data/clark_y_ar6_yaw_minus20.csv"
    )
    # Restrict to the first stall event; retain the full source table separately.
    data = data[data.alpha <= 30].reset_index(drop=True)
    peak = int(onp.argmax(data.CL.to_numpy()))
    return {"lift": data[["alpha", "CL"]], "drag": data.iloc[: peak + 1][["CD", "CL"]]}
