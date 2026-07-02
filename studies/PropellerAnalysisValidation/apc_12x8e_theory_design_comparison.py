from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import aerosandbox as asb

from studies.PropellerAnalysisValidation.apc_12x8e_validation_maps import (
    OUTPUTS,
    parse_apc_geometry_file,
)
from studies.PropellerAnalysisValidation.apc_12x8e_direct_root_chord_twist_optimization import (
    CHORD_TWIST_OPTIMIZATION_OUTPUTS,
    DESIGN_LABEL,
    DESIGN_RPM,
    DESIGN_VELOCITY_MPH,
    DESIGN_VELOCITY_MPS,
    MAX_CHORD_M,
    MAX_THICKNESS_RATIO,
    MIN_CHORD_M,
    MIN_THICKNESS_RATIO,
    N_TWIST_CONTROL_POINTS,
    REPORT_RADIAL_RESOLUTION,
    SPLINE_DEGREE,
    make_chord_twist_optimized_propeller,
    open_uniform_knot_vector,
)


THEORY_OUTPUTS = OUTPUTS / "theory_design_comparison_low_j"
THEORY_RADIAL_STATION_COUNT = 36
THEORY_EVALUATION_RADIAL_RESOLUTION = 24
TARGET_SECTION_CL = 0.65
INITIAL_DESIGN_ALPHA_DEG = 2.25
ALPHA_GRID_DEG = onp.linspace(-6.0, 12.0, 49)
INCH = 0.0254


CASE_STYLES = {
    "APC baseline": {"color": "black", "linestyle": "-", "linewidth": 2.2},
    "Numerical optimum": {
        "color": "#1f77b4",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    "Constant CL inverse": {
        "color": "#d62728",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "Minimum induced loss": {
        "color": "#2ca02c",
        "linestyle": "-.",
        "linewidth": 2.0,
    },
}


def output_array(output: dict, key: str) -> onp.ndarray:
    return onp.asarray(output[key], dtype=float).reshape(-1)


def evaluate_propeller(
    propeller: asb.Propeller,
    radial_resolution: int = THEORY_EVALUATION_RADIAL_RESOLUTION,
    fast: bool = False,
) -> dict:
    return asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=DESIGN_VELOCITY_MPS),
        rpm=DESIGN_RPM,
        radial_resolution=radial_resolution,
        newton_iterations=5 if fast else 10,
        bracketing_iterations=12 if fast else 30,
        model_size="xsmall",
        residual_tolerance=1e-3 if fast else 1e-5,
        include_root_loss=True,
        include_post_stall_confidence_blending=False,
    ).run()


def prandtl_loss_factor(f: onp.ndarray) -> onp.ndarray:
    f = onp.maximum(f, 1e-12)
    return 2 / onp.pi * onp.arccos(onp.exp(-f))


def make_design_stations(propeller_apc: asb.Propeller) -> onp.ndarray:
    return onp.linspace(
        float(propeller_apc.radial_stations[0]),
        1.0,
        THEORY_RADIAL_STATION_COUNT,
    )


def structural_chord_bounds(
    propeller_apc: asb.Propeller,
    r_over_R: onp.ndarray,
) -> tuple[onp.ndarray, onp.ndarray]:
    absolute_thickness = onp.asarray(
        propeller_apc.chord(r_over_R) * propeller_apc.thickness(r_over_R),
        dtype=float,
    )
    lower = onp.maximum(MIN_CHORD_M, absolute_thickness / MAX_THICKNESS_RATIO)
    upper = onp.minimum(MAX_CHORD_M, absolute_thickness / MIN_THICKNESS_RATIO)
    return lower, upper


def enforce_structural_chord_bounds(
    propeller_apc: asb.Propeller,
    r_over_R: onp.ndarray,
    chord: onp.ndarray,
) -> onp.ndarray:
    lower, upper = structural_chord_bounds(propeller_apc, r_over_R)
    return onp.minimum(onp.maximum(chord, lower), upper)


def make_propeller_from_design_arrays(
    propeller_apc: asb.Propeller,
    name: str,
    r_over_R: onp.ndarray,
    chord: onp.ndarray,
    twist: onp.ndarray,
) -> asb.Propeller:
    chord = enforce_structural_chord_bounds(propeller_apc, r_over_R, chord)
    def thickness_function(query_r_over_R):
        query_chord = onp.interp(query_r_over_R, r_over_R, chord)
        query_absolute_thickness = onp.asarray(
            propeller_apc.chord(query_r_over_R)
            * propeller_apc.thickness(query_r_over_R),
            dtype=float,
        )
        return query_absolute_thickness / query_chord

    return asb.Propeller(
        name=name,
        radius=propeller_apc.radius,
        hub_radius=propeller_apc.hub_radius,
        blade_count=propeller_apc.blade_count,
        radial_stations=r_over_R,
        chord_distribution=chord,
        twist_distribution=twist,
        thickness_distribution=thickness_function,
        airfoil_distribution=propeller_apc.airfoil_distribution,
        distribution_interpolation_method="linear",
    )


def alpha_for_target_cl(
    propeller: asb.Propeller,
    r_over_R: onp.ndarray,
    Re: onp.ndarray,
    mach: onp.ndarray,
    target_cl: float = TARGET_SECTION_CL,
) -> onp.ndarray:
    alphas = []
    for station, Re_i, mach_i in zip(r_over_R, Re, mach):
        airfoil = propeller.airfoil(float(station))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            aero = airfoil.get_aero_from_neuralfoil(
                alpha=ALPHA_GRID_DEG,
                Re=max(float(Re_i), 1e4),
                mach=max(float(mach_i), 0.0),
                model_size="xsmall",
                include_360_deg_effects=True,
            )
        cl = onp.asarray(aero["CL"], dtype=float).reshape(-1)
        confidence = onp.asarray(
            aero.get("analysis_confidence", onp.ones_like(cl)),
            dtype=float,
        ).reshape(-1)

        attached_score = (
            onp.abs(cl - target_cl)
            + 0.1 * onp.maximum(0.0, 0.75 - confidence)
            + 0.004 * onp.maximum(0.0, ALPHA_GRID_DEG - 7.0) ** 2
        )
        i_best = int(onp.argmin(attached_score))
        alpha_best = float(ALPHA_GRID_DEG[i_best])

        for i in range(len(ALPHA_GRID_DEG) - 1):
            cl_a = cl[i] - target_cl
            cl_b = cl[i + 1] - target_cl
            if cl_a == 0:
                alpha_best = float(ALPHA_GRID_DEG[i])
                break
            if cl_a * cl_b <= 0:
                alpha_best = float(
                    onp.interp(
                        target_cl,
                        [cl[i], cl[i + 1]],
                        [ALPHA_GRID_DEG[i], ALPHA_GRID_DEG[i + 1]],
                    )
                )
                break
        alphas.append(alpha_best)

    return onp.asarray(alphas)


def scale_chord_to_target_thrust(
    propeller_apc: asb.Propeller,
    name: str,
    r_over_R: onp.ndarray,
    chord_shape: onp.ndarray,
    twist: onp.ndarray,
    target_thrust_N: float,
) -> tuple[asb.Propeller, float, dict]:
    def evaluate_scale(scale: float) -> tuple[float, asb.Propeller, dict]:
        propeller = make_propeller_from_design_arrays(
            propeller_apc=propeller_apc,
            name=name,
            r_over_R=r_over_R,
            chord=scale * chord_shape,
            twist=twist,
        )
        output = evaluate_propeller(
            propeller,
            radial_resolution=12,
            fast=True,
        )
        return float(output["thrust"]), propeller, output

    scale = 1.0
    propeller = None
    output = None
    for _ in range(4):
        thrust, propeller, output = evaluate_scale(scale)
        if not onp.isfinite(thrust) or abs(thrust) < 1e-6:
            scale *= 1.5
        else:
            scale *= float(onp.clip(target_thrust_N / thrust, 0.35, 2.8))
            scale = float(onp.clip(scale, 0.05, 20.0))

    thrust, propeller, output = evaluate_scale(scale)
    return propeller, scale, output


def trim_chord_to_target_with_full_solver(
    propeller_apc: asb.Propeller,
    propeller: asb.Propeller,
    target_thrust_N: float,
    case_name: str,
    n_iterations: int = 4,
) -> tuple[asb.Propeller, float]:
    r_over_R = onp.asarray(propeller.radial_stations, dtype=float)
    chord = onp.asarray(propeller.chord(r_over_R), dtype=float)
    twist = onp.asarray(propeller.twist(r_over_R), dtype=float)
    trim_scale = 1.0

    for _ in range(n_iterations):
        output = evaluate_propeller(
            propeller,
            radial_resolution=REPORT_RADIAL_RESOLUTION,
        )
        thrust = float(output["thrust"])
        if not onp.isfinite(thrust) or abs(thrust) < 1e-6:
            break
        scale = float(onp.clip(target_thrust_N / thrust, 0.9, 1.1))
        trim_scale *= scale
        chord = chord * scale
        propeller = make_propeller_from_design_arrays(
            propeller_apc=propeller_apc,
            name=case_name,
            r_over_R=r_over_R,
            chord=chord,
            twist=twist,
        )

    return propeller, trim_scale


def make_constant_cl_inverse_design(
    propeller_apc: asb.Propeller,
    target_thrust_N: float,
) -> tuple[asb.Propeller, dict]:
    r_over_R = make_design_stations(propeller_apc)
    r = r_over_R * propeller_apc.radius
    omega = DESIGN_RPM * 2 * onp.pi / 60
    rho = float(asb.OperatingPoint(velocity=DESIGN_VELOCITY_MPS).atmosphere.density())
    mu = float(
        asb.OperatingPoint(velocity=DESIGN_VELOCITY_MPS).atmosphere.dynamic_viscosity()
    )
    speed_of_sound = float(
        asb.OperatingPoint(velocity=DESIGN_VELOCITY_MPS).atmosphere.speed_of_sound()
    )

    disk_area = onp.pi * (propeller_apc.radius**2 - propeller_apc.hub_radius**2)
    induced_velocity = (
        -0.5 * DESIGN_VELOCITY_MPS
        + (
            (0.5 * DESIGN_VELOCITY_MPS) ** 2
            + target_thrust_N / (2 * rho * disk_area)
        )
        ** 0.5
    )
    Wa = DESIGN_VELOCITY_MPS + induced_velocity
    Wt = omega * r
    W = (Wa**2 + Wt**2) ** 0.5
    phi = onp.degrees(onp.arctan2(Wa, Wt))

    s = (r - propeller_apc.hub_radius) / (
        propeller_apc.radius - propeller_apc.hub_radius
    )
    loading_shape = onp.sin(onp.pi * onp.clip(s, 0, 1)) ** 0.65
    loading_shape = onp.maximum(loading_shape, 0.04)
    dT_dr_target = target_thrust_N * loading_shape / onp.trapezoid(loading_shape, r)
    dynamic_pressure = 0.5 * rho * W**2
    chord_shape = dT_dr_target / (
        propeller_apc.blade_count
        * dynamic_pressure
        * TARGET_SECTION_CL
        * onp.maximum(onp.cos(onp.radians(phi)), 0.1)
    )
    chord_shape = enforce_structural_chord_bounds(
        propeller_apc,
        r_over_R,
        chord_shape,
    )
    twist = phi + INITIAL_DESIGN_ALPHA_DEG

    propeller = None
    output = None
    scale = 1.0
    for _ in range(2):
        propeller, scale, output = scale_chord_to_target_thrust(
            propeller_apc=propeller_apc,
            name="Constant CL inverse",
            r_over_R=r_over_R,
            chord_shape=chord_shape,
            twist=twist,
            target_thrust_N=target_thrust_N,
        )
        phi_eval = onp.interp(
            r_over_R,
            output_array(output, "r_over_R"),
            output_array(output, "phi"),
        )
        twist = phi_eval + INITIAL_DESIGN_ALPHA_DEG

    propeller, scale, output = scale_chord_to_target_thrust(
        propeller_apc=propeller_apc,
        name="Constant CL inverse",
        r_over_R=r_over_R,
        chord_shape=chord_shape,
        twist=twist,
        target_thrust_N=target_thrust_N,
    )
    propeller, full_solver_trim_scale = trim_chord_to_target_with_full_solver(
        propeller_apc=propeller_apc,
        propeller=propeller,
        target_thrust_N=target_thrust_N,
        case_name="Constant CL inverse",
    )

    metadata = {
        "theory": "actuator-disk inverse loading with constant target section CL",
        "target_cl": TARGET_SECTION_CL,
        "chord_scale": scale,
        "full_solver_trim_scale": full_solver_trim_scale,
        "induced_velocity_mps": induced_velocity,
    }
    return propeller, metadata


def qmil_geometry_for_eta(
    propeller_apc: asb.Propeller,
    eta_induced: float,
    alpha_design: onp.ndarray,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray]:
    r_over_R = make_design_stations(propeller_apc)
    r = r_over_R * propeller_apc.radius
    omega = DESIGN_RPM * 2 * onp.pi / 60
    V = DESIGN_VELOCITY_MPS
    U = (V**2 + (omega * r) ** 2) ** 0.5
    psi_low = onp.arctan2(V, omega * r)
    psi_high = onp.full_like(psi_low, onp.pi / 2 - 1e-7)

    def residual(psi: onp.ndarray) -> onp.ndarray:
        Wa = 0.5 * V + 0.5 * U * onp.sin(psi)
        Wt = 0.5 * omega * r + 0.5 * U * onp.cos(psi)
        return eta_induced * omega * r * Wa - V * Wt

    residual_low = residual(psi_low)
    residual_high = residual(psi_high)
    no_bracket = residual_low * residual_high > 0
    psi_low = onp.where(no_bracket, 1e-7, psi_low)
    psi_high = onp.where(no_bracket, onp.pi - 1e-7, psi_high)
    residual_low = residual(psi_low)

    for _ in range(80):
        psi_mid = 0.5 * (psi_low + psi_high)
        residual_mid = residual(psi_mid)
        same_sign = residual_low * residual_mid > 0
        psi_low = onp.where(same_sign, psi_mid, psi_low)
        residual_low = onp.where(same_sign, residual_mid, residual_low)
        psi_high = onp.where(same_sign, psi_high, psi_mid)

    psi = 0.5 * (psi_low + psi_high)
    Wa = 0.5 * V + 0.5 * U * onp.sin(psi)
    Wt = 0.5 * omega * r + 0.5 * U * onp.cos(psi)
    vt = omega * r - Wt
    W = (Wa**2 + Wt**2) ** 0.5
    phi = onp.degrees(onp.arctan2(Wa, Wt))
    lambda_w = r_over_R * Wa / onp.maximum(Wt, 1e-8)
    lambda_w = onp.maximum(lambda_w, 1e-8)
    tip_loss = prandtl_loss_factor(
        propeller_apc.blade_count / 2 * (1 - r_over_R) / lambda_w
    )
    root_loss = prandtl_loss_factor(
        propeller_apc.blade_count
        / 2
        * (r_over_R - propeller_apc.hub_radius / propeller_apc.radius)
        / lambda_w
    )
    finite_blade_loss = tip_loss * root_loss
    gamma = (
        vt
        * 4
        * onp.pi
        * r
        / propeller_apc.blade_count
        * finite_blade_loss
        * (
            1
            + (
                4
                * lambda_w
                * propeller_apc.radius
                / (onp.pi * propeller_apc.blade_count * r)
            )
            ** 2
        )
        ** 0.5
    )
    chord = 2 * gamma / onp.maximum(W * TARGET_SECTION_CL, 1e-8)
    chord = enforce_structural_chord_bounds(propeller_apc, r_over_R, chord)
    twist = phi + alpha_design
    return r_over_R, chord, twist, phi, W


def solve_minimum_induced_loss_design(
    propeller_apc: asb.Propeller,
    target_thrust_N: float,
) -> tuple[asb.Propeller, dict]:
    r_over_R = make_design_stations(propeller_apc)
    alpha_design = onp.full_like(r_over_R, INITIAL_DESIGN_ALPHA_DEG)

    def build_and_evaluate(eta_induced: float) -> tuple[asb.Propeller, dict]:
        propeller = build_propeller_for_eta(eta_induced)
        output = evaluate_propeller(
            propeller,
            radial_resolution=12,
            fast=True,
        )
        return propeller, output

    def build_propeller_for_eta(eta_induced: float) -> asb.Propeller:
        r_design, chord, twist, _, _ = qmil_geometry_for_eta(
            propeller_apc=propeller_apc,
            eta_induced=eta_induced,
            alpha_design=alpha_design,
        )
        return make_propeller_from_design_arrays(
            propeller_apc=propeller_apc,
            name="Minimum induced loss",
            r_over_R=r_design,
            chord=chord,
            twist=twist,
        )

    eta_samples = onp.linspace(0.24, 0.94, 8)
    sample_outputs = [build_and_evaluate(float(eta)) for eta in eta_samples]
    thrusts = onp.array([float(out["thrust"]) for _, out in sample_outputs])
    finite = onp.isfinite(thrusts)

    if onp.any(finite):
        thrusts_finite = thrusts[finite]
        eta_finite = eta_samples[finite]
        order = onp.argsort(thrusts_finite)
        if thrusts_finite[order][0] <= target_thrust_N <= thrusts_finite[order][-1]:
            eta_induced = float(
                onp.interp(target_thrust_N, thrusts_finite[order], eta_finite[order])
            )
            propeller, output = build_and_evaluate(eta_induced)
        else:
            i_best = int(onp.argmin(onp.abs(thrusts - target_thrust_N)))
            eta_induced = float(eta_samples[i_best])
            propeller, output = sample_outputs[i_best]
    else:
        eta_induced = 0.55
        propeller, output = build_and_evaluate(eta_induced)

    eta_probe = onp.unique(
        onp.clip(
            onp.array([eta_induced - 0.06, eta_induced, eta_induced + 0.06]),
            0.24,
            0.94,
        )
    )
    probe_propellers = []
    probe_thrusts = []
    for eta in eta_probe:
        probe_propeller = build_propeller_for_eta(float(eta))
        probe_output = evaluate_propeller(
            probe_propeller,
            radial_resolution=REPORT_RADIAL_RESOLUTION,
        )
        probe_propellers.append(probe_propeller)
        probe_thrusts.append(float(probe_output["thrust"]))

    probe_thrusts = onp.asarray(probe_thrusts)
    if onp.ptp(probe_thrusts) > 1e-6:
        order = onp.argsort(probe_thrusts)
        if (
            probe_thrusts[order][0]
            <= target_thrust_N
            <= probe_thrusts[order][-1]
        ):
            eta_induced = float(
                onp.interp(target_thrust_N, probe_thrusts[order], eta_probe[order])
            )
        else:
            coefficients = onp.polyfit(probe_thrusts, eta_probe, deg=1)
            eta_induced = float(onp.polyval(coefficients, target_thrust_N))
        eta_induced = float(onp.clip(eta_induced, 0.24, 0.94))
        propeller = build_propeller_for_eta(eta_induced)
    else:
        i_best = int(onp.argmin(onp.abs(probe_thrusts - target_thrust_N)))
        eta_induced = float(eta_probe[i_best])
        propeller = probe_propellers[i_best]

    metadata = {
        "theory": "QMIL-style constant induced efficiency and target section CL",
        "target_cl": TARGET_SECTION_CL,
        "eta_induced_parameter": eta_induced,
    }
    return propeller, metadata


def load_numerical_optimum(propeller_apc: asb.Propeller) -> asb.Propeller:
    control_file = (
        CHORD_TWIST_OPTIMIZATION_OUTPUTS
        / "apc_12x8e_direct_root_chord_twist_optimization_control_points.csv"
    )
    controls = pd.read_csv(control_file)
    knots = open_uniform_knot_vector(
        x_min=float(propeller_apc.radial_stations[0]),
        x_max=float(propeller_apc.radial_stations[-1]),
        n_control_points=N_TWIST_CONTROL_POINTS,
        degree=SPLINE_DEGREE,
    )
    propeller = make_chord_twist_optimized_propeller(
        propeller_apc=propeller_apc,
        chord_control_points=controls["chord_optimized_m"].values,
        twist_control_points=controls["twist_optimized_deg"].values,
        knots=knots,
    )
    propeller.name = "Numerical optimum"
    return propeller


def make_geometry_table(cases: dict[str, asb.Propeller]) -> pd.DataFrame:
    rows = []
    r_min = min(float(propeller.radial_stations[0]) for propeller in cases.values())
    r_plot = onp.linspace(r_min, 1.0, 260)
    for case_name, propeller in cases.items():
        chord = onp.asarray(propeller.chord(r_plot), dtype=float)
        thickness = onp.asarray(propeller.thickness(r_plot), dtype=float)
        twist = onp.asarray(propeller.twist(r_plot), dtype=float)
        for i, station in enumerate(r_plot):
            rows.append(
                {
                    "case": case_name,
                    "r_over_R": station,
                    "chord_m": chord[i],
                    "twist_deg": twist[i],
                    "thickness_ratio": thickness[i],
                    "absolute_thickness_m": chord[i] * thickness[i],
                    "solidity": propeller.blade_count
                    * chord[i]
                    / (2 * onp.pi * station * propeller.radius),
                }
            )
    return pd.DataFrame(rows)


def make_spanwise_table(outputs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for case_name, output in outputs.items():
        r = output_array(output, "r_over_R")
        omega = float(output["omega"])
        for i, station in enumerate(r):
            rows.append(
                {
                    "case": case_name,
                    "r_over_R": station,
                    "r_m": output_array(output, "r")[i],
                    "chord_m": output_array(output, "chord")[i],
                    "twist_deg": output_array(output, "twist")[i],
                    "thickness_ratio": output_array(output, "thickness")[i],
                    "alpha_deg": output_array(output, "alpha")[i],
                    "phi_deg": output_array(output, "phi")[i],
                    "CL": output_array(output, "CL")[i],
                    "CD": output_array(output, "CD")[i],
                    "L_over_D": output_array(output, "CL")[i]
                    / max(output_array(output, "CD")[i], 1e-12),
                    "Re": output_array(output, "Re")[i],
                    "mach": output_array(output, "mach")[i],
                    "analysis_confidence": output_array(output, "analysis_confidence")[
                        i
                    ],
                    "Gamma_m2_s": output_array(output, "Gamma")[i],
                    "dT_dr_N_m": output_array(output, "thrust_per_radius")[i],
                    "dP_dr_W_m": omega * output_array(output, "torque_per_radius")[i],
                    "finite_blade_loss_factor": output_array(
                        output,
                        "finite_blade_loss_factor",
                    )[i],
                    "root_loss_factor": output_array(output, "root_loss_factor")[i],
                    "tip_loss_factor": output_array(output, "tip_loss_factor")[i],
                    "residual_m2_s": output_array(output, "residual")[i],
                }
            )
    return pd.DataFrame(rows)


def make_summary_table(
    cases: dict[str, asb.Propeller],
    outputs: dict[str, dict],
    metadata: dict[str, dict],
    target_thrust_N: float,
) -> pd.DataFrame:
    rows = []
    for case_name, propeller in cases.items():
        output = outputs[case_name]
        r = output_array(output, "r_over_R")
        cl = output_array(output, "CL")
        design_region = (r >= 0.25) & (r <= 0.95)
        rows.append(
            {
                "case": case_name,
                "design_label": DESIGN_LABEL,
                "rpm": DESIGN_RPM,
                "velocity_mph": DESIGN_VELOCITY_MPH,
                "velocity_mps": DESIGN_VELOCITY_MPS,
                "target_thrust_N": target_thrust_N,
                "thrust_N": float(output["thrust"]),
                "power_W": float(output["power"]),
                "eta": float(output["eta"]),
                "J": float(output["J"]),
                "Ct": float(output["Ct"]),
                "Cp": float(output["Cp"]),
                "max_abs_residual": float(output["max_abs_residual"]),
                "mean_CL_025_095R": float(onp.mean(cl[design_region])),
                "std_CL_025_095R": float(onp.std(cl[design_region])),
                "min_analysis_confidence": float(
                    onp.min(output_array(output, "analysis_confidence"))
                ),
                "min_chord_m": float(
                    onp.min(onp.asarray(propeller.chord(propeller.radial_stations)))
                ),
                "max_chord_m": float(
                    onp.max(onp.asarray(propeller.chord(propeller.radial_stations)))
                ),
                "min_twist_deg": float(
                    onp.min(onp.asarray(propeller.twist(propeller.radial_stations)))
                ),
                "max_twist_deg": float(
                    onp.max(onp.asarray(propeller.twist(propeller.radial_stations)))
                ),
                **metadata.get(case_name, {}),
            }
        )
    return pd.DataFrame(rows)


def plot_geometry_distributions(geometry_table: pd.DataFrame, filename: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    axes = axes.ravel()
    fields = [
        ("chord_m", "Chord [in]", lambda x: x / INCH),
        ("twist_deg", "Twist [deg]", lambda x: x),
        ("thickness_ratio", "Thickness ratio t/c [-]", lambda x: x),
        ("absolute_thickness_m", "Absolute thickness [in]", lambda x: x / INCH),
    ]
    for ax, (field, ylabel, transform) in zip(axes, fields):
        for case_name, group in geometry_table.groupby("case", sort=False):
            style = CASE_STYLES[case_name]
            ax.plot(
                group["r_over_R"],
                transform(group[field].values),
                label=case_name,
                **style,
            )
        ax.set_xlabel("Station r/R [-]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(geometry_table["r_over_R"].min(), 1.0)
    axes[0].legend(loc="best")
    fig.suptitle(f"12x8E Theory Geometry Comparison ({DESIGN_LABEL})")
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_spanwise_distributions(spanwise_table: pd.DataFrame, filename: Path) -> None:
    distributions = [
        ("dT_dr_N_m", "dT/dr [N/m]"),
        ("dP_dr_W_m", "dP/dr [W/m]"),
        ("Gamma_m2_s", "Circulation [m^2/s]"),
        ("CL", "CL [-]"),
        ("CD", "CD [-]"),
        ("L_over_D", "L/D [-]"),
        ("alpha_deg", "Alpha [deg]"),
        ("phi_deg", "Inflow angle [deg]"),
        ("Re", "Re [millions]", lambda values: values / 1e6),
        ("analysis_confidence", "NeuralFoil confidence [-]"),
        ("finite_blade_loss_factor", "Finite-blade loss [-]"),
        ("residual_m2_s", "Residual [m^2/s]"),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(15.5, 12.5), constrained_layout=True)
    axes = axes.ravel()
    for ax, item in zip(axes, distributions):
        if len(item) == 2:
            field, ylabel = item
            transform = lambda values: values
        else:
            field, ylabel, transform = item
        for case_name, group in spanwise_table.groupby("case", sort=False):
            style = CASE_STYLES[case_name]
            ax.plot(
                group["r_over_R"],
                transform(group[field].values),
                label=case_name,
                **style,
            )
        ax.set_xlabel("Station r/R [-]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(spanwise_table["r_over_R"].min(), 1.0)
    axes[0].legend(loc="best")
    fig.suptitle(f"12x8E Theory Spanwise Comparison ({DESIGN_LABEL})")
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def section_coordinates_inches(
    propeller: asb.Propeller,
    r_over_R: float,
    rotate_by_twist: bool = True,
) -> tuple[onp.ndarray, onp.ndarray]:
    airfoil = propeller.airfoil(float(r_over_R)).to_airfoil(n_coordinates_per_side=90)
    coordinates = onp.asarray(airfoil.coordinates, dtype=float)
    chord_in = float(propeller.chord(r_over_R)) / INCH
    x = coordinates[:, 0] * chord_in
    y = coordinates[:, 1] * chord_in

    if rotate_by_twist:
        theta = onp.radians(float(propeller.twist(r_over_R)))
        x, y = (
            x * onp.cos(theta) - y * onp.sin(theta),
            x * onp.sin(theta) + y * onp.cos(theta),
        )

    return x, y


def plot_sections(cases: dict[str, asb.Propeller], filename: Path) -> None:
    section_stations = [0.25, 0.50, 0.75]
    fig, axes = plt.subplots(
        len(section_stations),
        1,
        figsize=(9.5, 8.0),
        constrained_layout=True,
    )
    for ax, station in zip(axes, section_stations):
        for case_name, propeller in cases.items():
            x, y = section_coordinates_inches(propeller, station)
            ax.plot(x, y, label=case_name, **CASE_STYLES[case_name])
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("y [in]")
        ax.set_title(f"r/R = {station:.2f}")
    axes[-1].set_xlabel("x [in]")
    axes[0].legend(loc="best")
    fig.suptitle("Dimensional Sections Rotated by Local Twist")
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_three_view(propeller: asb.Propeller, filename: Path, title: str) -> None:
    axs = propeller.draw_three_view(
        style="shaded",
        radial_resolution=40,
        n_coordinates_per_side=56,
        show=False,
    )
    fig = axs[0, 0].figure
    fig.suptitle(title, y=0.965)
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def slugify_case_name(case_name: str) -> str:
    return case_name.lower().replace(" ", "_").replace("-", "_")


def run_theory_design_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    THEORY_OUTPUTS.mkdir(parents=True, exist_ok=True)

    propeller_apc = parse_apc_geometry_file()
    propeller_apc.name = "APC baseline"
    output_apc_report = evaluate_propeller(
        propeller=propeller_apc,
        radial_resolution=REPORT_RADIAL_RESOLUTION,
    )
    target_thrust_N = float(output_apc_report["thrust"])

    propeller_numerical = load_numerical_optimum(propeller_apc)
    propeller_constant_cl, constant_cl_metadata = make_constant_cl_inverse_design(
        propeller_apc=propeller_apc,
        target_thrust_N=target_thrust_N,
    )
    propeller_mil, mil_metadata = solve_minimum_induced_loss_design(
        propeller_apc=propeller_apc,
        target_thrust_N=target_thrust_N,
    )

    cases = {
        "APC baseline": propeller_apc,
        "Numerical optimum": propeller_numerical,
        "Constant CL inverse": propeller_constant_cl,
        "Minimum induced loss": propeller_mil,
    }
    metadata = {
        "APC baseline": {"theory": "APC measured geometry"},
        "Numerical optimum": {
            "theory": "ASB numerical chord/twist optimization with constant absolute thickness"
        },
        "Constant CL inverse": constant_cl_metadata,
        "Minimum induced loss": mil_metadata,
    }

    outputs = {
        case_name: (
            output_apc_report
            if case_name == "APC baseline"
            else evaluate_propeller(propeller, radial_resolution=REPORT_RADIAL_RESOLUTION)
        )
        for case_name, propeller in cases.items()
    }

    geometry_table = make_geometry_table(cases)
    spanwise_table = make_spanwise_table(outputs)
    summary = make_summary_table(
        cases=cases,
        outputs=outputs,
        metadata=metadata,
        target_thrust_N=target_thrust_N,
    )

    summary.to_csv(
        THEORY_OUTPUTS / "apc_12x8e_theory_design_comparison_summary.csv",
        index=False,
    )
    geometry_table.to_csv(
        THEORY_OUTPUTS / "apc_12x8e_theory_design_comparison_geometry.csv",
        index=False,
    )
    spanwise_table.to_csv(
        THEORY_OUTPUTS / "apc_12x8e_theory_design_comparison_spanwise.csv",
        index=False,
    )

    plot_geometry_distributions(
        geometry_table=geometry_table,
        filename=THEORY_OUTPUTS
        / "apc_12x8e_theory_design_geometry_distributions.png",
    )
    plot_spanwise_distributions(
        spanwise_table=spanwise_table,
        filename=THEORY_OUTPUTS
        / "apc_12x8e_theory_design_spanwise_distributions.png",
    )
    plot_sections(
        cases=cases,
        filename=THEORY_OUTPUTS / "apc_12x8e_theory_design_sections.png",
    )
    for case_name, propeller in cases.items():
        plot_three_view(
            propeller=propeller,
            filename=THEORY_OUTPUTS
            / f"apc_12x8e_theory_design_three_view_{slugify_case_name(case_name)}.png",
            title=f"12x8E {case_name} Three-View ({DESIGN_LABEL})",
        )

    print(summary.to_string(index=False))
    print(f"Wrote theory design comparison outputs to: {THEORY_OUTPUTS}")
    return summary, geometry_table, spanwise_table


if __name__ == "__main__":
    run_theory_design_comparison()
