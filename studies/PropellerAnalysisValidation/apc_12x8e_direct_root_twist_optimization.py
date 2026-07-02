from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.modeling.splines import bspline, bspline_basis_matrix

from studies.PropellerAnalysisValidation.apc_12x8e_validation_maps import (
    OUTPUTS,
    parse_apc_geometry_file,
)


TWIST_OPTIMIZATION_OUTPUTS = OUTPUTS / "twist_optimization_direct_root_low_j"

# Low-J wind-on point inside the UIUC data envelope. The direct + root model
# is well behaved here, while the stock 12x8E is far below its best efficiency.
DESIGN_RPM = 5000
DESIGN_VELOCITY_MPH = 10.5
DESIGN_VELOCITY_MPS = DESIGN_VELOCITY_MPH * 0.44704
DESIGN_LABEL = "5000 RPM, 10.5 mph low-J climb"

N_TWIST_CONTROL_POINTS = 8
SPLINE_DEGREE = 3
OPTIMIZATION_RADIAL_RESOLUTION = 12
REPORT_RADIAL_RESOLUTION = 24
CHECK_RADIAL_RESOLUTION = 32
THRUST_MATCH_FACTOR = 1.0
THRUST_TARGET_MARGIN_N = 0.03


def open_uniform_knot_vector(
    x_min: float,
    x_max: float,
    n_control_points: int,
    degree: int,
) -> onp.ndarray:
    n_internal_knots = n_control_points - degree - 1
    if n_internal_knots > 0:
        internal_knots = onp.linspace(x_min, x_max, n_internal_knots + 2)[1:-1]
    else:
        internal_knots = onp.array([])

    return onp.concatenate(
        [
            onp.full(degree + 1, x_min),
            internal_knots,
            onp.full(degree + 1, x_max),
        ]
    )


def fit_bspline_control_points(
    x: onp.ndarray,
    y: onp.ndarray,
    knots: onp.ndarray,
    n_control_points: int,
    degree: int,
    preserve_endpoints: bool = True,
) -> onp.ndarray:
    basis = bspline_basis_matrix(
        x=x,
        n_control_points=n_control_points,
        degree=degree,
        knots=knots,
    )

    if preserve_endpoints:
        control_points = onp.empty(n_control_points)
        control_points[0] = y[0]
        control_points[-1] = y[-1]
        rhs = y - basis[:, 0] * control_points[0] - basis[:, -1] * control_points[-1]
        control_points[1:-1], *_ = onp.linalg.lstsq(
            basis[:, 1:-1],
            rhs,
            rcond=None,
        )
    else:
        control_points, *_ = onp.linalg.lstsq(basis, y, rcond=None)

    return control_points


def make_bspline_twist_function(control_points, knots: onp.ndarray):
    return lambda r_over_R: bspline(
        x=r_over_R,
        y_control_points=control_points,
        degree=SPLINE_DEGREE,
        knots=knots,
        extrapolation="clip",
    )


def make_twist_optimized_propeller(
    propeller_apc: asb.Propeller,
    twist_control_points,
    knots: onp.ndarray,
) -> asb.Propeller:
    # B-spline twist is supplied as a callable input. Chord and thickness remain
    # the direct APC tabulated distributions; the Propeller object itself does
    # not natively build a B-spline distribution.
    return asb.Propeller(
        name="APC 12x8E direct + root, twist-only optimized",
        radius=propeller_apc.radius,
        hub_radius=propeller_apc.hub_radius,
        blade_count=propeller_apc.blade_count,
        radial_stations=propeller_apc.radial_stations,
        chord_distribution=propeller_apc.chord_distribution,
        twist_distribution=make_bspline_twist_function(
            control_points=twist_control_points,
            knots=knots,
        ),
        thickness_distribution=propeller_apc.thickness_distribution,
        airfoil_distribution=propeller_apc.airfoil_distribution,
        distribution_interpolation_method="linear",
    )


def evaluate_propeller(
    propeller: asb.Propeller,
    radial_resolution: int,
) -> dict:
    return asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=DESIGN_VELOCITY_MPS),
        rpm=DESIGN_RPM,
        radial_resolution=radial_resolution,
        newton_iterations=8,
        bracketing_iterations=24,
        model_size="xsmall",
        residual_tolerance=1e-4,
        include_root_loss=True,
        include_post_stall_confidence_blending=False,
    ).run()


def section_coordinates_inches(
    propeller: asb.Propeller,
    r_over_R: float,
    rotate_by_twist: bool,
) -> tuple[onp.ndarray, onp.ndarray]:
    inch = 0.0254
    airfoil = propeller.airfoil(float(r_over_R)).to_airfoil(n_coordinates_per_side=90)
    coordinates = onp.asarray(airfoil.coordinates, dtype=float)
    chord_in = float(propeller.chord(r_over_R)) / inch
    x = coordinates[:, 0] * chord_in
    y = coordinates[:, 1] * chord_in

    if rotate_by_twist:
        theta = onp.radians(float(propeller.twist(r_over_R)))
        x, y = (
            x * onp.cos(theta) - y * onp.sin(theta),
            x * onp.sin(theta) + y * onp.cos(theta),
        )

    return x, y


def output_array(output: dict, key: str) -> onp.ndarray:
    return onp.asarray(output[key], dtype=float).reshape(-1)


def write_spanwise_csv(output_apc: dict, output_optimized: dict, filename: Path) -> None:
    rows = []
    for label, output in [("apc", output_apc), ("optimized", output_optimized)]:
        r = output_array(output, "r_over_R")
        for i in range(len(r)):
            rows.append(
                {
                    "case": label,
                    "r_over_R": r[i],
                    "r_m": output_array(output, "r")[i],
                    "chord_m": output_array(output, "chord")[i],
                    "twist_deg": output_array(output, "twist")[i],
                    "thickness": output_array(output, "thickness")[i],
                    "alpha_deg": output_array(output, "alpha")[i],
                    "phi_deg": output_array(output, "phi")[i],
                    "CL": output_array(output, "CL")[i],
                    "CD": output_array(output, "CD")[i],
                    "Re": output_array(output, "Re")[i],
                    "mach": output_array(output, "mach")[i],
                    "analysis_confidence": output_array(output, "analysis_confidence")[
                        i
                    ],
                    "root_loss_factor": output_array(output, "root_loss_factor")[i],
                    "tip_loss_factor": output_array(output, "tip_loss_factor")[i],
                    "finite_blade_loss_factor": output_array(
                        output,
                        "finite_blade_loss_factor",
                    )[i],
                    "Gamma_m2_s": output_array(output, "Gamma")[i],
                    "dT_N": output_array(output, "dT")[i],
                    "dQ_Nm": output_array(output, "dQ")[i],
                    "thrust_per_radius_N_m": output_array(
                        output,
                        "thrust_per_radius",
                    )[i],
                    "power_per_radius_W_m": float(output["omega"])
                    * output_array(output, "torque_per_radius")[i],
                    "residual_m2_s": output_array(output, "residual")[i],
                }
            )
    pd.DataFrame(rows).to_csv(filename, index=False)


def plot_geometry(
    propeller_apc: asb.Propeller,
    propeller_optimized: asb.Propeller,
    geometry_table: pd.DataFrame,
    filename: Path,
    target_thrust_N: float,
) -> None:
    inch = 0.0254
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    axes[0].plot(
        geometry_table["r_over_R"],
        geometry_table["chord_apc_m"] / inch,
        color="black",
        linewidth=2,
        label="APC",
    )
    axes[0].plot(
        geometry_table["r_over_R"],
        geometry_table["chord_optimized_m"] / inch,
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label="Optimized",
    )
    axes[0].set_ylabel("Chord [in]")

    axes[1].plot(
        geometry_table["r_over_R"],
        geometry_table["twist_apc_deg"],
        color="black",
        linewidth=2,
        label="APC",
    )
    axes[1].plot(
        geometry_table["r_over_R"],
        geometry_table["twist_optimized_deg"],
        color="#1f77b4",
        linewidth=2,
        label="Optimized",
    )
    axes[1].set_ylabel("Twist [deg]")

    axes[2].plot(
        geometry_table["r_over_R"],
        geometry_table["thickness_apc"],
        color="black",
        linewidth=2,
        label="APC",
    )
    axes[2].plot(
        geometry_table["r_over_R"],
        geometry_table["thickness_optimized"],
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label="Optimized",
    )
    axes[2].set_ylabel("Thickness t/c [-]")

    for ax in axes:
        ax.set_xlabel("Station r/R [-]")
        ax.set_xlim(propeller_apc.radial_stations[0], 1.0)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="best")
    fig.suptitle(
        "Direct + Root Twist-Only Optimization "
        f"({DESIGN_LABEL}, {target_thrust_N:.2f} N matched thrust)"
    )
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_twisted_sections(
    propeller_apc: asb.Propeller,
    propeller_optimized: asb.Propeller,
    filename: Path,
) -> None:
    section_stations = [0.25, 0.50, 0.75]
    fig, axes = plt.subplots(
        len(section_stations),
        1,
        figsize=(8.5, 7.0),
        constrained_layout=True,
    )

    for ax, r_over_R in zip(axes, section_stations):
        x_apc, y_apc = section_coordinates_inches(
            propeller=propeller_apc,
            r_over_R=r_over_R,
            rotate_by_twist=True,
        )
        x_opt, y_opt = section_coordinates_inches(
            propeller=propeller_optimized,
            r_over_R=r_over_R,
            rotate_by_twist=True,
        )

        ax.plot(x_apc, y_apc, color="black", linewidth=1.7, label="APC")
        ax.plot(x_opt, y_opt, color="#1f77b4", linewidth=1.7, label="Optimized")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("y [in]")
        ax.set_title(
            f"r/R = {r_over_R:.2f}: "
            f"APC twist = {float(propeller_apc.twist(r_over_R)):.2f} deg, "
            f"optimized twist = {float(propeller_optimized.twist(r_over_R)):.2f} deg"
        )

    axes[-1].set_xlabel("x [in]")
    axes[0].legend(loc="best")
    fig.suptitle("Dimensional Sections Rotated by Local Twist")
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_spanwise_distributions(
    output_apc: dict,
    output_optimized: dict,
    filename: Path,
) -> None:
    omega = float(output_apc["omega"])
    distributions = [
        (
            "thrust_per_radius",
            lambda output: output_array(output, "thrust_per_radius"),
            "dT/dr [N/m]",
        ),
        (
            "power_per_radius",
            lambda output: omega * output_array(output, "torque_per_radius"),
            "dP/dr [W/m]",
        ),
        ("alpha", lambda output: output_array(output, "alpha"), "Alpha [deg]"),
        ("phi", lambda output: output_array(output, "phi"), "Inflow angle [deg]"),
        ("CL", lambda output: output_array(output, "CL"), "CL [-]"),
        ("CD", lambda output: output_array(output, "CD"), "CD [-]"),
        ("Re", lambda output: output_array(output, "Re") / 1e6, "Re [millions]"),
        ("mach", lambda output: output_array(output, "mach"), "Mach [-]"),
        (
            "analysis_confidence",
            lambda output: output_array(output, "analysis_confidence"),
            "NeuralFoil confidence [-]",
        ),
        (
            "finite_blade_loss_factor",
            lambda output: output_array(output, "finite_blade_loss_factor"),
            "Finite-blade loss [-]",
        ),
        ("Gamma", lambda output: output_array(output, "Gamma"), "Circulation [m^2/s]"),
        (
            "residual",
            lambda output: output_array(output, "residual"),
            "Residual [m^2/s]",
        ),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
    axes_flat = axes.ravel()

    r_apc = output_array(output_apc, "r_over_R")
    r_optimized = output_array(output_optimized, "r_over_R")

    for ax, (_, value_function, ylabel) in zip(axes_flat, distributions):
        ax.plot(
            r_apc,
            value_function(output_apc),
            color="black",
            linewidth=1.8,
            label="APC",
        )
        ax.plot(
            r_optimized,
            value_function(output_optimized),
            color="#1f77b4",
            linewidth=1.8,
            label="Optimized",
        )
        ax.set_xlabel("Station r/R [-]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes_flat[0].legend(loc="best")
    fig.suptitle(
        "Direct + Root Spanwise Distributions at "
        f"{DESIGN_RPM:.0f} RPM, {DESIGN_VELOCITY_MPH:g} mph"
    )
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_three_view(
    propeller: asb.Propeller,
    filename: Path,
    title: str,
) -> None:
    axs = propeller.draw_three_view(
        style="shaded",
        radial_resolution=36,
        n_coordinates_per_side=56,
        show=False,
    )
    fig = axs[0, 0].figure
    fig.suptitle(title, y=0.965)
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def run_twist_optimization() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TWIST_OPTIMIZATION_OUTPUTS.mkdir(parents=True, exist_ok=True)

    propeller_apc = parse_apc_geometry_file()
    r_over_R_apc = onp.asarray(propeller_apc.radial_stations, dtype=float)
    twist_apc = onp.asarray(propeller_apc.twist_distribution, dtype=float)

    knots = open_uniform_knot_vector(
        x_min=float(r_over_R_apc[0]),
        x_max=float(r_over_R_apc[-1]),
        n_control_points=N_TWIST_CONTROL_POINTS,
        degree=SPLINE_DEGREE,
    )
    twist_control_points_initial = fit_bspline_control_points(
        x=r_over_R_apc,
        y=twist_apc,
        knots=knots,
        n_control_points=N_TWIST_CONTROL_POINTS,
        degree=SPLINE_DEGREE,
        preserve_endpoints=True,
    )

    output_apc_optimization_mesh = evaluate_propeller(
        propeller=propeller_apc,
        radial_resolution=OPTIMIZATION_RADIAL_RESOLUTION,
    )
    optimization_mesh_thrust_target_N = (
        THRUST_MATCH_FACTOR * float(output_apc_optimization_mesh["thrust"])
        + THRUST_TARGET_MARGIN_N
    )

    opti = asb.Opti()
    twist_control_points = opti.variable(
        init_guess=twist_control_points_initial,
        scale=10,
        lower_bound=0,
        upper_bound=70,
    )

    propeller_optimized_symbolic = make_twist_optimized_propeller(
        propeller_apc=propeller_apc,
        twist_control_points=twist_control_points,
        knots=knots,
    )

    geometry_constraint_r = onp.linspace(r_over_R_apc[0], r_over_R_apc[-1], 90)
    twist_apc_grid = onp.interp(geometry_constraint_r, r_over_R_apc, twist_apc)
    twist_optimized_grid = propeller_optimized_symbolic.twist(geometry_constraint_r)
    opti.subject_to(
        [
            twist_optimized_grid >= 0,
            twist_optimized_grid <= 70,
            twist_optimized_grid >= onp.maximum(0, twist_apc_grid - 30),
            twist_optimized_grid <= twist_apc_grid + 25,
        ]
    )

    optimization_output = asb.PropellerAnalysis(
        propeller=propeller_optimized_symbolic,
        op_point=asb.OperatingPoint(velocity=DESIGN_VELOCITY_MPS),
        rpm=DESIGN_RPM,
        radial_resolution=OPTIMIZATION_RADIAL_RESOLUTION,
        newton_iterations=5,
        bracketing_iterations=0,
        model_size="xsmall",
        residual_tolerance=1e-3,
        include_root_loss=True,
        include_post_stall_confidence_blending=False,
    ).run()

    opti.subject_to(optimization_output["thrust"] >= optimization_mesh_thrust_target_N)
    regularization = 3e-5 * np.sum(
        ((twist_control_points - twist_control_points_initial) / 10) ** 2
    )
    opti.minimize(optimization_output["power"] / 100 + regularization)

    solution = opti.solve(
        max_iter=250,
        verbose=False,
        options={
            "ipopt.mu_strategy": "monotone",
            "ipopt.tol": 1e-6,
            "ipopt.constr_viol_tol": 1e-6,
        },
    )

    twist_control_points_optimized = solution(twist_control_points)
    propeller_optimized = make_twist_optimized_propeller(
        propeller_apc=propeller_apc,
        twist_control_points=twist_control_points_optimized,
        knots=knots,
    )

    output_apc_report = evaluate_propeller(
        propeller=propeller_apc,
        radial_resolution=REPORT_RADIAL_RESOLUTION,
    )
    output_optimized_report = evaluate_propeller(
        propeller=propeller_optimized,
        radial_resolution=REPORT_RADIAL_RESOLUTION,
    )
    output_optimized_check = evaluate_propeller(
        propeller=propeller_optimized,
        radial_resolution=CHECK_RADIAL_RESOLUTION,
    )

    report_thrust_target_N = THRUST_MATCH_FACTOR * float(output_apc_report["thrust"])

    dense_r = onp.linspace(r_over_R_apc[0], r_over_R_apc[-1], 240)
    geometry_table = pd.DataFrame(
        {
            "r_over_R": dense_r,
            "chord_apc_m": onp.asarray(propeller_apc.chord(dense_r), dtype=float),
            "chord_optimized_m": onp.asarray(
                propeller_optimized.chord(dense_r),
                dtype=float,
            ),
            "twist_apc_deg": onp.asarray(propeller_apc.twist(dense_r), dtype=float),
            "twist_optimized_deg": onp.asarray(
                propeller_optimized.twist(dense_r),
                dtype=float,
            ),
            "thickness_apc": onp.asarray(propeller_apc.thickness(dense_r), dtype=float),
            "thickness_optimized": onp.asarray(
                propeller_optimized.thickness(dense_r),
                dtype=float,
            ),
        }
    )

    control_points = pd.DataFrame(
        {
            "control_point_index": onp.arange(N_TWIST_CONTROL_POINTS),
            "r_over_R_plot_location": onp.linspace(
                r_over_R_apc[0],
                r_over_R_apc[-1],
                N_TWIST_CONTROL_POINTS,
            ),
            "twist_initial_deg": twist_control_points_initial,
            "twist_optimized_deg": twist_control_points_optimized,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "design_label": DESIGN_LABEL,
                "rpm": DESIGN_RPM,
                "velocity_mph": DESIGN_VELOCITY_MPH,
                "velocity_mps": DESIGN_VELOCITY_MPS,
                "advance_ratio": float(output_apc_report["J"]),
                "target_basis": "match direct-root APC baseline thrust",
                "thrust_match_factor": THRUST_MATCH_FACTOR,
                "optimization_mesh_thrust_margin_N": THRUST_TARGET_MARGIN_N,
                "report_thrust_target_N": report_thrust_target_N,
                "optimization_mesh_thrust_target_N": optimization_mesh_thrust_target_N,
                "n_twist_control_points": N_TWIST_CONTROL_POINTS,
                "optimization_radial_resolution": OPTIMIZATION_RADIAL_RESOLUTION,
                "report_radial_resolution": REPORT_RADIAL_RESOLUTION,
                "check_radial_resolution": CHECK_RADIAL_RESOLUTION,
                "apc_report_thrust_N": float(output_apc_report["thrust"]),
                "apc_report_power_W": float(output_apc_report["power"]),
                "apc_report_efficiency": float(output_apc_report["eta"]),
                "optimized_report_thrust_N": float(output_optimized_report["thrust"]),
                "optimized_report_power_W": float(output_optimized_report["power"]),
                "optimized_report_efficiency": float(output_optimized_report["eta"]),
                "optimized_check_thrust_N": float(output_optimized_check["thrust"]),
                "optimized_check_power_W": float(output_optimized_check["power"]),
                "optimized_check_efficiency": float(output_optimized_check["eta"]),
                "power_delta_W": float(
                    output_optimized_report["power"] - output_apc_report["power"]
                ),
                "power_delta_percent": float(
                    100
                    * (
                        output_optimized_report["power"]
                        / output_apc_report["power"]
                        - 1
                    )
                ),
                "efficiency_delta": float(
                    output_optimized_report["eta"] - output_apc_report["eta"]
                ),
                "apc_max_abs_residual": float(output_apc_report["max_abs_residual"]),
                "optimized_max_abs_residual": float(
                    output_optimized_report["max_abs_residual"]
                ),
                "optimized_check_max_abs_residual": float(
                    output_optimized_check["max_abs_residual"]
                ),
                "min_chord_delta_m": float(
                    onp.min(
                        geometry_table["chord_optimized_m"]
                        - geometry_table["chord_apc_m"]
                    )
                ),
                "max_chord_delta_m": float(
                    onp.max(
                        geometry_table["chord_optimized_m"]
                        - geometry_table["chord_apc_m"]
                    )
                ),
                "min_thickness_delta": float(
                    onp.min(
                        geometry_table["thickness_optimized"]
                        - geometry_table["thickness_apc"]
                    )
                ),
                "max_thickness_delta": float(
                    onp.max(
                        geometry_table["thickness_optimized"]
                        - geometry_table["thickness_apc"]
                    )
                ),
            }
        ]
    )

    summary.to_csv(
        TWIST_OPTIMIZATION_OUTPUTS
        / "apc_12x8e_direct_root_twist_optimization_summary.csv",
        index=False,
    )
    control_points.to_csv(
        TWIST_OPTIMIZATION_OUTPUTS
        / "apc_12x8e_direct_root_twist_optimization_control_points.csv",
        index=False,
    )
    geometry_table.to_csv(
        TWIST_OPTIMIZATION_OUTPUTS
        / "apc_12x8e_direct_root_twist_optimization_geometry.csv",
        index=False,
    )
    write_spanwise_csv(
        output_apc=output_apc_report,
        output_optimized=output_optimized_report,
        filename=TWIST_OPTIMIZATION_OUTPUTS
        / "apc_12x8e_direct_root_twist_optimization_spanwise.csv",
    )

    plot_geometry(
        propeller_apc=propeller_apc,
        propeller_optimized=propeller_optimized,
        geometry_table=geometry_table,
        filename=TWIST_OPTIMIZATION_OUTPUTS
        / "apc_12x8e_direct_root_twist_optimized_geometry.png",
        target_thrust_N=report_thrust_target_N,
    )
    plot_twisted_sections(
        propeller_apc=propeller_apc,
        propeller_optimized=propeller_optimized,
        filename=TWIST_OPTIMIZATION_OUTPUTS
        / "apc_12x8e_direct_root_twist_optimized_sections.png",
    )
    plot_spanwise_distributions(
        output_apc=output_apc_report,
        output_optimized=output_optimized_report,
        filename=TWIST_OPTIMIZATION_OUTPUTS
        / "apc_12x8e_direct_root_twist_optimized_spanwise_distributions.png",
    )
    plot_three_view(
        propeller=propeller_optimized,
        filename=TWIST_OPTIMIZATION_OUTPUTS
        / "apc_12x8e_direct_root_twist_optimized_three_view.png",
        title="APC 12x8E Direct + Root Twist Optimized Three-View",
    )

    print(summary.to_string(index=False))
    print(f"Wrote twist optimization outputs to: {TWIST_OPTIMIZATION_OUTPUTS}")

    return summary, control_points, geometry_table


if __name__ == "__main__":
    run_twist_optimization()
