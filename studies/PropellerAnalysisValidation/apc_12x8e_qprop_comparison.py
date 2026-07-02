from pathlib import Path
import re
import subprocess
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import aerosandbox as asb
import aerosandbox.numpy as np

from studies.PropellerAnalysisValidation.apc_12x8e_validation_maps import (
    APC_GEOMETRY_FILE,
    APC_PERFORMANCE_FILE,
    MAX_MODELED_THICKNESS_RATIO,
    OUTPUTS,
    PLOT_RPM_LABEL,
    PLOT_RPM_MAX,
    PLOT_RPM_MIN,
    parse_apc_geometry_file,
    parse_apc_performance_file,
)


QPROP_EXE = Path.home() / "Downloads" / "qprop1.22" / "qprop.exe"
QPROP_OUTPUTS = OUTPUTS / "qprop"

QPROP_POLAR = {
    "CL0": 0.50,
    "CLa": 5.8,
    "CLmin": -0.3,
    "CLmax": 1.2,
    "CD0": 0.028,
    "CD2u": 0.050,
    "CD2l": 0.020,
    "CLCD0": 0.50,
    "REref": 70000.0,
    "REexp": -0.7,
}

SUBSET_RPMS = [3000, 6000, 9000, 12000]
SENSITIVITY_RPMS = [9000, 11000, 13000]
SPANWISE_RPM = 12000
SPANWISE_TARGET_J = 0.08


def parse_apc_geometry_table(
    filename: Path = APC_GEOMETRY_FILE,
) -> tuple[pd.DataFrame, dict[str, float]]:
    text = filename.read_text(errors="ignore")
    rows = []
    metadata = {}

    for line in text.splitlines():
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.?", line)
        if len(numbers) == 14:
            rows.append([float(number) for number in numbers])

        if "RADIUS:" in line:
            metadata["radius_in"] = float(numbers[0])
        elif "HUBRAD:" in line:
            metadata["hub_radius_in"] = float(numbers[0])
        elif "BLADES:" in line:
            metadata["blade_count"] = int(float(numbers[0]))
        elif "INNER LIMIT" in line:
            metadata["inner_limit_radius_ratio"] = float(numbers[0])

    geometry = pd.DataFrame(
        rows,
        columns=[
            "station_in",
            "chord_in",
            "pitch_quoted_in",
            "pitch_le_te_in",
            "pitch_prather_in",
            "sweep_y_in",
            "rake_z_in",
            "thickness_ratio",
            "twist_deg",
            "max_thickness_in",
            "cross_section_in2",
            "zhigh_in",
            "cgy_in",
            "cgz_in",
        ],
    )

    if len({"radius_in", "hub_radius_in", "blade_count"} - set(metadata)):
        raise ValueError(f"Could not parse APC geometry metadata from {filename}.")

    geometry["r_over_R"] = geometry["station_in"] / metadata["radius_in"]
    return geometry, metadata


def qprop_section_aerodynamics(airfoil, alpha, Re, mach, r_over_R):
    alpha_rad = np.radians(alpha)
    beta = np.maximum((1 - mach**2) ** 0.5, 0.05)
    CL_unclipped = (QPROP_POLAR["CL0"] + QPROP_POLAR["CLa"] * alpha_rad) / beta
    CL = np.minimum(
        np.maximum(CL_unclipped, QPROP_POLAR["CLmin"]),
        QPROP_POLAR["CLmax"],
    )

    CD2 = np.where(CL > QPROP_POLAR["CLCD0"], QPROP_POLAR["CD2u"], QPROP_POLAR["CD2l"])
    Re_ratio = np.maximum(Re / QPROP_POLAR["REref"], 1e-3)
    CD = (
        QPROP_POLAR["CD0"] + CD2 * (CL - QPROP_POLAR["CLCD0"]) ** 2
    ) * Re_ratio ** QPROP_POLAR["REexp"]

    alpha_cd0 = (QPROP_POLAR["CLCD0"] - QPROP_POLAR["CL0"]) / QPROP_POLAR["CLa"]
    stalled = np.logical_or(
        CL_unclipped < QPROP_POLAR["CLmin"],
        CL_unclipped > QPROP_POLAR["CLmax"],
    )
    CD = CD + np.where(stalled, 2 * np.sin(alpha_rad - alpha_cd0) ** 2, 0)

    return {
        "CL": CL,
        "CD": CD,
        "CM": 0.0,
        "analysis_confidence": 1.0,
    }


def write_qprop_inputs(
    geometry: pd.DataFrame,
    metadata: dict[str, float],
    directory: Path = QPROP_OUTPUTS,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    prop_file = directory / "apc_12x8e_qprop.prop"
    motor_file = directory / "dummy_motor.motor"
    qcon_file = directory / "qcon.def"

    active_geometry = geometry[
        geometry["station_in"]
        >= max(
            metadata["hub_radius_in"],
            metadata.get("inner_limit_radius_ratio", metadata["hub_radius_in"] / metadata["radius_in"])
            * metadata["radius_in"],
        )
    ].copy()
    if MAX_MODELED_THICKNESS_RATIO is not None:
        active_geometry = active_geometry[
            active_geometry["thickness_ratio"] <= MAX_MODELED_THICKNESS_RATIO
        ].copy()

    prop_lines = [
        "APC 12x8E from APC geometry, generic QPROP polar",
        "",
        f"{metadata['blade_count']} {metadata['radius_in']:.6f} ! Nblades R(in)",
        "",
        f"{QPROP_POLAR['CL0']:.6f} {QPROP_POLAR['CLa']:.6f} ! CL0 CLa",
        f"{QPROP_POLAR['CLmin']:.6f} {QPROP_POLAR['CLmax']:.6f} ! CLmin CLmax",
        "",
        (
            f"{QPROP_POLAR['CD0']:.6f} {QPROP_POLAR['CD2u']:.6f} "
            f"{QPROP_POLAR['CD2l']:.6f} {QPROP_POLAR['CLCD0']:.6f} "
            "! CD0 CD2u CD2l CLCD0"
        ),
        f"{QPROP_POLAR['REref']:.6f} {QPROP_POLAR['REexp']:.6f} ! REref REexp",
        "",
        "0.0254 0.0254 1.0 ! Rfac Cfac Bfac",
        "0.0 0.0 0.0 ! Radd Cadd Badd",
        "",
        "# r(in) chord(in) beta(deg)",
    ]
    for _, row in active_geometry.iterrows():
        prop_lines.append(
            f"{row.station_in:10.5f} {row.chord_in:10.5f} {row.twist_deg:10.5f}"
        )
    prop_file.write_text("\n".join(prop_lines) + "\n")

    motor_file.write_text(
        "\n".join(
            [
                "Dummy high-Kv motor for fixed-RPM aerodynamic QPROP runs",
                "",
                "1",
                "0.001",
                "0.0",
                "100.0",
                "",
            ]
        )
    )

    qcon_file.write_text("\n".join(["1.225", "1.7894E-5", "340.294"]) + "\n")

    return prop_file, motor_file


def parse_qprop_output(stdout: str) -> tuple[dict[str, float], pd.DataFrame]:
    summary = None
    radial_rows = []
    in_radial_table = False

    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if "radius" in stripped and "chord" in stripped and "beta" in stripped:
                in_radial_table = True
            stripped = stripped[1:].strip()
            if not stripped:
                continue

        parts = stripped.split()
        try:
            values = [float(part.replace("D", "E")) for part in parts]
        except ValueError:
            continue

        is_column_index_row = len(values) >= 19 and all(
            abs(values[i] - (i + 1)) < 1e-12 for i in range(19)
        )
        if len(values) >= 19 and summary is None and not is_column_index_row:
            summary = {
                "velocity_mps": values[0],
                "rpm": values[1],
                "thrust_N_qprop": values[3],
                "torque_Nm_qprop": values[4],
                "power_W_qprop": values[5],
                "eta_qprop": values[9],
                "cl_avg_qprop": values[17],
                "cd_avg_qprop": values[18],
            }
        elif in_radial_table and len(values) >= 12:
            radial_rows.append(
                {
                    "r": values[0],
                    "chord": values[1],
                    "beta": values[2],
                    "CL_qprop": values[3],
                    "CD_qprop": values[4],
                    "Re_qprop": values[5],
                    "mach_qprop": values[6],
                    "effi_qprop": values[7],
                    "effp_qprop": values[8],
                    "Wa_qprop": values[9],
                    "Aswirl_qprop": values[10],
                    "adv_wake_qprop": values[11],
                }
            )

    if summary is None:
        raise ValueError(f"Could not parse QPROP output:\n{stdout}")

    return summary, pd.DataFrame(radial_rows)


def run_qprop_case(
    prop_file: Path,
    motor_file: Path,
    velocity_mps: float,
    rpm: float,
    radius: float,
    rho: float = 1.225,
) -> tuple[dict[str, float], pd.DataFrame]:
    result = subprocess.run(
        [
            str(QPROP_EXE),
            str(prop_file.name),
            str(motor_file.name),
            f"{velocity_mps:.10g}",
            f"{rpm:.10g}",
        ],
        cwd=prop_file.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    summary, radial = parse_qprop_output(result.stdout)

    n = rpm / 60
    diameter = 2 * radius
    summary["rpm"] = rpm
    summary["J"] = velocity_mps / (n * diameter)
    summary["Ct_qprop"] = summary["thrust_N_qprop"] / (rho * n**2 * diameter**4)
    summary["Cp_qprop"] = summary["power_W_qprop"] / (rho * n**3 * diameter**5)
    summary["eta_qprop"] = (
        velocity_mps * summary["thrust_N_qprop"] / summary["power_W_qprop"]
        if summary["power_W_qprop"] > 0
        else 0.0
    )

    return summary, radial


def run_aerosandbox_qprop_polar_case(propeller, velocity_mps: float, rpm: float):
    result = asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=velocity_mps),
        rpm=rpm,
        radial_resolution=32,
        newton_iterations=8,
        bracketing_iterations=24,
        residual_tolerance=1e-5,
        section_aerodynamics=qprop_section_aerodynamics,
    ).run()

    return {
        "Ct_asb_qprop_polar": float(result["Ct"]),
        "Cp_asb_qprop_polar": float(result["Cp"]),
        "eta_asb_qprop_polar": float(result["eta"]),
        "thrust_N_asb_qprop_polar": float(result["thrust"]),
        "torque_Nm_asb_qprop_polar": float(result["torque"]),
        "power_W_asb_qprop_polar": float(result["power"]),
        "max_abs_residual_asb_qprop_polar": float(result["max_abs_residual"]),
    }


def run_aerosandbox_neuralfoil_case(propeller, velocity_mps: float, rpm: float):
    result = asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=velocity_mps),
        rpm=rpm,
        radial_resolution=32,
        newton_iterations=8,
        bracketing_iterations=24,
        residual_tolerance=1e-5,
        model_size="xsmall",
    ).run()

    return {
        "Ct_asb_neuralfoil": float(result["Ct"]),
        "Cp_asb_neuralfoil": float(result["Cp"]),
        "eta_asb_neuralfoil": float(result["eta"]),
        "thrust_N_asb_neuralfoil": float(result["thrust"]),
        "torque_Nm_asb_neuralfoil": float(result["torque"]),
        "power_W_asb_neuralfoil": float(result["power"]),
        "max_abs_residual_asb_neuralfoil": float(result["max_abs_residual"]),
    }


def make_subset_cases(apc_data: pd.DataFrame) -> pd.DataFrame:
    subset = apc_data[apc_data["rpm"].isin(SUBSET_RPMS)].copy()
    target_js = np.array([0.0, 0.08, 0.16, 0.28, 0.42, 0.56, 0.70, 0.80])
    rows = []
    for rpm, data in subset.groupby("rpm"):
        for target_j in target_js:
            index = (data["J"] - target_j).abs().idxmin()
            rows.append(data.loc[index])
    return pd.DataFrame(rows).drop_duplicates(["rpm", "J"]).sort_values(["rpm", "J"])


def run_subset_comparison(force_rerun: bool = False) -> pd.DataFrame:
    QPROP_OUTPUTS.mkdir(parents=True, exist_ok=True)
    output_csv = QPROP_OUTPUTS / "apc_12x8e_qprop_subset_comparison.csv"
    if output_csv.exists() and not force_rerun:
        return pd.read_csv(output_csv)
    cached = None if force_rerun else pd.read_csv(output_csv) if output_csv.exists() else None

    geometry, metadata = parse_apc_geometry_table()
    prop_file, motor_file = write_qprop_inputs(geometry, metadata)
    apc_data = parse_apc_performance_file()
    propeller = parse_apc_geometry_file()
    subset = make_subset_cases(apc_data)
    radius = metadata["radius_in"] * 0.0254

    rows = []
    total = len(subset)
    for i, (_, case) in enumerate(subset.iterrows(), start=1):
        print(
            f"QPROP comparison case {i}/{total}: "
            f"{case.rpm:.0f} RPM, J={case.J:.3f}",
            flush=True,
        )
        row = case.to_dict()
        cached_row = None
        if cached is not None:
            matches = cached[
                (cached["rpm"] == case.rpm)
                & np.isclose(cached["J"], case.J, rtol=0, atol=1e-10)
            ]
            if len(matches) > 0:
                cached_row = matches.iloc[0]

        try:
            qprop_summary, _ = run_qprop_case(
                prop_file=prop_file,
                motor_file=motor_file,
                velocity_mps=case.velocity_mps,
                rpm=case.rpm,
                radius=radius,
            )
            row.update(qprop_summary)
        except Exception as e:
            warnings.warn(f"QPROP failed for {case.rpm:.0f} RPM, J={case.J:.3f}: {e}")

        asb_qprop_columns = [
            "Ct_asb_qprop_polar",
            "Cp_asb_qprop_polar",
            "eta_asb_qprop_polar",
            "thrust_N_asb_qprop_polar",
            "torque_Nm_asb_qprop_polar",
            "power_W_asb_qprop_polar",
            "max_abs_residual_asb_qprop_polar",
        ]
        if cached_row is not None and all(column in cached_row for column in asb_qprop_columns):
            row.update({column: cached_row[column] for column in asb_qprop_columns})
        else:
            row.update(
                run_aerosandbox_qprop_polar_case(
                    propeller=propeller,
                    velocity_mps=case.velocity_mps,
                    rpm=case.rpm,
                )
            )

        asb_neuralfoil_columns = [
            "Ct_asb_neuralfoil",
            "Cp_asb_neuralfoil",
            "eta_asb_neuralfoil",
            "thrust_N_asb_neuralfoil",
            "torque_Nm_asb_neuralfoil",
            "power_W_asb_neuralfoil",
            "max_abs_residual_asb_neuralfoil",
        ]
        if cached_row is not None and all(
            column in cached_row for column in asb_neuralfoil_columns
        ):
            row.update({column: cached_row[column] for column in asb_neuralfoil_columns})
        else:
            row.update(
                run_aerosandbox_neuralfoil_case(
                    propeller=propeller,
                    velocity_mps=case.velocity_mps,
                    rpm=case.rpm,
                )
            )
        rows.append(row)

    comparison = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)
    return comparison


def plot_qprop_comparison(comparison: pd.DataFrame):
    comparison = comparison[
        comparison["rpm"].between(PLOT_RPM_MIN, PLOT_RPM_MAX, inclusive="both")
    ].copy()

    quantities = [
        ("Ct", "Ct_qprop", "Ct_asb_qprop_polar", "Ct_asb_neuralfoil", "$C_T$ [-]"),
        ("Cp", "Cp_qprop", "Cp_asb_qprop_polar", "Cp_asb_neuralfoil", "$C_P$ [-]"),
        ("eta", "eta_qprop", "eta_asb_qprop_polar", "eta_asb_neuralfoil", "Efficiency [-]"),
        (
            "thrust_N",
            "thrust_N_qprop",
            "thrust_N_asb_qprop_polar",
            "thrust_N_asb_neuralfoil",
            "Thrust [N]",
        ),
        (
            "torque_Nm",
            "torque_Nm_qprop",
            "torque_Nm_asb_qprop_polar",
            "torque_Nm_asb_neuralfoil",
            "Torque [N-m]",
        ),
        (
            "power_W",
            "power_W_qprop",
            "power_W_asb_qprop_polar",
            "power_W_asb_neuralfoil",
            "Power [W]",
        ),
    ]

    rpms = sorted(comparison["rpm"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    norm = Normalize(vmin=min(rpms), vmax=max(rpms))
    mapper = ScalarMappable(norm=norm, cmap="viridis")
    mapper.set_array([])

    for ax, (apc_col, qprop_col, asb_qprop_col, asb_nf_col, ylabel) in zip(
        axes.flatten(), quantities
    ):
        for rpm, data in comparison.groupby("rpm"):
            data = data.sort_values("J")
            color = mapper.cmap(norm(rpm))
            ax.plot(data["J"], data[apc_col], color=color, linewidth=1.8)
            ax.plot(data["J"], data[qprop_col], color=color, linewidth=1.3, linestyle="--")
            ax.plot(data["J"], data[asb_qprop_col], color=color, linewidth=1.2, linestyle=":")
            ax.plot(data["J"], data[asb_nf_col], color=color, linewidth=1.2, linestyle="-.")
        ax.set_xlabel("Advance ratio $J$ [-]")
        ax.set_ylabel(ylabel)
        if apc_col == "eta":
            ax.set_ylim(-0.25, 1.05)
        ax.grid(True, alpha=0.25)

    axes.flat[0].plot([], [], color="black", label="APC")
    axes.flat[0].plot([], [], color="black", linestyle="--", label="QPROP")
    axes.flat[0].plot([], [], color="black", linestyle=":", label="ASB QPROP polar")
    axes.flat[0].plot([], [], color="black", linestyle="-.", label="ASB NeuralFoil")
    axes.flat[0].legend(loc="best")
    cbar = fig.colorbar(mapper, ax=axes.flatten(), shrink=0.82, pad=0.015)
    cbar.set_label("RPM")
    fig.suptitle(f"APC 12x8E Subset ({PLOT_RPM_LABEL}): APC vs QPROP vs AeroSandbox")
    fig.savefig(QPROP_OUTPUTS / "apc_12x8e_qprop_subset_comparison_vs_J.png", dpi=220)
    plt.close(fig)


def plot_geometry():
    geometry, _ = parse_apc_geometry_table()
    propeller = parse_apc_geometry_file()
    active = geometry[geometry["r_over_R"] >= propeller.radial_stations[0]].copy()
    r = np.linspace(propeller.radial_stations[0], 1, 300)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True, constrained_layout=True)
    axes[0].plot(r, propeller.chord(r) / 0.0254, label="Direct APC interpolation")
    axes[0].scatter(active["r_over_R"], active["chord_in"], s=16, color="black", label="APC table")
    axes[0].set_ylabel("Chord [in]")
    axes[0].legend(loc="best")

    axes[1].plot(r, propeller.twist(r), label="Direct APC interpolation")
    axes[1].scatter(active["r_over_R"], active["twist_deg"], s=16, color="black")
    axes[1].set_ylabel("Twist [deg]")

    axes[2].plot(r, propeller.thickness(r), label="Direct APC interpolation")
    axes[2].scatter(active["r_over_R"], active["thickness_ratio"], s=16, color="black")
    axes[2].set_ylabel("Thickness ratio [-]")
    axes[2].set_xlabel("Station $r/R$ [-]")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.suptitle("APC 12x8E Geometry")
    fig.savefig(OUTPUTS / "apc_12x8e_geometry.png", dpi=220)
    plt.close(fig)


def plot_spanwise_mid_sweep():
    apc_data = parse_apc_performance_file()
    candidates = apc_data.loc[apc_data["rpm"].eq(SPANWISE_RPM)].copy()
    case = candidates.iloc[(candidates["J"] - SPANWISE_TARGET_J).abs().argmin()]
    propeller = parse_apc_geometry_file()

    output = asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=case.velocity_mps),
        rpm=case.rpm,
        radial_resolution=32,
        newton_iterations=8,
        bracketing_iterations=24,
        residual_tolerance=1e-5,
        model_size="xsmall",
    ).run()

    quantities = [
        ("chord", output["chord"] / 0.0254, "Chord [in]"),
        ("twist", output["twist"], "Twist [deg]"),
        ("thickness", output["thickness"], "Thickness ratio [-]"),
        ("Re", output["Re"] / 1e6, "Reynolds number [millions]"),
        ("alpha", output["alpha"], "Angle of attack [deg]"),
        ("CL", output["CL"], "$C_L$ [-]"),
        ("CD", output["CD"], "$C_D$ [-]"),
        (
            "analysis_confidence",
            output["analysis_confidence"],
            "NeuralFoil confidence [-]",
        ),
        ("dTdr", output["thrust_per_radius"], "dT/dr [N/m]"),
        ("dQdr", output["torque_per_radius"], "dQ/dr [N]"),
        ("dPdr", output["omega"] * output["torque_per_radius"], "dP/dr [W/m]"),
        ("residual", output["residual"], "Residual"),
    ]

    fig, axes = plt.subplots(6, 2, figsize=(12, 15), sharex=True, constrained_layout=True)
    for ax, (_, y, ylabel) in zip(axes.flatten(), quantities):
        ax.plot(output["r_over_R"], y, marker="o", markersize=3)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("Station $r/R$ [-]")
    fig.suptitle(
        f"APC 12x8E Spanwise Quantities, {case.rpm:.0f} RPM, "
        f"J={case.J:.3f}, V={case.velocity_mph:.1f} mph"
    )
    fig.savefig(OUTPUTS / "apc_12x8e_spanwise_mid_sweep.png", dpi=220)
    plt.close(fig)


def run_solver_sensitivity():
    apc_data = parse_apc_performance_file()
    propeller = parse_apc_geometry_file()
    cases = apc_data[
        apc_data["rpm"].isin(SENSITIVITY_RPMS)
        & apc_data["J"].between(0.0, 0.28)
    ].copy()
    cases = cases.groupby("rpm", group_keys=False).head(6)

    configs = [
        {"radial_resolution": 16, "newton_iterations": 6, "bracketing_iterations": 14},
        {"radial_resolution": 16, "newton_iterations": 12, "bracketing_iterations": 24},
        {"radial_resolution": 24, "newton_iterations": 8, "bracketing_iterations": 24},
        {"radial_resolution": 32, "newton_iterations": 8, "bracketing_iterations": 24},
    ]

    rows = []
    for config in configs:
        label = (
            f"r{config['radial_resolution']}_n{config['newton_iterations']}"
            f"_b{config['bracketing_iterations']}"
        )
        for _, case in cases.iterrows():
            output = asb.PropellerAnalysis(
                propeller=propeller,
                op_point=asb.OperatingPoint(velocity=case.velocity_mps),
                rpm=case.rpm,
                residual_tolerance=1e-5,
                model_size="xsmall",
                **config,
            ).run()
            rows.append(
                {
                    "config": label,
                    "rpm": case.rpm,
                    "J": case.J,
                    "Ct": float(output["Ct"]),
                    "Cp": float(output["Cp"]),
                    "eta": float(output["eta"]),
                    "max_abs_residual": float(output["max_abs_residual"]),
                }
            )

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(QPROP_OUTPUTS / "apc_12x8e_solver_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, quantity in zip(axes, ["Ct", "Cp", "max_abs_residual"]):
        for label, data in sensitivity.groupby("config"):
            ax.plot(
                data["J"],
                data[quantity],
                marker="o",
                linestyle="",
                markersize=4,
                label=label,
                alpha=0.8,
            )
        ax.set_xlabel("Advance ratio $J$ [-]")
        ax.set_ylabel(quantity)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Low-J Solver Sensitivity, {PLOT_RPM_LABEL}")
    fig.savefig(QPROP_OUTPUTS / "apc_12x8e_solver_sensitivity.png", dpi=220)
    plt.close(fig)
    return sensitivity


def main(force_rerun: bool = False):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    QPROP_OUTPUTS.mkdir(parents=True, exist_ok=True)
    comparison = run_subset_comparison(force_rerun=force_rerun)
    plot_qprop_comparison(comparison)
    plot_geometry()
    plot_spanwise_mid_sweep()
    sensitivity = run_solver_sensitivity()

    metrics = []
    comparison_for_metrics = comparison[
        comparison["rpm"].between(PLOT_RPM_MIN, PLOT_RPM_MAX, inclusive="both")
    ].copy()
    for model_name, ct_col, cp_col in [
        ("QPROP", "Ct_qprop", "Cp_qprop"),
        ("ASB QPROP polar", "Ct_asb_qprop_polar", "Cp_asb_qprop_polar"),
        ("ASB NeuralFoil", "Ct_asb_neuralfoil", "Cp_asb_neuralfoil"),
    ]:
        metrics.append(
            {
                "model": model_name,
                "Ct_rmse_vs_APC": np.sqrt(
                    np.nanmean(
                        (comparison_for_metrics[ct_col] - comparison_for_metrics["Ct"])
                        ** 2
                    )
                ),
                "Cp_rmse_vs_APC": np.sqrt(
                    np.nanmean(
                        (comparison_for_metrics[cp_col] - comparison_for_metrics["Cp"])
                        ** 2
                    )
                ),
            }
        )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(QPROP_OUTPUTS / "apc_12x8e_qprop_subset_metrics.csv", index=False)

    print(metrics.to_string(index=False))
    print(
        "Sensitivity max residuals:\n"
        + sensitivity.groupby("config")["max_abs_residual"].max().to_string()
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    main(force_rerun=args.force_rerun)
