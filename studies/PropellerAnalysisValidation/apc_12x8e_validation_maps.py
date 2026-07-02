import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
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


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
APC_COMPARISON_OUTPUTS = OUTPUTS / "12x8e_nf_root"
TIP_THICKNESS_TO_ROOT_OUTPUTS = OUTPUTS / "12x8e_tip_nf_root"
APC_COMPARISON_POST_STALL_BLEND_OUTPUTS = OUTPUTS / "12x8e_blend_root"
TIP_THICKNESS_TO_ROOT_POST_STALL_BLEND_OUTPUTS = OUTPUTS / "12x8e_tip_blend_root"
APC_GEOMETRY_FILE = Path.home() / "Downloads" / "12x8E-PERF.txt"
APC_PERFORMANCE_FILE = Path.home() / "Downloads" / "PER3_12x8E.txt"
VALIDATION_RADIAL_RESOLUTION = 16
VALIDATION_NEWTON_ITERATIONS = 8
VALIDATION_BRACKETING_ITERATIONS = 24
VALIDATION_MODEL_SIZE = "xsmall"
VALIDATION_RESIDUAL_TOLERANCE = 1e-3
MAX_MODELED_THICKNESS_RATIO = 0.35
PLOT_RPM_MIN = 3000
PLOT_RPM_MAX = 13000
PLOT_RPM_LABEL = f"{PLOT_RPM_MIN}-{PLOT_RPM_MAX} RPM"
SPANWISE_RPM = 12000
SPANWISE_TARGET_J = 0.08
SPANWISE_RADIAL_RESOLUTION = 32
_PARALLEL_PROPELLER = None


def make_geometry_cache_tag(
    max_modeled_thickness_ratio: float,
    tip_thickness_to_root: bool = False,
    post_stall_confidence_blending: bool = False,
) -> str:
    if max_modeled_thickness_ratio is None:
        thickness_label = "uncapped_tc"
    else:
        thickness_label = f"tc_le_{max_modeled_thickness_ratio:g}"

    if tip_thickness_to_root:
        geometry_label = f"direct_apc_table_tip_thickness_to_root_{thickness_label}"
    else:
        geometry_label = f"direct_apc_table_{thickness_label}"

    model_label = (
        "_neuralfoil_anchored_viterna_post_stall_c075_w022"
        if post_stall_confidence_blending
        else ""
    )
    return f"{geometry_label}_camber_preserved_thickness_rootloss{model_label}"


GEOMETRY_CACHE_TAG = make_geometry_cache_tag(
    max_modeled_thickness_ratio=MAX_MODELED_THICKNESS_RATIO,
    tip_thickness_to_root=False,
    post_stall_confidence_blending=False,
)


def make_comparison_label(
    tip_thickness_to_root: bool = False,
    post_stall_confidence_blending: bool = False,
) -> str:
    if tip_thickness_to_root and post_stall_confidence_blending:
        return "APC 12x8E tip-t/c blend + root"
    elif tip_thickness_to_root:
        return "APC 12x8E tip-t/c NF + root"
    elif post_stall_confidence_blending:
        return "APC 12x8E blend + root"
    else:
        return "APC 12x8E NF + root"


def _initialize_parallel_validation_worker(propeller: asb.Propeller) -> None:
    global _PARALLEL_PROPELLER
    _PARALLEL_PROPELLER = propeller


def _run_single_aerosandbox_validation_case(
    case: dict,
    propeller: asb.Propeller,
    radial_resolution: int,
    newton_iterations: int,
    model_size: str,
    residual_tolerance: float,
    geometry_cache_tag: str = GEOMETRY_CACHE_TAG,
    post_stall_confidence_blending: bool = False,
) -> dict:
    try:
        result = asb.PropellerAnalysis(
            propeller=propeller,
            op_point=asb.OperatingPoint(velocity=case["velocity_mps"]),
            rpm=case["rpm"],
            radial_resolution=radial_resolution,
            newton_iterations=newton_iterations,
            bracketing_iterations=VALIDATION_BRACKETING_ITERATIONS,
            model_size=model_size,
            residual_tolerance=residual_tolerance,
            include_post_stall_confidence_blending=post_stall_confidence_blending,
        ).run()

        max_abs_residual = float(result["max_abs_residual"])
        return {
            "rpm": case["rpm"],
            "velocity_mph": case["velocity_mph"],
            "velocity_mps": case["velocity_mps"],
            "J": case["J"],
            "Ct_asb": float(result["Ct"]),
            "Cp_asb": float(result["Cp"]),
            "eta_asb": float(result["eta"]),
            "thrust_N_asb": float(result["thrust"]),
            "torque_Nm_asb": float(result["torque"]),
            "power_W_asb": float(result["power"]),
            "max_abs_residual": max_abs_residual,
            "converged": max_abs_residual <= residual_tolerance,
            "min_analysis_confidence": float(np.min(result["analysis_confidence"])),
            "min_root_loss_factor": float(np.min(result["root_loss_factor"])),
            "min_finite_blade_loss_factor": float(
                np.min(result["finite_blade_loss_factor"])
            ),
            "geometry_cache_tag": geometry_cache_tag,
            "geometry_hub_over_R": float(propeller.hub_radius / propeller.radius),
            "geometry_station_count": len(propeller.radial_stations),
            "validation_radial_resolution": radial_resolution,
            "validation_newton_iterations": newton_iterations,
            "validation_bracketing_iterations": VALIDATION_BRACKETING_ITERATIONS,
        }
    except Exception as e:
        warnings.warn(
            f"PropellerAnalysis failed at {case['rpm']:.0f} RPM, "
            f"{case['velocity_mph']:.2f} mph: {e}",
            stacklevel=2,
        )
        return {
            "rpm": case["rpm"],
            "velocity_mph": case["velocity_mph"],
            "velocity_mps": case["velocity_mps"],
            "J": case["J"],
            "Ct_asb": np.nan,
            "Cp_asb": np.nan,
            "eta_asb": np.nan,
            "thrust_N_asb": np.nan,
            "torque_Nm_asb": np.nan,
            "power_W_asb": np.nan,
            "max_abs_residual": np.nan,
            "converged": False,
            "min_analysis_confidence": np.nan,
            "min_root_loss_factor": np.nan,
            "min_finite_blade_loss_factor": np.nan,
            "geometry_cache_tag": geometry_cache_tag,
            "geometry_hub_over_R": float(propeller.hub_radius / propeller.radius),
            "geometry_station_count": len(propeller.radial_stations),
            "validation_radial_resolution": radial_resolution,
            "validation_newton_iterations": newton_iterations,
            "validation_bracketing_iterations": VALIDATION_BRACKETING_ITERATIONS,
        }


def _run_single_aerosandbox_validation_case_parallel(
    case: dict,
    radial_resolution: int,
    newton_iterations: int,
    model_size: str,
    residual_tolerance: float,
    geometry_cache_tag: str = GEOMETRY_CACHE_TAG,
    post_stall_confidence_blending: bool = False,
) -> dict:
    return _run_single_aerosandbox_validation_case(
        case=case,
        propeller=_PARALLEL_PROPELLER,
        radial_resolution=radial_resolution,
        newton_iterations=newton_iterations,
        model_size=model_size,
        residual_tolerance=residual_tolerance,
        geometry_cache_tag=geometry_cache_tag,
        post_stall_confidence_blending=post_stall_confidence_blending,
    )


def parse_apc_geometry_file(
    filename: Path = APC_GEOMETRY_FILE,
    inner_model_radius_ratio: float = None,
    max_modeled_thickness_ratio: float = MAX_MODELED_THICKNESS_RATIO,
    tip_thickness_to_root: bool = False,
) -> asb.Propeller:
    text = filename.read_text(errors="ignore")
    rows = []
    radius_in = None
    hub_radius_in = None
    blade_count = None
    file_inner_limit_radius_ratio = None

    for line in text.splitlines():
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.?", line)
        if len(numbers) == 14:
            values = [float(number) for number in numbers]
            rows.append(values)

        if "RADIUS:" in line:
            radius_in = float(re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.?", line)[0])
        elif "HUBRAD:" in line:
            hub_radius_in = float(
                re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.?", line)[0]
            )
        elif "BLADES:" in line:
            blade_count = int(float(re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.?", line)[0]))
        elif "INNER LIMIT" in line:
            file_inner_limit_radius_ratio = float(numbers[0])

    if radius_in is None or hub_radius_in is None or blade_count is None:
        raise ValueError(f"Could not parse propeller metadata from {filename}.")

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
    geometry = geometry.sort_values("station_in").reset_index(drop=True)
    tip_thickness_ratio = float(geometry["thickness_ratio"].iloc[-1])
    if inner_model_radius_ratio is None:
        inner_model_radius_ratio = (
            file_inner_limit_radius_ratio
            if file_inner_limit_radius_ratio is not None
            else hub_radius_in / radius_in
        )

    active_inner_radius_in = max(hub_radius_in, inner_model_radius_ratio * radius_in)
    if tip_thickness_to_root:
        if geometry["station_in"].iloc[0] <= active_inner_radius_in <= geometry[
            "station_in"
        ].iloc[-1]:
            first_active_station = geometry.loc[
                geometry["station_in"] >= active_inner_radius_in,
                "station_in",
            ].iloc[0]
            if first_active_station > active_inner_radius_in + 1e-9:
                inner_row = {"station_in": active_inner_radius_in}
                for column in geometry.columns:
                    if column == "station_in":
                        continue
                    inner_row[column] = np.interp(
                        active_inner_radius_in,
                        geometry["station_in"],
                        geometry[column],
                    )
                geometry = pd.concat(
                    [geometry, pd.DataFrame([inner_row])],
                    ignore_index=True,
                ).sort_values("station_in")

        geometry = geometry[geometry["station_in"] >= active_inner_radius_in].copy()
        if max_modeled_thickness_ratio is not None:
            tip_thickness_ratio = min(
                tip_thickness_ratio,
                max_modeled_thickness_ratio,
            )
        geometry["thickness_ratio"] = tip_thickness_ratio
    else:
        geometry = geometry[geometry["station_in"] >= active_inner_radius_in].copy()
        if max_modeled_thickness_ratio is not None:
            geometry = geometry[
                geometry["thickness_ratio"] <= max_modeled_thickness_ratio
            ].copy()

    geometry = geometry.reset_index(drop=True)
    active_hub_radius_in = max(hub_radius_in, geometry["station_in"].iloc[0])

    inch = 0.0254
    return asb.Propeller.from_tabulated_geometry(
        name=(
            "APC 12x8E, tip thickness retained to root"
            if tip_thickness_to_root
            else "APC 12x8E"
        ),
        r=geometry["station_in"].to_numpy() * inch,
        chord=geometry["chord_in"].to_numpy() * inch,
        twist=geometry["twist_deg"].to_numpy(),
        radius=radius_in * inch,
        hub_radius=active_hub_radius_in * inch,
        blade_count=blade_count,
        thickness=geometry["thickness_ratio"].to_numpy(),
        airfoil_distribution=[
            (1.10 / radius_in, "e63"),
            (4.39 / radius_in, "naca4412"),
        ],
        interpolation_method="linear",
    )


def parse_apc_performance_file(filename: Path = APC_PERFORMANCE_FILE) -> pd.DataFrame:
    rows = []
    rpm = None
    columns = [
        "velocity_mph",
        "J",
        "eta",
        "Ct",
        "Cp",
        "power_hp",
        "torque_in_lbf",
        "thrust_lbf",
        "power_W",
        "torque_Nm",
        "thrust_N",
        "thrust_per_power_g_per_W",
        "mach_tip",
        "Re_75",
        "FOM",
    ]

    for line in filename.read_text(errors="ignore").splitlines():
        rpm_match = re.search(r"PROP RPM\s*=\s*([0-9.]+)", line)
        if rpm_match:
            rpm = int(float(rpm_match.group(1)))
            continue

        if rpm is None:
            continue

        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.?", line)
        if len(numbers) >= len(columns):
            row = {name: float(value) for name, value in zip(columns, numbers)}
            row["rpm"] = rpm
            rows.append(row)

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError(f"No performance data found in {filename}.")

    data["velocity_mps"] = data["velocity_mph"] * 0.44704
    return data


def run_aerosandbox_validation(
    apc_data: pd.DataFrame,
    propeller: asb.Propeller,
    radial_resolution: int = VALIDATION_RADIAL_RESOLUTION,
    newton_iterations: int = VALIDATION_NEWTON_ITERATIONS,
    model_size: str = VALIDATION_MODEL_SIZE,
    residual_tolerance: float = VALIDATION_RESIDUAL_TOLERANCE,
    parallel_workers: int = 1,
    geometry_cache_tag: str = GEOMETRY_CACHE_TAG,
    post_stall_confidence_blending: bool = False,
) -> pd.DataFrame:
    case_records = apc_data.reset_index(drop=True).to_dict(orient="records")
    total = len(apc_data)
    rows = [None] * total
    parallel_workers = int(parallel_workers)

    if parallel_workers > 1:
        print(f"Running {total} AeroSandbox cases on {parallel_workers} workers.")
        with ProcessPoolExecutor(
            max_workers=parallel_workers,
            initializer=_initialize_parallel_validation_worker,
            initargs=(propeller,),
        ) as executor:
            futures = {
                executor.submit(
                    _run_single_aerosandbox_validation_case_parallel,
                    case,
                    radial_resolution,
                    newton_iterations,
                    model_size,
                    residual_tolerance,
                    geometry_cache_tag,
                    post_stall_confidence_blending,
                ): i
                for i, case in enumerate(case_records)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                i = futures[future]
                rows[i] = future.result()
                if completed % 25 == 0 or completed == 1 or completed == total:
                    print(
                        f"Finished case {completed}/{total}: "
                        f"{rows[i]['rpm']:.0f} RPM"
                    )
    else:
        for i, case in enumerate(case_records):
            if (i + 1) % 25 == 0 or i == 0 or i + 1 == total:
                print(f"Running case {i + 1}/{total}: {case['rpm']:.0f} RPM")

            rows[i] = _run_single_aerosandbox_validation_case(
                case=case,
                propeller=propeller,
                radial_resolution=radial_resolution,
                newton_iterations=newton_iterations,
                model_size=model_size,
                residual_tolerance=residual_tolerance,
                geometry_cache_tag=geometry_cache_tag,
                post_stall_confidence_blending=post_stall_confidence_blending,
            )

    model_data = pd.DataFrame(rows)
    validation = apc_data.merge(
        model_data,
        on=["rpm", "velocity_mph", "velocity_mps", "J"],
        how="left",
    )

    pairs = {
        "Ct": ("Ct", "Ct_asb"),
        "Cp": ("Cp", "Cp_asb"),
        "eta": ("eta", "eta_asb"),
        "thrust_N": ("thrust_N", "thrust_N_asb"),
        "torque_Nm": ("torque_Nm", "torque_Nm_asb"),
        "power_W": ("power_W", "power_W_asb"),
    }
    for name, (apc_column, asb_column) in pairs.items():
        validation[f"{name}_error"] = validation[asb_column] - validation[apc_column]
        validation[f"{name}_relative_error"] = validation[f"{name}_error"] / np.maximum(
            np.abs(validation[apc_column]), 1e-9
        )

    return validation


def validation_cache_is_current(
    validation: pd.DataFrame,
    geometry_cache_tag: str = GEOMETRY_CACHE_TAG,
) -> bool:
    expected_columns = {
        "geometry_cache_tag",
        "validation_radial_resolution",
        "validation_newton_iterations",
        "validation_bracketing_iterations",
    }
    if not expected_columns.issubset(validation.columns):
        return False

    return bool(
        validation["geometry_cache_tag"].eq(geometry_cache_tag).all()
        and validation["validation_radial_resolution"].eq(VALIDATION_RADIAL_RESOLUTION).all()
        and validation["validation_newton_iterations"].eq(VALIDATION_NEWTON_ITERATIONS).all()
        and validation["validation_bracketing_iterations"].eq(
            VALIDATION_BRACKETING_ITERATIONS
        ).all()
    )


def summarize_by_rpm(validation: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for rpm, data in validation.groupby("rpm"):
        summaries.append(
            {
                "rpm": rpm,
                "n_cases": len(data),
                "Ct_rmse": np.sqrt(np.nanmean(data["Ct_error"] ** 2)),
                "Cp_rmse": np.sqrt(np.nanmean(data["Cp_error"] ** 2)),
                "eta_rmse": np.sqrt(np.nanmean(data["eta_error"] ** 2)),
                "Ct_mean_error": np.nanmean(data["Ct_error"]),
                "Cp_mean_error": np.nanmean(data["Cp_error"]),
                "eta_mean_error": np.nanmean(data["eta_error"]),
                "n_nonconverged": int((~data["converged"].astype(bool)).sum()),
                "max_abs_residual": np.nanmax(data["max_abs_residual"]),
                "min_analysis_confidence": np.nanmin(data["min_analysis_confidence"]),
            }
        )
    return pd.DataFrame(summaries)


def filter_plot_rpm_range(validation: pd.DataFrame) -> pd.DataFrame:
    return validation[
        validation["rpm"].between(PLOT_RPM_MIN, PLOT_RPM_MAX, inclusive="both")
    ].copy()


def add_rpm_colorbar(fig, axes, rpms):
    norm = Normalize(vmin=min(rpms), vmax=max(rpms))
    mapper = ScalarMappable(norm=norm, cmap="viridis")
    mapper.set_array([])
    cbar = fig.colorbar(mapper, ax=axes, shrink=0.82, pad=0.015)
    cbar.set_label("RPM")
    return norm, mapper


def plot_quantity_maps(
    validation: pd.DataFrame,
    x_column: str,
    filename: Path,
    comparison_label: str = "APC 12x8E Validation Maps",
):
    quantities = [
        ("Ct", "Ct_asb", "$C_T$ [-]"),
        ("Cp", "Cp_asb", "$C_P$ [-]"),
        ("eta", "eta_asb", "Efficiency [-]"),
        ("thrust_N", "thrust_N_asb", "Thrust [N]"),
        ("torque_Nm", "torque_Nm_asb", "Torque [N-m]"),
        ("power_W", "power_W_asb", "Power [W]"),
    ]
    x_label = "Velocity [mph]" if x_column == "velocity_mph" else "Advance ratio $J$ [-]"
    rpms = sorted(validation["rpm"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    axes_flat = axes.flatten()
    norm, mapper = add_rpm_colorbar(fig, axes_flat, rpms)
    has_unconverged = (
        "converged" in validation.columns
        and (~validation["converged"].astype(bool)).any()
    )

    for ax, (apc_column, asb_column, y_label) in zip(axes_flat, quantities):
        for rpm, data in validation.groupby("rpm"):
            data = data.sort_values(x_column)
            color = mapper.cmap(norm(rpm))
            ax.plot(
                data[x_column],
                data[apc_column],
                color=color,
                linewidth=1.8,
                alpha=0.9,
            )
            ax.plot(
                data[x_column],
                data[asb_column],
                color=color,
                linewidth=1.4,
                linestyle="--",
                alpha=0.9,
            )
            if has_unconverged and "converged" in data.columns:
                unconverged = ~data["converged"].astype(bool)
                if unconverged.any():
                    ax.scatter(
                        data.loc[unconverged, x_column],
                        data.loc[unconverged, asb_column],
                        color=color,
                        marker="x",
                        s=18,
                        linewidths=0.9,
                        zorder=4,
                    )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if apc_column == "eta":
            ax.set_ylim(-0.25, 1.05)
        ax.grid(True, alpha=0.25)

    axes_flat[0].plot([], [], color="black", label="APC")
    axes_flat[0].plot([], [], color="black", linestyle="--", label="AeroSandbox")
    if has_unconverged:
        axes_flat[0].scatter([], [], color="black", marker="x", label="Non-converged")
    axes_flat[0].legend(loc="best")
    fig.suptitle(
        f"{comparison_label} "
        f"({PLOT_RPM_LABEL}, "
        f"{VALIDATION_RADIAL_RESOLUTION} radial stations, "
        f"{VALIDATION_NEWTON_ITERATIONS} Newton + "
        f"{VALIDATION_BRACKETING_ITERATIONS} bracket iterations)"
    )
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_error_maps(
    validation: pd.DataFrame,
    x_column: str,
    filename: Path,
    comparison_label: str = "APC 12x8E Validation Maps",
):
    quantities = [
        ("Ct_error", "$C_T$ error [-]"),
        ("Cp_error", "$C_P$ error [-]"),
        ("eta_error", "Efficiency error [-]"),
        ("thrust_N_error", "Thrust error [N]"),
        ("torque_Nm_error", "Torque error [N-m]"),
        ("power_W_error", "Power error [W]"),
    ]
    x_label = "Velocity [mph]" if x_column == "velocity_mph" else "Advance ratio $J$ [-]"

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for ax, (error_column, label) in zip(axes.flatten(), quantities):
        values = validation[error_column].to_numpy()
        finite_values = values[np.isfinite(values)]
        scale = np.nanpercentile(np.abs(finite_values), 95) if len(finite_values) else 1
        scale = max(scale, 1e-12)
        scatter = ax.scatter(
            validation[x_column],
            validation["rpm"],
            c=validation[error_column],
            s=18,
            cmap="coolwarm",
            vmin=-scale,
            vmax=scale,
            linewidths=0,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("RPM")
        ax.set_title(label)
        ax.grid(True, alpha=0.2)
        fig.colorbar(scatter, ax=ax, shrink=0.85)

    fig.suptitle(f"{comparison_label}: AeroSandbox - APC Errors ({PLOT_RPM_LABEL})")
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_residual_map(
    validation: pd.DataFrame,
    x_column: str,
    filename: Path,
    comparison_label: str = "APC 12x8E Validation Maps",
):
    x_label = "Velocity [mph]" if x_column == "velocity_mph" else "Advance ratio $J$ [-]"
    residual = validation["max_abs_residual"].to_numpy()
    finite_residual = residual[np.isfinite(residual)]
    vmax = np.nanpercentile(finite_residual, 95) if len(finite_residual) else 1
    vmax = max(vmax, VALIDATION_RESIDUAL_TOLERANCE)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    scatter = ax.scatter(
        validation[x_column],
        validation["rpm"],
        c=validation["max_abs_residual"],
        s=24,
        cmap="magma_r",
        vmin=0,
        vmax=vmax,
        linewidths=0,
    )
    unconverged = ~validation["converged"].astype(bool)
    if unconverged.any():
        ax.scatter(
            validation.loc[unconverged, x_column],
            validation.loc[unconverged, "rpm"],
            facecolors="none",
            edgecolors="black",
            s=52,
            linewidths=0.9,
            label="Non-converged",
        )
        ax.legend(loc="best")

    ax.set_xlabel(x_label)
    ax.set_ylabel("RPM")
    ax.set_title(f"{comparison_label}: Maximum Spanwise Residual ({PLOT_RPM_LABEL})")
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Max |residual|")
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_individual_rpm_quantity_comparisons(
    validation: pd.DataFrame,
    x_column: str,
    output_directory: Path,
    comparison_label: str = "APC 12x8E",
) -> None:
    quantities = [
        ("Ct", "Ct_asb", "$C_T$ [-]"),
        ("Cp", "Cp_asb", "$C_P$ [-]"),
        ("eta", "eta_asb", "Efficiency [-]"),
        ("thrust_N", "thrust_N_asb", "Thrust [N]"),
        ("torque_Nm", "torque_Nm_asb", "Torque [N-m]"),
        ("power_W", "power_W_asb", "Power [W]"),
    ]
    x_label = "Velocity [mph]" if x_column == "velocity_mph" else "Advance ratio $J$ [-]"
    filename_suffix = "velocity" if x_column == "velocity_mph" else "J"

    output_directory.mkdir(parents=True, exist_ok=True)

    for rpm, data in validation.groupby("rpm"):
        data = data.sort_values(x_column)
        rpm_integer = int(round(rpm))

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for ax, (apc_column, asb_column, y_label) in zip(axes.flatten(), quantities):
            ax.plot(
                data[x_column],
                data[apc_column],
                color="black",
                linewidth=2.0,
                label="APC",
            )
            ax.plot(
                data[x_column],
                data[asb_column],
                color="#1f77b4",
                linewidth=1.8,
                linestyle="--",
                label="AeroSandbox",
            )

            if "converged" in data.columns:
                unconverged = ~data["converged"].astype(bool)
                if unconverged.any():
                    ax.scatter(
                        data.loc[unconverged, x_column],
                        data.loc[unconverged, asb_column],
                        color="#d62728",
                        marker="x",
                        s=28,
                        linewidths=1.0,
                        zorder=4,
                        label="Non-converged",
                    )

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if apc_column == "eta":
                ax.set_ylim(-0.25, 1.05)
            ax.grid(True, alpha=0.25)

        axes.flat[0].legend(loc="best")
        fig.suptitle(
            f"{comparison_label}, {rpm_integer} RPM: APC vs AeroSandbox "
            f"({VALIDATION_RADIAL_RESOLUTION} radial stations, "
            f"{VALIDATION_NEWTON_ITERATIONS} Newton + "
            f"{VALIDATION_BRACKETING_ITERATIONS} bracket iterations)"
        )
        fig.savefig(
            output_directory
            / f"apc_12x8e_{rpm_integer:05d}rpm_quantities_vs_{filename_suffix}.png",
            dpi=220,
        )
        plt.close(fig)


def plot_geometry(
    propeller: asb.Propeller,
    filename: Path,
    comparison_label: str = "APC 12x8E",
) -> None:
    r = np.linspace(propeller.hub_radius / propeller.radius, 1.0, 300)
    inch = 0.0254

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.5, 9.0),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(r, propeller.chord(r) / inch, linewidth=2)
    axes[0].scatter(
        propeller.radial_stations,
        propeller.chord_distribution / inch,
        color="black",
        s=14,
        zorder=3,
    )
    axes[0].set_ylabel("Chord [in]")

    axes[1].plot(r, propeller.twist(r), linewidth=2)
    axes[1].scatter(
        propeller.radial_stations,
        propeller.twist_distribution,
        color="black",
        s=14,
        zorder=3,
    )
    axes[1].set_ylabel("Twist [deg]")

    axes[2].plot(r, propeller.thickness(r), linewidth=2)
    axes[2].scatter(
        propeller.radial_stations,
        propeller.thickness_distribution,
        color="black",
        s=14,
        zorder=3,
    )
    axes[2].set_ylabel("Modeled t/c [-]")
    axes[2].set_xlabel("Station r/R [-]")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(propeller.hub_radius / propeller.radius, 1.0)

    fig.suptitle(f"{comparison_label}: Modeled Geometry")
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_spanwise_mid_sweep(
    propeller: asb.Propeller,
    output_directory: Path,
    comparison_label: str = "APC 12x8E",
    post_stall_confidence_blending: bool = False,
) -> pd.DataFrame:
    apc_data = parse_apc_performance_file()
    candidates = apc_data.loc[apc_data["rpm"].eq(SPANWISE_RPM)].copy()
    case = candidates.iloc[(candidates["J"] - SPANWISE_TARGET_J).abs().argmin()]

    output = asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=float(case["velocity_mps"])),
        rpm=float(case["rpm"]),
        radial_resolution=SPANWISE_RADIAL_RESOLUTION,
        newton_iterations=VALIDATION_NEWTON_ITERATIONS,
        bracketing_iterations=VALIDATION_BRACKETING_ITERATIONS,
        residual_tolerance=1e-5,
        model_size=VALIDATION_MODEL_SIZE,
        include_post_stall_confidence_blending=post_stall_confidence_blending,
    ).run()

    def arr(key: str) -> np.ndarray:
        return np.asarray(output[key], dtype=float).reshape(-1)

    spanwise = pd.DataFrame(
        {
            "r_over_R": arr("r_over_R"),
            "r_m": arr("r"),
            "dr_m": arr("dr"),
            "chord_m": arr("chord"),
            "chord_in": arr("chord") / 0.0254,
            "twist_deg": arr("twist"),
            "thickness": arr("thickness"),
            "Re": arr("Re"),
            "mach": arr("mach"),
            "alpha_deg": arr("alpha"),
            "phi_deg": arr("phi"),
            "CL": arr("CL"),
            "CD": arr("CD"),
            "CM": arr("CM"),
            "CL_neuralfoil": arr("CL_neuralfoil"),
            "CD_neuralfoil": arr("CD_neuralfoil"),
            "CL_post_stall": arr("CL_post_stall"),
            "CD_post_stall": arr("CD_post_stall"),
            "post_stall_blend_fraction": arr("post_stall_blend_fraction"),
            "analysis_confidence": arr("analysis_confidence"),
            "Gamma_m2_s": arr("Gamma"),
            "tip_loss_factor": arr("tip_loss_factor"),
            "root_loss_factor": arr("root_loss_factor"),
            "finite_blade_loss_factor": arr("finite_blade_loss_factor"),
            "dT_N": arr("dT"),
            "dQ_Nm": arr("dQ"),
            "thrust_per_radius_N_m": arr("thrust_per_radius"),
            "torque_per_radius_N": arr("torque_per_radius"),
            "power_per_radius_W_m": float(output["omega"]) * arr("torque_per_radius"),
            "residual_m2_s": arr("residual"),
        }
    )
    spanwise.to_csv(
        output_directory / "apc_12x8e_spanwise_mid_sweep.csv",
        index=False,
    )

    quantities = [
        ("chord_in", spanwise["chord_in"], "Chord [in]"),
        ("twist_deg", spanwise["twist_deg"], "Twist [deg]"),
        ("thickness", spanwise["thickness"], "Thickness ratio [-]"),
        ("Re", spanwise["Re"] / 1e6, "Reynolds number [millions]"),
        ("alpha_deg", spanwise["alpha_deg"], "Angle of attack [deg]"),
        ("phi_deg", spanwise["phi_deg"], "Inflow angle [deg]"),
        (
            "finite_blade_loss_factor",
            spanwise["finite_blade_loss_factor"],
            "Finite-blade loss factor [-]",
        ),
        (
            "analysis_confidence",
            spanwise["analysis_confidence"],
            "NeuralFoil confidence [-]",
        ),
        (
            "post_stall_blend_fraction",
            spanwise["post_stall_blend_fraction"],
            "Post-stall blend fraction [-]",
        ),
        ("CL", spanwise["CL"], "$C_L$ [-]"),
        ("CD", spanwise["CD"], "$C_D$ [-]"),
        ("dTdr", spanwise["thrust_per_radius_N_m"], "dT/dr [N/m]"),
        ("dPdr", spanwise["power_per_radius_W_m"], "dP/dr [W/m]"),
        ("residual", spanwise["residual_m2_s"], "Residual [m^2/s]"),
    ]

    fig, axes = plt.subplots(
        7,
        2,
        figsize=(12, 17),
        sharex=True,
        constrained_layout=True,
    )
    for ax, (name, y, ylabel) in zip(axes.flatten(), quantities):
        if name == "CL":
            ax.plot(
                spanwise["r_over_R"],
                spanwise["CL_neuralfoil"],
                color="0.45",
                linestyle="--",
                linewidth=1.2,
                label="NeuralFoil",
            )
            ax.plot(
                spanwise["r_over_R"],
                spanwise["CL_post_stall"],
                color="#d62728",
                linestyle=":",
                linewidth=1.4,
                label="Post-stall",
            )
            ax.plot(
                spanwise["r_over_R"],
                y,
                color="#1f77b4",
                marker="o",
                markersize=3,
                label="Blended",
            )
            ax.legend(loc="best")
        elif name == "CD":
            ax.plot(
                spanwise["r_over_R"],
                spanwise["CD_neuralfoil"],
                color="0.45",
                linestyle="--",
                linewidth=1.2,
                label="NeuralFoil",
            )
            ax.plot(
                spanwise["r_over_R"],
                spanwise["CD_post_stall"],
                color="#d62728",
                linestyle=":",
                linewidth=1.4,
                label="Post-stall",
            )
            ax.plot(
                spanwise["r_over_R"],
                y,
                color="#1f77b4",
                marker="o",
                markersize=3,
                label="Blended",
            )
        elif name == "finite_blade_loss_factor":
            ax.plot(
                spanwise["r_over_R"],
                spanwise["tip_loss_factor"],
                color="0.45",
                linestyle="--",
                linewidth=1.2,
                label="Tip",
            )
            ax.plot(
                spanwise["r_over_R"],
                spanwise["root_loss_factor"],
                color="#d62728",
                linestyle=":",
                linewidth=1.4,
                label="Root",
            )
            ax.plot(
                spanwise["r_over_R"],
                y,
                color="#1f77b4",
                marker="o",
                markersize=3,
                label="Combined",
            )
            ax.legend(loc="best")
        else:
            ax.plot(spanwise["r_over_R"], y, marker="o", markersize=3)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    for ax in axes.flatten()[len(quantities):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Station $r/R$ [-]")
    fig.suptitle(
        f"{comparison_label}: Spanwise Quantities, {case['rpm']:.0f} RPM, "
        f"J={case['J']:.3f}, V={case['velocity_mph']:.1f} mph"
    )
    fig.savefig(output_directory / "apc_12x8e_spanwise_mid_sweep.png", dpi=220)
    plt.close(fig)

    return spanwise


def plot_post_stall_blend_section_polar(
    propeller: asb.Propeller,
    output_directory: Path,
    comparison_label: str = "APC 12x8E",
    post_stall_confidence_blending: bool = False,
) -> pd.DataFrame:
    output_directory.mkdir(parents=True, exist_ok=True)

    apc_data = parse_apc_performance_file()
    candidates = apc_data.loc[apc_data["rpm"].eq(SPANWISE_RPM)].copy()
    case = candidates.iloc[(candidates["J"] - SPANWISE_TARGET_J).abs().argmin()]

    operating_analysis = asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=float(case["velocity_mps"])),
        rpm=float(case["rpm"]),
        radial_resolution=SPANWISE_RADIAL_RESOLUTION,
        newton_iterations=VALIDATION_NEWTON_ITERATIONS,
        bracketing_iterations=VALIDATION_BRACKETING_ITERATIONS,
        residual_tolerance=1e-5,
        model_size=VALIDATION_MODEL_SIZE,
        include_post_stall_confidence_blending=post_stall_confidence_blending,
    )
    output = operating_analysis.run()

    def arr(key: str) -> np.ndarray:
        return np.asarray(output[key], dtype=float).reshape(-1)

    blend = arr("post_stall_blend_fraction")
    alpha = arr("alpha")
    if post_stall_confidence_blending and np.max(blend) > 0:
        station_index = int(np.argmax(blend))
    else:
        station_index = int(np.argmax(np.abs(alpha)))

    r_over_R = float(arr("r_over_R")[station_index])
    Re = float(arr("Re")[station_index])
    mach = float(arr("mach")[station_index])
    operating_alpha = float(alpha[station_index])
    airfoil = propeller.airfoil(r_over_R)

    section_analysis = asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=float(case["velocity_mps"])),
        rpm=float(case["rpm"]),
        radial_resolution=1,
        model_size=VALIDATION_MODEL_SIZE,
        include_post_stall_confidence_blending=post_stall_confidence_blending,
    )

    rows = []
    for alpha_deg in np.linspace(-30, 95, 251):
        aero = section_analysis._section_aero(
            airfoil=airfoil,
            alpha=float(alpha_deg),
            Re=Re,
            mach=mach,
            r_over_R=r_over_R,
        )
        rows.append(
            {
                "alpha_deg": float(alpha_deg),
                "CL": float(np.reshape(np.array(aero["CL"]), -1)[0]),
                "CD": float(np.reshape(np.array(aero["CD"]), -1)[0]),
                "CM": float(np.reshape(np.array(aero["CM"]), -1)[0]),
                "CL_neuralfoil": float(
                    np.reshape(np.array(aero["CL_neuralfoil"]), -1)[0]
                ),
                "CD_neuralfoil": float(
                    np.reshape(np.array(aero["CD_neuralfoil"]), -1)[0]
                ),
                "CL_post_stall": float(
                    np.reshape(np.array(aero["CL_post_stall"]), -1)[0]
                ),
                "CD_post_stall": float(
                    np.reshape(np.array(aero["CD_post_stall"]), -1)[0]
                ),
                "CL_post_stall_anchor": float(
                    np.reshape(np.array(aero["CL_post_stall_anchor"]), -1)[0]
                ),
                "CD_post_stall_anchor": float(
                    np.reshape(np.array(aero["CD_post_stall_anchor"]), -1)[0]
                ),
                "post_stall_blend_fraction": float(
                    np.reshape(np.array(aero["post_stall_blend_fraction"]), -1)[0]
                ),
                "analysis_confidence": float(
                    np.reshape(np.array(aero["analysis_confidence"]), -1)[0]
                ),
                "r_over_R": r_over_R,
                "Re": Re,
                "mach": mach,
                "operating_alpha_deg": operating_alpha,
                "operating_J": float(case["J"]),
                "operating_rpm": float(case["rpm"]),
                "operating_velocity_mph": float(case["velocity_mph"]),
            }
        )

    polar = pd.DataFrame(rows)
    polar_csv = output_directory / "apc_12x8e_post_stall_blend_section_polar.csv"
    polar.to_csv(polar_csv, index=False)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.0, 10.5),
        sharex=True,
        constrained_layout=True,
    )
    curve_styles = [
        ("neuralfoil", "NeuralFoil", "0.45", "--", 1.5),
        ("post_stall", "NF-anchored Viterna", "#d62728", ":", 1.8),
        ("", "Blended", "#1f77b4", "-", 2.0),
    ]

    for suffix, label, color, linestyle, linewidth in curve_styles:
        cl_column = "CL" if suffix == "" else f"CL_{suffix}"
        cd_column = "CD" if suffix == "" else f"CD_{suffix}"
        axes[0].plot(
            polar["alpha_deg"],
            polar[cl_column],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
        axes[1].plot(
            polar["alpha_deg"],
            polar[cd_column],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )

    axes[2].plot(
        polar["alpha_deg"],
        polar["analysis_confidence"],
        color="0.35",
        linestyle="--",
        linewidth=1.5,
        label="NeuralFoil confidence",
    )
    axes[2].plot(
        polar["alpha_deg"],
        polar["post_stall_blend_fraction"],
        color="#1f77b4",
        linewidth=2.0,
        label="Post-stall blend fraction",
    )

    for ax in axes:
        ax.axvline(
            operating_alpha,
            color="black",
            linewidth=1.0,
            alpha=0.75,
            label="Operating alpha" if ax is axes[0] else None,
        )
        ax.axvline(
            90,
            color="0.2",
            linewidth=0.9,
            linestyle="-.",
            alpha=0.6,
            label="90 deg" if ax is axes[0] else None,
        )
        ax.axvline(
            section_analysis.post_stall_alpha_stall,
            color="#d62728",
            linewidth=0.9,
            linestyle="-.",
            alpha=0.6,
            label="Stall anchor" if ax is axes[0] else None,
        )
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-30, 95)

    axes[0].set_ylabel("$C_L$ [-]")
    axes[1].set_ylabel("$C_D$ [-]")
    axes[2].set_ylabel("Weight / confidence [-]")
    axes[2].set_xlabel("Angle of attack [deg]")
    axes[2].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="best")
    axes[2].legend(loc="best")
    fig.suptitle(
        f"{comparison_label}: Post-Stall Section Blend at r/R={r_over_R:.3f}, "
        f"Re={Re / 1e3:.0f}k, Mach={mach:.3f}, "
        f"{case['rpm']:.0f} RPM, J={case['J']:.3f}"
    )
    polar_png = output_directory / "apc_12x8e_post_stall_blend_section_polar.png"
    fig.savefig(polar_png, dpi=220)
    plt.close(fig)

    print(f"Wrote section polar: {polar_png}")
    print(f"Wrote section polar data: {polar_csv}")
    return polar


def make_validation_maps(
    force_rerun: bool = False,
    parallel_workers: int = 1,
    tip_thickness_to_root: bool = False,
    post_stall_confidence_blending: bool = False,
    output_directory: Path = None,
):
    if output_directory is None:
        if post_stall_confidence_blending and tip_thickness_to_root:
            output_directory = TIP_THICKNESS_TO_ROOT_POST_STALL_BLEND_OUTPUTS
        elif post_stall_confidence_blending:
            output_directory = APC_COMPARISON_POST_STALL_BLEND_OUTPUTS
        elif tip_thickness_to_root:
            output_directory = TIP_THICKNESS_TO_ROOT_OUTPUTS
        else:
            output_directory = APC_COMPARISON_OUTPUTS

    output_directory.mkdir(parents=True, exist_ok=True)
    geometry_cache_tag = make_geometry_cache_tag(
        max_modeled_thickness_ratio=MAX_MODELED_THICKNESS_RATIO,
        tip_thickness_to_root=tip_thickness_to_root,
        post_stall_confidence_blending=post_stall_confidence_blending,
    )
    comparison_label = make_comparison_label(
        tip_thickness_to_root=tip_thickness_to_root,
        post_stall_confidence_blending=post_stall_confidence_blending,
    )

    validation_csv = output_directory / "apc_12x8e_validation.csv"
    summary_csv = output_directory / "apc_12x8e_summary_by_rpm.csv"

    if validation_csv.exists() and not force_rerun:
        validation = pd.read_csv(validation_csv)
        if not validation_cache_is_current(
            validation,
            geometry_cache_tag=geometry_cache_tag,
        ):
            propeller = parse_apc_geometry_file(
                tip_thickness_to_root=tip_thickness_to_root
            )
            apc_data = parse_apc_performance_file()
            validation = run_aerosandbox_validation(
                apc_data=apc_data,
                propeller=propeller,
                parallel_workers=parallel_workers,
                geometry_cache_tag=geometry_cache_tag,
                post_stall_confidence_blending=post_stall_confidence_blending,
            )
            validation.to_csv(validation_csv, index=False)
    else:
        propeller = parse_apc_geometry_file(
            tip_thickness_to_root=tip_thickness_to_root
        )
        apc_data = parse_apc_performance_file()
        validation = run_aerosandbox_validation(
            apc_data=apc_data,
            propeller=propeller,
            parallel_workers=parallel_workers,
            geometry_cache_tag=geometry_cache_tag,
            post_stall_confidence_blending=post_stall_confidence_blending,
        )
        validation.to_csv(validation_csv, index=False)

    propeller = parse_apc_geometry_file(tip_thickness_to_root=tip_thickness_to_root)
    plot_geometry(
        propeller=propeller,
        filename=output_directory / "apc_12x8e_geometry.png",
        comparison_label=comparison_label,
    )
    plot_spanwise_mid_sweep(
        propeller=propeller,
        output_directory=output_directory,
        comparison_label=comparison_label,
        post_stall_confidence_blending=post_stall_confidence_blending,
    )
    if post_stall_confidence_blending:
        plot_post_stall_blend_section_polar(
            propeller=propeller,
            output_directory=output_directory,
            comparison_label=comparison_label,
            post_stall_confidence_blending=post_stall_confidence_blending,
        )

    plot_validation = filter_plot_rpm_range(validation)

    filtered_validation_csv = (
        output_directory
        / f"apc_12x8e_validation_{PLOT_RPM_MIN}_{PLOT_RPM_MAX}rpm.csv"
    )
    plot_validation.to_csv(filtered_validation_csv, index=False)

    summary = summarize_by_rpm(plot_validation)
    summary.to_csv(summary_csv, index=False)

    plot_quantity_maps(
        validation=plot_validation,
        x_column="velocity_mph",
        filename=output_directory / "apc_12x8e_quantities_vs_velocity.png",
        comparison_label=comparison_label,
    )
    plot_quantity_maps(
        validation=plot_validation,
        x_column="J",
        filename=output_directory / "apc_12x8e_quantities_vs_J.png",
        comparison_label=comparison_label,
    )
    plot_error_maps(
        validation=plot_validation,
        x_column="velocity_mph",
        filename=output_directory / "apc_12x8e_errors_vs_velocity.png",
        comparison_label=comparison_label,
    )
    plot_error_maps(
        validation=plot_validation,
        x_column="J",
        filename=output_directory / "apc_12x8e_errors_vs_J.png",
        comparison_label=comparison_label,
    )
    plot_residual_map(
        validation=plot_validation,
        x_column="velocity_mph",
        filename=output_directory / "apc_12x8e_residuals_vs_velocity.png",
        comparison_label=comparison_label,
    )
    plot_residual_map(
        validation=plot_validation,
        x_column="J",
        filename=output_directory / "apc_12x8e_residuals_vs_J.png",
        comparison_label=comparison_label,
    )
    individual_output_directory = output_directory / "per_rpm_comparison"
    plot_individual_rpm_quantity_comparisons(
        validation=plot_validation,
        x_column="velocity_mph",
        output_directory=individual_output_directory,
        comparison_label=comparison_label,
    )
    plot_individual_rpm_quantity_comparisons(
        validation=plot_validation,
        x_column="J",
        output_directory=individual_output_directory,
        comparison_label=comparison_label,
    )

    print(f"Wrote validation table: {validation_csv}")
    print(f"Wrote filtered table:   {filtered_validation_csv}")
    print(f"Wrote RPM summary:      {summary_csv}")
    print(f"Wrote per-RPM plots:    {individual_output_directory}")
    print(summary.to_string(index=False))

    return validation, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Recompute every AeroSandbox operating point instead of reusing cached CSV data.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of worker processes to use for independent AeroSandbox cases.",
    )
    parser.add_argument(
        "--tip-thickness-to-root",
        action="store_true",
        help=(
            "Use the APC tip thickness ratio as the modeled thickness ratio at "
            "all stations, including the root, and write to a separate "
            "comparison folder."
        ),
    )
    parser.add_argument(
        "--enable-post-stall-blend",
        action="store_true",
        help=(
            "Enable the experimental NeuralFoil-confidence-based Viterna "
            "post-stall blend. By default, the validation uses NeuralFoil "
            "directly, including its post-stall model."
        ),
    )
    parser.add_argument(
        "--disable-post-stall-blend",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--only-post-stall-polar",
        action="store_true",
        help=(
            "Only regenerate the post-stall section polar diagnostic. Does not "
            "rerun validation sweeps."
        ),
    )
    args = parser.parse_args()
    post_stall_confidence_blending = (
        args.enable_post_stall_blend and not args.disable_post_stall_blend
    )

    if args.only_post_stall_polar:
        if post_stall_confidence_blending and args.tip_thickness_to_root:
            output_directory = TIP_THICKNESS_TO_ROOT_POST_STALL_BLEND_OUTPUTS
        elif post_stall_confidence_blending:
            output_directory = APC_COMPARISON_POST_STALL_BLEND_OUTPUTS
        elif args.tip_thickness_to_root:
            output_directory = TIP_THICKNESS_TO_ROOT_OUTPUTS
        else:
            output_directory = APC_COMPARISON_OUTPUTS

        propeller = parse_apc_geometry_file(
            tip_thickness_to_root=args.tip_thickness_to_root
        )
        plot_post_stall_blend_section_polar(
            propeller=propeller,
            output_directory=output_directory,
            comparison_label=make_comparison_label(
                tip_thickness_to_root=args.tip_thickness_to_root,
                post_stall_confidence_blending=post_stall_confidence_blending,
            ),
            post_stall_confidence_blending=post_stall_confidence_blending,
        )
        sys.exit()

    make_validation_maps(
        force_rerun=args.force_rerun,
        parallel_workers=args.parallel_workers,
        tip_thickness_to_root=args.tip_thickness_to_root,
        post_stall_confidence_blending=post_stall_confidence_blending,
    )
