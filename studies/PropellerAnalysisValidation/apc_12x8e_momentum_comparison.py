import argparse
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import aerosandbox as asb
import aerosandbox.numpy as np

from studies.PropellerAnalysisValidation.apc_12x8e_validation_maps import (
    OUTPUTS,
    VALIDATION_BRACKETING_ITERATIONS,
    VALIDATION_MODEL_SIZE,
    parse_apc_geometry_file,
    parse_apc_performance_file,
)


COMPARISON_OUTPUTS = OUTPUTS / "12x8e_momentum_bem"
DEFAULT_RPMS = [3000, 5000, 8000, 11000, 13000]
DEFAULT_MAX_CASES_PER_RPM = 13
DEFAULT_RADIAL_RESOLUTION = 14
DEFAULT_QPROP_LIKE_ITERATIONS = 7
DEFAULT_MOMENTUM_ITERATIONS = 14
DEFAULT_SPANWISE_RPM = 8000
DEFAULT_SPANWISE_VELOCITY_MPH = 40

MODEL_LABELS = {
    "qprop_like": "ASB QPROP-like",
    "bem_momentum": "ASB BEM momentum",
}
MODEL_COLORS = {
    "qprop_like": "#276FBF",
    "bem_momentum": "#D95F02",
}
MODEL_LINESTYLES = {
    "qprop_like": "-",
    "bem_momentum": "--",
}

QUANTITIES = [
    ("Ct", "Ct", "$C_T$"),
    ("Cp", "Cp", "$C_P$"),
    ("eta", "eta", "$\\eta$"),
    ("thrust_N", "thrust_N", "Thrust [N]"),
    ("torque_Nm", "torque_Nm", "Torque [N m]"),
    ("power_W", "power_W", "Power [W]"),
]


def select_comparison_cases(
    apc_data: pd.DataFrame,
    rpms: list[int],
    max_cases_per_rpm: int = DEFAULT_MAX_CASES_PER_RPM,
) -> pd.DataFrame:
    rows = []
    subset = apc_data[apc_data["rpm"].isin(rpms)].copy()
    for _, data in subset.groupby("rpm"):
        data = data.sort_values("J")
        if max_cases_per_rpm is not None and len(data) > max_cases_per_rpm:
            indices = np.linspace(0, len(data) - 1, max_cases_per_rpm)
            indices = sorted(set(int(round(index)) for index in indices))
            data = data.iloc[indices]
        rows.append(data)

    if not rows:
        raise ValueError(f"No APC cases found for RPMs {rpms}.")

    return (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates(["rpm", "J"])
        .sort_values(["rpm", "J"])
        .reset_index(drop=True)
    )


def _extract_model_result(result: dict, model_key: str) -> dict:
    return {
        f"Ct_{model_key}": float(result["Ct"]),
        f"Cp_{model_key}": float(result["Cp"]),
        f"eta_{model_key}": float(result["eta"]),
        f"thrust_N_{model_key}": float(result["thrust"]),
        f"torque_Nm_{model_key}": float(result["torque"]),
        f"power_W_{model_key}": float(result["power"]),
        f"max_abs_residual_{model_key}": float(result["max_abs_residual"]),
        f"converged_{model_key}": bool(result["converged"]),
        f"min_analysis_confidence_{model_key}": float(
            np.min(result["analysis_confidence"])
        ),
        f"min_finite_blade_loss_factor_{model_key}": float(
            np.min(result["finite_blade_loss_factor"])
        ),
    }


def _failed_model_result(model_key: str) -> dict:
    output = {}
    for quantity, _, _ in QUANTITIES:
        output[f"{quantity}_{model_key}"] = np.nan
    output[f"max_abs_residual_{model_key}"] = np.nan
    output[f"converged_{model_key}"] = False
    output[f"min_analysis_confidence_{model_key}"] = np.nan
    output[f"min_finite_blade_loss_factor_{model_key}"] = np.nan
    return output


def run_case(
    propeller: asb.Propeller,
    case: pd.Series,
    radial_resolution: int,
    qprop_like_iterations: int,
    momentum_iterations: int,
    model_size: str,
) -> dict:
    row = case.to_dict()
    op_point = asb.OperatingPoint(velocity=case.velocity_mps)

    try:
        qprop_like = asb.PropellerAnalysis(
            propeller=propeller,
            op_point=op_point,
            rpm=case.rpm,
            radial_resolution=radial_resolution,
            newton_iterations=qprop_like_iterations,
            bracketing_iterations=VALIDATION_BRACKETING_ITERATIONS,
            residual_tolerance=1e-4,
            model_size=model_size,
            include_root_loss=True,
        ).run()
        row.update(_extract_model_result(qprop_like, "qprop_like"))
    except Exception as e:
        warnings.warn(
            f"QPROP-like ASB solve failed at {case.rpm:.0f} RPM, "
            f"J={case.J:.3f}: {e}",
            stacklevel=2,
        )
        row.update(_failed_model_result("qprop_like"))

    try:
        bem_momentum = asb.PropellerMomentumAnalysis(
            propeller=propeller,
            op_point=op_point,
            rpm=case.rpm,
            radial_resolution=radial_resolution,
            newton_iterations=momentum_iterations,
            residual_tolerance=0.05,
            model_size=model_size,
            include_root_loss=True,
            relaxation=0.35,
        ).run()
        row.update(_extract_model_result(bem_momentum, "bem_momentum"))
    except Exception as e:
        warnings.warn(
            f"BEM/momentum ASB solve failed at {case.rpm:.0f} RPM, "
            f"J={case.J:.3f}: {e}",
            stacklevel=2,
        )
        row.update(_failed_model_result("bem_momentum"))

    return row


def add_error_columns(comparison: pd.DataFrame) -> pd.DataFrame:
    comparison = comparison.copy()
    for model_key in MODEL_LABELS:
        for quantity, apc_column, _ in QUANTITIES:
            model_column = f"{quantity}_{model_key}"
            error_column = f"{quantity}_error_{model_key}"
            relative_error_column = f"{quantity}_relative_error_{model_key}"
            reference_floor = 0.05 * np.nanmax(np.abs(comparison[apc_column]))
            comparison[error_column] = comparison[model_column] - comparison[apc_column]
            denominator = np.maximum(
                np.maximum(np.abs(comparison[apc_column]), reference_floor),
                1e-9,
            )
            comparison[relative_error_column] = comparison[error_column] / denominator
    return comparison


def summarize_errors(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_key, model_label in MODEL_LABELS.items():
        for quantity, apc_column, label in QUANTITIES:
            model_column = f"{quantity}_{model_key}"
            error = comparison[model_column] - comparison[apc_column]
            reference_floor = 0.05 * np.nanmax(np.abs(comparison[apc_column]))
            denominator = np.maximum(
                np.maximum(np.abs(comparison[apc_column]), reference_floor),
                1e-9,
            )
            relative_error = error / denominator
            rows.append(
                {
                    "model": model_label,
                    "model_key": model_key,
                    "quantity": quantity,
                    "label": label,
                    "n": int(np.sum(np.isfinite(error))),
                    "bias": float(np.nanmean(error)),
                    "mae": float(np.nanmean(np.abs(error))),
                    "rmse": float(np.sqrt(np.nanmean(error**2))),
                    "mean_abs_relative_error": float(
                        np.nanmean(np.abs(relative_error))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _comparison_cache_tag(
    rpms: list[int],
    max_cases_per_rpm: int,
    radial_resolution: int,
    qprop_like_iterations: int,
    momentum_iterations: int,
    model_size: str,
) -> str:
    return (
        f"rpms={','.join(str(rpm) for rpm in sorted(rpms))};"
        f"max_cases_per_rpm={max_cases_per_rpm};"
        f"radial_resolution={radial_resolution};"
        f"qprop_like_iterations={qprop_like_iterations};"
        f"momentum_iterations={momentum_iterations};"
        f"model_size={model_size}"
    )


def comparison_cache_is_current(
    comparison: pd.DataFrame,
    cache_tag: str,
) -> bool:
    return (
        "comparison_cache_tag" in comparison.columns
        and comparison["comparison_cache_tag"].eq(cache_tag).all()
    )


def plot_overview(comparison: pd.DataFrame, filename: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharex=True)
    axes = axes.flatten()
    rpms = sorted(comparison["rpm"].unique())
    colors = plt.cm.viridis(np.linspace(0.12, 0.90, len(rpms)))
    color_by_rpm = dict(zip(rpms, colors))

    for axis, (quantity, apc_column, label) in zip(axes, QUANTITIES):
        for rpm in rpms:
            data = comparison[comparison["rpm"] == rpm].sort_values("J")
            color = color_by_rpm[rpm]
            axis.plot(
                data["J"],
                data[apc_column],
                color=color,
                marker="o",
                linestyle="None",
                markersize=3.2,
                alpha=0.85,
            )
            for model_key in MODEL_LABELS:
                axis.plot(
                    data["J"],
                    data[f"{quantity}_{model_key}"],
                    color=color,
                    linestyle=MODEL_LINESTYLES[model_key],
                    linewidth=1.6,
                    alpha=0.95,
                )

        axis.set_ylabel(label)
        axis.grid(True, alpha=0.28)

    for axis in axes[-3:]:
        axis.set_xlabel("Advance ratio $J$")

    style_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="0.2",
            label="APC data",
            markersize=4,
        ),
        *[
            Line2D(
                [0],
                [0],
                color="0.2",
                linestyle=MODEL_LINESTYLES[model_key],
                label=MODEL_LABELS[model_key],
                linewidth=1.8,
            )
            for model_key in MODEL_LABELS
        ],
    ]
    rpm_handles = [
        Line2D([0], [0], color=color_by_rpm[rpm], linewidth=3, label=f"{rpm:.0f} RPM")
        for rpm in rpms
    ]
    fig.legend(
        handles=style_handles + rpm_handles,
        loc="lower center",
        ncol=min(8, len(style_handles) + len(rpm_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle("APC 12x8E: APC Data vs. QPROP-like Closure vs. BEM Momentum")
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_per_rpm(comparison: pd.DataFrame, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    filenames = []
    for rpm, data in comparison.groupby("rpm"):
        data = data.sort_values("J")
        fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), sharex=True)
        axes = axes.flatten()
        for axis, (quantity, apc_column, label) in zip(axes, QUANTITIES):
            axis.plot(
                data["J"],
                data[apc_column],
                color="0.1",
                marker="o",
                linestyle="None",
                label="APC data",
                markersize=4,
            )
            for model_key, model_label in MODEL_LABELS.items():
                axis.plot(
                    data["J"],
                    data[f"{quantity}_{model_key}"],
                    color=MODEL_COLORS[model_key],
                    linestyle=MODEL_LINESTYLES[model_key],
                    linewidth=1.9,
                    label=model_label,
                )
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.28)

        for axis in axes[-3:]:
            axis.set_xlabel("Advance ratio $J$")
        axes[0].legend(frameon=False)
        fig.suptitle(f"APC 12x8E Momentum-Closure Comparison, {rpm:.0f} RPM")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        filename = directory / f"apc_12x8e_momentum_comparison_{rpm:.0f}rpm.png"
        fig.savefig(filename, dpi=200)
        plt.close(fig)
        filenames.append(filename)
    return filenames


def plot_error_summary(summary: pd.DataFrame, filename: Path) -> None:
    labels = [label for _, _, label in QUANTITIES]
    x = np.arange(len(labels))
    width = 0.34

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    for i, model_key in enumerate(MODEL_LABELS):
        model_summary = summary[summary["model_key"] == model_key].set_index(
            "quantity"
        )
        offset = (i - 0.5) * width
        axes[0].bar(
            x + offset,
            [model_summary.loc[quantity, "mean_abs_relative_error"] for quantity, _, _ in QUANTITIES],
            width=width,
            label=MODEL_LABELS[model_key],
            color=MODEL_COLORS[model_key],
            alpha=0.85,
        )
        axes[1].bar(
            x + offset,
            [model_summary.loc[quantity, "rmse"] for quantity, _, _ in QUANTITIES],
            width=width,
            label=MODEL_LABELS[model_key],
            color=MODEL_COLORS[model_key],
            alpha=0.85,
        )

    axes[0].set_ylabel("Mean abs. relative error")
    axes[1].set_ylabel("RMSE, native units")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.28)
    axes[0].legend(frameon=False)
    fig.suptitle("APC 12x8E Comparison Error Summary")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def choose_spanwise_case(
    comparison: pd.DataFrame,
    rpm: int = DEFAULT_SPANWISE_RPM,
    velocity_mph: float = DEFAULT_SPANWISE_VELOCITY_MPH,
) -> pd.Series:
    data = comparison.copy()
    data["spanwise_distance"] = (
        (data["rpm"] - rpm).abs() / max(rpm, 1)
        + (data["velocity_mph"] - velocity_mph).abs() / max(velocity_mph, 1)
    )
    return data.loc[data["spanwise_distance"].idxmin()]


def plot_spanwise_comparison(
    propeller: asb.Propeller,
    case: pd.Series,
    filename: Path,
    radial_resolution: int,
    qprop_like_iterations: int,
    momentum_iterations: int,
    model_size: str,
) -> None:
    op_point = asb.OperatingPoint(velocity=case.velocity_mps)
    qprop_like = asb.PropellerAnalysis(
        propeller=propeller,
        op_point=op_point,
        rpm=case.rpm,
        radial_resolution=radial_resolution,
        newton_iterations=qprop_like_iterations,
        bracketing_iterations=VALIDATION_BRACKETING_ITERATIONS,
        residual_tolerance=1e-4,
        model_size=model_size,
        include_root_loss=True,
    ).run()
    bem_momentum = asb.PropellerMomentumAnalysis(
        propeller=propeller,
        op_point=op_point,
        rpm=case.rpm,
        radial_resolution=radial_resolution,
        newton_iterations=momentum_iterations,
        residual_tolerance=0.05,
        model_size=model_size,
        include_root_loss=True,
        relaxation=0.35,
    ).run()

    omega = case.rpm * 2 * np.pi / 60
    qprop_axial_induced = qprop_like["Wa"] - case.velocity_mps
    qprop_tangential_induced = omega * qprop_like["r"] - qprop_like["Wt"]

    distributions = [
        ("alpha", "Angle of attack [deg]", qprop_like["alpha"], bem_momentum["alpha"]),
        ("CL", "$C_L$", qprop_like["CL"], bem_momentum["CL"]),
        ("CD", "$C_D$", qprop_like["CD"], bem_momentum["CD"]),
        (
            "analysis_confidence",
            "NeuralFoil confidence",
            qprop_like["analysis_confidence"],
            bem_momentum["analysis_confidence"],
        ),
        (
            "axial_induced_velocity",
            "Axial induced velocity [m/s]",
            qprop_axial_induced,
            bem_momentum["axial_induced_velocity"],
        ),
        (
            "tangential_induced_velocity",
            "Tangential induced velocity [m/s]",
            qprop_tangential_induced,
            bem_momentum["tangential_induced_velocity"],
        ),
        (
            "thrust_per_radius",
            "dT/dr [N/m]",
            qprop_like["thrust_per_radius"],
            bem_momentum["thrust_per_radius"],
        ),
        (
            "torque_per_radius",
            "dQ/dr [N]",
            qprop_like["torque_per_radius"],
            bem_momentum["torque_per_radius"],
        ),
        (
            "finite_blade_loss_factor",
            "Finite-blade loss",
            qprop_like["finite_blade_loss_factor"],
            bem_momentum["finite_blade_loss_factor"],
        ),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13.0, 9.2), sharex=True)
    axes = axes.flatten()
    for axis, (_, ylabel, qprop_values, bem_values) in zip(axes, distributions):
        axis.plot(
            qprop_like["r_over_R"],
            qprop_values,
            color=MODEL_COLORS["qprop_like"],
            linestyle=MODEL_LINESTYLES["qprop_like"],
            linewidth=1.9,
            label=MODEL_LABELS["qprop_like"],
        )
        axis.plot(
            bem_momentum["r_over_R"],
            bem_values,
            color=MODEL_COLORS["bem_momentum"],
            linestyle=MODEL_LINESTYLES["bem_momentum"],
            linewidth=1.9,
            label=MODEL_LABELS["bem_momentum"],
        )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.28)

    for axis in axes[-3:]:
        axis.set_xlabel("$r/R$")
    axes[0].legend(frameon=False)
    fig.suptitle(
        "APC 12x8E Spanwise Comparison, "
        f"{case.rpm:.0f} RPM, {case.velocity_mph:.1f} mph, J={case.J:.3f}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def run_momentum_comparison(
    rpms: list[int],
    max_cases_per_rpm: int,
    radial_resolution: int,
    qprop_like_iterations: int,
    momentum_iterations: int,
    model_size: str,
    force_rerun: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    COMPARISON_OUTPUTS.mkdir(parents=True, exist_ok=True)
    csv_file = COMPARISON_OUTPUTS / "apc_12x8e_momentum_comparison.csv"
    summary_file = COMPARISON_OUTPUTS / "apc_12x8e_momentum_comparison_summary.csv"
    cache_tag = _comparison_cache_tag(
        rpms=rpms,
        max_cases_per_rpm=max_cases_per_rpm,
        radial_resolution=radial_resolution,
        qprop_like_iterations=qprop_like_iterations,
        momentum_iterations=momentum_iterations,
        model_size=model_size,
    )

    propeller = parse_apc_geometry_file()

    if csv_file.exists() and not force_rerun:
        comparison = pd.read_csv(csv_file)
        if not comparison_cache_is_current(comparison, cache_tag):
            comparison = None
    else:
        comparison = None

    if comparison is None:
        apc_data = parse_apc_performance_file()
        cases = select_comparison_cases(
            apc_data=apc_data,
            rpms=rpms,
            max_cases_per_rpm=max_cases_per_rpm,
        )
        rows = []
        total = len(cases)
        for i, (_, case) in enumerate(cases.iterrows(), start=1):
            print(
                f"Momentum comparison {i}/{total}: "
                f"{case.rpm:.0f} RPM, J={case.J:.3f}",
                flush=True,
            )
            rows.append(
                run_case(
                    propeller=propeller,
                    case=case,
                    radial_resolution=radial_resolution,
                    qprop_like_iterations=qprop_like_iterations,
                    momentum_iterations=momentum_iterations,
                    model_size=model_size,
                )
            )
            rows[-1]["comparison_cache_tag"] = cache_tag
            rows[-1]["comparison_radial_resolution"] = radial_resolution
            rows[-1]["comparison_qprop_like_iterations"] = qprop_like_iterations
            rows[-1]["comparison_momentum_iterations"] = momentum_iterations
            rows[-1]["comparison_model_size"] = model_size
        comparison = add_error_columns(pd.DataFrame(rows))
        comparison.to_csv(csv_file, index=False)

    summary = summarize_errors(comparison)
    summary.to_csv(summary_file, index=False)

    overview_file = COMPARISON_OUTPUTS / "apc_12x8e_momentum_comparison_overview.png"
    error_file = COMPARISON_OUTPUTS / "apc_12x8e_momentum_comparison_errors.png"
    spanwise_file = COMPARISON_OUTPUTS / "apc_12x8e_momentum_spanwise_comparison.png"
    per_rpm_directory = COMPARISON_OUTPUTS / "per_rpm"

    plot_overview(comparison, overview_file)
    per_rpm_files = plot_per_rpm(comparison, per_rpm_directory)
    plot_error_summary(summary, error_file)
    plot_spanwise_comparison(
        propeller=propeller,
        case=choose_spanwise_case(comparison),
        filename=spanwise_file,
        radial_resolution=max(radial_resolution, 24),
        qprop_like_iterations=qprop_like_iterations,
        momentum_iterations=momentum_iterations,
        model_size=model_size,
    )

    files = {
        "csv": csv_file,
        "summary_csv": summary_file,
        "overview": overview_file,
        "errors": error_file,
        "spanwise": spanwise_file,
        "per_rpm_directory": per_rpm_directory,
    }
    if per_rpm_files:
        files["first_per_rpm"] = per_rpm_files[0]

    return comparison, summary, files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare APC 12x8E data against ASB QPROP-like and BEM momentum closures."
    )
    parser.add_argument("--force", action="store_true", help="Rerun analyses.")
    parser.add_argument(
        "--rpms",
        nargs="+",
        type=int,
        default=DEFAULT_RPMS,
        help="RPMs to compare.",
    )
    parser.add_argument(
        "--max-cases-per-rpm",
        type=int,
        default=DEFAULT_MAX_CASES_PER_RPM,
        help="Downsample each RPM to this many J stations.",
    )
    parser.add_argument(
        "--radial-resolution",
        type=int,
        default=DEFAULT_RADIAL_RESOLUTION,
        help="Blade elements for integrated comparison cases.",
    )
    parser.add_argument(
        "--qprop-like-iterations",
        type=int,
        default=DEFAULT_QPROP_LIKE_ITERATIONS,
        help="Newton iterations for ASB QPROP-like closure.",
    )
    parser.add_argument(
        "--momentum-iterations",
        type=int,
        default=DEFAULT_MOMENTUM_ITERATIONS,
        help="Fixed-point iterations for ASB BEM momentum closure.",
    )
    parser.add_argument(
        "--model-size",
        default=VALIDATION_MODEL_SIZE,
        help="NeuralFoil model size.",
    )
    args = parser.parse_args()

    comparison, summary, files = run_momentum_comparison(
        rpms=args.rpms,
        max_cases_per_rpm=args.max_cases_per_rpm,
        radial_resolution=args.radial_resolution,
        qprop_like_iterations=args.qprop_like_iterations,
        momentum_iterations=args.momentum_iterations,
        model_size=args.model_size,
        force_rerun=args.force,
    )

    print("\nError summary:")
    print(
        summary.pivot(
            index="quantity",
            columns="model",
            values="mean_abs_relative_error",
        ).round(4)
    )
    print("\nGenerated files:")
    for label, file in files.items():
        print(f"{label}: {file}")


if __name__ == "__main__":
    main()
