from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import aerosandbox as asb
import aerosandbox.numpy as np

from studies.PropellerAnalysisValidation.apc_12x8e_validation_maps import (
    OUTPUTS,
    VALIDATION_BRACKETING_ITERATIONS,
    VALIDATION_MODEL_SIZE,
    VALIDATION_NEWTON_ITERATIONS,
    parse_apc_geometry_file,
    parse_apc_performance_file,
)


ROOT = Path(__file__).resolve().parent
UIUC_DATA = ROOT / "data" / "uiuc_volume4_apce_12x8"
OUTPUT_DIR = OUTPUTS / "uiuc_experiment_comparison"
RADIAL_RESOLUTION = 16
PARALLEL_WORKERS = 4
INCH = 0.0254
UIUC_VOLUME4_URL = "https://m-selig.ae.illinois.edu/props/volume-4/propDB-volume-4.html"
UIUC_PAPER_URL = (
    "https://m-selig.ae.illinois.edu/pubs/"
    "DantskerCaccamoDetersSelig-2022-AIAA-Paper-2022-4058-Propeller-Testing.pdf"
)


def parse_uiuc_experiment_files(data_directory: Path = UIUC_DATA) -> pd.DataFrame:
    rows = []

    for filename in sorted(data_directory.glob("apce_12x8_*.txt")):
        text = filename.read_text(errors="ignore")
        numeric_rows = []
        for line in text.splitlines():
            values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.?", line)
            if not values:
                continue
            try:
                numeric_rows.append([float(value) for value in values])
            except ValueError:
                continue

        if "static" in filename.name:
            for values in numeric_rows:
                if len(values) == 3:
                    rpm, CT, CP = values
                    rows.append(
                        {
                            "source_file": filename.name,
                            "run_type": "static",
                            "rpm": rpm,
                            "J": 0.0,
                            "Ct_exp": CT,
                            "Cp_exp": CP,
                            "eta_exp": 0.0,
                        }
                    )
        else:
            rpm_match = re.search(r"_(\d{4})\.txt$", filename.name)
            if rpm_match is None:
                warnings.warn(f"Could not infer RPM from {filename.name}.")
                continue
            rpm = float(rpm_match.group(1))
            for values in numeric_rows:
                if len(values) == 4:
                    J, CT, CP, eta = values
                    rows.append(
                        {
                            "source_file": filename.name,
                            "run_type": "wind_on",
                            "rpm": rpm,
                            "J": J,
                            "Ct_exp": CT,
                            "Cp_exp": CP,
                            "eta_exp": eta,
                        }
                    )

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError(f"No UIUC data rows found in {data_directory}.")

    data = (
        data.drop_duplicates(["source_file", "rpm", "J", "Ct_exp", "Cp_exp", "eta_exp"])
        .sort_values(["run_type", "rpm", "J"])
        .reset_index(drop=True)
    )
    diameter = 12 * INCH
    data["velocity_mps"] = data["J"] * (data["rpm"] / 60) * diameter
    data["velocity_mph"] = data["velocity_mps"] / 0.44704
    return data


def interpolate_apc_map_to_experiment(
    experiment: pd.DataFrame,
    apc_map: pd.DataFrame,
) -> pd.DataFrame:
    apc_by_rpm = {
        float(rpm): data.sort_values("J").drop_duplicates("J")
        for rpm, data in apc_map.groupby("rpm")
    }
    rpm_values = np.array(sorted(apc_by_rpm.keys()), dtype=float)
    columns = {
        "Ct_apc_interp": "Ct",
        "Cp_apc_interp": "Cp",
        "eta_apc_interp": "eta",
        "thrust_N_apc_interp": "thrust_N",
        "power_W_apc_interp": "power_W",
    }

    def interp_at_rpm(rpm: float, J: float, source_column: str):
        data = apc_by_rpm[rpm]
        if J < data["J"].min() or J > data["J"].max():
            return np.nan
        return float(np.interp(J, data["J"], data[source_column]))

    rows = []
    for _, row in experiment.iterrows():
        rpm = float(row["rpm"])
        J = float(row["J"])
        out = row.to_dict()

        if rpm < rpm_values[0] or rpm > rpm_values[-1]:
            for output_column in columns:
                out[output_column] = np.nan
            rows.append(out)
            continue

        upper_index = int(np.searchsorted(rpm_values, rpm, side="left"))
        if upper_index < len(rpm_values) and rpm_values[upper_index] == rpm:
            rpm_low = rpm_high = rpm_values[upper_index]
        else:
            rpm_low = rpm_values[max(upper_index - 1, 0)]
            rpm_high = rpm_values[min(upper_index, len(rpm_values) - 1)]

        for output_column, source_column in columns.items():
            low_value = interp_at_rpm(rpm_low, J, source_column)
            high_value = interp_at_rpm(rpm_high, J, source_column)
            if np.isnan(low_value) or np.isnan(high_value):
                out[output_column] = np.nan
            elif rpm_low == rpm_high:
                out[output_column] = low_value
            else:
                rpm_fraction = (rpm - rpm_low) / (rpm_high - rpm_low)
                out[output_column] = (
                    (1 - rpm_fraction) * low_value + rpm_fraction * high_value
                )
        rows.append(out)

    return pd.DataFrame(rows)


def _run_asb_case(case: dict, tip_thickness_to_root: bool) -> dict:
    propeller = parse_apc_geometry_file(tip_thickness_to_root=tip_thickness_to_root)
    try:
        result = asb.PropellerAnalysis(
            propeller=propeller,
            op_point=asb.OperatingPoint(velocity=float(case["velocity_mps"])),
            rpm=float(case["rpm"]),
            radial_resolution=RADIAL_RESOLUTION,
            newton_iterations=VALIDATION_NEWTON_ITERATIONS,
            bracketing_iterations=VALIDATION_BRACKETING_ITERATIONS,
            model_size=VALIDATION_MODEL_SIZE,
            residual_tolerance=1e-4,
            include_post_stall_confidence_blending=False,
        ).run()

        return {
            "run_type": case["run_type"],
            "rpm": case["rpm"],
            "J": case["J"],
            "tip_thickness_to_root": tip_thickness_to_root,
            "Ct_asb": float(result["Ct"]),
            "Cp_asb": float(result["Cp"]),
            "eta_asb": float(result["eta"]),
            "thrust_N_asb": float(result["thrust"]),
            "power_W_asb": float(result["power"]),
            "max_abs_residual": float(result["max_abs_residual"]),
            "converged": bool(result["converged"]),
            "min_analysis_confidence": float(np.min(result["analysis_confidence"])),
            "min_root_loss_factor": float(np.min(result["root_loss_factor"])),
            "min_finite_blade_loss_factor": float(
                np.min(result["finite_blade_loss_factor"])
            ),
        }
    except Exception as e:
        warnings.warn(
            f"ASB failed at {case['rpm']:.0f} RPM, J={case['J']:.3f}: {e}",
            stacklevel=2,
        )
        return {
            "run_type": case["run_type"],
            "rpm": case["rpm"],
            "J": case["J"],
            "tip_thickness_to_root": tip_thickness_to_root,
            "Ct_asb": np.nan,
            "Cp_asb": np.nan,
            "eta_asb": np.nan,
            "thrust_N_asb": np.nan,
            "power_W_asb": np.nan,
            "max_abs_residual": np.nan,
            "converged": False,
            "min_analysis_confidence": np.nan,
            "min_root_loss_factor": np.nan,
            "min_finite_blade_loss_factor": np.nan,
        }


def run_asb_at_experiment_points(
    experiment: pd.DataFrame,
    parallel_workers: int = PARALLEL_WORKERS,
) -> pd.DataFrame:
    case_records = experiment.to_dict(orient="records")
    rows = []
    total = 2 * len(case_records)

    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = []
        for case in case_records:
            for tip_thickness_to_root in [False, True]:
                futures.append(
                    executor.submit(_run_asb_case, case, tip_thickness_to_root)
                )

        for i, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if i == 1 or i % 20 == 0 or i == total:
                print(f"Finished ASB UIUC case {i}/{total}", flush=True)

    return pd.DataFrame(rows)


def add_errors(comparison: pd.DataFrame) -> pd.DataFrame:
    for reference in ["apc_interp", "asb"]:
        for quantity in ["Ct", "Cp", "eta"]:
            comparison[f"{quantity}_{reference}_error_vs_exp"] = (
                comparison[f"{quantity}_{reference}"] - comparison[f"{quantity}_exp"]
            )
    return comparison


def make_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_type, run_data in comparison.groupby("run_type"):
        for model_name, model_data in run_data.groupby("model_name"):
            row = {
                "run_type": run_type,
                "model_name": model_name,
                "n_cases": len(model_data),
                "n_nonconverged": int((~model_data["converged"].astype(bool)).sum()),
                "max_abs_residual": np.nanmax(model_data["max_abs_residual"]),
                "min_root_loss_factor": np.nanmin(model_data["min_root_loss_factor"]),
            }
            for reference in ["apc_interp", "asb"]:
                for quantity in ["Ct", "Cp", "eta"]:
                    errors = model_data[f"{quantity}_{reference}_error_vs_exp"]
                    row[f"{quantity}_{reference}_rmse_vs_exp"] = np.sqrt(
                        np.nanmean(errors**2)
                    )
                    row[f"{quantity}_{reference}_mean_error_vs_exp"] = np.nanmean(
                        errors
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_wind_on_by_rpm(comparison: pd.DataFrame, output_dir: Path) -> None:
    wind_on = comparison[comparison["run_type"].eq("wind_on")].copy()
    rpms = sorted(wind_on["rpm"].unique())
    n_rows = len(rpms)
    quantities = [
        ("Ct", "$C_T$ [-]"),
        ("Cp", "$C_P$ [-]"),
        ("eta", "Efficiency [-]"),
    ]

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(13.5, 2.35 * n_rows),
        sharex=False,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    for row_index, rpm in enumerate(rpms):
        rpm_data = wind_on[wind_on["rpm"].eq(rpm)].sort_values("J")
        direct = rpm_data[rpm_data["model_name"].eq("ASB direct + root")]
        tip = rpm_data[rpm_data["model_name"].eq("ASB tip-t/c + root")]
        for ax, (quantity, ylabel) in zip(axes[row_index, :], quantities):
            ax.plot(
                direct["J"],
                direct[f"{quantity}_exp"],
                color="black",
                marker="o",
                linestyle="None",
                markersize=4,
                label="UIUC experiment",
            )
            ax.plot(
                direct["J"],
                direct[f"{quantity}_apc_interp"],
                color="black",
                linestyle="-",
                linewidth=1.6,
                label="APC map interp.",
            )
            ax.plot(
                direct["J"],
                direct[f"{quantity}_asb"],
                color="#1f77b4",
                linestyle="--",
                linewidth=1.5,
                label="ASB direct + root",
            )
            ax.plot(
                tip["J"],
                tip[f"{quantity}_asb"],
                color="#2ca02c",
                linestyle="-.",
                linewidth=1.5,
                label="ASB tip-t/c + root",
            )
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"{rpm:.0f} RPM")
            if quantity == "eta":
                ax.set_ylim(-0.25, 1.05)

    for ax in axes[-1, :]:
        ax.set_xlabel("Advance ratio $J$ [-]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.008),
        ncol=4,
    )
    fig.savefig(output_dir / "uiuc_wind_on_by_rpm.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_static(comparison: pd.DataFrame, output_dir: Path) -> None:
    static = comparison[comparison["run_type"].eq("static")].sort_values("rpm")
    direct = static[static["model_name"].eq("ASB direct + root")]
    tip = static[static["model_name"].eq("ASB tip-t/c + root")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, quantity, ylabel in [
        (axes[0], "Ct", "$C_T$ [-]"),
        (axes[1], "Cp", "$C_P$ [-]"),
    ]:
        ax.plot(
            direct["rpm"],
            direct[f"{quantity}_exp"],
            color="black",
            marker="o",
            linestyle="None",
            label="UIUC experiment",
        )
        ax.plot(
            direct["rpm"],
            direct[f"{quantity}_apc_interp"],
            color="black",
            linewidth=1.7,
            label="APC map interp.",
        )
        ax.plot(
            direct["rpm"],
            direct[f"{quantity}_asb"],
            color="#1f77b4",
            linestyle="--",
            linewidth=1.6,
            label="ASB direct + root",
        )
        ax.plot(
            tip["rpm"],
            tip[f"{quantity}_asb"],
            color="#2ca02c",
            linestyle="-.",
            linewidth=1.6,
            label="ASB tip-t/c + root",
        )
        ax.set_xlabel("RPM")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="best")
    fig.suptitle("APC 12x8E Static: UIUC Experiment vs APC Map and AeroSandbox")
    fig.savefig(output_dir / "uiuc_static_vs_rpm.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parity(comparison: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    quantities = [
        ("Ct", "$C_T$ [-]"),
        ("Cp", "$C_P$ [-]"),
        ("eta", "Efficiency [-]"),
    ]
    for col_index, (quantity, label) in enumerate(quantities):
        ax = axes[0, col_index]
        direct = comparison[comparison["model_name"].eq("ASB direct + root")]
        ax.scatter(
            direct[f"{quantity}_exp"],
            direct[f"{quantity}_apc_interp"],
            s=18,
            color="black",
            alpha=0.65,
            label="APC map interp.",
        )
        for model_name, color, marker in [
            ("ASB direct + root", "#1f77b4", "o"),
            ("ASB tip-t/c + root", "#2ca02c", "^"),
        ]:
            data = comparison[comparison["model_name"].eq(model_name)]
            ax.scatter(
                data[f"{quantity}_exp"],
                data[f"{quantity}_asb"],
                s=18,
                color=color,
                marker=marker,
                alpha=0.65,
                label=model_name,
            )
        values = pd.concat(
            [
                comparison[f"{quantity}_exp"],
                comparison[f"{quantity}_apc_interp"],
                comparison[f"{quantity}_asb"],
            ]
        ).dropna()
        lim_min = float(values.min())
        lim_max = float(values.max())
        padding = 0.05 * (lim_max - lim_min + 1e-12)
        ax.plot(
            [lim_min - padding, lim_max + padding],
            [lim_min - padding, lim_max + padding],
            color="0.5",
            linewidth=1,
        )
        ax.set_xlabel(f"UIUC experiment {label}")
        ax.set_ylabel(f"Model/map {label}")
        ax.grid(True, alpha=0.25)
        if col_index == 0:
            ax.legend(fontsize=8)

        ax = axes[1, col_index]
        direct = comparison[comparison["model_name"].eq("ASB direct + root")]
        ax.scatter(
            direct[f"{quantity}_exp"],
            direct[f"{quantity}_apc_interp_error_vs_exp"],
            s=18,
            color="black",
            alpha=0.65,
            label="APC map interp.",
        )
        for model_name, color, marker in [
            ("ASB direct + root", "#1f77b4", "o"),
            ("ASB tip-t/c + root", "#2ca02c", "^"),
        ]:
            data = comparison[comparison["model_name"].eq(model_name)]
            ax.scatter(
                data[f"{quantity}_exp"],
                data[f"{quantity}_asb_error_vs_exp"],
                s=18,
                color=color,
                marker=marker,
                alpha=0.65,
                label=model_name,
            )
        ax.axhline(0, color="0.5", linewidth=1)
        ax.set_xlabel(f"UIUC experiment {label}")
        ax.set_ylabel(f"Error in {label}")
        ax.grid(True, alpha=0.25)

    fig.suptitle("APC 12x8E: Parity and Errors vs UIUC Experiment")
    fig.savefig(output_dir / "uiuc_parity_and_errors.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "uiuc_sources.txt").write_text(
        "\n".join(
            [
                "UIUC Propeller Database Volume 4:",
                UIUC_VOLUME4_URL,
                "",
                "Associated AIAA paper:",
                UIUC_PAPER_URL,
                "",
                "Downloaded files:",
                *[path.name for path in sorted(UIUC_DATA.glob('apce_12x8_*.txt'))],
                "",
            ]
        )
    )
    experiment = parse_uiuc_experiment_files()
    experiment.to_csv(OUTPUT_DIR / "uiuc_apce_12x8_experiment.csv", index=False)

    apc_interp = interpolate_apc_map_to_experiment(
        experiment=experiment,
        apc_map=parse_apc_performance_file(),
    )
    asb_results = run_asb_at_experiment_points(experiment)
    asb_results["model_name"] = np.where(
        asb_results["tip_thickness_to_root"],
        "ASB tip-t/c + root",
        "ASB direct + root",
    )

    comparison = apc_interp.merge(
        asb_results,
        on=["run_type", "rpm", "J"],
        how="left",
        suffixes=("", "_asbcase"),
    )
    comparison = add_errors(comparison)
    comparison.to_csv(OUTPUT_DIR / "uiuc_apc_asb_comparison.csv", index=False)

    metrics = make_metrics(comparison)
    metrics.to_csv(OUTPUT_DIR / "uiuc_apc_asb_metrics.csv", index=False)

    plot_wind_on_by_rpm(comparison, OUTPUT_DIR)
    plot_static(comparison, OUTPUT_DIR)
    plot_parity(comparison, OUTPUT_DIR)

    print(f"Wrote experiment comparison: {OUTPUT_DIR}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
