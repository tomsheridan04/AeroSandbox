from pathlib import Path
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
from aerosandbox.geometry.propeller import _scale_kulfan_thickness_about_camber

from studies.PropellerAnalysisValidation.apc_12x8e_qprop_comparison import (
    qprop_section_aerodynamics,
)
from studies.PropellerAnalysisValidation.apc_12x8e_validation_maps import (
    OUTPUTS,
    parse_apc_geometry_file,
    parse_apc_performance_file,
)


OPTION_OUTPUTS = OUTPUTS / "low_confidence_options"
REPRESENTATIVE_RPMS = [3000, 6000, 9000, 12000, 13000]
TARGET_JS = np.array([0.0, 0.08, 0.16, 0.28, 0.42])
CONFIDENCE_THRESHOLD = 0.5
THICKNESS_CAP = 0.18


def neuralfoil_aero(airfoil, alpha, Re, mach, r_over_R):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="overflow encountered in exp",
            category=RuntimeWarning,
            module=r"neuralfoil\..*",
        )
        return airfoil.get_aero_from_neuralfoil(
            alpha=alpha,
            Re=Re,
            mach=mach,
            model_size="xsmall",
            include_360_deg_effects=True,
        )


def confidence_gated_analytical_aero(airfoil, alpha, Re, mach, r_over_R):
    nf = neuralfoil_aero(airfoil, alpha, Re, mach, r_over_R)
    analytical = qprop_section_aerodynamics(airfoil, alpha, Re, mach, r_over_R)
    use_analytical = nf["analysis_confidence"] < CONFIDENCE_THRESHOLD

    return {
        "CL": np.where(use_analytical, analytical["CL"], nf["CL"]),
        "CD": np.where(use_analytical, analytical["CD"], nf["CD"]),
        "CM": np.where(use_analytical, analytical["CM"], nf["CM"]),
        "analysis_confidence": nf["analysis_confidence"],
    }


def confidence_blended_analytical_aero(airfoil, alpha, Re, mach, r_over_R):
    nf = neuralfoil_aero(airfoil, alpha, Re, mach, r_over_R)
    analytical = qprop_section_aerodynamics(airfoil, alpha, Re, mach, r_over_R)
    analytical_weight = 0.5 * (
        1 - np.tanh((nf["analysis_confidence"] - CONFIDENCE_THRESHOLD) / 0.1)
    )

    return {
        "CL": analytical_weight * analytical["CL"]
        + (1 - analytical_weight) * nf["CL"],
        "CD": analytical_weight * analytical["CD"]
        + (1 - analytical_weight) * nf["CD"],
        "CM": analytical_weight * analytical["CM"]
        + (1 - analytical_weight) * nf["CM"],
        "analysis_confidence": nf["analysis_confidence"],
    }


def thickness_capped_neuralfoil_aero(airfoil, alpha, Re, mach, r_over_R):
    thickness = float(airfoil.max_thickness())
    if thickness > THICKNESS_CAP:
        airfoil = _scale_kulfan_thickness_about_camber(
            airfoil=airfoil,
            thickness_scale=THICKNESS_CAP / thickness,
        )
    return neuralfoil_aero(airfoil, alpha, Re, mach, r_over_R)


def make_representative_cases(apc_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rpm in REPRESENTATIVE_RPMS:
        data = apc_data[apc_data["rpm"].eq(rpm)]
        for target_j in TARGET_JS:
            rows.append(data.iloc[(data["J"] - target_j).abs().argmin()])
    return pd.DataFrame(rows).drop_duplicates(["rpm", "J"]).sort_values(["rpm", "J"])


def run_option_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    OPTION_OUTPUTS.mkdir(parents=True, exist_ok=True)

    propeller = parse_apc_geometry_file()
    cases = make_representative_cases(parse_apc_performance_file())
    options = {
        "NeuralFoil baseline": None,
        f"Hard analytical if confidence < {CONFIDENCE_THRESHOLD:g}": confidence_gated_analytical_aero,
        f"Smooth analytical blend near confidence {CONFIDENCE_THRESHOLD:g}": confidence_blended_analytical_aero,
        f"NeuralFoil with t/c <= {THICKNESS_CAP:g}": thickness_capped_neuralfoil_aero,
    }

    rows = []
    total = len(cases) * len(options)
    i = 0
    for option_name, section_aero in options.items():
        for _, case in cases.iterrows():
            i += 1
            print(
                f"Low-confidence option case {i}/{total}: "
                f"{option_name}, {case.rpm:.0f} RPM, J={case.J:.3f}",
                flush=True,
            )
            output = asb.PropellerAnalysis(
                propeller=propeller,
                op_point=asb.OperatingPoint(velocity=case.velocity_mps),
                rpm=case.rpm,
                radial_resolution=16,
                newton_iterations=8,
                bracketing_iterations=24,
                residual_tolerance=1e-5,
                model_size="xsmall",
                section_aerodynamics=section_aero,
            ).run()

            rows.append(
                {
                    "option": option_name,
                    "rpm": case.rpm,
                    "J": case.J,
                    "velocity_mph": case.velocity_mph,
                    "Ct_apc": case.Ct,
                    "Cp_apc": case.Cp,
                    "power_W_apc": case.power_W,
                    "Ct_model": float(output["Ct"]),
                    "Cp_model": float(output["Cp"]),
                    "power_W_model": float(output["power"]),
                    "max_abs_residual": float(output["max_abs_residual"]),
                    "min_analysis_confidence": float(np.min(output["analysis_confidence"])),
                }
            )

    results = pd.DataFrame(rows)
    results["Ct_error"] = results["Ct_model"] - results["Ct_apc"]
    results["Cp_error"] = results["Cp_model"] - results["Cp_apc"]
    results["power_W_error"] = results["power_W_model"] - results["power_W_apc"]
    results.to_csv(OPTION_OUTPUTS / "apc_12x8e_low_confidence_options.csv", index=False)

    metrics = []
    for option, data in results.groupby("option"):
        metrics.append(
            {
                "option": option,
                "n_cases": len(data),
                "Ct_rmse": np.sqrt(np.nanmean(data["Ct_error"] ** 2)),
                "Cp_rmse": np.sqrt(np.nanmean(data["Cp_error"] ** 2)),
                "power_W_rmse": np.sqrt(np.nanmean(data["power_W_error"] ** 2)),
                "max_abs_residual": np.nanmax(data["max_abs_residual"]),
                "min_analysis_confidence": np.nanmin(data["min_analysis_confidence"]),
            }
        )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OPTION_OUTPUTS / "apc_12x8e_low_confidence_option_metrics.csv", index=False)

    return results, metrics


def plot_option_sweep(results: pd.DataFrame, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, quantity, ylabel in [
        (axes[0], "Ct_rmse", "$C_T$ RMSE"),
        (axes[1], "Cp_rmse", "$C_P$ RMSE"),
        (axes[2], "power_W_rmse", "Power RMSE [W]"),
    ]:
        ordered = metrics.sort_values(quantity)
        ax.barh(ordered["option"], ordered[quantity])
        ax.set_xlabel(ylabel)
        ax.grid(True, axis="x", alpha=0.25)
    fig.suptitle("APC 12x8E Low-Confidence Section-Aero Options")
    fig.savefig(OPTION_OUTPUTS / "apc_12x8e_low_confidence_option_metrics.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for option, data in results[results["rpm"].eq(12000)].groupby("option"):
        data = data.sort_values("J")
        axes[0].plot(data["J"], data["Ct_model"], marker="o", label=option)
        axes[1].plot(data["J"], data["Cp_model"], marker="o", label=option)
    apc = results[results["rpm"].eq(12000)].drop_duplicates(["rpm", "J"]).sort_values("J")
    axes[0].plot(apc["J"], apc["Ct_apc"], color="black", linewidth=2, label="APC")
    axes[1].plot(apc["J"], apc["Cp_apc"], color="black", linewidth=2, label="APC")
    axes[0].set_ylabel("$C_T$")
    axes[1].set_ylabel("$C_P$")
    axes[1].set_xlabel("Advance ratio $J$")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Low-Confidence Options at 12000 RPM")
    fig.savefig(OPTION_OUTPUTS / "apc_12x8e_low_confidence_options_12000rpm.png", dpi=220)
    plt.close(fig)


def main():
    results, metrics = run_option_sweep()
    plot_option_sweep(results, metrics)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
