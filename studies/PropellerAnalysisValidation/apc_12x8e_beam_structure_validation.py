from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import aerosandbox as asb

from aerosandbox.structures.propeller_beam import (
    PropellerBeamStructuralAnalysis,
    airfoil_section_properties_over_span,
)
from studies.PropellerAnalysisValidation.apc_12x8e_validation_maps import (
    APC_GEOMETRY_FILE,
    OUTPUTS,
    parse_apc_geometry_file,
    parse_apc_performance_file,
)


BEAM_OUTPUTS = OUTPUTS / "beam_structure_validation_apc"
VALIDATION_RPM = 8000
VALIDATION_TARGET_VELOCITY_MPH = 40.0
VALIDATION_RADIAL_RESOLUTION = 32
BEAM_RADIAL_RESOLUTION = 301
ASSUMED_POISSON_RATIO = 0.35

INCH = 0.0254
IN2_TO_M2 = INCH**2
PSI_TO_PA = 6894.757293168361
LB_IN3_TO_KG_M3 = 0.45359237 / INCH**3


def output_array(output: dict, key: str) -> onp.ndarray:
    return onp.asarray(output[key], dtype=float).reshape(-1)


def parse_float_from_text(
    text: str,
    pattern: str,
    name: str,
    default=None,
):
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    if default is not None:
        return default
    raise ValueError(f"Could not parse {name} from APC file.")


def parse_apc_structural_file(
    filename: Path = APC_GEOMETRY_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    text = filename.read_text(errors="ignore")
    rows = []
    for line in text.splitlines():
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.?", line)
        if len(numbers) == 14:
            rows.append([float(number) for number in numbers])

    if not rows:
        raise ValueError(f"No structural section rows found in {filename}.")

    table = pd.DataFrame(
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
    ).sort_values("station_in")

    metadata = {
        "radius_in": parse_float_from_text(
            text,
            r"RADIUS:\s*([0-9.]+)",
            "radius",
        ),
        "hub_radius_in": parse_float_from_text(
            text,
            r"HUBRAD:\s*([0-9.]+)",
            "hub radius",
        ),
        "blade_count": int(
            parse_float_from_text(
                text,
                r"BLADES:\s*([0-9.]+)",
                "blade count",
            )
        ),
        "total_mass_kg": parse_float_from_text(
            text,
            r"TOTAL WEIGHT \(Kg\)\s*=\s*([0-9.]+)",
            "total mass",
        ),
        "total_volume_in3": parse_float_from_text(
            text,
            r"TOTAL VOLUME \(IN\*\*3\)\s*=\s*([0-9.]+)",
            "total volume",
        ),
        "polar_inertia_kg_m2": parse_float_from_text(
            text,
            r"MOMENT OF INERTIA \(Kg-M\*\*2\)\s*=\s*([0-9.Ee+-]+)",
            "polar inertia",
        ),
        "density_lb_in3": parse_float_from_text(
            text,
            r"DENSITY \(INPUT FILE, LB/IN\*\*3\)\s*=\s*([0-9.Ee+-]+)",
            "density",
        ),
        "specific_gravity": parse_float_from_text(
            text,
            r"DENSITY \(SPECIFIC GRAVITY, INPUT FILE\)\s*=\s*([0-9.]+)",
            "specific gravity",
        ),
        "elastic_modulus_mpsi": parse_float_from_text(
            text,
            r"BASED ON MODULUS \(MILLION\)\s*=\s*([0-9.]+)",
            "elastic modulus",
        ),
        "first_bending_frequency_rpm": parse_float_from_text(
            text,
            r"LOWEST NATURAL BENDING FREQUENCY \(IN TERMS OF RPM\)\s*=\s*([0-9.]+)",
            "first bending frequency",
            default=onp.nan,
        ),
    }
    metadata["density_kg_m3"] = metadata["density_lb_in3"] * LB_IN3_TO_KG_M3
    metadata["elastic_modulus_pa"] = (
        metadata["elastic_modulus_mpsi"] * 1e6 * PSI_TO_PA
    )
    metadata["shear_modulus_pa"] = metadata["elastic_modulus_pa"] / (
        2 * (1 + ASSUMED_POISSON_RATIO)
    )

    for column in [
        "station",
        "chord",
        "sweep_y",
        "rake_z",
        "max_thickness",
        "zhigh",
        "cgy",
        "cgz",
    ]:
        table[f"{column}_m"] = table[f"{column}_in"] * INCH
    table["cross_section_m2"] = table["cross_section_in2"] * IN2_TO_M2

    root_station_in = metadata["hub_radius_in"]
    if not onp.any(onp.isclose(table["station_in"], root_station_in)):
        root_row = {"station_in": root_station_in}
        for column in table.columns:
            if column == "station_in":
                continue
            root_row[column] = onp.interp(
                root_station_in,
                table["station_in"],
                table[column],
            )
        table = pd.concat(
            [table, pd.DataFrame([root_row])],
            ignore_index=True,
        ).sort_values("station_in")

    beam_table = table[
        table["station_in"].between(
            metadata["hub_radius_in"],
            metadata["radius_in"],
            inclusive="both",
        )
    ].copy()
    beam_table = beam_table[beam_table["cross_section_m2"] > 0].copy()
    beam_table = beam_table.sort_values("station_in").reset_index(drop=True)

    return table.reset_index(drop=True), beam_table, metadata


def choose_validation_case() -> pd.Series:
    apc_data = parse_apc_performance_file()
    candidates = apc_data[apc_data["rpm"].eq(VALIDATION_RPM)].copy()
    if candidates.empty:
        raise ValueError(f"No APC performance data for {VALIDATION_RPM} rpm.")
    index = (
        candidates["velocity_mph"] - VALIDATION_TARGET_VELOCITY_MPH
    ).abs().argmin()
    return candidates.iloc[index]


def run_propeller_case(propeller: asb.Propeller, case: pd.Series) -> dict:
    return asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=float(case["velocity_mps"])),
        rpm=float(case["rpm"]),
        radial_resolution=VALIDATION_RADIAL_RESOLUTION,
        newton_iterations=10,
        bracketing_iterations=30,
        residual_tolerance=1e-5,
        model_size="xsmall",
        include_root_loss=True,
        include_post_stall_confidence_blending=False,
    ).run()


def section_integral_validation(
    full_table: pd.DataFrame,
    metadata: dict,
) -> dict:
    r = full_table["station_m"].to_numpy()
    area = full_table["cross_section_m2"].to_numpy()
    density = metadata["density_kg_m3"]
    blade_count = metadata["blade_count"]
    total_mass = blade_count * onp.trapezoid(density * area, r)
    total_volume = blade_count * onp.trapezoid(area, r)
    polar_inertia = blade_count * onp.trapezoid(density * area * r**2, r)

    return {
        "section_integrated_mass_kg": total_mass,
        "section_integrated_volume_m3": total_volume,
        "section_integrated_volume_in3": total_volume / INCH**3,
        "section_integrated_polar_inertia_kg_m2": polar_inertia,
        "mass_error_percent": 100 * (total_mass / metadata["total_mass_kg"] - 1),
        "volume_error_percent": 100
        * (total_volume / (metadata["total_volume_in3"] * INCH**3) - 1),
        "polar_inertia_error_percent": 100
        * (polar_inertia / metadata["polar_inertia_kg_m2"] - 1),
    }


def run_beam_analysis(
    beam_table: pd.DataFrame,
    metadata: dict,
    aero_output: dict,
    case: pd.Series,
) -> dict:
    op_point = asb.OperatingPoint(velocity=float(case["velocity_mps"]))
    structural_propeller = parse_apc_geometry_file(
        inner_model_radius_ratio=metadata["hub_radius_in"] / metadata["radius_in"],
        max_modeled_thickness_ratio=None,
    )
    section_properties = airfoil_section_properties_over_span(
        r_over_R=beam_table["station_m"].to_numpy()
        / (metadata["radius_in"] * INCH),
        chord=beam_table["chord_m"].to_numpy(),
        area=beam_table["cross_section_m2"].to_numpy(),
        airfoil_function=structural_propeller.airfoil,
    )
    return PropellerBeamStructuralAnalysis(
        radius_stations=beam_table["station_m"].to_numpy(),
        chord=beam_table["chord_m"].to_numpy(),
        twist=beam_table["twist_deg"].to_numpy(),
        area=beam_table["cross_section_m2"].to_numpy(),
        max_thickness=beam_table["max_thickness_m"].to_numpy(),
        cg_chordwise=beam_table["cgy_m"].to_numpy(),
        cg_thickness=beam_table["cgz_m"].to_numpy(),
        density=metadata["density_kg_m3"],
        elastic_modulus=metadata["elastic_modulus_pa"],
        shear_modulus=metadata["shear_modulus_pa"],
        aerodynamic_output=aero_output,
        air_density=float(op_point.atmosphere.density()),
        blade_count=metadata["blade_count"],
        radial_resolution=BEAM_RADIAL_RESOLUTION,
        section_properties=section_properties,
    ).run()


def make_spanwise_table(beam_output: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            key: output_array(beam_output, key)
            for key in [
                "r",
                "r_over_R",
                "chord",
                "twist",
                "area",
                "max_thickness",
                "mass_per_length",
                "I_chord_axis",
                "I_thickness_axis",
                "I_chord_thickness_product",
                "torsion_constant",
                "section_area",
                "raw_airfoil_area",
                "area_scale_factor",
                "section_centroid_chordwise",
                "section_centroid_thickness",
                "max_chordwise_distance",
                "max_thickness_distance",
                "EI_chord_axis",
                "EI_thickness_axis",
                "GJ",
                "q_thrust",
                "q_tangential",
                "q_chordwise",
                "q_thickness",
                "q_centrifugal",
                "axial_force",
                "shear_chordwise",
                "shear_thickness",
                "moment_about_chord_axis",
                "moment_about_thickness_axis",
                "torque_about_span",
                "deflection_chordwise",
                "deflection_thickness",
                "torsion_angle",
                "axial_stress",
                "bending_stress",
                "combined_tensile_stress",
                "combined_compressive_stress",
            ]
        }
    )


def make_summary(
    case: pd.Series,
    aero_output: dict,
    beam_output: dict,
    metadata: dict,
    section_validation: dict,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rpm": float(case["rpm"]),
                "velocity_mph": float(case["velocity_mph"]),
                "velocity_mps": float(case["velocity_mps"]),
                "J_apc": float(case["J"]),
                "J_asb": float(aero_output["J"]),
                "apc_thrust_N": float(case["thrust_N"]),
                "asb_thrust_N": float(aero_output["thrust"]),
                "thrust_error_percent": 100
                * (float(aero_output["thrust"]) / float(case["thrust_N"]) - 1),
                "apc_power_W": float(case["power_W"]),
                "asb_power_W": float(aero_output["power"]),
                "power_error_percent": 100
                * (float(aero_output["power"]) / float(case["power_W"]) - 1),
                "apc_torque_Nm": float(case["torque_Nm"]),
                "asb_torque_Nm": float(aero_output["torque"]),
                "max_abs_prop_residual": float(aero_output["max_abs_residual"]),
                "density_kg_m3": metadata["density_kg_m3"],
                "elastic_modulus_GPa": metadata["elastic_modulus_pa"] / 1e9,
                "shear_modulus_GPa": metadata["shear_modulus_pa"] / 1e9,
                "apc_total_mass_kg": metadata["total_mass_kg"],
                "section_integrated_mass_kg": section_validation[
                    "section_integrated_mass_kg"
                ],
                "mass_error_percent": section_validation["mass_error_percent"],
                "apc_total_volume_in3": metadata["total_volume_in3"],
                "section_integrated_volume_in3": section_validation[
                    "section_integrated_volume_in3"
                ],
                "volume_error_percent": section_validation["volume_error_percent"],
                "apc_polar_inertia_kg_m2": metadata["polar_inertia_kg_m2"],
                "section_integrated_polar_inertia_kg_m2": section_validation[
                    "section_integrated_polar_inertia_kg_m2"
                ],
                "polar_inertia_error_percent": section_validation[
                    "polar_inertia_error_percent"
                ],
                "apc_first_bending_frequency_rpm": metadata[
                    "first_bending_frequency_rpm"
                ],
                "root_axial_force_N": float(beam_output["axial_force"][0]),
                "root_shear_chordwise_N": float(beam_output["shear_chordwise"][0]),
                "root_shear_thickness_N": float(beam_output["shear_thickness"][0]),
                "root_moment_about_chord_axis_Nm": float(
                    beam_output["moment_about_chord_axis"][0]
                ),
                "root_moment_about_thickness_axis_Nm": float(
                    beam_output["moment_about_thickness_axis"][0]
                ),
                "root_torque_about_span_Nm": float(
                    beam_output["torque_about_span"][0]
                ),
                "tip_deflection_chordwise_mm": float(
                    beam_output["deflection_chordwise"][-1] * 1e3
                ),
                "tip_deflection_thickness_mm": float(
                    beam_output["deflection_thickness"][-1] * 1e3
                ),
                "tip_torsion_deg": float(
                    onp.degrees(beam_output["torsion_angle"][-1])
                ),
                "max_combined_tensile_stress_MPa": float(
                    onp.max(beam_output["combined_tensile_stress"]) / 1e6
                ),
                "min_combined_compressive_stress_MPa": float(
                    onp.min(beam_output["combined_compressive_stress"]) / 1e6
                ),
            }
        ]
    )


def plot_summary(summary: pd.DataFrame, filename: Path) -> None:
    row = summary.iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)

    labels = ["Thrust", "Power", "Torque"]
    ratios = [
        row["asb_thrust_N"] / row["apc_thrust_N"],
        row["asb_power_W"] / row["apc_power_W"],
        row["asb_torque_Nm"] / row["apc_torque_Nm"],
    ]
    x = onp.arange(len(labels))
    axes[0, 0].bar(x, ratios, color="#1f77b4")
    axes[0, 0].axhline(1.0, color="black", linewidth=1)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("ASB / APC [-]")
    axes[0, 0].grid(True, axis="y", alpha=0.25)
    axes[0, 0].set_title("Aerodynamic Validation")
    for i, ratio in enumerate(ratios):
        axes[0, 0].text(
            i,
            ratio + 0.004,
            f"{100 * (ratio - 1):+.1f}%",
            ha="center",
            va="bottom",
        )

    labels = ["Mass", "Volume", "Polar inertia"]
    ratios = [
        row["section_integrated_mass_kg"] / row["apc_total_mass_kg"],
        row["section_integrated_volume_in3"] / row["apc_total_volume_in3"],
        row["section_integrated_polar_inertia_kg_m2"]
        / row["apc_polar_inertia_kg_m2"],
    ]
    # APC mass and volume use the same density, so the first two ratios are
    # intentionally nearly identical; keeping both catches unit mistakes.
    x = onp.arange(len(labels))
    axes[0, 1].bar(x, ratios, color="#2ca02c")
    axes[0, 1].axhline(1.0, color="black", linewidth=1)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("Section integral / APC [-]")
    axes[0, 1].grid(True, axis="y", alpha=0.25)
    axes[0, 1].set_title("Structural Property Validation")
    for i, ratio in enumerate(ratios):
        axes[0, 1].text(
            i,
            ratio + 0.004,
            f"{100 * (ratio - 1):+.1f}%",
            ha="center",
            va="bottom",
        )

    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_axis_off()

    root_text = "\n".join(
        [
            "Root Loads",
            f"Axial tension: {row['root_axial_force_N']:.1f} N",
            f"Chordwise shear: {row['root_shear_chordwise_N']:.2f} N",
            f"Thickness shear: {row['root_shear_thickness_N']:.2f} N",
            f"Flap moment: {row['root_moment_about_chord_axis_Nm']:.3f} N m",
            f"Edge moment: {row['root_moment_about_thickness_axis_Nm']:.3f} N m",
            f"Torsion: {row['root_torque_about_span_Nm']:.4f} N m",
        ]
    )
    axes[1, 0].text(
        0.02,
        0.98,
        root_text,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )

    response_text = "\n".join(
        [
            "Beam Response",
            f"Tip flap deflection: {row['tip_deflection_thickness_mm']:.2f} mm",
            f"Tip chord deflection: {row['tip_deflection_chordwise_mm']:.3f} mm",
            f"Tip elastic torsion: {row['tip_torsion_deg']:.3f} deg",
            f"Max tensile stress: {row['max_combined_tensile_stress_MPa']:.1f} MPa",
            f"Min compressive stress: {row['min_combined_compressive_stress_MPa']:.1f} MPa",
            f"APC first bending freq: {row['apc_first_bending_frequency_rpm']:.0f} rpm",
        ]
    )
    axes[1, 1].text(
        0.02,
        0.98,
        response_text,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )

    fig.suptitle(
        "APC 12x8E Beam Validation: "
        f"{row['rpm']:.0f} rpm, {row['velocity_mph']:.2f} mph"
    )
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_structural_properties(
    beam_output: dict,
    filename: Path,
) -> None:
    r_over_R = output_array(beam_output, "r_over_R")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(r_over_R, output_array(beam_output, "area") / IN2_TO_M2)
    axes[0].set_ylabel("Area [in^2]")

    axes[1].plot(r_over_R, output_array(beam_output, "mass_per_length"))
    axes[1].set_ylabel("Mass / length [kg/m]")

    axes[2].plot(r_over_R, output_array(beam_output, "cg_chordwise") / INCH, label="CGY")
    axes[2].plot(r_over_R, output_array(beam_output, "cg_thickness") / INCH, label="CGZ")
    axes[2].set_ylabel("CG offset [in]")
    axes[2].legend(loc="best")

    axes[3].semilogy(
        r_over_R,
        output_array(beam_output, "EI_chord_axis"),
        label="EI chord axis",
    )
    axes[3].semilogy(
        r_over_R,
        output_array(beam_output, "EI_thickness_axis"),
        label="EI thickness axis",
    )
    axes[3].set_ylabel("Bending stiffness [N m^2]")
    axes[3].legend(loc="best")

    axes[4].semilogy(r_over_R, output_array(beam_output, "GJ"))
    axes[4].set_ylabel("Torsional stiffness GJ [N m^2]")

    axes[5].plot(
        r_over_R,
        output_array(beam_output, "equivalent_width") / INCH,
        label="width",
    )
    axes[5].plot(
        r_over_R,
        output_array(beam_output, "equivalent_height") / INCH,
        label="height",
    )
    axes[5].set_ylabel("Equivalent rectangle [in]")
    axes[5].legend(loc="best")

    for ax in axes:
        ax.set_xlabel("Station r/R [-]")
        ax.grid(True, alpha=0.25)
    fig.suptitle("APC 12x8E Beam Section Properties")
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_loads(beam_output: dict, filename: Path) -> None:
    r_over_R = output_array(beam_output, "r_over_R")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(r_over_R, output_array(beam_output, "q_thrust"), label="thrust")
    axes[0].plot(
        r_over_R,
        output_array(beam_output, "q_tangential"),
        label="tangential",
    )
    axes[0].set_ylabel("Global aero load [N/m]")
    axes[0].legend(loc="best")

    axes[1].plot(
        r_over_R,
        output_array(beam_output, "q_chordwise"),
        label="chordwise",
    )
    axes[1].plot(
        r_over_R,
        output_array(beam_output, "q_thickness"),
        label="thickness",
    )
    axes[1].set_ylabel("Local aero load [N/m]")
    axes[1].legend(loc="best")

    axes[2].plot(r_over_R, output_array(beam_output, "q_centrifugal"))
    axes[2].set_ylabel("Centrifugal load [N/m]")

    axes[3].plot(
        r_over_R,
        output_array(beam_output, "shear_chordwise"),
        label="chordwise",
    )
    axes[3].plot(
        r_over_R,
        output_array(beam_output, "shear_thickness"),
        label="thickness",
    )
    axes[3].set_ylabel("Shear force [N]")
    axes[3].legend(loc="best")

    axes[4].plot(
        r_over_R,
        output_array(beam_output, "moment_about_chord_axis"),
        label="about chord axis",
    )
    axes[4].plot(
        r_over_R,
        output_array(beam_output, "moment_about_thickness_axis"),
        label="about thickness axis",
    )
    axes[4].set_ylabel("Bending moment [N m]")
    axes[4].legend(loc="best")

    axes[5].plot(r_over_R, output_array(beam_output, "axial_force"), label="axial")
    axes[5].plot(
        r_over_R,
        output_array(beam_output, "torque_about_span"),
        label="torsion",
    )
    axes[5].set_ylabel("Axial force [N] / torque [N m]")
    axes[5].legend(loc="best")

    for ax in axes:
        ax.set_xlabel("Station r/R [-]")
        ax.grid(True, alpha=0.25)
    fig.suptitle("APC 12x8E Beam Loads")
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_response(beam_output: dict, filename: Path) -> None:
    r_over_R = output_array(beam_output, "r_over_R")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(
        r_over_R,
        output_array(beam_output, "deflection_chordwise") * 1e3,
        label="chordwise",
    )
    axes[0].plot(
        r_over_R,
        output_array(beam_output, "deflection_thickness") * 1e3,
        label="thickness",
    )
    axes[0].set_ylabel("Deflection [mm]")
    axes[0].legend(loc="best")

    axes[1].plot(r_over_R, onp.degrees(output_array(beam_output, "torsion_angle")))
    axes[1].set_ylabel("Elastic torsion [deg]")

    axes[2].plot(
        r_over_R,
        output_array(beam_output, "axial_stress") / 1e6,
        label="centrifugal axial",
    )
    axes[2].plot(
        r_over_R,
        output_array(beam_output, "bending_stress") / 1e6,
        label="bending",
    )
    axes[2].set_ylabel("Stress component [MPa]")
    axes[2].legend(loc="best")

    axes[3].plot(
        r_over_R,
        output_array(beam_output, "combined_tensile_stress") / 1e6,
        label="tensile side",
    )
    axes[3].plot(
        r_over_R,
        output_array(beam_output, "combined_compressive_stress") / 1e6,
        label="compressive side",
    )
    axes[3].set_ylabel("Combined stress [MPa]")
    axes[3].legend(loc="best")

    axes[4].plot(
        r_over_R,
        output_array(beam_output, "curvature_thickness"),
        label="thickness deflection",
    )
    axes[4].plot(
        r_over_R,
        output_array(beam_output, "curvature_chordwise"),
        label="chordwise deflection",
    )
    axes[4].set_ylabel("Curvature [1/m]")
    axes[4].legend(loc="best")

    axes[5].plot(r_over_R, output_array(beam_output, "twist_rate"))
    axes[5].set_ylabel("Twist rate [rad/m]")

    for ax in axes:
        ax.set_xlabel("Station r/R [-]")
        ax.grid(True, alpha=0.25)
    fig.suptitle("APC 12x8E Beam Response")
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def run_apc_beam_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    BEAM_OUTPUTS.mkdir(parents=True, exist_ok=True)

    full_table, beam_table, metadata = parse_apc_structural_file()
    propeller = parse_apc_geometry_file()
    case = choose_validation_case()
    aero_output = run_propeller_case(propeller=propeller, case=case)
    beam_output = run_beam_analysis(
        beam_table=beam_table,
        metadata=metadata,
        aero_output=aero_output,
        case=case,
    )
    section_validation = section_integral_validation(
        full_table=full_table,
        metadata=metadata,
    )

    spanwise = make_spanwise_table(beam_output)
    summary = make_summary(
        case=case,
        aero_output=aero_output,
        beam_output=beam_output,
        metadata=metadata,
        section_validation=section_validation,
    )

    full_table.to_csv(BEAM_OUTPUTS / "apc_12x8e_structural_table.csv", index=False)
    spanwise.to_csv(BEAM_OUTPUTS / "apc_12x8e_beam_spanwise.csv", index=False)
    summary.to_csv(BEAM_OUTPUTS / "apc_12x8e_beam_summary.csv", index=False)

    plot_summary(
        summary=summary,
        filename=BEAM_OUTPUTS / "apc_12x8e_beam_validation_summary.png",
    )
    plot_structural_properties(
        beam_output=beam_output,
        filename=BEAM_OUTPUTS / "apc_12x8e_beam_structural_properties.png",
    )
    plot_loads(
        beam_output=beam_output,
        filename=BEAM_OUTPUTS / "apc_12x8e_beam_loads.png",
    )
    plot_response(
        beam_output=beam_output,
        filename=BEAM_OUTPUTS / "apc_12x8e_beam_response.png",
    )

    print(summary.to_string(index=False))
    print(f"Wrote APC beam validation outputs to: {BEAM_OUTPUTS}")
    return summary, spanwise


if __name__ == "__main__":
    run_apc_beam_validation()
