from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from aerosandbox.structures.propeller_beam import PropellerBeamStructuralAnalysis
from studies.PropellerAnalysisValidation.apc_12x8e_validation_maps import OUTPUTS


ANALYTICAL_OUTPUTS = OUTPUTS / "beam_structure_validation_analytical"


def rectangle_torsion_constant(width: float, height: float) -> float:
    long_side = max(width, height)
    short_side = min(width, height)
    ratio = short_side / long_side
    return (
        long_side
        * short_side**3
        / 3
        * (1 - 0.63 * ratio + 0.052 * ratio**5)
    )


def make_uniform_rectangle_case() -> tuple[dict, dict, dict]:
    length = 1.2
    chord = 0.12
    thickness = 0.024
    area = chord * thickness
    elastic_modulus = 70e9
    shear_modulus = 26e9
    q_thrust = 80.0
    aerodynamic_center_fraction = 0.65
    centroid_fraction = 0.50
    lever_arm = (aerodynamic_center_fraction - centroid_fraction) * chord
    torsion_per_length = lever_arm * q_thrust

    I_chord_axis = chord * thickness**3 / 12
    I_thickness_axis = thickness * chord**3 / 12
    torsion_constant = rectangle_torsion_constant(chord, thickness)

    stations = onp.array([0.0, length])
    section_properties = {
        "section_area": area * onp.ones_like(stations),
        "section_centroid_chordwise": centroid_fraction
        * chord
        * onp.ones_like(stations),
        "section_centroid_thickness": onp.zeros_like(stations),
        "equivalent_width": chord * onp.ones_like(stations),
        "equivalent_height": thickness * onp.ones_like(stations),
        "I_chord_axis": I_chord_axis * onp.ones_like(stations),
        "I_thickness_axis": I_thickness_axis * onp.ones_like(stations),
        "I_chord_thickness_product": onp.zeros_like(stations),
        "torsion_constant": torsion_constant * onp.ones_like(stations),
        "max_chordwise_distance": 0.5 * chord * onp.ones_like(stations),
        "max_thickness_distance": 0.5 * thickness * onp.ones_like(stations),
        "raw_airfoil_area": area * onp.ones_like(stations),
        "area_scale_factor": onp.ones_like(stations),
    }

    aero_r = onp.linspace(0, length, 51)
    aerodynamic_output = {
        "omega": 0.0,
        "r": aero_r,
        "thrust_per_radius": q_thrust * onp.ones_like(aero_r),
        "torque_per_radius": onp.zeros_like(aero_r),
    }

    beam = PropellerBeamStructuralAnalysis(
        radius_stations=stations,
        chord=chord * onp.ones_like(stations),
        twist=onp.zeros_like(stations),
        area=area * onp.ones_like(stations),
        max_thickness=thickness * onp.ones_like(stations),
        cg_chordwise=centroid_fraction * chord * onp.ones_like(stations),
        cg_thickness=onp.zeros_like(stations),
        density=1.0,
        elastic_modulus=elastic_modulus,
        shear_modulus=shear_modulus,
        aerodynamic_output=aerodynamic_output,
        aerodynamic_center_chord_fraction=aerodynamic_center_fraction,
        radial_resolution=401,
        section_properties=section_properties,
    )
    numerical = beam.run()

    x = numerical["r"]
    analytical = {
        "r": x,
        "shear_thickness": q_thrust * (length - x),
        "moment_about_chord_axis": q_thrust * (length - x) ** 2 / 2,
        "deflection_thickness": q_thrust
        * x**2
        * (6 * length**2 - 4 * length * x + x**2)
        / (24 * elastic_modulus * I_chord_axis),
        "torque_about_span": torsion_per_length * (length - x),
        "torsion_angle": torsion_per_length
        * (length * x - 0.5 * x**2)
        / (shear_modulus * torsion_constant),
        "bending_stress": (
            q_thrust
            * (length - x) ** 2
            / 2
            * (0.5 * thickness)
            / I_chord_axis
        ),
    }
    metadata = {
        "length": length,
        "chord": chord,
        "thickness": thickness,
        "area": area,
        "elastic_modulus": elastic_modulus,
        "shear_modulus": shear_modulus,
        "I_chord_axis": I_chord_axis,
        "I_thickness_axis": I_thickness_axis,
        "torsion_constant": torsion_constant,
        "q_thrust": q_thrust,
        "lever_arm": lever_arm,
        "torsion_per_length": torsion_per_length,
    }
    return numerical, analytical, metadata


def max_relative_error(numerical, analytical, key: str) -> float:
    denominator = max(onp.max(onp.abs(analytical[key])), 1e-30)
    return float(onp.max(onp.abs(numerical[key] - analytical[key])) / denominator)


def make_tables(numerical: dict, analytical: dict, metadata: dict):
    spanwise = pd.DataFrame(
        {
            "r_m": numerical["r"],
            "r_over_L": numerical["r"] / metadata["length"],
            "shear_thickness_numerical_N": numerical["shear_thickness"],
            "shear_thickness_analytical_N": analytical["shear_thickness"],
            "moment_about_chord_axis_numerical_Nm": numerical[
                "moment_about_chord_axis"
            ],
            "moment_about_chord_axis_analytical_Nm": analytical[
                "moment_about_chord_axis"
            ],
            "deflection_thickness_numerical_m": numerical["deflection_thickness"],
            "deflection_thickness_analytical_m": analytical["deflection_thickness"],
            "torque_about_span_numerical_Nm": numerical["torque_about_span"],
            "torque_about_span_analytical_Nm": analytical["torque_about_span"],
            "torsion_angle_numerical_rad": numerical["torsion_angle"],
            "torsion_angle_analytical_rad": analytical["torsion_angle"],
            "bending_stress_numerical_Pa": numerical["bending_stress"],
            "bending_stress_analytical_Pa": analytical["bending_stress"],
        }
    )
    summary = pd.DataFrame(
        [
            {
                **metadata,
                "max_relative_error_shear": max_relative_error(
                    numerical, analytical, "shear_thickness"
                ),
                "max_relative_error_moment": max_relative_error(
                    numerical, analytical, "moment_about_chord_axis"
                ),
                "max_relative_error_deflection": max_relative_error(
                    numerical, analytical, "deflection_thickness"
                ),
                "max_relative_error_torque": max_relative_error(
                    numerical, analytical, "torque_about_span"
                ),
                "max_relative_error_torsion": max_relative_error(
                    numerical, analytical, "torsion_angle"
                ),
                "max_relative_error_bending_stress": max_relative_error(
                    numerical, analytical, "bending_stress"
                ),
                "tip_deflection_numerical_m": float(
                    numerical["deflection_thickness"][-1]
                ),
                "tip_deflection_analytical_m": float(
                    analytical["deflection_thickness"][-1]
                ),
                "tip_torsion_numerical_rad": float(numerical["torsion_angle"][-1]),
                "tip_torsion_analytical_rad": float(analytical["torsion_angle"][-1]),
            }
        ]
    )
    return summary, spanwise


def plot_validation(
    numerical: dict,
    analytical: dict,
    metadata: dict,
    filename: Path,
) -> None:
    r_over_L = numerical["r"] / metadata["length"]
    fields = [
        ("shear_thickness", "Shear [N]", 1.0),
        ("moment_about_chord_axis", "Moment [N m]", 1.0),
        ("deflection_thickness", "Deflection [mm]", 1e3),
        ("torque_about_span", "Torque [N m]", 1.0),
        ("torsion_angle", "Torsion [deg]", 180 / onp.pi),
        ("bending_stress", "Bending stress [MPa]", 1e-6),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, (field, ylabel, scale) in zip(axes.ravel(), fields):
        ax.plot(
            r_over_L,
            analytical[field] * scale,
            color="black",
            linewidth=2,
            label="Analytical",
        )
        ax.plot(
            r_over_L,
            numerical[field] * scale,
            color="#1f77b4",
            linestyle="--",
            linewidth=1.8,
            label="Numerical beam",
        )
        ax.set_xlabel("x / L [-]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes.ravel()[0].legend(loc="best")
    fig.suptitle("PropellerBeamStructuralAnalysis Analytical Cantilever Validation")
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def run_analytical_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    ANALYTICAL_OUTPUTS.mkdir(parents=True, exist_ok=True)
    numerical, analytical, metadata = make_uniform_rectangle_case()
    summary, spanwise = make_tables(numerical, analytical, metadata)
    summary.to_csv(
        ANALYTICAL_OUTPUTS / "propeller_beam_analytical_summary.csv",
        index=False,
    )
    spanwise.to_csv(
        ANALYTICAL_OUTPUTS / "propeller_beam_analytical_spanwise.csv",
        index=False,
    )
    plot_validation(
        numerical=numerical,
        analytical=analytical,
        metadata=metadata,
        filename=ANALYTICAL_OUTPUTS / "propeller_beam_analytical_validation.png",
    )
    print(summary.to_string(index=False))
    print(f"Wrote analytical beam validation outputs to: {ANALYTICAL_OUTPUTS}")
    return summary, spanwise


if __name__ == "__main__":
    run_analytical_validation()
