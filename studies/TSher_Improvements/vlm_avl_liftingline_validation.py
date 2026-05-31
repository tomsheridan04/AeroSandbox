"""
Validation sweep for AeroSandbox VortexLatticeMethod against AVL and LiftingLine.

This script intentionally lives in studies/TSher_Improvements so it can use the
local avl352.exe binary without requiring AVL to be on PATH.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import aerosandbox as asb
import aerosandbox.numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent
AVL_EXE = STUDY_DIR / "avl352.exe"
OUTPUT_DIR = STUDY_DIR / "vlm_avl_liftingline_validation_outputs"
AVL_WORK_DIR = OUTPUT_DIR / "avl_work"

SPANWISE_RESOLUTION = 12
CHORDWISE_RESOLUTION = 10
VELOCITY = 45.0

COEFFICIENT_KEYS = ["CL", "CD", "CY", "Cl", "Cm", "Cn"]
DERIVATIVE_SUFFIXES = ["a", "b", "p", "q", "r"]
DERIVATIVE_KEYS = [
    f"{coefficient}{suffix}"
    for suffix in DERIVATIVE_SUFFIXES
    for coefficient in COEFFICIENT_KEYS
]


def ensure_avl_tempfile_patch() -> None:
    """
    The desktop sandbox cannot clean Python's default temp folders. The AVL
    wrapper creates a TemporaryDirectory even when working_directory is supplied,
    so patch that context manager to a no-op and keep all real AVL I/O in
    AVL_WORK_DIR.
    """
    noop_temp_dir = OUTPUT_DIR / "tmp_noop"
    noop_temp_dir.mkdir(parents=True, exist_ok=True)
    tempfile.TemporaryDirectory = lambda *args, **kwargs: contextlib.nullcontext(  # type: ignore[assignment]
        str(noop_temp_dir)
    )


def make_xsec(
    xyz_le: List[float],
    chord: float,
    twist: float,
    airfoil: asb.Airfoil,
) -> asb.WingXSec:
    return asb.WingXSec(
        xyz_le=xyz_le,
        chord=chord,
        twist=twist,
        airfoil=airfoil,
        analysis_specific_options={
            asb.AVL: {
                # Keep AVL's section lift-curve slope on the same thin-airfoil
                # footing as VLM and LiftingLine for this inviscid validation.
                "cl_alpha_factor": 1.0,
            }
        },
    )


def make_airplane() -> Tuple[asb.Airplane, Dict[str, Any]]:
    main_airfoil = asb.Airfoil("naca0012")
    tail_airfoil = asb.Airfoil("naca0010")

    main_wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            make_xsec([0.00, 0.00, 0.00], 1.50, 1.0, main_airfoil),
            make_xsec([0.15, 2.50, 0.15], 1.15, 0.0, main_airfoil),
            make_xsec([0.48, 5.00, 0.35], 0.65, -2.0, main_airfoil),
        ],
        analysis_specific_options={
            asb.AVL: {
                "spanwise_resolution": SPANWISE_RESOLUTION,
                "chordwise_resolution": CHORDWISE_RESOLUTION,
            }
        },
    )

    horizontal_tail = asb.Wing(
        name="Horizontal Tail",
        symmetric=True,
        xsecs=[
            make_xsec([4.80, 0.00, 0.25], 0.70, 0.0, tail_airfoil),
            make_xsec([4.98, 1.45, 0.32], 0.42, 0.0, tail_airfoil),
        ],
        analysis_specific_options={
            asb.AVL: {
                "spanwise_resolution": SPANWISE_RESOLUTION,
                "chordwise_resolution": CHORDWISE_RESOLUTION,
            }
        },
    )

    vertical_tail = asb.Wing(
        name="Vertical Tail",
        symmetric=False,
        xsecs=[
            make_xsec([4.65, 0.00, 0.20], 0.85, 0.0, tail_airfoil),
            make_xsec([5.00, 0.00, 1.55], 0.35, 0.0, tail_airfoil),
        ],
        analysis_specific_options={
            asb.AVL: {
                "spanwise_resolution": SPANWISE_RESOLUTION,
                "chordwise_resolution": CHORDWISE_RESOLUTION,
            }
        },
    )

    s_ref = float(main_wing.area())
    c_ref = float(main_wing.mean_aerodynamic_chord())
    b_ref = float(main_wing.span(include_centerline_distance=True))

    mac_le = main_wing.aerodynamic_center(chord_fraction=0.0)
    cg_fraction_mac = 0.28
    xyz_ref = np.array(
        [
            float(mac_le[0] + cg_fraction_mac * c_ref),
            0.0,
            float(mac_le[2]),
        ]
    )

    airplane = asb.Airplane(
        name="Notional Conventional AVL Validation Aircraft",
        xyz_ref=xyz_ref,
        wings=[main_wing, horizontal_tail, vertical_tail],
        s_ref=s_ref,
        c_ref=c_ref,
        b_ref=b_ref,
    )

    metadata = {
        "cg_fraction_mac": cg_fraction_mac,
        "xyz_ref": list(map(float, airplane.xyz_ref)),
        "s_ref": airplane.s_ref,
        "c_ref": airplane.c_ref,
        "b_ref": airplane.b_ref,
        "main_wing_mac_le": list(map(float, mac_le)),
        "wing_names": [wing.name for wing in airplane.wings],
    }
    return airplane, metadata


def make_op_point(
    *,
    alpha: float = 4.0,
    beta: float = 0.0,
    p: float = 0.0,
    q: float = 0.0,
    r: float = 0.0,
) -> asb.OperatingPoint:
    return asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=0),
        velocity=VELOCITY,
        alpha=alpha,
        beta=beta,
        p=p,
        q=q,
        r=r,
    )


def force_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    for alpha in [-4.0, 0.0, 4.0, 8.0, 12.0]:
        cases.append(
            {
                "name": f"alpha_{alpha:+.0f}",
                "alpha": alpha,
                "beta": 0.0,
                "p": 0.0,
                "q": 0.0,
                "r": 0.0,
            }
        )

    for beta in [-8.0, -4.0, 4.0, 8.0]:
        cases.append(
            {
                "name": f"beta_{beta:+.0f}",
                "alpha": 4.0,
                "beta": beta,
                "p": 0.0,
                "q": 0.0,
                "r": 0.0,
            }
        )

    for rate_name, p, q, r in [
        ("p_pos", 0.50, 0.0, 0.0),
        ("p_neg", -0.50, 0.0, 0.0),
        ("q_pos", 0.0, 2.00, 0.0),
        ("q_neg", 0.0, -2.00, 0.0),
        ("r_pos", 0.0, 0.0, 0.50),
        ("r_neg", 0.0, 0.0, -0.50),
    ]:
        cases.append(
            {
                "name": rate_name,
                "alpha": 4.0,
                "beta": 0.0,
                "p": p,
                "q": q,
                "r": r,
            }
        )

    return cases


def derivative_cases() -> List[Dict[str, Any]]:
    return [
        {"name": "neutral", "alpha": 0.0, "beta": 0.0},
        {"name": "design", "alpha": 4.0, "beta": 0.0},
        {"name": "sideslip", "alpha": 4.0, "beta": 5.0},
    ]


def run_vlm(airplane: asb.Airplane, op_point: asb.OperatingPoint) -> Dict[str, Any]:
    return asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=airplane.xyz_ref,
        spanwise_resolution=SPANWISE_RESOLUTION,
        chordwise_resolution=CHORDWISE_RESOLUTION,
        align_trailing_vortices_with_wind=False,
    ).run()


def run_vlm_derivatives(
    airplane: asb.Airplane, op_point: asb.OperatingPoint
) -> Dict[str, Any]:
    return asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=airplane.xyz_ref,
        spanwise_resolution=SPANWISE_RESOLUTION,
        chordwise_resolution=CHORDWISE_RESOLUTION,
        align_trailing_vortices_with_wind=False,
    ).run_with_stability_derivatives()


def run_lifting_line(
    airplane: asb.Airplane, op_point: asb.OperatingPoint
) -> Dict[str, Any]:
    return asb.LiftingLine(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=airplane.xyz_ref,
        spanwise_resolution=SPANWISE_RESOLUTION,
        align_trailing_vortices_with_wind=False,
        model_size="xsmall",
    ).run()


def run_lifting_line_derivatives(
    airplane: asb.Airplane, op_point: asb.OperatingPoint
) -> Dict[str, Any]:
    return asb.LiftingLine(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=airplane.xyz_ref,
        spanwise_resolution=SPANWISE_RESOLUTION,
        align_trailing_vortices_with_wind=False,
        model_size="xsmall",
    ).run_with_stability_derivatives()


def run_avl(airplane: asb.Airplane, op_point: asb.OperatingPoint) -> Dict[str, Any]:
    ensure_avl_tempfile_patch()
    AVL_WORK_DIR.mkdir(parents=True, exist_ok=True)
    return asb.AVL(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=airplane.xyz_ref,
        avl_command=str(AVL_EXE.resolve()),
        working_directory=str(AVL_WORK_DIR.resolve()),
        timeout=30,
    ).run()


def scalar(value: Any) -> float:
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).reshape(-1)[0])
    return float(value)


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-12)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare_force_coefficients(
    case: Dict[str, Any], results: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    value_rows = []
    comparison_rows = []

    for method, result in results.items():
        row = {
            "case": case["name"],
            "method": method,
            "alpha": case["alpha"],
            "beta": case["beta"],
            "p": case["p"],
            "q": case["q"],
            "r": case["r"],
        }
        for key in COEFFICIENT_KEYS:
            row[key] = scalar(result[key])
        value_rows.append(row)

    for method in ["VLM", "LiftingLine"]:
        row = {
            "case": case["name"],
            "method": method,
            "reference": "AVL",
        }
        for key in COEFFICIENT_KEYS:
            value = scalar(results[method][key])
            reference = scalar(results["AVL"][key])
            row[f"{key}_abs_error"] = value - reference
            row[f"{key}_rel_error"] = relative_error(value, reference)
        comparison_rows.append(row)

    return value_rows, comparison_rows


def compare_derivatives(
    case: Dict[str, Any], results: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    value_rows = []
    comparison_rows = []

    for method, result in results.items():
        row = {
            "case": case["name"],
            "method": method,
            "alpha": case["alpha"],
            "beta": case["beta"],
        }
        for key in COEFFICIENT_KEYS + DERIVATIVE_KEYS + ["x_np", "Xnp"]:
            if key in result:
                row[key] = scalar(result[key])
        value_rows.append(row)

    for method in ["VLM", "LiftingLine"]:
        for key in DERIVATIVE_KEYS:
            if key not in results[method] or key not in results["AVL"]:
                continue
            value = scalar(results[method][key])
            reference = scalar(results["AVL"][key])
            comparison_rows.append(
                {
                    "case": case["name"],
                    "method": method,
                    "reference": "AVL",
                    "derivative": key,
                    "value": value,
                    "reference_value": reference,
                    "abs_error": value - reference,
                    "rel_error": relative_error(value, reference),
                }
            )

    return value_rows, comparison_rows


def spanwise_distribution_checks(
    airplane: asb.Airplane,
    method: str,
    result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []

    for wing_index, wing in enumerate(airplane.wings):
        y = np.asarray(result["y"][wing_index], dtype=float)
        dy = np.asarray(result["dy"][wing_index], dtype=float)
        lift = np.asarray(result["lift"][wing_index], dtype=float)
        cl = np.asarray(result["cl"][wing_index], dtype=float)
        clc_over_cref = np.asarray(result["clc_over_cref"][wing_index], dtype=float)

        has_projected_y_span = bool(np.max(np.abs(dy)) > 1e-10)
        finite_projected_y_values = bool(
            np.all(np.isfinite(y))
            and np.all(np.isfinite(dy))
            and np.all(np.isfinite(lift))
            and (
                not has_projected_y_span
                or (np.all(np.isfinite(cl)) and np.all(np.isfinite(clc_over_cref)))
            )
        )

        if has_projected_y_span:
            magnitudes = np.abs(clc_over_cref)
            finite_magnitudes = magnitudes[np.isfinite(magnitudes)]
            median_magnitude = float(np.median(finite_magnitudes) + 1e-12)
            spike_ratio = float(np.max(finite_magnitudes) / median_magnitude)
            max_abs_clc_over_cref = float(np.max(finite_magnitudes))
        else:
            spike_ratio = math.nan
            max_abs_clc_over_cref = math.nan

        rows.append(
            {
                "method": method,
                "wing": wing.name,
                "has_projected_y_span": has_projected_y_span,
                "n_strips": len(y),
                "finite_horizontal_lift_distribution": finite_projected_y_values,
                "max_abs_lift": float(np.max(np.abs(lift))),
                "max_abs_clc_over_cref": max_abs_clc_over_cref,
                "clc_over_cref_spike_ratio": spike_ratio,
                "sum_lift": float(np.sum(lift)),
            }
        )

    return rows


def audit_cg_written_to_avl(metadata: Dict[str, Any]) -> Dict[str, Any]:
    avl_file = AVL_WORK_DIR / "airplane.avl"
    lines = avl_file.read_text().splitlines()
    xref_index = next(
        i for i, line in enumerate(lines) if line.strip().startswith("#Xref")
    )
    xref_values = [float(value) for value in lines[xref_index + 1].split()[:3]]
    expected = metadata["xyz_ref"]
    errors = [xref_values[i] - expected[i] for i in range(3)]
    return {
        "avl_file": str(avl_file),
        "xref_values": xref_values,
        "expected_xyz_ref": expected,
        "errors": errors,
        "max_abs_error": max(abs(error) for error in errors),
    }


def shifted_cg_moment_check(airplane: asb.Airplane) -> Dict[str, Any]:
    op_point = make_op_point(alpha=4.0, beta=0.0)
    base = run_vlm(airplane, op_point)

    dx = 0.10 * airplane.c_ref
    shifted_xyz_ref = np.array(airplane.xyz_ref) + np.array([dx, 0.0, 0.0])
    shifted = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=shifted_xyz_ref,
        spanwise_resolution=SPANWISE_RESOLUTION,
        chordwise_resolution=CHORDWISE_RESOLUTION,
        align_trailing_vortices_with_wind=False,
    ).run()

    qS = op_point.dynamic_pressure() * airplane.s_ref
    expected_delta_cm = scalar(base["F_g"][2]) * dx / (qS * airplane.c_ref)
    actual_delta_cm = scalar(shifted["Cm"]) - scalar(base["Cm"])

    return {
        "dx": dx,
        "base_Cm": scalar(base["Cm"]),
        "shifted_Cm": scalar(shifted["Cm"]),
        "actual_delta_Cm": actual_delta_cm,
        "expected_delta_Cm": expected_delta_cm,
        "abs_error": actual_delta_cm - expected_delta_cm,
    }


def make_lift_distribution_plot(
    airplane: asb.Airplane,
    vlm_result: Dict[str, Any],
    lifting_line_result: Dict[str, Any],
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    horizontal_wing_indices = [
        i
        for i, dy in enumerate(vlm_result["dy"])
        if np.max(np.abs(np.asarray(dy, dtype=float))) > 1e-10
    ]

    fig, axes = plt.subplots(
        len(horizontal_wing_indices),
        1,
        figsize=(8, 3.0 * len(horizontal_wing_indices)),
        squeeze=False,
    )

    for axis, wing_index in zip(axes[:, 0], horizontal_wing_indices):
        wing = airplane.wings[wing_index]
        for method, result in [
            ("VLM", vlm_result),
            ("LiftingLine", lifting_line_result),
        ]:
            axis.plot(
                result["y"][wing_index],
                result["clc_over_cref"][wing_index],
                marker="o",
                label=method,
            )
        axis.set_title(wing.name)
        axis.set_xlabel("y [m]")
        axis.set_ylabel("cl * c / cref [-]")
        axis.grid(True, alpha=0.35)
        axis.legend()

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "lift_distributions.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def summarize_errors(
    force_comparisons: List[Dict[str, Any]],
    derivative_comparisons: List[Dict[str, Any]],
    distribution_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for method in ["VLM", "LiftingLine"]:
        method_rows = [row for row in force_comparisons if row["method"] == method]
        summary[f"{method}_force_max_abs_error"] = {
            key: max(abs(row[f"{key}_abs_error"]) for row in method_rows)
            for key in COEFFICIENT_KEYS
        }
        summary[f"{method}_force_max_rel_error"] = {
            key: max(row[f"{key}_rel_error"] for row in method_rows)
            for key in COEFFICIENT_KEYS
        }

        derivative_rows = [
            row for row in derivative_comparisons if row["method"] == method
        ]
        summary[f"{method}_derivative_max_abs_error"] = max(
            abs(row["abs_error"]) for row in derivative_rows
        )
        summary[f"{method}_derivative_max_rel_error"] = max(
            row["rel_error"] for row in derivative_rows
        )
        worst_derivative = max(
            derivative_rows,
            key=lambda row: abs(row["abs_error"]),
        )
        summary[f"{method}_worst_derivative_abs_error"] = worst_derivative

    summary["horizontal_distribution_checks_pass"] = all(
        row["finite_horizontal_lift_distribution"]
        and (
            not row["has_projected_y_span"]
            or row["clc_over_cref_spike_ratio"] < 8.0
        )
        for row in distribution_rows
    )
    return summary


def main() -> None:
    if not AVL_EXE.exists():
        raise FileNotFoundError(f"AVL executable not found: {AVL_EXE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AVL_WORK_DIR.mkdir(parents=True, exist_ok=True)
    ensure_avl_tempfile_patch()

    airplane, metadata = make_airplane()

    force_value_rows: List[Dict[str, Any]] = []
    force_comparison_rows: List[Dict[str, Any]] = []

    for case in force_cases():
        op_point = make_op_point(
            alpha=case["alpha"],
            beta=case["beta"],
            p=case["p"],
            q=case["q"],
            r=case["r"],
        )
        results = {
            "AVL": run_avl(airplane, op_point),
            "VLM": run_vlm(airplane, op_point),
            "LiftingLine": run_lifting_line(airplane, op_point),
        }
        values, comparisons = compare_force_coefficients(case, results)
        force_value_rows.extend(values)
        force_comparison_rows.extend(comparisons)

    derivative_value_rows: List[Dict[str, Any]] = []
    derivative_comparison_rows: List[Dict[str, Any]] = []

    for case in derivative_cases():
        op_point = make_op_point(alpha=case["alpha"], beta=case["beta"])
        results = {
            "AVL": run_avl(airplane, op_point),
            "VLM": run_vlm_derivatives(airplane, op_point),
            "LiftingLine": run_lifting_line_derivatives(airplane, op_point),
        }
        values, comparisons = compare_derivatives(case, results)
        derivative_value_rows.extend(values)
        derivative_comparison_rows.extend(comparisons)

    distribution_op_point = make_op_point(alpha=4.0, beta=0.0)
    vlm_distribution_result = run_vlm(airplane, distribution_op_point)
    lifting_line_distribution_result = run_lifting_line(airplane, distribution_op_point)
    distribution_rows = []
    distribution_rows.extend(
        spanwise_distribution_checks(airplane, "VLM", vlm_distribution_result)
    )
    distribution_rows.extend(
        spanwise_distribution_checks(
            airplane, "LiftingLine", lifting_line_distribution_result
        )
    )

    lift_plot_path = make_lift_distribution_plot(
        airplane, vlm_distribution_result, lifting_line_distribution_result
    )

    cg_audit = audit_cg_written_to_avl(metadata)
    shifted_cg_check = shifted_cg_moment_check(airplane)

    summary = summarize_errors(
        force_comparison_rows,
        derivative_comparison_rows,
        distribution_rows,
    )
    summary["metadata"] = metadata
    summary["cg_audit"] = cg_audit
    summary["shifted_cg_moment_check"] = shifted_cg_check
    summary["lift_distribution_plot"] = str(lift_plot_path) if lift_plot_path else None

    write_csv(OUTPUT_DIR / "force_coefficients.csv", force_value_rows)
    write_csv(OUTPUT_DIR / "force_coefficient_comparisons.csv", force_comparison_rows)
    write_csv(OUTPUT_DIR / "stability_derivatives.csv", derivative_value_rows)
    write_csv(
        OUTPUT_DIR / "stability_derivative_comparisons.csv",
        derivative_comparison_rows,
    )
    write_csv(OUTPUT_DIR / "spanwise_distribution_checks.csv", distribution_rows)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
