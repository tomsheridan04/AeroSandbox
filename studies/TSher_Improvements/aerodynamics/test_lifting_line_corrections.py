"""Integration and differentiability checks for the experimental study solver."""

import aerosandbox as asb
import casadi as cas
import numpy as np
import pytest

from studies.TSher_Improvements.aerodynamics.lifting_line_corrections import (
    CorrectedLiftingLine,
)


def wing_airplane(parameters, include_tail=False):
    alpha, speed, root, tip, span = [parameters[i] for i in range(5)]
    foil = asb.Airfoil("naca0012")
    wings = [
        asb.Wing(
            name="Wing",
            symmetric=True,
            xsecs=[
                asb.WingXSec(xyz_le=[0, 0, 0], chord=root, airfoil=foil),
                asb.WingXSec(xyz_le=[0.2, span / 2, 0], chord=tip, airfoil=foil),
            ],
        )
    ]
    if include_tail:
        wings += [
            asb.Wing(
                name="Tail",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[2, 0, 0.1], chord=0.35, airfoil=foil),
                    asb.WingXSec(xyz_le=[2.1, 0.7, 0.1], chord=0.25, airfoil=foil),
                ],
            ),
            asb.Wing(
                name="Fin",
                symmetric=False,
                xsecs=[
                    asb.WingXSec(xyz_le=[1.8, 0, 0.1], chord=0.4, airfoil=foil),
                    asb.WingXSec(xyz_le=[2, 0, 0.5], chord=0.2, airfoil=foil),
                ],
            ),
        ]
    return asb.Airplane(
        name="Test",
        xyz_ref=[0.25, 0, 0],
        wings=wings,
        s_ref=span * (root + tip) / 2,
        c_ref=0.8,
        b_ref=span,
    )


def test_zero_corrections_reproduces_original_swept_wing():
    plane = wing_airplane([4, 30, 1, 0.6, 4])
    common = dict(
        airplane=plane,
        op_point=asb.OperatingPoint(velocity=30, alpha=4),
        spanwise_resolution=4,
        model_size="small",
        n_crit=5,
        xtr_upper=0.3,
        xtr_lower=0.3,
    )
    original = asb.LiftingLine(**common).run()
    clone = CorrectedLiftingLine(
        **common, slope_model="two_pi", circulation_corrections=0
    ).run()
    for key in ["CL", "CD", "CY", "Cl", "Cm", "Cn", "spanwise_lift", "spanwise_cl"]:
        np.testing.assert_allclose(clone[key], original[key], rtol=1e-10, atol=1e-12)


def test_corrections_reduce_sectional_consistency_error():
    plane = wing_airplane([5, 30, 1, 0.6, 4])
    common = dict(
        airplane=plane,
        op_point=asb.OperatingPoint(velocity=30, alpha=5),
        spanwise_resolution=4,
        model_size="small",
    )
    initial = CorrectedLiftingLine(**common, circulation_corrections=0).run()
    corrected = CorrectedLiftingLine(**common, circulation_corrections=2).run()
    assert np.max(np.abs(corrected["sectional_residual"])) < 0.3 * np.max(
        np.abs(initial["sectional_residual"])
    )
    np.testing.assert_allclose(
        corrected["CD"], corrected["CD_induced"] + corrected["CD_profile"], atol=1e-12
    )


@pytest.mark.parametrize(
    "kinematics,slope_model", [("legacy", "attached"), ("normal", "local")]
)
def test_symbolic_geometry_values_and_jacobians_match_numeric(kinematics, slope_model):
    x = cas.MX.sym("design", 5)
    plane = wing_airplane(x)
    common = dict(
        model_size="small",
        spanwise_resolution=3,
        kinematics=kinematics,
        slope_model=slope_model,
        circulation_corrections=2,
    )
    symbolic = CorrectedLiftingLine(
        airplane=plane, op_point=asb.OperatingPoint(alpha=x[0], velocity=x[1]), **common
    ).run()
    values = cas.vertcat(*[symbolic[key] for key in ["CL", "CD", "Cm"]])
    function = cas.Function("aero", [x], [values, cas.jacobian(values, x)])
    for point in [np.array([4, 30, 1, 0.6, 4]), np.array([10, 24, 0.9, 0.55, 3.5])]:
        value, jacobian = [np.asarray(v) for v in function(point)]
        numeric = CorrectedLiftingLine(
            airplane=wing_airplane(point),
            op_point=asb.OperatingPoint(alpha=point[0], velocity=point[1]),
            **common
        ).run()
        np.testing.assert_allclose(
            value.ravel(),
            [numeric[key] for key in ["CL", "CD", "Cm"]],
            rtol=1e-8,
            atol=1e-10,
        )
        finite_difference = np.empty((3, 5))
        for j in range(5):
            step = 1e-5 * max(1, abs(point[j]))
            delta = np.zeros(5)
            delta[j] = step
            finite_difference[:, j] = (
                np.asarray(function(point + delta)[0]).ravel()
                - np.asarray(function(point - delta)[0]).ravel()
            ) / (2 * step)
        assert np.isfinite(jacobian).all()
        np.testing.assert_allclose(jacobian, finite_difference, rtol=2e-3, atol=2e-5)


def test_ctol_interface_multiple_surfaces_fuselage_and_derivatives():
    plane = wing_airplane([4, 25, 1, 0.6, 4], include_tail=True)
    plane.fuselages = [
        asb.Fuselage(
            xsecs=[
                asb.FuselageXSec(xyz_c=[-0.2, 0, -0.1], radius=0.01),
                asb.FuselageXSec(xyz_c=[0.5, 0, -0.1], radius=0.12),
                asb.FuselageXSec(xyz_c=[2.1, 0, -0.1], radius=0.01),
            ]
        )
    ]
    analysis = CorrectedLiftingLine(
        airplane=plane,
        op_point=asb.OperatingPoint(velocity=25, alpha=4),
        model_size="small",
        spanwise_resolution=3,
        kinematics="normal",
        n_crit=5,
        xtr_upper=0.3,
        xtr_lower=0.3,
    )
    result = analysis.run_with_stability_derivatives(
        alpha=True, beta=False, p=False, q=True, r=False
    )
    for key in ["CL", "L", "CD", "D", "Cm", "x_np", "CLa", "Cma", "CLq", "Cmq"]:
        assert np.isfinite(result[key]).all(), key
    assert len(analysis.vortex_strengths) == 15
    assert len(result["y"]) == 3
    assert analysis["CL"] == result["CL"]
    velocities = analysis.get_velocity_at_points(
        np.array([[0.3, 0.4, 0.1], [1.0, 0.6, 0.2]])
    )
    assert np.isfinite(velocities).all()
    assert len(result["fuselage_aero_components"]) == 1


def test_normal_flow_rate_response_is_translation_invariant():
    original = wing_airplane([4, 25, 1, 0.6, 4], include_tail=True)
    # Symmetric Wing objects mirror about global y=0; preserve that plane.
    delta = np.array([3, 0, 0.7])
    translated = asb.Airplane(
        xyz_ref=np.array(original.xyz_ref) + delta,
        wings=[w.translate(delta) for w in original.wings],
        s_ref=original.s_ref,
        c_ref=original.c_ref,
        b_ref=original.b_ref,
    )
    common = dict(
        op_point=asb.OperatingPoint(velocity=25, alpha=4, beta=3, p=0.1, q=0.2, r=0.05),
        spanwise_resolution=3,
        model_size="small",
        kinematics="normal",
    )
    a = CorrectedLiftingLine(airplane=original, **common).run()
    b = CorrectedLiftingLine(airplane=translated, **common).run()
    for key in ["CL", "CD", "CY", "Cl", "Cm", "Cn"]:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-9, atol=1e-11)


def test_symbolic_control_deflection_and_antisymmetric_mirroring():
    deflection = cas.MX.sym("aileron_degrees")
    foil = asb.Airfoil("naca0012")
    surface = asb.ControlSurface(
        symmetric=False, hinge_point=0.75, deflection=deflection
    )
    plane = asb.Airplane(
        wings=[
            asb.Wing(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=1,
                        airfoil=foil,
                        control_surfaces=[surface],
                    ),
                    asb.WingXSec(
                        xyz_le=[0.1, 2, 0],
                        chord=0.7,
                        airfoil=foil,
                        control_surfaces=[surface],
                    ),
                ],
            )
        ]
    )
    result = CorrectedLiftingLine(
        airplane=plane,
        op_point=asb.OperatingPoint(velocity=25, alpha=4),
        spanwise_resolution=3,
        model_size="small",
        kinematics="normal",
    ).run()
    function = cas.Function(
        "control", [deflection], [result["Cl"], cas.jacobian(result["Cl"], deflection)]
    )
    positive = np.array(function(2)).ravel()
    negative = np.array(function(-2)).ravel()
    assert np.isfinite(positive).all()
    assert abs(positive[1]) > 1e-4
    np.testing.assert_allclose(positive[0], -negative[0], atol=1e-7)


def test_symbolic_stability_derivatives_and_neutral_point():
    x = cas.MX.sym("ctol_design", 5)
    analysis = CorrectedLiftingLine(
        airplane=wing_airplane(x, include_tail=True),
        op_point=asb.OperatingPoint(alpha=x[0], velocity=x[1]),
        spanwise_resolution=2,
        model_size="xsmall",
        kinematics="normal",
    )
    result = analysis.run_with_stability_derivatives(
        alpha=True, beta=False, p=False, q=False, r=False
    )
    output = cas.vertcat(
        *[result[key] for key in ["CL", "CD", "Cm", "CLa", "Cma", "x_np"]]
    )
    function = cas.Function("ctol_stability", [x], [output, cas.jacobian(output, x)])
    point = np.array([4, 25, 1, 0.6, 4])
    value, jacobian = [np.asarray(v) for v in function(point)]
    assert np.isfinite(value).all()
    assert np.isfinite(jacobian).all()
    # Parent stability derivatives are per radian; the design alpha is degrees.
    np.testing.assert_allclose(
        value[3, 0], jacobian[0, 0] * 180 / np.pi, rtol=2e-4, atol=1e-5
    )
