"""Normal-flow API, force, differentiability, and broadcast regressions."""

import aerosandbox as asb
import casadi as cas
import numpy as np
import pytest


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


def test_symmetric_airfoil_force_symmetry_and_public_defaults():
    plane = wing_airplane([4, 30, 1, 0.6, 4])
    answers = []
    for alpha in [-4, 4]:
        analysis = asb.NormalFlowLiftingLine(
            airplane=plane,
            op_point=asb.OperatingPoint(velocity=30, alpha=alpha),
            spanwise_resolution=4,
            model_size="small",
        )
        answers.append(analysis.run())
        assert analysis.circulation_corrections == 2
        assert analysis.relaxation == 0.8
    assert answers[1]["CL"] > 0
    assert answers[1]["CD"] > 0
    np.testing.assert_allclose(answers[0]["CL"], -answers[1]["CL"], atol=1e-6)
    np.testing.assert_allclose(answers[0]["CD"], answers[1]["CD"], atol=1e-6)


def test_corrections_reduce_sectional_error_and_preserve_force_integrals():
    plane = wing_airplane([5, 30, 1, 0.6, 4])
    common = dict(
        airplane=plane,
        op_point=asb.OperatingPoint(velocity=30, alpha=5),
        spanwise_resolution=4,
        model_size="small",
    )
    initial = asb.NormalFlowLiftingLine(**common, circulation_corrections=0).run()
    analysis = asb.NormalFlowLiftingLine(**common)
    result = analysis.run()
    assert np.max(np.abs(result["sectional_residual"])) < 0.3 * np.max(
        np.abs(initial["sectional_residual"])
    )
    np.testing.assert_allclose(
        result["CD"], result["CD_induced"] + result["CD_profile"], atol=1e-12
    )
    wind = analysis.op_point.convert_axes(
        *analysis.force_inviscid_geometry, from_axes="geometry", to_axes="wind"
    )
    np.testing.assert_allclose(np.sum(result["spanwise_lift"]), -wind[2], atol=1e-10)
    assert np.all(np.diff(result["y"][0]) > 0)


@pytest.mark.parametrize("corrections", [0, 2])
def test_symbolic_geometry_values_and_derivatives(corrections):
    x = cas.MX.sym("design", 5)
    common = dict(
        model_size="small", spanwise_resolution=3, circulation_corrections=corrections
    )
    symbolic = asb.NormalFlowLiftingLine(
        airplane=wing_airplane(x),
        op_point=asb.OperatingPoint(alpha=x[0], velocity=x[1]),
        **common
    ).run()
    values = cas.vertcat(*[symbolic[key] for key in ["CL", "CD", "Cm"]])
    function = cas.Function("normal_flow", [x], [values, cas.jacobian(values, x)])
    for point in [np.array([4, 30, 1, 0.6, 4]), np.array([10, 24, 0.9, 0.55, 3.5])]:
        value, jacobian = [np.asarray(v) for v in function(point)]
        numeric = asb.NormalFlowLiftingLine(
            airplane=wing_airplane(point),
            op_point=asb.OperatingPoint(alpha=point[0], velocity=point[1]),
            **common
        ).run()
        np.testing.assert_allclose(
            value.ravel(),
            [numeric[k] for k in ["CL", "CD", "Cm"]],
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


@pytest.mark.parametrize("corrections", [-1, 9, 1.5, True])
def test_invalid_correction_count(corrections):
    with pytest.raises(ValueError, match="circulation_corrections"):
        asb.NormalFlowLiftingLine(
            airplane=wing_airplane([4, 25, 1, 0.6, 4]),
            op_point=asb.OperatingPoint(velocity=25),
            circulation_corrections=corrections,
        )


@pytest.mark.parametrize("relaxation", [0, -0.1, 1.1])
def test_invalid_relaxation(relaxation):
    with pytest.raises(ValueError, match="relaxation"):
        asb.NormalFlowLiftingLine(
            airplane=wing_airplane([4, 25, 1, 0.6, 4]),
            op_point=asb.OperatingPoint(velocity=25),
            relaxation=relaxation,
        )


@pytest.mark.parametrize("analysis_type", [asb.LiftingLine, asb.VortexLatticeMethod])
def test_stashed_symbolic_coordinate_broadcast_fix(analysis_type):
    chord = cas.MX.sym("chord")
    plane = wing_airplane([4, 25, chord, 0.6 * chord, 4])
    kwargs = dict(
        airplane=plane,
        op_point=asb.OperatingPoint(velocity=25, alpha=4, q=0.2),
        spanwise_resolution=2,
    )
    if analysis_type is asb.LiftingLine:
        kwargs["model_size"] = "xsmall"
    else:
        kwargs["chordwise_resolution"] = 2
    analysis = analysis_type(**kwargs)
    result = analysis.run()
    velocities = analysis.get_velocity_at_points(analysis.vortex_centers)
    values = cas.vertcat(result["CL"], result["CD"], result["Cm"], cas.vec(velocities))
    function = cas.Function("coordinate_broadcast", [chord], [values])
    assert np.isfinite(np.asarray(function(1))).all()


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
    analysis = asb.NormalFlowLiftingLine(
        airplane=plane,
        op_point=asb.OperatingPoint(velocity=25, alpha=4),
        model_size="small",
        spanwise_resolution=3,
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
    )
    a = asb.NormalFlowLiftingLine(airplane=original, **common).run()
    b = asb.NormalFlowLiftingLine(airplane=translated, **common).run()
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
    result = asb.NormalFlowLiftingLine(
        airplane=plane,
        op_point=asb.OperatingPoint(velocity=25, alpha=4),
        spanwise_resolution=3,
        model_size="small",
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
    analysis = asb.NormalFlowLiftingLine(
        airplane=wing_airplane(x, include_tail=True),
        op_point=asb.OperatingPoint(alpha=x[0], velocity=x[1]),
        spanwise_resolution=2,
        model_size="xsmall",
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
