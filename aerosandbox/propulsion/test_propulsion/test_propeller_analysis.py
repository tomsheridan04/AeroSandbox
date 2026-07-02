import pytest

import aerosandbox as asb
import aerosandbox.numpy as np


def _smooth_section_aerodynamics(airfoil, alpha, Re, mach, r_over_R):
    cl_linear = 2 * np.pi * np.radians(alpha)
    CL = 1.35 * np.tanh(cl_linear / 1.35)
    CD = 0.018 + 0.04 * CL**2

    return {
        "CL": CL,
        "CD": CD,
        "CM": -0.04 * CL,
        "analysis_confidence": 1.0,
    }


def _test_propeller():
    return asb.Propeller(
        name="Test Propeller",
        radius=0.1524,
        hub_radius=0.018,
        blade_count=2,
        radial_stations=np.linspace(0.12, 1.0, 5),
        chord_distribution=[0.020, 0.026, 0.023, 0.015, 0.006],
        twist_distribution=[38, 31, 24, 18, 12],
        thickness_distribution=[0.18, 0.15, 0.12, 0.10, 0.09],
        airfoil_distribution=[
            (0.12, "e63"),
            (0.75, "naca4412"),
            (1.00, "naca4412"),
        ],
    )


def test_propeller_geometry_evaluates_smooth_distributions():
    propeller = _test_propeller()
    r_over_R = np.linspace(0.15, 0.95, 7)

    chord = propeller.chord(r_over_R)
    twist = propeller.twist(r_over_R)
    thickness = propeller.thickness(r_over_R)

    assert len(chord) == len(r_over_R)
    assert len(twist) == len(r_over_R)
    assert len(thickness) == len(r_over_R)
    assert np.all(chord > 0)
    assert np.all(thickness > 0)

    section_airfoil = propeller.airfoil(0.5)
    assert section_airfoil.max_thickness() == pytest.approx(propeller.thickness(0.5))


def test_propeller_from_tabulated_geometry_preserves_table_values_by_default():
    inch = 0.0254
    r = np.array([1, 2, 3, 4, 5, 6]) * inch
    chord = np.array([0.7, 1.0, 1.1, 0.8, 0.4, 0.02]) * inch
    twist = np.array([45, 32, 24, 18, 14, 12])

    propeller = asb.Propeller.from_tabulated_geometry(
        name="Endpoint Test Propeller",
        r=r,
        chord=chord,
        twist=twist,
        radius=6 * inch,
        hub_radius=0.4 * inch,
        n_control_points=6,
    )

    r_over_R = r / propeller.radius

    assert propeller.distribution_interpolation_method == "linear"
    assert propeller.chord(r_over_R[0]) == pytest.approx(chord[0])
    assert propeller.chord(r_over_R[2]) == pytest.approx(chord[2])
    assert propeller.chord(1.0) == pytest.approx(chord[-1])
    assert propeller.twist(r_over_R[0]) == pytest.approx(twist[0])
    assert propeller.twist(r_over_R[3]) == pytest.approx(twist[3])
    assert propeller.twist(1.0) == pytest.approx(twist[-1])


def test_propeller_allows_symbolic_b_spline_control_points():
    opti = asb.Opti()
    chord_control_points = opti.variable(
        init_guess=[0.020, 0.026, 0.020, 0.008],
        lower_bound=0.001,
    )
    twist_control_points = opti.variable(
        init_guess=[35, 25, 18, 12],
    )

    propeller = asb.Propeller(
        radius=0.1524,
        hub_radius=0.018,
        blade_count=2,
        radial_stations=np.linspace(0.12, 1.0, 4),
        chord_distribution=chord_control_points,
        twist_distribution=twist_control_points,
        distribution_interpolation_method="bspline",
    )

    r_over_R = np.linspace(0.2, 0.9, 5)
    assert np.is_casadi_type(propeller.chord(r_over_R), recursive=False)
    assert np.is_casadi_type(propeller.twist(r_over_R), recursive=False)


def test_propeller_thickness_scaling_preserves_camber_line():
    base_propeller = asb.Propeller(
        radius=1.0,
        hub_radius=0.1,
        blade_count=2,
        radial_stations=[0.1, 1.0],
        chord_distribution=0.1,
        twist_distribution=20,
        airfoil_distribution="e63",
    )
    base_airfoil = base_propeller.airfoil(0.5)
    target_thickness = 1.3 * base_airfoil.max_thickness()

    thickened_propeller = asb.Propeller(
        radius=1.0,
        hub_radius=0.1,
        blade_count=2,
        radial_stations=[0.1, 1.0],
        chord_distribution=0.1,
        twist_distribution=20,
        thickness_distribution=target_thickness,
        airfoil_distribution="e63",
    )
    thickened_airfoil = thickened_propeller.airfoil(0.5)

    x_over_c = np.linspace(0.05, 0.95, 20)
    assert thickened_airfoil.max_thickness() == pytest.approx(target_thickness)
    assert thickened_airfoil.local_camber(x_over_c) == pytest.approx(
        base_airfoil.local_camber(x_over_c)
    )


def test_propeller_rejects_nonphysical_geometry():
    with pytest.raises(ValueError):
        asb.Propeller(
            radius=1,
            hub_radius=1.1,
            chord_distribution=0.1,
            twist_distribution=20,
        )

    with pytest.raises(ValueError):
        asb.Propeller.from_tabulated_geometry(
            name="Bad Propeller",
            r=[0.2, 0.4, 0.6],
            chord=[0.1, -0.1, 0.05],
            twist=[30, 20, 10],
            radius=0.6,
        )


def test_propeller_analysis_returns_integrated_and_spanwise_loads():
    propeller = _test_propeller()
    analysis = asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=10),
        rpm=6500,
        radial_resolution=8,
        newton_iterations=7,
        model_size="xsmall",
        section_aerodynamics=_smooth_section_aerodynamics,
    )
    output = analysis.run()

    assert output["thrust"] > 0
    assert output["torque"] > 0
    assert output["power"] > 0
    assert output["C_T"] > 0
    assert output["C_P"] > 0
    assert len(output["r"]) == 8
    assert len(output["alpha"]) == 8
    assert len(output["post_stall_blend_fraction"]) == 8
    assert np.all(output["post_stall_blend_fraction"] == 0)
    assert len(output["root_loss_factor"]) == 8
    assert len(output["finite_blade_loss_factor"]) == 8
    assert analysis["thrust"] is output["thrust"]
    assert np.sum(output["dT"]) == pytest.approx(output["thrust"])
    assert np.sum(output["dQ"]) == pytest.approx(output["torque"])
    assert np.max(np.abs(output["residual"])) < 1e-3


def test_propeller_momentum_analysis_returns_integrated_and_spanwise_loads():
    propeller = _test_propeller()
    analysis = asb.PropellerMomentumAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=10),
        rpm=6500,
        radial_resolution=8,
        newton_iterations=12,
        residual_tolerance=0.1,
        section_aerodynamics=_smooth_section_aerodynamics,
    )
    output = analysis.run()

    assert output["thrust"] > 0
    assert output["torque"] > 0
    assert output["power"] > 0
    assert output["C_T"] > 0
    assert output["C_P"] > 0
    assert len(output["r"]) == 8
    assert len(output["axial_induced_velocity"]) == 8
    assert len(output["tangential_induced_velocity"]) == 8
    assert len(output["root_loss_factor"]) == 8
    assert analysis["thrust"] is output["thrust"]
    assert np.sum(output["dT"]) == pytest.approx(output["thrust"])
    assert np.sum(output["dQ"]) == pytest.approx(output["torque"])
    assert np.all(output["axial_induced_velocity"] >= 0)
    assert np.all(output["tangential_induced_velocity"] >= 0)
    assert np.max(output["finite_blade_loss_factor"]) <= 1 + 1e-12
    assert np.max(output["residual"]) < 0.1


def test_propeller_analysis_root_loss_tapers_inner_loading():
    propeller = _test_propeller()

    with_root_loss = asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=10),
        rpm=6500,
        radial_resolution=10,
        newton_iterations=7,
        section_aerodynamics=_smooth_section_aerodynamics,
        include_root_loss=True,
    ).run()

    without_root_loss = asb.PropellerAnalysis(
        propeller=propeller,
        op_point=asb.OperatingPoint(velocity=10),
        rpm=6500,
        radial_resolution=10,
        newton_iterations=7,
        section_aerodynamics=_smooth_section_aerodynamics,
        include_root_loss=False,
    ).run()

    assert np.all(with_root_loss["root_loss_factor"] <= 1)
    assert np.all(with_root_loss["root_loss_factor"] >= 0)
    assert with_root_loss["root_loss_factor"][0] < 0.6
    assert with_root_loss["root_loss_factor"][-1] > with_root_loss["root_loss_factor"][0]
    assert without_root_loss["root_loss_factor"] == pytest.approx(
        np.ones_like(without_root_loss["root_loss_factor"])
    )
    assert np.all(
        with_root_loss["finite_blade_loss_factor"]
        <= with_root_loss["tip_loss_factor"] + 1e-12
    )


def test_propeller_analysis_can_be_used_in_an_optimization_loop():
    opti = asb.Opti()
    rpm = opti.variable(init_guess=5500, lower_bound=2500, upper_bound=9000)

    output = asb.PropellerAnalysis(
        propeller=_test_propeller(),
        op_point=asb.OperatingPoint(velocity=8),
        rpm=rpm,
        radial_resolution=5,
        newton_iterations=5,
        section_aerodynamics=_smooth_section_aerodynamics,
    ).run()

    opti.subject_to(output["thrust"] >= 0.15)
    opti.minimize(output["power"])

    sol = opti.solve()

    assert sol(output["thrust"]) >= 0.15 - 1e-4
    assert 2500 - 1e-3 <= sol(rpm) <= 9000 + 1e-3


def _apc_12x8e_propeller():
    inch = 0.0254
    r_in = np.array(
        [0.6600, 1.1000, 1.8293, 2.6034, 3.3776, 4.1517, 4.9259, 5.6935, 6.0000]
    )
    chord_in = np.array(
        [0.6555, 0.7846, 1.0214, 1.1041, 1.0134, 0.8200, 0.5944, 0.4086, 0.0180]
    )
    twist_deg = np.array(
        [
            33.6194,
            47.2147,
            34.8387,
            26.0614,
            20.6548,
            17.0496,
            14.4926,
            12.6057,
            11.9867,
        ]
    )
    thickness = np.array(
        [0.3141, 0.2338, 0.1496, 0.1173, 0.1088, 0.1030, 0.0999, 0.0994, 0.0999]
    )

    return asb.Propeller.from_tabulated_geometry(
        name="APC 12x8E",
        r=r_in * inch,
        chord=chord_in * inch,
        twist=twist_deg,
        radius=6.0 * inch,
        hub_radius=0.40 * inch,
        blade_count=2,
        thickness=thickness,
        airfoil_distribution=[
            (1.10 / 6.0, "e63"),
            (4.39 / 6.0, "naca4412"),
        ],
        n_control_points=8,
    )


def test_apc_12x8e_static_coefficients_with_neuralfoil_smoke_validation():
    pytest.importorskip("neuralfoil")

    output = asb.PropellerAnalysis(
        propeller=_apc_12x8e_propeller(),
        op_point=asb.OperatingPoint(velocity=0),
        rpm=3000,
        radial_resolution=10,
        newton_iterations=7,
        model_size="xsmall",
    ).run()

    # APC PER3_12x8E.txt gives, at 3000 RPM static, Ct = 0.1076 and Cp = 0.0445.
    # The outboard APC12 section is proprietary, so this check uses a generic
    # cambered section plus the measured thickness distribution.
    assert output["Ct"] == pytest.approx(0.1076, rel=0.2)
    assert output["Cp"] == pytest.approx(0.0445, rel=0.2)


if __name__ == "__main__":
    pytest.main()
