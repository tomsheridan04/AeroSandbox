import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def _rectangular_wing_airplane(include_tail=False):
    airfoil = asb.Airfoil("naca0012")
    wings = [
        asb.Wing(
            name="Main Wing",
            symmetric=True,
            xsecs=[
                asb.WingXSec(xyz_le=[0, 0, 0], chord=1, airfoil=airfoil),
                asb.WingXSec(xyz_le=[0, 2, 0], chord=1, airfoil=airfoil),
            ],
        )
    ]

    if include_tail:
        wings.append(
            asb.Wing(
                name="Horizontal Tail",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[3, 0, 0], chord=0.5, airfoil=airfoil),
                    asb.WingXSec(xyz_le=[3, 1, 0], chord=0.5, airfoil=airfoil),
                ],
            )
        )

    return asb.Airplane(
        name="Spanwise Distribution Test Airplane",
        wings=wings,
    )


def test_lifting_line_spanwise_lift_distribution_integrates_to_inviscid_lift():
    airplane = _rectangular_wing_airplane()
    op_point = asb.OperatingPoint(velocity=10, alpha=5)
    spanwise_resolution = 3
    analysis = asb.LiftingLine(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=spanwise_resolution,
        model_size="xsmall",
    )
    aero = analysis.run()

    expected_spanwise_strips = 2 * spanwise_resolution
    inviscid_force_wind = op_point.convert_axes(
        *analysis.force_inviscid_geometry,
        from_axes="geometry",
        to_axes="wind",
    )
    inviscid_lift = -inviscid_force_wind[2]

    assert len(aero["spanwise_y"]) == expected_spanwise_strips
    assert len(aero["y"]) == 1
    assert len(aero["y"][0]) == expected_spanwise_strips
    assert len(aero["cl"][0]) == expected_spanwise_strips
    assert len(aero["clc_over_cref"][0]) == expected_spanwise_strips
    assert np.all(np.diff(aero["y"][0]) > 0)
    assert analysis["y"][0] is aero["y"][0]
    assert np.sum(aero["spanwise_lift"]) == pytest.approx(inviscid_lift)
    assert np.sum(aero["spanwise_lift_per_y"] * aero["spanwise_dy"]) == pytest.approx(
        inviscid_lift
    )
    expected_clc_over_cref = (
        aero["spanwise_lift_per_y"] / op_point.dynamic_pressure() / airplane.c_ref
    )
    for actual, expected in zip(aero["spanwise_clc_over_cref"], expected_clc_over_cref):
        assert actual == pytest.approx(expected)


def test_lifting_line_spanwise_lift_distribution_tracks_multiple_wings():
    airplane = _rectangular_wing_airplane(include_tail=True)
    spanwise_resolution = 2
    aero = asb.LiftingLine(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=10, alpha=5),
        spanwise_resolution=spanwise_resolution,
        model_size="xsmall",
    ).run()

    assert set(aero["spanwise_wing_index"]) == {0, 1}
    assert len(aero["y"]) == 2
    assert len(aero["y"][0]) == 2 * spanwise_resolution
    assert len(aero["y"][1]) == 2 * spanwise_resolution
    assert np.max(np.abs(aero["y"][1])) < np.max(np.abs(aero["y"][0]))

    for wing_index in [0, 1]:
        wing_mask = aero["spanwise_wing_index"] == wing_index
        assert np.sum(wing_mask) == 2 * spanwise_resolution
        assert set(aero["spanwise_side"][wing_mask]) == {-1, 1}


def test_lifting_line_rate_response_is_invariant_to_geometry_translation():
    airplane = _rectangular_wing_airplane()
    translated_airplane = asb.Airplane(
        name="Translated",
        xyz_ref=np.array([10.25, 0, 0]),
        wings=[wing.translate([10, 0, 0]) for wing in airplane.wings],
        s_ref=airplane.s_ref,
        c_ref=airplane.c_ref,
        b_ref=airplane.b_ref,
    )
    airplane.xyz_ref = np.array([0.25, 0, 0])

    op_point = asb.OperatingPoint(velocity=10, alpha=4, q=1.0)
    kwargs = dict(
        op_point=op_point,
        spanwise_resolution=3,
        model_size="xsmall",
    )

    baseline = asb.LiftingLine(airplane=airplane, **kwargs).run()
    translated = asb.LiftingLine(
        airplane=translated_airplane,
        **kwargs,
    ).run()

    for key in ["CL", "CD", "CY", "Cl", "Cm", "Cn"]:
        assert translated[key] == pytest.approx(baseline[key], abs=1e-12)


if __name__ == "__main__":
    test_lifting_line_spanwise_lift_distribution_integrates_to_inviscid_lift()
    # pytest.main()
