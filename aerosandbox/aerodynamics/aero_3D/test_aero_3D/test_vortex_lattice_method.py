import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def _rectangular_wing_airplane(include_tail=False):
    wings = [
        asb.Wing(
            name="Main Wing",
            symmetric=True,
            xsecs=[
                asb.WingXSec(xyz_le=[0, 0, 0], chord=1),
                asb.WingXSec(xyz_le=[0, 2, 0], chord=1),
            ],
        )
    ]

    if include_tail:
        wings.append(
            asb.Wing(
                name="Horizontal Tail",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[3, 0, 0], chord=0.5),
                    asb.WingXSec(xyz_le=[3, 1, 0], chord=0.5),
                ],
            )
        )

    return asb.Airplane(
        name="Spanwise Distribution Test Airplane",
        wings=wings,
    )


def test_conventional():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import (
        airplane,
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_vanilla():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.vanilla import (
        airplane,
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_flat_plate():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate import (
        airplane,
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_flat_plate_mirrored():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate_mirrored import (
        airplane,
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
        spanwise_resolution=1,
        chordwise_resolution=3,
    )
    return analysis.run()


def test_vlm_spanwise_lift_distribution_integrates_to_total_lift():
    airplane = _rectangular_wing_airplane()
    op_point = asb.OperatingPoint(velocity=10, alpha=5)
    spanwise_resolution = 3
    chordwise_resolution = 4
    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=chordwise_resolution,
    )
    aero = analysis.run()

    expected_spanwise_strips = 2 * spanwise_resolution
    expected_panels = expected_spanwise_strips * chordwise_resolution

    assert len(aero["spanwise_y"]) == expected_spanwise_strips
    assert len(aero["y"]) == 1
    assert len(aero["y"][0]) == expected_spanwise_strips
    assert len(aero["cl"][0]) == expected_spanwise_strips
    assert len(aero["clc_over_cref"][0]) == expected_spanwise_strips
    assert np.all(np.diff(aero["y"][0]) > 0)
    assert analysis["y"][0] is aero["y"][0]
    assert len(analysis.vortex_strengths) == expected_panels
    assert len(aero["spanwise_y"]) != len(analysis.vortex_strengths)
    assert np.sum(aero["spanwise_lift"]) == pytest.approx(aero["L"])
    assert np.sum(aero["spanwise_lift_per_y"] * aero["spanwise_dy"]) == pytest.approx(
        aero["L"]
    )
    expected_clc_over_cref = (
        aero["spanwise_lift_per_y"] / op_point.dynamic_pressure() / airplane.c_ref
    )
    for actual, expected in zip(aero["spanwise_clc_over_cref"], expected_clc_over_cref):
        assert actual == pytest.approx(expected)


def test_vlm_spanwise_lift_distribution_tracks_multiple_wings():
    airplane = _rectangular_wing_airplane(include_tail=True)
    spanwise_resolution = 2
    aero = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=10, alpha=5),
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=3,
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


def test_vlm_spanwise_lift_distribution_supports_opti_variables():
    opti = asb.Opti()
    alpha = opti.variable(init_guess=5, lower_bound=-10, upper_bound=10)
    airplane = _rectangular_wing_airplane()
    aero = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=10, alpha=alpha),
        spanwise_resolution=2,
        chordwise_resolution=2,
    ).run()

    opti.minimize((aero["CL"] - 0.4) ** 2 + 1e-6 * np.sum(aero["cl"][0] ** 2))
    sol = opti.solve(verbose=False)

    assert np.isfinite(sol(aero["cl"][0][0]))


def test_vlm_rate_response_is_invariant_to_geometry_translation():
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
        chordwise_resolution=3,
    )

    baseline = asb.VortexLatticeMethod(airplane=airplane, **kwargs).run()
    translated = asb.VortexLatticeMethod(
        airplane=translated_airplane,
        **kwargs,
    ).run()

    for key in ["CL", "CD", "CY", "Cl", "Cm", "Cn"]:
        assert translated[key] == pytest.approx(baseline[key], abs=1e-12)


if __name__ == "__main__":
    # test_conventional()
    # test_vanilla()
    # test_flat_plate()['CL']
    # test_flat_plate_mirrored()
    # pytest.main()
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import (
        airplane,
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    aero = analysis.run()
    analysis.draw()
