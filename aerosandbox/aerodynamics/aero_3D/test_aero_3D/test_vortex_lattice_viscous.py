import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def _generic_wing_airplane(include_tail=False):
    main_airfoil = asb.Airfoil("naca2412")
    tail_airfoil = asb.Airfoil("naca0012")

    wings = [
        asb.Wing(
            name="Main Wing",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1.0,
                    twist=2,
                    airfoil=main_airfoil,
                ),
                asb.WingXSec(
                    xyz_le=[0.12, 2.0, 0.12],
                    chord=0.55,
                    twist=-2,
                    airfoil=main_airfoil,
                ),
            ],
        )
    ]

    if include_tail:
        wings.append(
            asb.Wing(
                name="Horizontal Tail",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[3.0, 0, 0.1],
                        chord=0.45,
                        twist=-1,
                        airfoil=tail_airfoil,
                    ),
                    asb.WingXSec(
                        xyz_le=[3.08, 0.9, 0.1],
                        chord=0.28,
                        twist=-1,
                        airfoil=tail_airfoil,
                    ),
                ],
            )
        )

    return asb.Airplane(
        name="VortexLatticeViscous Test Airplane",
        wings=wings,
    )


def test_vortex_lattice_viscous_public_import():
    from aerosandbox.aerodynamics.aero_3D import VortexLatticeViscous

    assert asb.VortexLatticeViscous is VortexLatticeViscous


def test_vortex_lattice_viscous_outputs_and_drag():
    airplane = _generic_wing_airplane()
    aero = asb.VortexLatticeViscous(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=20, alpha=5),
        spanwise_resolution=2,
        chordwise_resolution=3,
        model_size="xsmall",
    ).run()

    for key in [
        "F_g",
        "F_b",
        "F_w",
        "M_g",
        "M_b",
        "M_w",
        "CL",
        "CD",
        "CY",
        "Cl",
        "Cm",
        "Cn",
        "spanwise_alpha",
        "spanwise_Re",
        "spanwise_mach",
        "spanwise_cl_inviscid",
        "spanwise_cl_viscous",
        "spanwise_cd_profile",
        "spanwise_cm",
        "spanwise_clmax",
        "spanwise_clmin",
        "spanwise_analysis_confidence",
        "CL_inviscid",
        "CD_inviscid",
        "Cm_inviscid",
        "CD_profile",
    ]:
        assert key in aero

    for key in ["CL", "CD", "CY", "Cl", "Cm", "Cn", "CL_inviscid", "CD_inviscid"]:
        assert np.isfinite(aero[key])

    assert aero["CD_profile"] > 0
    assert aero["CD"] > aero["CD_inviscid"]
    assert np.all(aero["spanwise_clmax"] > aero["spanwise_clmin"])


def test_vortex_lattice_viscous_tracks_vlm_lift_in_attached_flow():
    airplane = _generic_wing_airplane()
    op_point = asb.OperatingPoint(velocity=20, alpha=5)

    vlm_aero = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=2,
        chordwise_resolution=3,
    ).run()
    viscous_aero = asb.VortexLatticeViscous(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=2,
        chordwise_resolution=3,
        model_size="xsmall",
    ).run()

    assert viscous_aero["CL"] == pytest.approx(vlm_aero["CL"])
    assert np.all(
        viscous_aero["spanwise_cl_viscous"]
        == pytest.approx(vlm_aero["spanwise_cl"])
    )


def test_vortex_lattice_viscous_preserves_swept_vlm_attached_flow_geometry():
    airplane = asb.Airplane(
        wings=[
            asb.Wing(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=1.0,
                        airfoil=asb.Airfoil("naca2412"),
                    ),
                    asb.WingXSec(
                        xyz_le=[0.7, 2.0, 0],
                        chord=0.45,
                        airfoil=asb.Airfoil("naca2412"),
                    ),
                ],
            )
        ]
    )
    op_point = asb.OperatingPoint(velocity=30, alpha=4)

    vlm_aero = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=3,
        chordwise_resolution=4,
    ).run()
    viscous_aero = asb.VortexLatticeViscous(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=3,
        chordwise_resolution=4,
        model_size="xsmall",
    ).run()

    assert viscous_aero["CL"] == pytest.approx(vlm_aero["CL"])
    assert viscous_aero["CD"] > vlm_aero["CD"]
    assert viscous_aero["Cm"] == pytest.approx(vlm_aero["Cm"], abs=0.04)


def test_vortex_lattice_viscous_spanwise_lift_distribution():
    airplane = _generic_wing_airplane()
    op_point = asb.OperatingPoint(velocity=20, alpha=5)
    spanwise_resolution = 3
    chordwise_resolution = 3
    analysis = asb.VortexLatticeViscous(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=chordwise_resolution,
        model_size="xsmall",
    )
    aero = analysis.run()

    expected_spanwise_strips = 2 * spanwise_resolution
    expected_panels = expected_spanwise_strips * chordwise_resolution

    assert len(aero["spanwise_y"]) == expected_spanwise_strips
    assert len(aero["spanwise_alpha"]) == expected_spanwise_strips
    assert len(aero["spanwise_cl_inviscid"]) == expected_spanwise_strips
    assert len(aero["spanwise_cl_viscous"]) == expected_spanwise_strips
    assert len(aero["spanwise_cd_profile"]) == expected_spanwise_strips
    assert len(aero["spanwise_clmax"]) == expected_spanwise_strips
    assert len(aero["spanwise_analysis_confidence"]) == expected_spanwise_strips
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


def test_vortex_lattice_viscous_spanwise_metadata_tracks_multiple_wings():
    airplane = _generic_wing_airplane(include_tail=True)
    spanwise_resolution = 2
    aero = asb.VortexLatticeViscous(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=20, alpha=5),
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=3,
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


def test_vortex_lattice_viscous_reports_high_alpha_stall_margin():
    airplane = _generic_wing_airplane()
    op_point = asb.OperatingPoint(velocity=20, alpha=25)

    vlm_aero = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=2,
        chordwise_resolution=3,
    ).run()
    viscous_aero = asb.VortexLatticeViscous(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=2,
        chordwise_resolution=3,
        model_size="xsmall",
    ).run()

    assert viscous_aero["CL"] == pytest.approx(vlm_aero["CL"])
    assert np.any(viscous_aero["spanwise_cl"] > viscous_aero["spanwise_clmax"])
