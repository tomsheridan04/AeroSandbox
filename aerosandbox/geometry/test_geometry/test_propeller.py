import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import aerosandbox as asb
import aerosandbox.numpy as np


def make_propeller() -> asb.Propeller:
    return asb.Propeller(
        name="Test Propeller",
        radius=0.1524,
        hub_radius=0.018,
        blade_count=2,
        radial_stations=np.array([0.2, 0.45, 0.7, 1.0]),
        chord_distribution=np.array([0.030, 0.032, 0.024, 0.010]),
        twist_distribution=np.array([42.0, 28.0, 19.0, 12.0]),
        thickness_distribution=np.array([0.18, 0.14, 0.11, 0.08]),
        airfoil_distribution="naca4412",
        spinner_radius=0.018,
        spinner_length=0.025,
        distribution_interpolation_method="linear",
    )


def test_mesh_body():
    points, faces = make_propeller().mesh_body(
        radial_resolution=5,
        n_coordinates_per_side=8,
    )

    assert points.shape[1] == 3
    assert faces.shape[1] == 4
    assert len(points) > 0
    assert len(faces) > 0
    assert np.all(np.isfinite(points))


@pytest.mark.parametrize("style", ["shaded", "wireframe"])
def test_draw_three_view_smoke(style):
    axs = make_propeller().draw_three_view(
        style=style,
        radial_resolution=4,
        n_coordinates_per_side=6,
        show=False,
    )

    assert axs.shape == (2, 2)
    plt.close(axs[0, 0].figure)
