import casadi as cas
import pytest

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.modeling.splines import bspline, bspline_basis_matrix


def test_bspline_basis_is_partition_of_unity():
    x = np.linspace(0, 1, 11)
    basis = bspline_basis_matrix(x=x, n_control_points=11, degree=3)

    assert basis.shape == (11, 11)
    assert np.allclose(np.sum(basis, axis=1), 1)


def test_bspline_clamped_endpoints_match_end_control_points():
    x = np.linspace(0, 1, 7)
    y_control_points = np.array([3, 2, 2, 1, 1, 0, -1])
    y = bspline(x=x, y_control_points=y_control_points, degree=3)

    assert y[0] == pytest.approx(y_control_points[0])
    assert y[-1] == pytest.approx(y_control_points[-1])


def test_bspline_accepts_opti_variable_control_points():
    opti = asb.Opti()
    x = np.linspace(0, 1, 7)
    y_control_points = opti.variable(init_guess=np.ones_like(x))

    y = bspline(x=x, y_control_points=y_control_points, degree=3)

    assert isinstance(y, cas.MX)
    assert y.shape == (7, 1)

    opti.subject_to([y[0] == 1, y[-1] == 0])
    target_y = np.reshape(np.linspace(1, 0, 7), (-1, 1))
    opti.minimize(np.sum((y - target_y) ** 2))

    sol = opti.solve(verbose=False)
    assert sol(y[0]) == pytest.approx(1)
    assert sol(y[-1]) == pytest.approx(0)


def test_bspline_is_exposed_at_top_level():
    x = np.linspace(0, 1, 5)
    y_control_points = np.linspace(1, 0, 5)

    assert np.allclose(
        asb.bspline(x=x, y_control_points=y_control_points),
        bspline(x=x, y_control_points=y_control_points),
    )


def test_bspline_accepts_opti_variable_x_locations():
    opti = asb.Opti()

    span = opti.variable(init_guess=2, lower_bound=1)
    x = np.linspace(0, span, 7)
    y_control_points = opti.variable(init_guess=np.linspace(1, 0, 7))

    y = bspline(x=x, y_control_points=y_control_points, degree=3)

    assert isinstance(y, cas.MX)
    assert y.shape == (7, 1)
