from typing import Optional, Union

import casadi as cas
import numpy as _onp

import aerosandbox.numpy as np
from aerosandbox.numpy.determine_type import is_casadi_type


def _open_uniform_knot_vector(
    x_min: float,
    x_max: float,
    n_control_points: int,
    degree: int,
) -> _onp.ndarray:
    if n_control_points < degree + 1:
        raise ValueError(
            "`n_control_points` must be at least `degree + 1` for a B-spline."
        )

    if x_max <= x_min:
        raise ValueError("The B-spline x-domain must have nonzero positive width.")

    n_internal_knots = n_control_points - degree - 1

    if n_internal_knots > 0:
        internal_knots = _onp.linspace(
            x_min,
            x_max,
            n_internal_knots + 2,
        )[1:-1]
    else:
        internal_knots = _onp.array([])

    return _onp.concatenate(
        [
            _onp.full(degree + 1, x_min),
            internal_knots,
            _onp.full(degree + 1, x_max),
        ]
    )


def _open_uniform_unit_knot_vector(
    n_control_points: int,
    degree: int,
) -> _onp.ndarray:
    return _open_uniform_knot_vector(
        x_min=0.0,
        x_max=1.0,
        n_control_points=n_control_points,
        degree=degree,
    )


def _clip_symbolic(x, lower, upper):
    return np.minimum(np.maximum(x, lower), upper)


def _bspline_basis_values(
    x,
    knots: _onp.ndarray,
    n_control_points: int,
    degree: int,
    extrapolation: str,
):
    if extrapolation == "clip":
        x_eval = _clip_symbolic(x, knots[degree], knots[-degree - 1])
        outside_domain = None
    elif extrapolation == "zero":
        x_eval = x
        outside_domain = np.logical_or(x < knots[degree], x > knots[-degree - 1])
    elif extrapolation == "raise":
        raise ValueError(
            "`extrapolation='raise'` is not available when `x` is symbolic."
        )
    else:
        raise ValueError("`extrapolation` must be one of: 'clip', 'zero', 'raise'.")

    basis = [
        np.where(
            np.logical_and(knots[i] <= x_eval, x_eval < knots[i + 1]),
            1.0,
            0.0,
        )
        for i in range(len(knots) - 1)
    ]

    for p in range(1, degree + 1):
        next_basis = []

        for i in range(len(knots) - 1 - p):
            value = 0.0

            left_denominator = knots[i + p] - knots[i]
            right_denominator = knots[i + p + 1] - knots[i + 1]

            if left_denominator != 0:
                value += (x_eval - knots[i]) / left_denominator * basis[i]

            if right_denominator != 0:
                value += (
                    (knots[i + p + 1] - x_eval)
                    / right_denominator
                    * basis[i + 1]
                )

            next_basis.append(value)

        basis = next_basis

    basis = basis[:n_control_points]

    if extrapolation == "zero":
        basis = [np.where(outside_domain, 0.0, b) for b in basis]

    return basis


def _bspline_symbolic(
    x,
    y_control_points,
    n_control_points: int,
    degree: int,
    knots: Optional[np.ndarray],
    extrapolation: str,
):
    x = np.array(x)
    y_control_points = np.array(y_control_points)

    if knots is None:
        if np.length(x) < 2:
            raise ValueError(
                "When `x` is symbolic and `knots` is omitted, `x` must have "
                "at least two entries so `x[0]` and `x[-1]` can define the "
                "spline domain."
            )

        knots = _open_uniform_unit_knot_vector(
            n_control_points=n_control_points,
            degree=degree,
        )

        x_domain_min = x[0]
        x_domain_max = x[np.length(x) - 1]
        x_normalized = (x - x_domain_min) / (x_domain_max - x_domain_min)
    else:
        if is_casadi_type(knots, recursive=True):
            raise TypeError(
                "`knots` must be fixed numeric data when `x` is symbolic."
            )

        knots = _onp.asarray(knots, dtype=float)
        x_normalized = x

    expected_n_knots = n_control_points + degree + 1
    if len(knots) != expected_n_knots:
        raise ValueError(
            f"`knots` must have length `n_control_points + degree + 1`, "
            f"which is {expected_n_knots}; got {len(knots)}."
        )

    if _onp.any(_onp.diff(knots) < 0):
        raise ValueError("`knots` must be monotonically nondecreasing.")

    y = []
    for i in range(np.length(x_normalized)):
        basis = _bspline_basis_values(
            x=x_normalized[i],
            knots=knots,
            n_control_points=n_control_points,
            degree=degree,
            extrapolation=extrapolation,
        )

        value = sum(
            basis[j] * y_control_points[j]
            for j in range(n_control_points)
        )

        if extrapolation == "clip":
            value = np.where(
                x_normalized[i] >= knots[-degree - 1],
                y_control_points[n_control_points - 1],
                value,
            )
        elif extrapolation == "zero":
            value = np.where(
                x_normalized[i] == knots[-degree - 1],
                y_control_points[n_control_points - 1],
                value,
            )

        y.append(value)

    return np.array(y)


def bspline_basis_matrix(
    x: Union[float, np.ndarray],
    n_control_points: int,
    degree: int = 3,
    knots: Optional[np.ndarray] = None,
    extrapolation: str = "clip",
) -> _onp.ndarray:
    """
    Builds the matrix that maps B-spline control-point values to sampled values.

    This is intentionally a numeric matrix builder: `x` and `knots` should be fixed
    data, while the control-point values may be AeroSandbox/CasADi variables in
    `bspline()`.

    Args:
        x: Fixed scalar or array of x-locations where the spline should be evaluated.
        n_control_points: Number of B-spline control-point ordinates.
        degree: Polynomial degree of the B-spline. Cubic (`degree=3`) is the default.
        knots: Optional full knot vector. If omitted, an open-uniform knot vector is
            built over `[min(x), max(x)]`.
        extrapolation: Behavior outside the spline domain. One of:

            * `"clip"`: evaluate outside-domain points at the nearest endpoint.
            * `"zero"`: basis rows outside the domain are zero.
            * `"raise"`: raise a `ValueError` if any x-location is outside the domain.

    Returns:
        A dense matrix `B` with shape `(len(x), n_control_points)`, such that
        `B @ y_control_points` gives the sampled spline values.
    """
    if is_casadi_type([x, knots], recursive=True):
        raise TypeError(
            "`bspline_basis_matrix()` only builds fixed numeric matrices. Use "
            "`bspline()` directly when `x` or `y_control_points` are Opti/CasADi "
            "expressions."
        )

    if degree < 0:
        raise ValueError("`degree` must be nonnegative.")

    x = _onp.atleast_1d(_onp.asarray(x, dtype=float))

    if knots is None:
        knots = _open_uniform_knot_vector(
            x_min=float(_onp.min(x)),
            x_max=float(_onp.max(x)),
            n_control_points=n_control_points,
            degree=degree,
        )
    else:
        knots = _onp.asarray(knots, dtype=float)

    expected_n_knots = n_control_points + degree + 1
    if len(knots) != expected_n_knots:
        raise ValueError(
            f"`knots` must have length `n_control_points + degree + 1`, "
            f"which is {expected_n_knots}; got {len(knots)}."
        )

    if _onp.any(_onp.diff(knots) < 0):
        raise ValueError("`knots` must be monotonically nondecreasing.")

    domain_min = knots[degree]
    domain_max = knots[-degree - 1]

    x_eval = x.copy()
    outside_domain = (x_eval < domain_min) | (x_eval > domain_max)

    if extrapolation == "clip":
        x_eval = _onp.clip(x_eval, domain_min, domain_max)
    elif extrapolation == "zero":
        pass
    elif extrapolation == "raise":
        if _onp.any(outside_domain):
            raise ValueError("Some x-locations are outside the B-spline knot domain.")
    else:
        raise ValueError("`extrapolation` must be one of: 'clip', 'zero', 'raise'.")

    basis = _onp.zeros((len(x_eval), len(knots) - 1))
    for i in range(len(knots) - 1):
        basis[:, i] = (knots[i] <= x_eval) & (x_eval < knots[i + 1])

    for p in range(1, degree + 1):
        next_basis = _onp.zeros((len(x_eval), len(knots) - 1 - p))

        for i in range(next_basis.shape[1]):
            left_denominator = knots[i + p] - knots[i]
            right_denominator = knots[i + p + 1] - knots[i + 1]

            if left_denominator != 0:
                next_basis[:, i] += (
                    (x_eval - knots[i]) / left_denominator * basis[:, i]
                )

            if right_denominator != 0:
                next_basis[:, i] += (
                    (knots[i + p + 1] - x_eval)
                    / right_denominator
                    * basis[:, i + 1]
                )

        basis = next_basis

    basis = basis[:, :n_control_points]

    at_upper_bound = _onp.isclose(x_eval, domain_max)
    basis[at_upper_bound, :] = 0
    basis[at_upper_bound, n_control_points - 1] = 1

    if extrapolation == "zero":
        basis[outside_domain, :] = 0

    return basis


def bspline(
    x: Union[float, np.ndarray],
    y_control_points: Union[np.ndarray, cas.MX, cas.SX, cas.DM],
    degree: int = 3,
    knots: Optional[np.ndarray] = None,
    extrapolation: str = "clip",
) -> Union[_onp.ndarray, cas.MX, cas.SX, cas.DM]:
    """
    Evaluates a B-spline from x-locations and y-control-points.

    This is useful for optimization problems where you want a geometry vector, such
    as wing chord, to vary smoothly without directly optimizing every sampled value.
    If `x` and/or `y_control_points` are `opti.variable(...)` expressions, the
    returned vector is a differentiable CasADi expression that can be used in
    constraints, objectives, and AeroSandbox analyses.

    Args:
        x: Scalar or array of x-locations where the spline should be sampled.
            If `len(x) == len(y_control_points)`, the returned vector has the same
            length as the control-point vector. When `x` is symbolic and `knots` is
            omitted, `x[0]` and `x[-1]` define the spline domain.
        y_control_points: Numeric or CasADi/AeroSandbox vector of B-spline
            control-point ordinates.
        degree: Polynomial degree of the B-spline. Cubic (`degree=3`) is the default.
        knots: Optional full knot vector. If omitted, an open-uniform knot vector is
            built over `[min(x), max(x)]` for numeric `x`, or over `[x[0], x[-1]]`
            for symbolic `x`. If `x` is symbolic and `knots` is provided, `knots`
            must be fixed numeric data.
        extrapolation: Passed to `bspline_basis_matrix()`.

    Returns:
        The sampled B-spline values at `x`.
    """
    n_control_points = np.length(y_control_points)

    if is_casadi_type(x, recursive=True) or is_casadi_type(knots, recursive=True):
        if degree < 0:
            raise ValueError("`degree` must be nonnegative.")

        return _bspline_symbolic(
            x=x,
            y_control_points=y_control_points,
            n_control_points=n_control_points,
            degree=degree,
            knots=knots,
            extrapolation=extrapolation,
        )

    basis = bspline_basis_matrix(
        x=x,
        n_control_points=n_control_points,
        degree=degree,
        knots=knots,
        extrapolation=extrapolation,
    )

    if is_casadi_type(y_control_points, recursive=True):
        return cas.DM(basis) @ y_control_points
    else:
        return basis @ _onp.asarray(y_control_points)
