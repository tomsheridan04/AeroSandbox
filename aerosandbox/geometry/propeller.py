from aerosandbox.common import AeroSandboxObject
from aerosandbox.geometry.airfoil import Airfoil, KulfanAirfoil
from aerosandbox.modeling.splines import bspline
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import aerosandbox.numpy as np
import numpy as _onp
import copy


def _as_float_array_if_possible(value):
    try:
        return _onp.asarray(value, dtype=float)
    except Exception:
        return None


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

    n_internal_knots = n_control_points - degree - 1
    if n_internal_knots > 0:
        internal_knots = _onp.linspace(x_min, x_max, n_internal_knots + 2)[1:-1]
    else:
        internal_knots = _onp.array([])

    return _onp.concatenate(
        [
            _onp.full(degree + 1, x_min),
            internal_knots,
            _onp.full(degree + 1, x_max),
        ]
    )


def _as_kulfan_airfoil(airfoil: Union[Airfoil, KulfanAirfoil, str]) -> KulfanAirfoil:
    if isinstance(airfoil, str):
        return KulfanAirfoil(airfoil)
    if isinstance(airfoil, KulfanAirfoil):
        return airfoil
    if isinstance(airfoil, Airfoil):
        return airfoil.to_kulfan_airfoil()
    raise TypeError(
        "`airfoil_distribution` entries must be Airfoil, KulfanAirfoil, or airfoil-name strings."
    )


def _scale_kulfan_thickness_about_camber(
    airfoil: KulfanAirfoil,
    thickness_scale,
) -> KulfanAirfoil:
    """
    Scales thickness while preserving the Kulfan camber-line terms.
    """
    upper_weights = 0.5 * (
        (1 + thickness_scale) * airfoil.upper_weights
        + (1 - thickness_scale) * airfoil.lower_weights
    )
    lower_weights = 0.5 * (
        (1 - thickness_scale) * airfoil.upper_weights
        + (1 + thickness_scale) * airfoil.lower_weights
    )

    return KulfanAirfoil(
        name=airfoil.name,
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        leading_edge_weight=airfoil.leading_edge_weight,
        TE_thickness=airfoil.TE_thickness * thickness_scale,
        N1=airfoil.N1,
        N2=airfoil.N2,
    )


class Propeller(AeroSandboxObject):
    """
    Geometry definition for a propeller or rotor.

    Spanwise scalar distributions such as chord, twist, and thickness can be
    represented either as direct tabulated values or as B-spline control points
    over nondimensional radius ``r_over_R``.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        radius: float = 1.0,
        hub_radius: Optional[float] = None,
        hub_diameter: Optional[float] = None,
        blade_count: int = 2,
        radial_stations: Union[np.ndarray, List[float]] = None,
        chord_distribution: Union[float, np.ndarray, List[float], Callable] = 0.1,
        twist_distribution: Union[float, np.ndarray, List[float], Callable] = 20.0,
        thickness_distribution: Optional[
            Union[float, np.ndarray, List[float], Callable]
        ] = None,
        airfoil_distribution: Union[
            Airfoil,
            KulfanAirfoil,
            str,
            List[Union[Airfoil, KulfanAirfoil, str]],
            List[Tuple[float, Union[Airfoil, KulfanAirfoil, str]]],
            Dict[float, Union[Airfoil, KulfanAirfoil, str]],
        ] = None,
        spinner_radius: Optional[float] = None,
        spinner_length: float = 0.0,
        xyz_c: Union[np.ndarray, List[float]] = None,
        xyz_normal: Union[np.ndarray, List[float]] = None,
        color: Optional[Union[str, Tuple[float]]] = None,
        analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
        distribution_interpolation_method: str = "bspline",
        spline_degree: int = 3,
    ):
        """
        Args:
            radius: Propeller tip radius [m].
            hub_radius: Radius inside which no lifting blade is modeled [m].
            hub_diameter: Alternative to ``hub_radius`` [m].
            blade_count: Number of blades.
            radial_stations: Nondimensional ``r/R`` locations. For
                ``distribution_interpolation_method="linear"``, these are the
                tabulated stations. For ``"bspline"``, these define the B-spline
                domain.
            chord_distribution: Chord values [m], B-spline control points [m],
                scalar, or callable.
            twist_distribution: Geometric pitch/twist angle control points [deg],
                scalar, or callable.
            thickness_distribution: Optional target max thickness ratio ``t/c``.
                If supplied, section airfoil thickness is scaled about its camber
                line to this target.
            airfoil_distribution: One airfoil, a list of airfoils, a list of
                ``(r_over_R, airfoil)`` pairs, or a dict mapping ``r_over_R`` to
                airfoils. Adjacent entries are blended in Kulfan space.
            spinner_radius: Spinner radius [m]. Defaults to ``hub_radius``.
            spinner_length: Spinner length [m].
            distribution_interpolation_method: How vector-valued scalar
                distributions are evaluated. Options are ``"linear"`` for direct
                tabulated interpolation and ``"bspline"`` for smooth B-splines.
        """
        super().__init__()

        if name is None:
            name = "Untitled"

        if hub_radius is not None and hub_diameter is not None:
            raise ValueError("Specify either `hub_radius` or `hub_diameter`, not both.")
        if hub_radius is None:
            hub_radius = 0.0 if hub_diameter is None else hub_diameter / 2

        if spinner_radius is None:
            spinner_radius = hub_radius

        if xyz_c is None:
            xyz_c = np.array([0.0, 0.0, 0.0])
        if xyz_normal is None:
            xyz_normal = np.array([-1.0, 0.0, 0.0])
        if analysis_specific_options is None:
            analysis_specific_options = {}
        if distribution_interpolation_method not in {"linear", "bspline"}:
            raise ValueError(
                "`distribution_interpolation_method` must be either 'linear' or 'bspline'."
            )

        if radial_stations is None:
            radial_stations = _onp.linspace(max(hub_radius / radius, 0.0), 1.0, 6)
        radial_stations = _onp.asarray(radial_stations, dtype=float)

        if len(radial_stations) < 2:
            raise ValueError("`radial_stations` must contain at least two entries.")
        if _onp.any(_onp.diff(radial_stations) <= 0):
            raise ValueError("`radial_stations` must be strictly increasing.")
        if radial_stations[0] < 0 or radial_stations[-1] > 1:
            raise ValueError("`radial_stations` must lie within [0, 1].")

        radius_float = _as_float_array_if_possible(radius)
        hub_radius_float = _as_float_array_if_possible(hub_radius)
        spinner_radius_float = _as_float_array_if_possible(spinner_radius)
        chord_float = _as_float_array_if_possible(chord_distribution)
        twist_float = _as_float_array_if_possible(twist_distribution)
        thickness_float = _as_float_array_if_possible(thickness_distribution)

        if radius_float is not None and (
            not _onp.isfinite(float(radius_float)) or float(radius_float) <= 0
        ):
            raise ValueError("`radius` must be positive and finite.")
        if hub_radius_float is not None:
            if float(hub_radius_float) < 0:
                raise ValueError("`hub_radius` must be non-negative.")
            if radius_float is not None and float(hub_radius_float) >= float(radius_float):
                raise ValueError("`hub_radius` must be smaller than `radius`.")
        if not isinstance(blade_count, int) or blade_count < 1:
            raise ValueError("`blade_count` must be a positive integer.")
        if spinner_radius_float is not None:
            if float(spinner_radius_float) < 0:
                raise ValueError("`spinner_radius` must be non-negative.")
            if radius_float is not None and float(spinner_radius_float) > float(radius_float):
                raise ValueError("`spinner_radius` must not exceed `radius`.")
        if chord_float is not None and _onp.any(chord_float <= 0):
            raise ValueError("`chord_distribution` control points must be positive.")
        if twist_float is not None and not _onp.all(_onp.isfinite(twist_float)):
            raise ValueError("`twist_distribution` control points must be finite.")
        if thickness_float is not None and _onp.any(thickness_float <= 0):
            raise ValueError(
                "`thickness_distribution` control points must be positive."
            )
        if distribution_interpolation_method == "linear":
            for distribution, name in [
                (chord_distribution, "chord_distribution"),
                (twist_distribution, "twist_distribution"),
                (thickness_distribution, "thickness_distribution"),
            ]:
                if distribution is None or callable(distribution):
                    continue
                distribution_length = np.length(distribution)
                if distribution_length not in {1, len(radial_stations)}:
                    raise ValueError(
                        f"`{name}` must be scalar, callable, or have the same "
                        "length as `radial_stations` when using linear interpolation."
                    )

        self.name = name
        self.radius = radius
        self.hub_radius = hub_radius
        self.blade_count = blade_count
        self.radial_stations = radial_stations
        self.chord_distribution = chord_distribution
        self.twist_distribution = twist_distribution
        self.thickness_distribution = thickness_distribution
        self.airfoil_distribution = self._normalize_airfoil_distribution(
            airfoil_distribution
        )
        self.spinner_radius = spinner_radius
        self.spinner_length = spinner_length
        self.xyz_c = np.array(xyz_c)
        self.xyz_normal = np.array(xyz_normal)
        self.color = color
        self.analysis_specific_options = analysis_specific_options
        self.distribution_interpolation_method = distribution_interpolation_method
        self.spline_degree = spline_degree

    def __repr__(self) -> str:
        diameter = 2 * self.radius
        return (
            f"Propeller '{self.name}' "
            f"({self.blade_count} blades, diameter: {diameter})"
        )

    @property
    def diameter(self) -> float:
        return 2 * self.radius

    @property
    def hub_diameter(self) -> float:
        return 2 * self.hub_radius

    def disk_area(self) -> float:
        return np.pi * self.radius**2

    def annulus_area(self) -> float:
        return np.pi * (self.radius**2 - self.hub_radius**2)

    def translate(
        self,
        xyz: Union[np.ndarray, List[float]],
    ) -> "Propeller":
        new_propeller = copy.deepcopy(self)
        new_propeller.xyz_c = new_propeller.xyz_c + np.array(xyz)
        return new_propeller

    def compute_frame(self) -> Tuple[_onp.ndarray, _onp.ndarray, _onp.ndarray]:
        """
        Computes the local coordinate frame of the propeller, in aircraft geometry axes.

        The local x-axis is aligned with ``xyz_normal``. The local y- and z-axes
        lie in the propeller disk plane and are chosen to remain close to the
        global y- and z-axes when possible.
        """
        xyz_normal = _as_float_array_if_possible(self.xyz_normal)
        if xyz_normal is None:
            raise ValueError(
                "Propeller plotting requires numeric geometry; `xyz_normal` could "
                "not be converted to a float array."
            )

        xg_local = _onp.reshape(xyz_normal, 3)
        xg_local_norm = _onp.linalg.norm(xg_local)
        if xg_local_norm <= 0:
            raise ValueError("`xyz_normal` must have nonzero magnitude.")
        xg_local = xg_local / xg_local_norm

        zg_local = _onp.array([0.0, 0.0, 1.0])
        zg_local = zg_local - _onp.dot(zg_local, xg_local) * xg_local

        if _onp.linalg.norm(zg_local) < 1e-12:
            zg_local = _onp.array([0.0, 1.0, 0.0])
            zg_local = zg_local - _onp.dot(zg_local, xg_local) * xg_local

        zg_local = zg_local / _onp.linalg.norm(zg_local)
        yg_local = _onp.cross(zg_local, xg_local)
        yg_local = yg_local / _onp.linalg.norm(yg_local)

        return xg_local, yg_local, zg_local

    @staticmethod
    def _as_float_scalar(value, name: str) -> float:
        value_float = _as_float_array_if_possible(value)
        if value_float is None or _onp.size(value_float) != 1:
            raise ValueError(
                f"Propeller plotting requires numeric geometry; `{name}` could "
                "not be converted to a scalar float."
            )
        return float(value_float)

    def _section_coordinates(
        self,
        r_over_R: float,
        azimuth: float,
        n_coordinates_per_side: int,
    ) -> _onp.ndarray:
        radius = self._as_float_scalar(self.radius, "radius")
        chord = self._as_float_scalar(self.chord(r_over_R), "chord")
        twist = self._as_float_scalar(self.twist(r_over_R), "twist")

        xg_local, yg_local, zg_local = self.compute_frame()
        xyz_c = _as_float_array_if_possible(self.xyz_c)
        if xyz_c is None:
            raise ValueError(
                "Propeller plotting requires numeric geometry; `xyz_c` could not "
                "be converted to a float array."
            )
        xyz_c = _onp.reshape(xyz_c, 3)

        radial_unit = _onp.cos(azimuth) * yg_local + _onp.sin(azimuth) * zg_local
        tangential_unit = (
            -_onp.sin(azimuth) * yg_local + _onp.cos(azimuth) * zg_local
        )

        beta = _onp.deg2rad(twist)
        chord_unit = _onp.cos(beta) * tangential_unit + _onp.sin(beta) * xg_local
        thickness_unit = -_onp.sin(beta) * tangential_unit + _onp.cos(beta) * xg_local

        coordinates = _onp.asarray(
            self.airfoil(float(r_over_R))
            .to_airfoil(n_coordinates_per_side=n_coordinates_per_side)
            .coordinates,
            dtype=float,
        )
        if _onp.linalg.norm(coordinates[0] - coordinates[-1]) > 1e-10:
            coordinates = _onp.vstack([coordinates, coordinates[0]])

        x_over_c = coordinates[:, 0]
        y_over_c = coordinates[:, 1]

        # Station locations are assumed to lie on the quarter-chord stacking line
        # with no axial rake offset.
        section_origin = xyz_c + radial_unit * (r_over_R * radius)

        return (
            section_origin
            + chord_unit * ((0.25 - x_over_c) * chord)[:, None]
            + thickness_unit * (y_over_c * chord)[:, None]
        )

    def _spinner_mesh(
        self,
        n_azimuthal: int = 48,
    ) -> Tuple[_onp.ndarray, _onp.ndarray]:
        spinner_radius = self._as_float_scalar(self.spinner_radius, "spinner_radius")
        if spinner_radius <= 0:
            return _onp.empty((0, 3)), _onp.empty((0, 4), dtype=int)

        radius = self._as_float_scalar(self.radius, "radius")
        spinner_length = self._as_float_scalar(self.spinner_length, "spinner_length")
        xg_local, yg_local, zg_local = self.compute_frame()

        xyz_c = _as_float_array_if_possible(self.xyz_c)
        if xyz_c is None:
            raise ValueError(
                "Propeller plotting requires numeric geometry; `xyz_c` could not "
                "be converted to a float array."
            )
        xyz_c = _onp.reshape(xyz_c, 3)

        if spinner_length > 0:
            axial_stations = _onp.array([0.0, 0.55 * spinner_length, spinner_length])
            ring_radii = _onp.array([spinner_radius, 0.75 * spinner_radius, 0.0])
        else:
            visual_length = min(0.06 * radius, max(0.5 * spinner_radius, 1e-6))
            axial_stations = _onp.array([-0.5 * visual_length, 0.5 * visual_length])
            ring_radii = _onp.array([spinner_radius, spinner_radius])

        theta = _onp.linspace(0, 2 * _onp.pi, n_azimuthal + 1)
        points = []
        for axial_station, ring_radius in zip(axial_stations, ring_radii):
            ring = (
                xyz_c
                + axial_station * xg_local
                + ring_radius * _onp.cos(theta)[:, None] * yg_local
                + ring_radius * _onp.sin(theta)[:, None] * zg_local
            )
            points.append(ring)
        points = _onp.vstack(points)

        faces = []
        n_theta = len(theta)
        for i in range(len(axial_stations) - 1):
            for j in range(n_theta - 1):
                faces.append(
                    [
                        i * n_theta + j,
                        (i + 1) * n_theta + j,
                        (i + 1) * n_theta + j + 1,
                        i * n_theta + j + 1,
                    ]
                )

        cap_points = []
        for ring_i, reverse_winding in [
            (0, True),
            (len(axial_stations) - 1, False),
        ]:
            if ring_radii[ring_i] <= 0:
                continue

            ring_start = ring_i * n_theta
            ring_indices = _onp.arange(ring_start, ring_start + n_theta - 1)
            center_index = len(points) + len(cap_points)
            cap_points.append(_onp.mean(points[ring_indices], axis=0))

            for j in range(n_theta - 1):
                face = [
                    center_index,
                    ring_start + j,
                    ring_start + j + 1,
                    center_index,
                ]
                if reverse_winding:
                    face = face[::-1]
                faces.append(face)

        if len(cap_points) > 0:
            points = _onp.vstack([points, _onp.asarray(cap_points)])

        return points, _onp.asarray(faces, dtype=int)

    def mesh_body(
        self,
        radial_resolution: int = 24,
        n_coordinates_per_side: int = 40,
        include_spinner: bool = True,
        close_blade_ends: bool = True,
    ) -> Tuple[_onp.ndarray, _onp.ndarray]:
        """
        Generates a quad mesh of the propeller blades and spinner for visualization.

        Args:
            radial_resolution: Number of blade sections used from hub to tip.
            n_coordinates_per_side: Number of airfoil coordinates per side.
            include_spinner: Whether to include the spinner or hub in the mesh.
            close_blade_ends: Whether to cap the root and tip airfoil sections.

        Returns:
            ``points, faces`` where ``points`` has shape ``(n_points, 3)`` and
            ``faces`` has shape ``(n_faces, 4)``.
        """
        if radial_resolution < 2:
            raise ValueError("`radial_resolution` must be at least 2.")
        if n_coordinates_per_side < 3:
            raise ValueError("`n_coordinates_per_side` must be at least 3.")

        radius = self._as_float_scalar(self.radius, "radius")
        hub_radius = self._as_float_scalar(self.hub_radius, "hub_radius")
        hub_over_radius = hub_radius / radius
        r_over_R_min = max(hub_over_radius, float(self.radial_stations[0]))
        r_over_R_min = min(r_over_R_min, 1 - 1e-9)

        r_over_R = _onp.linspace(r_over_R_min, 1.0, radial_resolution)
        sections = []
        for blade_index in range(self.blade_count):
            azimuth = 2 * _onp.pi * blade_index / self.blade_count
            for section_r_over_R in r_over_R:
                sections.append(
                    self._section_coordinates(
                        r_over_R=float(section_r_over_R),
                        azimuth=azimuth,
                        n_coordinates_per_side=n_coordinates_per_side,
                    )
                )

        points = _onp.vstack(sections)
        n_coordinates = sections[0].shape[0]
        faces = []
        for blade_index in range(self.blade_count):
            blade_start = blade_index * radial_resolution * n_coordinates
            for i in range(radial_resolution - 1):
                section_start = blade_start + i * n_coordinates
                next_section_start = blade_start + (i + 1) * n_coordinates
                for j in range(n_coordinates - 1):
                    faces.append(
                        [
                            section_start + j,
                            next_section_start + j,
                            next_section_start + j + 1,
                            section_start + j + 1,
                        ]
                    )

        if close_blade_ends:
            cap_points = []
            for blade_index in range(self.blade_count):
                blade_start = blade_index * radial_resolution * n_coordinates
                for section_i, reverse_winding in [
                    (0, False),
                    (radial_resolution - 1, True),
                ]:
                    section_start = blade_start + section_i * n_coordinates
                    section_indices = _onp.arange(
                        section_start, section_start + n_coordinates - 1
                    )
                    center_index = len(points) + len(cap_points)
                    cap_points.append(_onp.mean(points[section_indices], axis=0))

                    for j in range(n_coordinates - 1):
                        face = [
                            center_index,
                            section_start + j,
                            section_start + j + 1,
                            center_index,
                        ]
                        if reverse_winding:
                            face = face[::-1]
                        faces.append(face)

            if len(cap_points) > 0:
                points = _onp.vstack([points, _onp.asarray(cap_points)])

        faces = _onp.asarray(faces, dtype=int)

        if include_spinner:
            spinner_points, spinner_faces = self._spinner_mesh()
            if len(spinner_points) > 0:
                spinner_faces = spinner_faces + len(points)
                points = _onp.vstack([points, spinner_points])
                faces = _onp.vstack([faces, spinner_faces])

        return points, faces

    def draw(
        self,
        backend: str = "matplotlib",
        ax=None,
        style: str = "shaded",
        radial_resolution: int = 24,
        n_coordinates_per_side: int = 40,
        include_spinner: bool = True,
        close_blade_ends: bool = True,
        show_reference_circle: bool = True,
        color: Optional[Union[str, Tuple[float]]] = None,
        set_lims: bool = True,
        set_equal: bool = True,
        set_axis_visibility: bool = None,
        use_preset_view_angle: str = None,
        show: bool = True,
    ):
        """
        Produces a 3D visualization of the propeller.

        Args:
            backend: Visualization backend. Currently, only ``"matplotlib"`` is supported.
            ax: Matplotlib 3D axis to draw on. If None, creates a new one.
            style: ``"shaded"`` or ``"wireframe"``.
            radial_resolution: Number of blade sections used from hub to tip.
            n_coordinates_per_side: Number of airfoil coordinates per side.
            include_spinner: Whether to include the spinner or hub in the mesh.
            close_blade_ends: Whether to cap the root and tip airfoil sections.
            show_reference_circle: Whether to draw thin hub and tip reference circles.
            color: Blade color. Defaults to ``self.color`` or a neutral gray.
            show: Whether to show the figure after drawing.
        """
        if backend != "matplotlib":
            raise NotImplementedError(
                "Propeller drawing currently supports only `backend='matplotlib'`."
            )
        if style not in {"shaded", "wireframe"}:
            raise ValueError("`style` must be either 'shaded' or 'wireframe'.")

        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if ax is None:
            _, ax = p.figure3d(figsize=(8, 8), computed_zorder=False)
        elif not p.ax_is_3d(ax):
            raise ValueError("`ax` must be a 3D axis.")

        plt.sca(ax)

        points, faces = self.mesh_body(
            radial_resolution=radial_resolution,
            n_coordinates_per_side=n_coordinates_per_side,
            include_spinner=include_spinner,
            close_blade_ends=close_blade_ends,
        )

        plot_color = color if color is not None else self.color
        if plot_color is None:
            plot_color = "lightgray"

        if style == "shaded":
            ax.add_collection(
                Poly3DCollection(
                    points[faces],
                    facecolors=plot_color,
                    edgecolors=(0, 0, 0, 0.12),
                    linewidths=0.35,
                    alpha=0.9,
                    shade=True,
                )
            )
        else:
            ax.add_collection(
                Poly3DCollection(
                    points[faces],
                    facecolors=(0, 0, 0, 0),
                    edgecolors=plot_color,
                    linewidths=0.35,
                )
            )

        if show_reference_circle:
            self._draw_reference_circles(ax=ax)

        if use_preset_view_angle is not None:
            p.set_preset_3d_view_angle(use_preset_view_angle)

        if set_lims:
            xyz_c = _onp.reshape(_as_float_array_if_possible(self.xyz_c), 3)
            radius = self._as_float_scalar(self.radius, "radius")
            limits_points = _onp.vstack(
                [
                    points,
                    xyz_c + _onp.array([[radius, radius, radius]]),
                    xyz_c - _onp.array([[radius, radius, radius]]),
                ]
            )
            margin = 0.04 * _onp.max(_onp.ptp(limits_points, axis=0))
            ax.set_xlim(
                limits_points[:, 0].min() - margin,
                limits_points[:, 0].max() + margin,
            )
            ax.set_ylim(
                limits_points[:, 1].min() - margin,
                limits_points[:, 1].max() + margin,
            )
            ax.set_zlim(
                limits_points[:, 2].min() - margin,
                limits_points[:, 2].max() + margin,
            )

        if set_equal:
            p.equal()

        ax.set_xlabel("$x_g$ [m]")
        ax.set_ylabel("$y_g$ [m]")
        ax.set_zlabel("$z_g$ [m]")

        if set_axis_visibility is not None:
            if set_axis_visibility:
                ax.set_axis_on()
            else:
                ax.set_axis_off()

        if show:
            p.show_plot()

        return ax

    def _draw_reference_circles(self, ax) -> None:
        radius = self._as_float_scalar(self.radius, "radius")
        hub_radius = self._as_float_scalar(self.hub_radius, "hub_radius")
        xyz_c = _onp.reshape(_as_float_array_if_possible(self.xyz_c), 3)
        _, yg_local, zg_local = self.compute_frame()

        theta = _onp.linspace(0, 2 * _onp.pi, 181)
        for circle_radius, alpha, linewidth in [
            (radius, 0.25, 0.6),
            (hub_radius, 0.18, 0.45),
        ]:
            if circle_radius <= 0:
                continue
            circle = (
                xyz_c
                + circle_radius * _onp.cos(theta)[:, None] * yg_local
                + circle_radius * _onp.sin(theta)[:, None] * zg_local
            )
            ax.plot(
                circle[:, 0],
                circle[:, 1],
                circle[:, 2],
                color=(0, 0, 0, alpha),
                linewidth=linewidth,
            )

    def draw_wireframe(self, *args, **kwargs):
        """
        Draws a wireframe of the propeller on a Matplotlib 3D axis.
        """
        kwargs.pop("style", None)
        return self.draw(*args, style="wireframe", **kwargs)

    def draw_three_view(
        self,
        axs=None,
        style: str = "shaded",
        radial_resolution: int = 24,
        n_coordinates_per_side: int = 40,
        include_spinner: bool = True,
        close_blade_ends: bool = True,
        show: bool = True,
    ) -> np.ndarray:
        """
        Draws a standard 4-panel three-view diagram of the propeller using Matplotlib.

        Args:
            axs: A 2D numpy array of Matplotlib 3D axes objects, with shape at
                least ``(2, 2)``. If None, creates a new figure.
            style: ``"shaded"`` or ``"wireframe"``.
            radial_resolution: Number of blade sections used from hub to tip.
            n_coordinates_per_side: Number of airfoil coordinates per side.
            include_spinner: Whether to include the spinner or hub in the mesh.
            close_blade_ends: Whether to cap the root and tip airfoil sections.
            show: Whether to show the figure after creating it.

        Returns:
            A 2D NumPy array of Matplotlib axes objects, with shape ``(2, 2)``.
        """
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        preset_view_angles = np.array(
            [["XZ", "-YZ"], ["XY", "left_isometric"]], dtype="O"
        )

        if axs is None:
            fig, axs = p.figure3d(
                nrows=preset_view_angles.shape[0],
                ncols=preset_view_angles.shape[1],
                figsize=(8, 8),
                computed_zorder=False,
            )
        else:
            if not len(axs.shape) == 2:
                raise ValueError(
                    f"`axs` must be a 2D array of axes; instead, it is: {axs}."
                )
            if axs.shape[0] < preset_view_angles.shape[0]:
                raise ValueError(
                    "`axs` must have at least as many rows as preset_view_angles "
                    f"({preset_view_angles.shape[0]})."
                )
            if axs.shape[1] < preset_view_angles.shape[1]:
                raise ValueError(
                    "`axs` must have at least as many columns as preset_view_angles "
                    f"({preset_view_angles.shape[1]})."
                )

        for i in range(preset_view_angles.shape[0]):
            for j in range(preset_view_angles.shape[1]):
                ax = axs[i, j]
                preset_view = preset_view_angles[i, j]

                self.draw(
                    backend="matplotlib",
                    ax=ax,
                    style=style,
                    radial_resolution=radial_resolution,
                    n_coordinates_per_side=n_coordinates_per_side,
                    include_spinner=include_spinner,
                    close_blade_ends=close_blade_ends,
                    set_axis_visibility=(
                        False if "isometric" in preset_view else None
                    ),
                    show=False,
                )

                p.set_preset_3d_view_angle(preset_view)

                if preset_view == "XY" or preset_view == "-XY":
                    ax.set_zticks([])
                if preset_view == "XZ" or preset_view == "-XZ":
                    ax.set_yticks([])
                if preset_view == "YZ" or preset_view == "-YZ":
                    ax.set_xticks([])

        axs[1, 0].set_xlabel("$x_g$ [m]")
        axs[1, 0].set_ylabel("$y_g$ [m]")
        axs[0, 0].set_zlabel("$z_g$ [m]")
        axs[0, 0].set_xticklabels([])
        axs[0, 1].set_yticklabels([])
        axs[0, 1].set_zticklabels([])

        plt.subplots_adjust(
            left=-0.08,
            right=1.08,
            bottom=-0.08,
            top=1.08,
            wspace=-0.38,
            hspace=-0.38,
        )

        if show:
            p.show_plot(tight_layout=False)

        return axs

    def _normalize_airfoil_distribution(
        self,
        airfoil_distribution,
    ) -> List[Tuple[float, KulfanAirfoil]]:
        if airfoil_distribution is None:
            return [(0.0, KulfanAirfoil("naca4412")), (1.0, KulfanAirfoil("naca4412"))]

        if isinstance(airfoil_distribution, (str, Airfoil, KulfanAirfoil)):
            airfoil = _as_kulfan_airfoil(airfoil_distribution)
            return [(0.0, airfoil), (1.0, airfoil)]

        if isinstance(airfoil_distribution, dict):
            entries = list(airfoil_distribution.items())
        else:
            entries = list(airfoil_distribution)
            if len(entries) > 0 and not isinstance(entries[0], tuple):
                stations = _onp.linspace(
                    self.radial_stations[0],
                    self.radial_stations[-1],
                    len(entries),
                )
                entries = list(zip(stations, entries))

        normalized = [
            (float(r_over_R), _as_kulfan_airfoil(airfoil))
            for r_over_R, airfoil in entries
        ]
        normalized = sorted(normalized, key=lambda entry: entry[0])

        if len(normalized) == 0:
            raise ValueError("`airfoil_distribution` must contain at least one airfoil.")

        return normalized

    def _evaluate_distribution(
        self,
        distribution,
        r_over_R: Union[float, np.ndarray],
        name: str,
    ):
        input_is_scalar = _onp.isscalar(r_over_R)

        if callable(distribution):
            result = distribution(r_over_R)
            if input_is_scalar and hasattr(result, "__len__"):
                return result[0]
            return result

        if np.length(distribution) == 1:
            return np.reshape(np.array(distribution), -1)[0] * np.ones_like(r_over_R)

        if self.distribution_interpolation_method == "linear":
            result = np.interp(
                x=r_over_R,
                xp=self.radial_stations,
                fp=distribution,
                left=np.reshape(np.array(distribution), -1)[0],
                right=np.reshape(np.array(distribution), -1)[-1],
            )
            if input_is_scalar and hasattr(result, "__len__"):
                return result[0]
            return result

        n_control_points = np.length(distribution)
        degree = min(self.spline_degree, n_control_points - 1)
        knots = _open_uniform_knot_vector(
            x_min=self.radial_stations[0],
            x_max=self.radial_stations[-1],
            n_control_points=n_control_points,
            degree=degree,
        )

        try:
            result = bspline(
                x=r_over_R,
                y_control_points=distribution,
                degree=degree,
                knots=knots,
                extrapolation="clip",
            )
            if input_is_scalar and hasattr(result, "__len__"):
                return result[0]
            return result
        except Exception as e:
            raise ValueError(f"Could not evaluate `{name}` as a B-spline.") from e

    def chord(
        self,
        r_over_R: Union[float, np.ndarray],
    ):
        return self._evaluate_distribution(
            distribution=self.chord_distribution,
            r_over_R=r_over_R,
            name="chord_distribution",
        )

    def twist(
        self,
        r_over_R: Union[float, np.ndarray],
    ):
        return self._evaluate_distribution(
            distribution=self.twist_distribution,
            r_over_R=r_over_R,
            name="twist_distribution",
        )

    def thickness(
        self,
        r_over_R: Union[float, np.ndarray],
    ):
        if self.thickness_distribution is None:
            return None
        return self._evaluate_distribution(
            distribution=self.thickness_distribution,
            r_over_R=r_over_R,
            name="thickness_distribution",
        )

    def airfoil(
        self,
        r_over_R: float,
    ) -> KulfanAirfoil:
        airfoil_entries = self.airfoil_distribution

        if len(airfoil_entries) == 1:
            section_airfoil = airfoil_entries[0][1]
        elif r_over_R <= airfoil_entries[0][0]:
            section_airfoil = airfoil_entries[0][1]
        elif r_over_R >= airfoil_entries[-1][0]:
            section_airfoil = airfoil_entries[-1][1]
        else:
            section_airfoil = airfoil_entries[-1][1]
            for i in range(len(airfoil_entries) - 1):
                r_a, airfoil_a = airfoil_entries[i]
                r_b, airfoil_b = airfoil_entries[i + 1]
                if r_a <= r_over_R <= r_b:
                    blend_fraction = (r_over_R - r_a) / (r_b - r_a)
                    section_airfoil = airfoil_a.blend_with_another_airfoil(
                        airfoil=airfoil_b,
                        blend_fraction=blend_fraction,
                    )
                    break

        target_thickness = self.thickness(r_over_R)
        if target_thickness is not None:
            current_thickness = section_airfoil.max_thickness()
            section_airfoil = _scale_kulfan_thickness_about_camber(
                airfoil=section_airfoil,
                thickness_scale=target_thickness / current_thickness,
            )

        return section_airfoil

    def validate_geometry(
        self,
        radial_resolution: int = 101,
        raise_errors: bool = True,
    ) -> Dict[str, Any]:
        """
        Checks numeric propeller geometry for basic physical consistency.
        """
        try:
            hub_over_R = float(self.hub_radius / self.radius)
            r_min = max(hub_over_R, float(self.radial_stations[0]))
            r_max = float(self.radial_stations[-1])
        except (TypeError, ValueError):
            return {
                "is_valid": True,
                "messages": ["Skipped numeric geometry checks for symbolic geometry."],
            }

        r_over_R = _onp.linspace(r_min, r_max, radial_resolution)
        checks = {
            "chord": self.chord(r_over_R),
            "twist": self.twist(r_over_R),
        }
        if self.thickness_distribution is not None:
            checks["thickness"] = self.thickness(r_over_R)

        messages = []
        for name, values in checks.items():
            values = _as_float_array_if_possible(values)
            if values is None:
                continue
            if not _onp.all(_onp.isfinite(values)):
                messages.append(f"`{name}` contains non-finite values.")
            if name in {"chord", "thickness"} and _onp.any(values <= 0):
                messages.append(f"`{name}` must be positive everywhere.")

        if messages and raise_errors:
            raise ValueError(
                f"Propeller '{self.name}' has nonphysical geometry: "
                + " ".join(messages)
            )

        return {
            "is_valid": len(messages) == 0,
            "messages": messages,
        }

    @staticmethod
    def fit_b_spline_control_points(
        x: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        n_control_points: int = 8,
        degree: int = 3,
        preserve_endpoints: bool = True,
    ) -> _onp.ndarray:
        """
        Fits B-spline control points to tabulated data in a least-squares sense.
        """
        from aerosandbox.modeling.splines.bspline import bspline_basis_matrix

        x = _onp.asarray(x, dtype=float)
        y = _onp.asarray(y, dtype=float)
        degree = min(degree, n_control_points - 1)
        knots = _open_uniform_knot_vector(
            x_min=float(_onp.min(x)),
            x_max=float(_onp.max(x)),
            n_control_points=n_control_points,
            degree=degree,
        )
        basis = bspline_basis_matrix(
            x=x,
            n_control_points=n_control_points,
            degree=degree,
            knots=knots,
        )
        if preserve_endpoints:
            control_points = _onp.empty(n_control_points)
            control_points[0] = y[0]
            control_points[-1] = y[-1]
            if n_control_points > 2:
                rhs = (
                    y
                    - basis[:, 0] * control_points[0]
                    - basis[:, -1] * control_points[-1]
                )
                control_points[1:-1], *_ = _onp.linalg.lstsq(
                    basis[:, 1:-1],
                    rhs,
                    rcond=None,
                )
        else:
            control_points, *_ = _onp.linalg.lstsq(basis, y, rcond=None)
        return control_points

    @classmethod
    def from_tabulated_geometry(
        cls,
        name: Optional[str],
        r: Union[np.ndarray, List[float]],
        chord: Union[np.ndarray, List[float]],
        twist: Union[np.ndarray, List[float]],
        radius: Optional[float] = None,
        hub_radius: float = 0.0,
        blade_count: int = 2,
        thickness: Optional[Union[np.ndarray, List[float]]] = None,
        airfoil_distribution=None,
        n_control_points: int = 8,
        spline_degree: int = 3,
        preserve_endpoints: bool = True,
        interpolation_method: str = "linear",
        **kwargs,
    ) -> "Propeller":
        """
        Creates a propeller from tabulated radial geometry.
        """
        r = _onp.asarray(r, dtype=float)
        chord = _onp.asarray(chord, dtype=float)
        twist = _onp.asarray(twist, dtype=float)

        if not (len(r) == len(chord) == len(twist)):
            raise ValueError("`r`, `chord`, and `twist` must have the same length.")
        if len(r) < 2:
            raise ValueError("At least two tabulated radial stations are required.")
        if _onp.any(_onp.diff(r) <= 0):
            raise ValueError("`r` must be strictly increasing.")
        if _onp.any(chord <= 0):
            raise ValueError("`chord` must be positive at every tabulated station.")
        if not _onp.all(_onp.isfinite(twist)):
            raise ValueError("`twist` must be finite at every tabulated station.")

        if radius is None:
            radius = float(_onp.max(r))
        if radius <= 0:
            raise ValueError("`radius` must be positive.")
        if _onp.any(r <= 0) or _onp.any(r > radius):
            raise ValueError("All `r` values must lie within (0, radius].")

        r_over_R = r / radius
        if interpolation_method not in {"linear", "bspline"}:
            raise ValueError("`interpolation_method` must be either 'linear' or 'bspline'.")

        if thickness is not None:
            thickness = _onp.asarray(thickness, dtype=float)
            if len(thickness) != len(r):
                raise ValueError("`thickness` must have the same length as `r`.")
            if _onp.any(thickness <= 0):
                raise ValueError("`thickness` must be positive.")

        if interpolation_method == "linear":
            radial_stations = r_over_R
            chord_distribution = chord
            twist_distribution = twist
            thickness_distribution = thickness
        else:
            radial_stations = _onp.linspace(
                float(_onp.min(r_over_R)),
                float(_onp.max(r_over_R)),
                n_control_points,
            )

            chord_distribution = cls.fit_b_spline_control_points(
                x=r_over_R,
                y=chord,
                n_control_points=n_control_points,
                degree=spline_degree,
                preserve_endpoints=preserve_endpoints,
            )
            twist_distribution = cls.fit_b_spline_control_points(
                x=r_over_R,
                y=twist,
                n_control_points=n_control_points,
                degree=spline_degree,
                preserve_endpoints=preserve_endpoints,
            )

            if thickness is None:
                thickness_distribution = None
            else:
                thickness_distribution = cls.fit_b_spline_control_points(
                    x=r_over_R,
                    y=thickness,
                    n_control_points=n_control_points,
                    degree=spline_degree,
                    preserve_endpoints=preserve_endpoints,
                )

        propeller = cls(
            name=name,
            radius=radius,
            hub_radius=hub_radius,
            blade_count=blade_count,
            radial_stations=radial_stations,
            chord_distribution=chord_distribution,
            twist_distribution=twist_distribution,
            thickness_distribution=thickness_distribution,
            airfoil_distribution=airfoil_distribution,
            distribution_interpolation_method=interpolation_method,
            spline_degree=spline_degree,
            **kwargs,
        )
        propeller.validate_geometry(raise_errors=True)
        return propeller
