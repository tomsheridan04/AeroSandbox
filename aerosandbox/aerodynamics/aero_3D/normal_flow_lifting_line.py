"""Fixed-work lifting line with NeuralFoil circulation corrections.

Geometry and output conventions follow :class:`LiftingLine`. The two analyses
remain separate so an existing design script can select either formulation.
"""

from dataclasses import dataclass
from typing import Any, Callable

import aerosandbox.numpy as np
from aerosandbox.geometry import Airplane
from aerosandbox.performance import OperatingPoint
from .lifting_line import LiftingLine
from .singularities.uniform_strength_horseshoe_singularities import (
    calculate_induced_velocity_horseshoe,
)


def _column(values):
    return np.reshape(values, (-1, 1))


def _row(values):
    return np.reshape(values, (1, -1))


def _limit_lift_slope(slope):
    """Smoothly limit the approximate Jacobian slope, not the airfoil lift."""
    lower, upper, smoothing = 0.5, 12.0, 0.05  # All in 1/rad.
    above_lower = (slope + lower + np.sqrt((slope - lower) ** 2 + smoothing**2)) / 2
    return (
        above_lower + upper - np.sqrt((above_lower - upper) ** 2 + smoothing**2)
    ) / 2


@dataclass
class _SectionalFlow:
    """Local flow and the approximate circulation Jacobian at one iterate."""

    velocities: Any  # N x 3, geometry axes, m/s.
    alpha: Any  # Degrees in each section's normal plane.
    reynolds: Any
    mach: Any
    circulation_to_cl: Any  # D: sectional Kutta-Joukowski cl = D * Gamma.
    angle_influence: Any  # B: d(alpha_i in radians) / d(Gamma_j).
    perpendicular_speed: Any
    perpendicular_velocity: Any  # N x 3, excluding velocity along the bound leg.


class NormalFlowLiftingLine(LiftingLine):
    """Lifting line with normal-plane flow and two circulation corrections.

    This explicit analysis uses the same geometry, operating-point arguments,
    force outputs, stability-derivative interface, and spanwise ordering as
    :class:`LiftingLine`. It supports numeric operating points and CasADi design
    variables. Loop over operating points to construct a numeric polar.

    Each airfoil is evaluated using flow perpendicular to its bound vortex.
    The normal chord is strip area divided by bound-leg length. The supplied
    airfoil polar is interpreted in that plane; its shape is not transformed
    for sweep.

    An initial linear circulation solve uses a NeuralFoil lift-curve slope
    sampled at +/-2 degrees. By default, two further linear solves reduce the
    mismatch between Kutta-Joukowski and NeuralFoil sectional lift. Their slope
    is smoothly limited to 0.5--12 per radian for the approximate Jacobian;
    the NeuralFoil lift coefficient itself is not limited.

    There is no internal nonlinear optimizer or convergence-based loop. Two
    corrections do not establish a converged nonlinear solution, particularly
    near stall. Inspect ``sectional_residual`` and mesh sensitivity. The method
    does not guarantee more accurate drag or CLmax for every configuration.

    Example:
        >>> analysis = NormalFlowLiftingLine(
        ...     airplane=airplane,
        ...     op_point=OperatingPoint(velocity=25, alpha=5),
        ...     spanwise_resolution=20,
        ...     n_crit=5,
        ...     xtr_upper=0.3,
        ...     xtr_lower=0.3,
        ... )
        >>> aero = analysis.run_with_stability_derivatives(
        ...     alpha=True, beta=False, p=False, q=False, r=False,
        ... )
    """

    def __init__(
        self,
        airplane: Airplane,
        op_point: OperatingPoint,
        xyz_ref=None,
        model_size: str = "medium",
        n_crit=9.0,
        xtr_upper=1.0,
        xtr_lower=1.0,
        run_symmetric_if_possible: bool = False,
        verbose: bool = False,
        spanwise_resolution: int = 4,
        spanwise_spacing_function: Callable = np.cosspace,
        vortex_core_radius: float = 1e-8,
        align_trailing_vortices_with_wind: bool = False,
        *,
        circulation_corrections: int = 2,
        relaxation: float = 0.8,
    ):
        """Initialize an analysis; call ``run()`` to evaluate it.

        Args:
            airplane: Lifting surfaces, fuselages, and coefficient references.
            op_point: OperatingPoint, including optional angular rates.
            xyz_ref: Moment reference; defaults to ``airplane.xyz_ref``.
            model_size: NeuralFoil model size.
            n_crit: NeuralFoil transition amplification factor.
            xtr_upper: Upper forced-transition position, as a chord fraction.
            xtr_lower: Lower forced-transition position, as a chord fraction.
            run_symmetric_if_possible: Inherited option; symmetry reduction is
                not implemented. Leave False.
            verbose: Print analysis progress.
            spanwise_resolution: Subdivisions per original wing section.
            spanwise_spacing_function: Spacing used for those subdivisions.
            vortex_core_radius: Horseshoe vortex regularization radius, metres.
            align_trailing_vortices_with_wind: Use freestream-aligned wakes
                instead of geometry-x-aligned wakes.
            circulation_corrections: Fixed additional linear solves, from 0
                through 8. The default of 2 gives 3 solves in total.
            relaxation: Fraction of each correction applied, in (0, 1]. This
                and the correction count are numerical settings, not design
                variables.
        """
        if (
            isinstance(circulation_corrections, bool)
            or not isinstance(circulation_corrections, int)
            or not 0 <= circulation_corrections <= 8
        ):
            raise ValueError("circulation_corrections must be an integer from 0 to 8.")
        if not 0 < relaxation <= 1:
            raise ValueError("relaxation must be in (0, 1].")
        self.circulation_corrections = circulation_corrections
        self.relaxation = relaxation
        super().__init__(
            airplane=airplane,
            op_point=op_point,
            xyz_ref=xyz_ref,
            model_size=model_size,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
            run_symmetric_if_possible=run_symmetric_if_possible,
            verbose=verbose,
            spanwise_resolution=spanwise_resolution,
            spanwise_spacing_function=spanwise_spacing_function,
            vortex_core_radius=vortex_core_radius,
            align_trailing_vortices_with_wind=align_trailing_vortices_with_wind,
        )

    def _initialize_geometry(self):
        """Follow LiftingLine's panel, airfoil, and mirrored-control ordering."""
        vertex_groups = [[], [], [], []]
        airfoils, controls, wing_ids, sides, strip_ids = [], [], [], [], []
        for wing_index, wing in enumerate(self.airplane.wings):
            if self.spanwise_resolution > 1:
                wing = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function,
                )
            points, faces = wing.mesh_thin_surface(
                method="quad",
                chordwise_resolution=1,
                add_camber=False,
            )
            for i in range(4):
                vertex_groups[i].append(points[faces[:, i], :])
            sections = list(zip(wing.xsecs[:-1], wing.xsecs[1:]))
            count = len(sections)
            section_airfoils = [
                a.airfoil.blend_with_another_airfoil(b.airfoil, blend_fraction=0.5)
                for a, b in sections
            ]
            section_controls = [a.control_surfaces for a, b in sections]
            for side in ([1, -1] if wing.symmetric else [1]):
                airfoils.extend(section_airfoils)
                if side == 1:
                    controls.extend(section_controls)
                else:
                    for surfaces in section_controls:
                        mirrored = []
                        for surface in surfaces:
                            copied = surface.copy()
                            if not copied.symmetric:
                                copied.deflection = -copied.deflection
                            mirrored.append(copied)
                        controls.append(mirrored)
                wing_ids.extend([wing_index] * count)
                sides.extend([side] * count)
                strip_ids.extend(range(count))

        front_left, back_left, back_right, front_right = [
            np.concatenate(parts, axis=0) for parts in vertex_groups
        ]
        self.front_left_vertices, self.back_left_vertices = front_left, back_left
        self.back_right_vertices, self.front_right_vertices = back_right, front_right
        self.left_vortex_vertices = 0.75 * front_left + 0.25 * back_left
        self.right_vortex_vertices = 0.75 * front_right + 0.25 * back_right
        self.vortex_centers = (
            self.left_vortex_vertices + self.right_vortex_vertices
        ) / 2
        self.vortex_bound_leg = self.right_vortex_vertices - self.left_vortex_vertices
        self._bound_leg_length = np.linalg.norm(self.vortex_bound_leg, axis=1)
        self._section_span = self.vortex_bound_leg / _column(self._bound_leg_length)
        cross = np.cross(front_right - back_left, front_left - back_right, axis=1)
        cross_norm = np.linalg.norm(cross, axis=1)
        self.normal_directions = cross / _column(cross_norm)
        self.areas = cross_norm / 2
        self.chord_vectors = (back_left + back_right - front_left - front_right) / 2
        self.chords = np.linalg.norm(self.chord_vectors, axis=1)
        self.local_forward_direction = np.cross(
            self.normal_directions,
            self._section_span,
            axis=1,
        )
        self._section_forward = -self.local_forward_direction / _column(
            np.linalg.norm(self.local_forward_direction, axis=1)
        )
        self._normal_chord = self.areas / self._bound_leg_length
        self.n_panels = int(self.areas.shape[0])
        self.airfoils, self.control_surfaces = airfoils, controls
        self.panel_wing_indices = np.array(wing_ids)
        self.panel_side_indices = np.array(sides)
        self.panel_spanwise_indices = np.array(strip_ids)
        self.spanwise_wing_index = self.panel_wing_indices
        self.spanwise_side = self.panel_side_indices
        self.spanwise_y = self.vortex_centers[:, 1]
        self.spanwise_dy = np.abs(
            self.right_vortex_vertices[:, 1] - self.left_vortex_vertices[:, 1]
        )
        self.spanwise_chord = self.chords
        self.steady_freestream_velocity = (
            self.op_point.compute_freestream_velocity_geometry_axes()
        )
        self.steady_freestream_direction = (
            self.steady_freestream_velocity
            / np.linalg.norm(self.steady_freestream_velocity)
        )
        offsets = self.vortex_centers - np.tile(
            _row(np.array(self.xyz_ref)), (self.n_panels, 1)
        )
        self.freestream_velocities = np.tile(
            _row(self.steady_freestream_velocity), (self.n_panels, 1)
        ) + self.op_point.compute_rotation_velocity_geometry_axes(offsets)

    def _initialize_velocity_influences(self):
        """Build the unit-circulation velocity matrices once per analysis."""
        self._velocity_influences = calculate_induced_velocity_horseshoe(
            x_field=_column(self.vortex_centers[:, 0]),
            y_field=_column(self.vortex_centers[:, 1]),
            z_field=_column(self.vortex_centers[:, 2]),
            x_left=_row(self.left_vortex_vertices[:, 0]),
            y_left=_row(self.left_vortex_vertices[:, 1]),
            z_left=_row(self.left_vortex_vertices[:, 2]),
            x_right=_row(self.right_vortex_vertices[:, 0]),
            y_right=_row(self.right_vortex_vertices[:, 1]),
            z_right=_row(self.right_vortex_vertices[:, 2]),
            trailing_vortex_direction=(
                self.steady_freestream_direction
                if self.align_trailing_vortices_with_wind
                else np.array([1.0, 0.0, 0.0])
            ),
            gamma=1.0,
            vortex_core_radius=self.vortex_core_radius,
        )
        self._normal_influence = sum(
            matrix * _column(self.normal_directions[:, axis])
            for axis, matrix in enumerate(self._velocity_influences)
        )
        self._forward_influence = sum(
            matrix * _column(self._section_forward[:, axis])
            for axis, matrix in enumerate(self._velocity_influences)
        )

    def _section_aero(self, alpha, reynolds, mach):
        """Evaluate each section with its inherited transition/control settings."""
        return [
            airfoil.get_aero_from_neuralfoil(
                alpha=alpha[i],
                Re=reynolds[i],
                mach=mach[i],
                n_crit=self.n_crit,
                xtr_upper=self.xtr_upper,
                xtr_lower=self.xtr_lower,
                control_surfaces=self.control_surfaces[i],
                model_size=self.model_size,
            )
            for i, airfoil in enumerate(self.airfoils)
        ]

    @staticmethod
    def _values(aeros, key):
        return np.reshape(np.array([aero[key] for aero in aeros]), (-1,))

    def _slope(self, alpha, reynolds, mach):
        """Attached-flow slope, in 1/rad, from a two-angle NeuralFoil query."""
        samples = self._section_aero(
            [np.array([-2.0, 2.0]) for _ in range(self.n_panels)],
            reynolds,
            mach,
        )
        slopes = np.array(
            [
                (sample["CL"][1] - sample["CL"][0]) / np.radians(4.0)
                for sample in samples
            ]
        )
        return _limit_lift_slope(slopes)

    def _flow_state(self, circulation):
        """Resolve local flow normal to the bound leg and its angle Jacobian."""
        velocities = self.freestream_velocities + np.stack(
            [influence @ circulation for influence in self._velocity_influences],
            axis=1,
        )
        forward_speed = np.sum(velocities * self._section_forward, axis=1)
        normal_speed = np.sum(velocities * self.normal_directions, axis=1)
        speed_squared = forward_speed**2 + normal_speed**2 + 1e-16
        speed = np.sqrt(speed_squared)
        return _SectionalFlow(
            velocities=velocities,
            alpha=np.degrees(np.arctan2(normal_speed, forward_speed)),
            reynolds=speed
            * self._normal_chord
            / self.op_point.atmosphere.kinematic_viscosity(),
            mach=speed / self.op_point.atmosphere.speed_of_sound(),
            circulation_to_cl=2 * self._bound_leg_length / speed / self.areas,
            angle_influence=(
                _column(forward_speed) * self._normal_influence
                - _column(normal_speed) * self._forward_influence
            )
            / _column(speed_squared),
            perpendicular_speed=speed,
            perpendicular_velocity=(
                _column(forward_speed) * self._section_forward
                + _column(normal_speed) * self.normal_directions
            ),
        )

    def _correction_slope(self, flow, attached_slope):
        """Reuse the attached-flow slope as an approximate Jacobian slope."""
        return attached_slope

    def _solve_circulation(self):
        """One predictor followed by a fixed number of relaxed corrections."""
        free = self._flow_state(np.zeros(self.n_panels))
        lift = self._values(
            self._section_aero(free.alpha, free.reynolds, free.mach), "CL"
        )
        slope = self._slope(free.alpha, free.reynolds, free.mach)
        attached_slope = slope
        # Each receiving section's slope scales an entire ROW of the angle
        # influence matrix. This matters when slopes differ between sections.
        circulation = np.linalg.solve(
            np.diag(free.circulation_to_cl) - _column(slope) * free.angle_influence,
            lift,
        )
        for _ in range(self.circulation_corrections):
            flow = self._flow_state(circulation)
            lift = self._values(
                self._section_aero(flow.alpha, flow.reynolds, flow.mach), "CL"
            )
            slope = self._correction_slope(flow, attached_slope)
            residual = lift - flow.circulation_to_cl * circulation
            # This approximate Jacobian neglects Re/Mach and speed derivatives.
            # CasADi still differentiates the complete fixed update sequence.
            correction = np.linalg.solve(
                np.diag(flow.circulation_to_cl) - _column(slope) * flow.angle_influence,
                residual,
            )
            circulation = circulation + self.relaxation * correction
        return circulation, slope

    def _profile_aerodynamics(self, flow, drag_coefficients, moment_coefficients):
        """Use the same normal-plane convention for drag and pitching couples."""
        rho = self.op_point.atmosphere.density()
        profile_forces = (
            0.5
            * rho
            * _column(flow.perpendicular_speed * self.areas * drag_coefficients)
            * flow.perpendicular_velocity
        )
        pitching_moments = (
            0.5
            * rho
            * _column(
                flow.perpendicular_speed**2
                * self.areas
                * self._normal_chord
                * moment_coefficients
            )
            * self._section_span
        )
        return profile_forces, pitching_moments

    def _integrate_forces(self, flow, aeros):
        """Integrate Kutta-Joukowski forces, profile forces, and moments."""
        forces = (
            self.op_point.atmosphere.density()
            * np.cross(flow.velocities, self.vortex_bound_leg, axis=1)
            * _column(self.vortex_strengths)
        )
        offsets = self.vortex_centers - np.tile(
            _row(np.array(self.xyz_ref)), (self.n_panels, 1)
        )
        moments = np.cross(offsets, forces, axis=1)
        profile_forces, pitching_moments = self._profile_aerodynamics(
            flow,
            self._values(aeros, "CD"),
            self._values(aeros, "CM"),
        )
        profile_moments = np.cross(offsets, profile_forces, axis=1)
        self.forces_inviscid_geometry = forces
        self.moments_inviscid_geometry = moments
        self.force_inviscid_geometry = np.sum(forces, axis=0)
        self.moment_inviscid_geometry = np.sum(moments, axis=0)
        self.forces_profile_geometry = profile_forces
        self.force_profile_geometry = np.sum(profile_forces, axis=0)
        self._update_spanwise(forces)
        total_force = self.force_inviscid_geometry + self.force_profile_geometry
        total_moment = self.moment_inviscid_geometry + np.sum(
            profile_moments + pitching_moments, axis=0
        )
        return self.AeroComponentResults(
            s_ref=self.airplane.s_ref,
            c_ref=self.airplane.c_ref,
            b_ref=self.airplane.b_ref,
            op_point=self.op_point,
            F_g=[total_force[i] for i in range(3)],
            M_g=[total_moment[i] for i in range(3)],
        )

    def _update_spanwise(self, forces):
        """Preserve the original projected-y distributions and per-wing order."""
        wind = self.op_point.convert_axes(
            forces[:, 0],
            forces[:, 1],
            forces[:, 2],
            from_axes="geometry",
            to_axes="wind",
        )
        self.spanwise_lift = -wind[2]
        safe_width = np.where(self.spanwise_dy == 0, np.nan, self.spanwise_dy)
        self.spanwise_lift_per_y = self.spanwise_lift / safe_width
        self.spanwise_cl = (
            self.spanwise_lift_per_y / self.op_point.dynamic_pressure() / self.chords
        )
        self.spanwise_clc_over_cref = (
            self.spanwise_lift_per_y
            / self.op_point.dynamic_pressure()
            / self.airplane.c_ref
        )
        indices_by_wing = []
        for wing_index in range(len(self.airplane.wings)):
            ordered = []
            for side in [-1, 1]:
                indices = [
                    i
                    for i in range(self.n_panels)
                    if self.panel_wing_indices[i] == wing_index
                    and self.panel_side_indices[i] == side
                ]
                ordered.extend(
                    sorted(
                        indices,
                        key=lambda i: self.panel_spanwise_indices[i],
                        reverse=side == -1,
                    )
                )
            indices_by_wing.append(ordered)
        for attribute, values in [
            ("y", self.spanwise_y),
            ("dy", self.spanwise_dy),
            ("chord", self.chords),
            ("lift", self.spanwise_lift),
            ("lift_per_y", self.spanwise_lift_per_y),
            ("cl", self.spanwise_cl),
            ("clc_over_cref", self.spanwise_clc_over_cref),
        ]:
            setattr(
                self,
                attribute,
                [np.array([values[i] for i in indices]) for indices in indices_by_wing],
            )

    def wing_aerodynamics(self):
        """Mesh, solve circulation, and evaluate forces at the final circulation."""
        if self.verbose:
            print("Meshing and constructing normal-flow velocity influences...")
        self._initialize_geometry()
        self._initialize_velocity_influences()
        self.vortex_strengths, self.sectional_slope = self._solve_circulation()
        flow = self._flow_state(self.vortex_strengths)
        aeros = self._section_aero(flow.alpha, flow.reynolds, flow.mach)
        self.sectional_alpha = flow.alpha
        self.sectional_reynolds = flow.reynolds
        self.sectional_mach = flow.mach
        self.sectional_cl_neuralfoil = self._values(aeros, "CL")
        self.sectional_cl_circulation = flow.circulation_to_cl * self.vortex_strengths
        self.sectional_residual = (
            self.sectional_cl_neuralfoil - self.sectional_cl_circulation
        )
        self.sectional_confidence = self._values(aeros, "analysis_confidence")
        return self._integrate_forces(flow, aeros)

    def run(self):
        """Return LiftingLine outputs plus sectional consistency diagnostics.

        Additional arrays, in panel order:

        * ``sectional_alpha``: Effective section angle, degrees.
        * ``sectional_reynolds`` and ``sectional_mach``: Section flow conditions.
        * ``sectional_cl_neuralfoil``: Lift from the final airfoil evaluations.
        * ``sectional_cl_circulation``: Lift implied by the final circulation.
        * ``sectional_residual``: NeuralFoil cl minus circulation cl.
        * ``sectional_slope``: Approximate Jacobian slope, per radian.
        * ``sectional_confidence``: NeuralFoil analysis confidence.

        ``circulation_corrections`` records the fixed correction count.
        ``CD_induced`` and ``CD_profile`` split lifting-surface drag; fuselage
        contributions remain in the inherited total ``CD``. Projected-y
        spanwise coefficients remain undefined for vertical surfaces, while
        their integrated forces and sectional consistency remain finite.
        """
        result = super().run()
        for name in [
            "sectional_alpha",
            "sectional_reynolds",
            "sectional_mach",
            "sectional_cl_neuralfoil",
            "sectional_cl_circulation",
            "sectional_residual",
            "sectional_slope",
            "sectional_confidence",
            "circulation_corrections",
        ]:
            result[name] = getattr(self, name)
        for label, force in [
            ("induced", self.force_inviscid_geometry),
            ("profile", self.force_profile_geometry),
        ]:
            wind = self.op_point.convert_axes(
                *[force[i] for i in range(3)],
                from_axes="geometry",
                to_axes="wind",
            )
            result[f"CD_{label}"] = (
                -wind[0] / self.op_point.dynamic_pressure() / self.airplane.s_ref
            )
        return result
