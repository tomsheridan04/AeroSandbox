"""Research ablations around the public NormalFlowLiftingLine implementation.

Use ``asb.NormalFlowLiftingLine`` in a design script. This subclass retains the
original study's legacy-flow and slope experiments so its comparison notebook
can be reproduced without maintaining a second mesher or force integrator.
"""

import aerosandbox.numpy as np
from aerosandbox.aerodynamics.aero_3D.normal_flow_lifting_line import (
    NormalFlowLiftingLine,
    _SectionalFlow,
    _column,
    _row,
    _limit_lift_slope,
)


class CorrectedLiftingLine(NormalFlowLiftingLine):
    """Study-only alternatives; historical default remains legacy kinematics."""

    def __init__(
        self,
        *args,
        slope_model="attached",
        kinematics="legacy",
        slope_scale=1.0,
        **kwargs
    ):
        if slope_model not in {"two_pi", "attached", "local"}:
            raise ValueError("slope_model must be 'two_pi', 'attached', or 'local'.")
        if kinematics not in {"legacy", "normal"}:
            raise ValueError("kinematics must be 'legacy' or 'normal'.")
        if slope_scale <= 0:
            raise ValueError("slope_scale must be positive.")
        self.slope_model = slope_model
        self.kinematics = kinematics
        self.slope_scale = slope_scale
        super().__init__(*args, **kwargs)

    def _slope(self, alpha, reynolds, mach):
        if self.slope_model == "two_pi":
            return np.ones(self.n_panels) * (2 * np.pi * self.slope_scale)
        if self.slope_model == "attached":
            return super()._slope(alpha, reynolds, mach) * self.slope_scale
        samples = self._section_aero(
            [np.array([alpha[i] - 0.5, alpha[i] + 0.5]) for i in range(self.n_panels)],
            reynolds,
            mach,
        )
        slope = np.array(
            [
                (sample["CL"][1] - sample["CL"][0]) / np.radians(1.0)
                for sample in samples
            ]
        )
        return _limit_lift_slope(slope) * self.slope_scale

    def _correction_slope(self, flow, attached_slope):
        if self.slope_model == "local":
            return self._slope(flow.alpha, flow.reynolds, flow.mach)
        return attached_slope

    def _initialize_velocity_influences(self):
        super()._initialize_velocity_influences()
        directions = np.tile(_row(self.steady_freestream_direction), (self.n_panels, 1))
        self._legacy_alpha = 90 - np.arccosd(
            np.sum(directions * self.normal_directions, axis=1)
        )
        self._legacy_speed = self.op_point.velocity * np.sum(
            directions * self._section_forward, axis=1
        )
        self._legacy_circulation_to_cl = (
            2
            * np.linalg.norm(
                np.cross(
                    np.tile(_row(self.steady_freestream_velocity), (self.n_panels, 1)),
                    self.vortex_bound_leg,
                    axis=1,
                ),
                axis=1,
            )
            / self._legacy_speed**2
            / self.areas
        )
        self._legacy_angle_influence = self._normal_influence / self.op_point.velocity

    def _flow_state(self, circulation):
        if self.kinematics == "normal":
            return super()._flow_state(circulation)
        velocities = self.freestream_velocities + np.stack(
            [influence @ circulation for influence in self._velocity_influences],
            axis=1,
        )
        return _SectionalFlow(
            velocities=velocities,
            alpha=self._legacy_alpha
            + np.degrees(self._legacy_angle_influence @ circulation),
            reynolds=self._legacy_speed
            * self.chords
            / self.op_point.atmosphere.kinematic_viscosity(),
            mach=self._legacy_speed / self.op_point.atmosphere.speed_of_sound(),
            circulation_to_cl=self._legacy_circulation_to_cl,
            angle_influence=self._legacy_angle_influence,
            perpendicular_speed=self._legacy_speed,
            perpendicular_velocity=None,  # Legacy profile forces use the full velocity.
        )

    def _profile_aerodynamics(self, flow, drag_coefficients, moment_coefficients):
        if self.kinematics == "normal":
            return super()._profile_aerodynamics(
                flow, drag_coefficients, moment_coefficients
            )
        speed = np.linalg.norm(flow.velocities, axis=1)
        rho = self.op_point.atmosphere.density()
        forces = (
            0.5
            * rho
            * flow.velocities
            * _column(speed * drag_coefficients * self.areas)
        )
        pitching_axis = np.stack(
            [
                0 * self.vortex_bound_leg[:, 0],
                self.vortex_bound_leg[:, 1],
                self.vortex_bound_leg[:, 2],
            ],
            axis=1,
        )
        moments = (
            0.5
            * rho
            * _column(speed**2 * moment_coefficients * self.chords**2)
            * pitching_axis
        )
        return forces, moments
