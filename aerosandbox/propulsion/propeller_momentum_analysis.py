from aerosandbox.propulsion.propeller_analysis import PropellerAnalysis
from typing import Any, Dict, Optional
import aerosandbox.numpy as np


class PropellerMomentumAnalysis(PropellerAnalysis):
    """
    Differentiable blade-element momentum propeller analysis.

    This is a deliberately simpler alternative to :class:`PropellerAnalysis`.
    Instead of Drela's QPROP-like circulation/vortex closure, each radial station
    solves for axial and tangential induced velocities using annular momentum
    balances. Section lift and drag still come from the same section model as
    ``PropellerAnalysis`` by default, so this class is useful for isolating the
    effect of the inflow closure.
    """

    def __init__(
        self,
        *args,
        axial_induced_velocity_initial_guess_fraction: Optional[float] = None,
        tangential_induced_velocity_initial_guess_fraction: float = 0.02,
        relaxation: float = 0.35,
        maximum_tangential_induction_fraction: float = 0.95,
        **kwargs,
    ):
        """
        Args:
            axial_induced_velocity_initial_guess_fraction: Initial axial induced
                velocity, nondimensionalized by ``abs(omega) * radius``. If
                omitted, this uses ``induced_velocity_initial_guess_fraction``.
            tangential_induced_velocity_initial_guess_fraction: Initial swirl
                induced velocity, nondimensionalized by local ``abs(omega) * r``.
            relaxation: Fixed-point relaxation factor for the induced velocities.
            maximum_tangential_induction_fraction: Upper bound on local swirl
                induced velocity as a fraction of ``abs(omega) * r``. This keeps
                the relative tangential velocity on the physical branch.
        """
        super().__init__(*args, **kwargs)

        if axial_induced_velocity_initial_guess_fraction is None:
            axial_induced_velocity_initial_guess_fraction = (
                self.induced_velocity_initial_guess_fraction
            )

        self.axial_induced_velocity_initial_guess_fraction = (
            axial_induced_velocity_initial_guess_fraction
        )
        self.tangential_induced_velocity_initial_guess_fraction = (
            tangential_induced_velocity_initial_guess_fraction
        )
        self.relaxation = relaxation
        self.maximum_tangential_induction_fraction = (
            maximum_tangential_induction_fraction
        )

    @staticmethod
    def _throughflow_preserving_root_of_x_times_x_plus_v_equals_y(V, y):
        """
        Returns ``u`` such that ``u * (V + u) = y`` on the branch that keeps
        ``V + u`` positive when possible.
        """
        V_abs = (V**2 + 1e-16) ** 0.5
        y = np.maximum(y, -0.25 * V_abs**2 + 1e-16)
        return -0.5 * V_abs + (0.25 * V_abs**2 + y) ** 0.5

    def run(self) -> Dict[str, Any]:
        """
        Runs the propeller analysis.

        Returns:
            A dictionary with the same main integrated and spanwise keys as
            ``PropellerAnalysis``. Additional keys include
            ``axial_induced_velocity``, ``tangential_induced_velocity``,
            ``momentum_dT`` and ``momentum_dQ``.
        """
        propeller = self.propeller
        op_point = self.op_point

        rho = op_point.atmosphere.density()
        mu = op_point.atmosphere.dynamic_viscosity()
        speed_of_sound = op_point.atmosphere.speed_of_sound()
        V = op_point.velocity
        omega = self.omega
        B = propeller.blade_count

        r_edges = self.radial_spacing_function(
            propeller.hub_radius,
            propeller.radius,
            self.radial_resolution + 1,
        )
        r = (r_edges[:-1] + r_edges[1:]) / 2
        dr = r_edges[1:] - r_edges[:-1]
        r_over_R = r / propeller.radius

        chord = propeller.chord(r_over_R)
        twist = propeller.twist(r_over_R) + self.pitch_offset
        thickness = propeller.thickness(r_over_R)
        section_airfoils = [
            propeller.airfoil(float(r_over_R[i])) for i in range(self.radial_resolution)
        ]

        Ua_external = V + self.external_axial_induced_velocity
        Ut_external = omega * r - self.external_tangential_induced_velocity
        Ut_magnitude = (Ut_external**2 + 1e-16) ** 0.5

        axial_induced_velocity = (
            self.axial_induced_velocity_initial_guess_fraction
            * np.abs(omega)
            * propeller.radius
            * np.ones_like(r)
        )
        tangential_induced_velocity = (
            self.tangential_induced_velocity_initial_guess_fraction
            * Ut_magnitude
        )

        def evaluate(ua, ut):
            ut_limit = self.maximum_tangential_induction_fraction * Ut_magnitude
            ua_lower = -0.95 * np.maximum(Ua_external, 0.0)
            ua = np.maximum(ua, ua_lower)
            ut = np.minimum(np.maximum(ut, -ut_limit), ut_limit)

            Wa = Ua_external + ua
            Wt = Ut_external - ut
            Wt = np.sign(Ut_external + 1e-16) * np.maximum(np.abs(Wt), 1e-8)
            W = (Wa**2 + Wt**2) ** 0.5

            phi = np.arctan2d(Wa, Wt)
            alpha = twist - phi
            Re = rho * W * chord / mu
            mach = W / speed_of_sound

            CL = []
            CD = []
            CM = []
            analysis_confidence = []
            CL_neuralfoil = []
            CD_neuralfoil = []
            CM_neuralfoil = []
            CL_post_stall = []
            CD_post_stall = []
            post_stall_blend_fraction = []
            for i in range(self.radial_resolution):
                aero = self._section_aero(
                    airfoil=section_airfoils[i],
                    alpha=alpha[i],
                    Re=Re[i],
                    mach=mach[i],
                    r_over_R=float(r_over_R[i]),
                )
                CL.append(np.reshape(np.array(aero["CL"]), -1)[0])
                CD.append(np.reshape(np.array(aero["CD"]), -1)[0])
                CM.append(np.reshape(np.array(aero.get("CM", 0.0)), -1)[0])
                analysis_confidence.append(
                    np.reshape(np.array(aero.get("analysis_confidence", 1.0)), -1)[0]
                )
                CL_neuralfoil.append(
                    np.reshape(np.array(aero.get("CL_neuralfoil", aero["CL"])), -1)[0]
                )
                CD_neuralfoil.append(
                    np.reshape(np.array(aero.get("CD_neuralfoil", aero["CD"])), -1)[0]
                )
                CM_neuralfoil.append(
                    np.reshape(
                        np.array(aero.get("CM_neuralfoil", aero.get("CM", 0.0))), -1
                    )[0]
                )
                CL_post_stall.append(
                    np.reshape(np.array(aero.get("CL_post_stall", aero["CL"])), -1)[0]
                )
                CD_post_stall.append(
                    np.reshape(np.array(aero.get("CD_post_stall", aero["CD"])), -1)[0]
                )
                post_stall_blend_fraction.append(
                    np.reshape(
                        np.array(aero.get("post_stall_blend_fraction", 0.0)), -1
                    )[0]
                )

            CL = np.array(CL)
            CD = np.array(CD)
            CM = np.array(CM)
            analysis_confidence = np.array(analysis_confidence)
            CL_neuralfoil = np.array(CL_neuralfoil)
            CD_neuralfoil = np.array(CD_neuralfoil)
            CM_neuralfoil = np.array(CM_neuralfoil)
            CL_post_stall = np.array(CL_post_stall)
            CD_post_stall = np.array(CD_post_stall)
            post_stall_blend_fraction = np.array(post_stall_blend_fraction)

            q = 0.5 * rho * W**2
            blade_element_dT_dr = (
                B * q * chord * (CL * np.cosd(phi) - CD * np.sind(phi))
            )
            blade_element_dQ_dr = (
                B * q * chord * r * (CL * np.sind(phi) + CD * np.cosd(phi))
            )

            lambda_w = r_over_R * Wa / Wt
            lambda_w_magnitude = (lambda_w**2 + 1e-16) ** 0.5

            if self.include_tip_loss:
                f_tip = B / 2 * (1 - r_over_R) / lambda_w_magnitude
                tip_loss_factor = self._prandtl_loss_factor(f_tip)
            else:
                tip_loss_factor = np.ones_like(r)

            if self.include_hub_loss:
                hub_over_R = propeller.hub_radius / propeller.radius
                f_hub = B / 2 * (r_over_R - hub_over_R) / lambda_w_magnitude
                root_loss_factor = self._prandtl_loss_factor(f_hub)
            else:
                root_loss_factor = np.ones_like(r)

            finite_blade_loss_factor = tip_loss_factor * root_loss_factor
            finite_blade_loss_factor = np.maximum(finite_blade_loss_factor, 1e-6)

            momentum_dT_dr = (
                4
                * np.pi
                * rho
                * r
                * finite_blade_loss_factor
                * ua
                * (Ua_external + ua)
            )
            momentum_dQ_dr = (
                4
                * np.pi
                * rho
                * r**2
                * finite_blade_loss_factor
                * ut
                * (Ua_external + ua)
            )

            thrust_momentum_coefficient = (
                blade_element_dT_dr
                / (4 * np.pi * rho * r * finite_blade_loss_factor + 1e-30)
            )
            ua_target = self._throughflow_preserving_root_of_x_times_x_plus_v_equals_y(
                Ua_external,
                thrust_momentum_coefficient,
            )
            ua_target = np.maximum(ua_target, ua_lower)

            torque_momentum_denominator = (
                4
                * np.pi
                * rho
                * r**2
                * finite_blade_loss_factor
                * (Ua_external + ua_target)
                + 1e-30
            )
            ut_target = blade_element_dQ_dr / torque_momentum_denominator
            ut_target = np.minimum(
                np.maximum(
                    ut_target,
                    -self.maximum_tangential_induction_fraction * Ut_magnitude,
                ),
                self.maximum_tangential_induction_fraction * Ut_magnitude,
            )

            return {
                "ua": ua,
                "ut": ut,
                "ua_target": ua_target,
                "ut_target": ut_target,
                "Wa": Wa,
                "Wt": Wt,
                "W": W,
                "phi": phi,
                "alpha": alpha,
                "Re": Re,
                "mach": mach,
                "CL": CL,
                "CD": CD,
                "CM": CM,
                "analysis_confidence": analysis_confidence,
                "CL_neuralfoil": CL_neuralfoil,
                "CD_neuralfoil": CD_neuralfoil,
                "CM_neuralfoil": CM_neuralfoil,
                "CL_post_stall": CL_post_stall,
                "CD_post_stall": CD_post_stall,
                "post_stall_blend_fraction": post_stall_blend_fraction,
                "tip_loss_factor": tip_loss_factor,
                "root_loss_factor": root_loss_factor,
                "hub_loss_factor": root_loss_factor,
                "finite_blade_loss_factor": finite_blade_loss_factor,
                "blade_element_dT_dr": blade_element_dT_dr,
                "blade_element_dQ_dr": blade_element_dQ_dr,
                "momentum_dT_dr": momentum_dT_dr,
                "momentum_dQ_dr": momentum_dQ_dr,
                "thrust_residual_per_radius": momentum_dT_dr
                - blade_element_dT_dr,
                "torque_residual_per_radius": momentum_dQ_dr
                - blade_element_dQ_dr,
            }

        for _ in range(self.newton_iterations):
            evaluated = evaluate(
                axial_induced_velocity,
                tangential_induced_velocity,
            )
            relaxation = self.relaxation
            axial_induced_velocity = (
                (1 - relaxation) * evaluated["ua"]
                + relaxation * evaluated["ua_target"]
            )
            tangential_induced_velocity = (
                (1 - relaxation) * evaluated["ut"]
                + relaxation * evaluated["ut_target"]
            )

        evaluated = evaluate(
            axial_induced_velocity,
            tangential_induced_velocity,
        )

        dT = evaluated["blade_element_dT_dr"] * dr
        dQ = evaluated["blade_element_dQ_dr"] * dr
        momentum_dT = evaluated["momentum_dT_dr"] * dr
        momentum_dQ = evaluated["momentum_dQ_dr"] * dr

        thrust = np.sum(dT)
        torque = np.sum(dQ)
        power = omega * torque

        n = omega / (2 * np.pi)
        D = propeller.diameter
        advance_ratio = V / (n * D)
        C_T = thrust / (rho * n**2 * D**4)
        C_Q = torque / (rho * n**2 * D**5)
        C_P = power / (rho * n**3 * D**5)
        efficiency = np.where(power > 0, V * thrust / power, 0.0)

        thrust_scale = np.maximum(np.abs(evaluated["blade_element_dT_dr"]), 1.0)
        torque_scale = np.maximum(np.abs(evaluated["blade_element_dQ_dr"]), 0.05)
        residual = np.maximum(
            np.abs(evaluated["thrust_residual_per_radius"]) / thrust_scale,
            np.abs(evaluated["torque_residual_per_radius"]) / torque_scale,
        )
        max_abs_residual = np.max(residual)

        output = {
            "thrust": thrust,
            "torque": torque,
            "power": power,
            "efficiency": efficiency,
            "eta": efficiency,
            "advance_ratio": advance_ratio,
            "J": advance_ratio,
            "C_T": C_T,
            "C_Q": C_Q,
            "C_P": C_P,
            "Ct": C_T,
            "Cq": C_Q,
            "Cp": C_P,
            "omega": omega,
            "rpm": self.rpm,
            "r": r,
            "r_over_R": r_over_R,
            "dr": dr,
            "chord": chord,
            "twist": twist,
            "thickness": thickness,
            "axial_induced_velocity": evaluated["ua"],
            "tangential_induced_velocity": evaluated["ut"],
            "ua": evaluated["ua"],
            "ut": evaluated["ut"],
            "Wa": evaluated["Wa"],
            "Wt": evaluated["Wt"],
            "W": evaluated["W"],
            "phi": evaluated["phi"],
            "alpha": evaluated["alpha"],
            "Re": evaluated["Re"],
            "mach": evaluated["mach"],
            "CL": evaluated["CL"],
            "CD": evaluated["CD"],
            "CM": evaluated["CM"],
            "CL_neuralfoil": evaluated["CL_neuralfoil"],
            "CD_neuralfoil": evaluated["CD_neuralfoil"],
            "CM_neuralfoil": evaluated["CM_neuralfoil"],
            "CL_post_stall": evaluated["CL_post_stall"],
            "CD_post_stall": evaluated["CD_post_stall"],
            "post_stall_blend_fraction": evaluated["post_stall_blend_fraction"],
            "tip_loss_factor": evaluated["tip_loss_factor"],
            "root_loss_factor": evaluated["root_loss_factor"],
            "hub_loss_factor": evaluated["hub_loss_factor"],
            "finite_blade_loss_factor": evaluated["finite_blade_loss_factor"],
            "dT": dT,
            "dQ": dQ,
            "momentum_dT": momentum_dT,
            "momentum_dQ": momentum_dQ,
            "thrust_per_radius": evaluated["blade_element_dT_dr"],
            "torque_per_radius": evaluated["blade_element_dQ_dr"],
            "momentum_thrust_per_radius": evaluated["momentum_dT_dr"],
            "momentum_torque_per_radius": evaluated["momentum_dQ_dr"],
            "thrust_residual_per_radius": evaluated["thrust_residual_per_radius"],
            "torque_residual_per_radius": evaluated["torque_residual_per_radius"],
            "residual": residual,
            "max_abs_residual": max_abs_residual,
            "converged": max_abs_residual <= self.residual_tolerance,
            "analysis_confidence": evaluated["analysis_confidence"],
        }

        self.output = output
        return output
