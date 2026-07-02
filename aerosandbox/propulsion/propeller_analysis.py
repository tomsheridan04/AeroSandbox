from aerosandbox.common import ExplicitAnalysis
from aerosandbox.geometry.propeller import Propeller
from aerosandbox.performance import OperatingPoint
from typing import Any, Callable, Dict, Optional
import aerosandbox.numpy as np
import warnings


class PropellerAnalysis(ExplicitAnalysis):
    """
    Differentiable, QPROP-like propeller analysis.

    This uses Drela's local blade-element/vortex closure with one scalar residual
    per radial station. The residual is solved by a fixed, unrolled Newton loop,
    so the result remains compatible with AeroSandbox/CasADi optimization.
    """

    def __init__(
        self,
        propeller: Propeller,
        op_point: OperatingPoint = None,
        rpm: Optional[float] = None,
        omega: Optional[float] = None,
        pitch_offset: float = 0.0,
        radial_resolution: int = 24,
        radial_spacing_function: Callable = np.cosspace,
        model_size: str = "large",
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        include_360_deg_effects: bool = True,
        newton_iterations: int = 8,
        newton_step_limit: float = 0.35,
        finite_difference_step: float = 1e-5,
        induced_velocity_initial_guess_fraction: float = 0.15,
        bracketing_iterations: int = 24,
        bracket_upper_psi: float = 1.5,
        minimum_psi: float = 1e-4,
        residual_tolerance: float = 1e-4,
        include_tip_loss: bool = True,
        include_hub_loss: bool = True,
        include_root_loss: Optional[bool] = None,
        external_axial_induced_velocity: float = 0.0,
        external_tangential_induced_velocity: float = 0.0,
        section_aerodynamics: Optional[Callable[..., Dict[str, Any]]] = None,
        include_post_stall_confidence_blending: bool = False,
        post_stall_confidence_threshold: float = 0.75,
        post_stall_confidence_width: float = 0.22,
        post_stall_alpha_start: float = 12.0,
        post_stall_alpha_width: float = 4.0,
        post_stall_alpha_stall: float = 15.0,
        post_stall_CL_stall: Optional[float] = None,
        post_stall_CD_stall: Optional[float] = None,
        post_stall_CD90: float = 1.98,
        verbose: bool = False,
    ):
        """
        Args:
            propeller: Propeller geometry.
            op_point: Freestream operating point.
            rpm: Rotational speed [rev/min]. Specify either ``rpm`` or ``omega``.
            omega: Rotational speed [rad/sec]. Specify either ``rpm`` or ``omega``.
            pitch_offset: Radially uniform blade pitch offset [deg].
            radial_resolution: Number of radial blade elements.
            model_size: NeuralFoil model size passed to ``get_aero_from_neuralfoil``.
            newton_iterations: Fixed number of unrolled Newton iterations.
            induced_velocity_initial_guess_fraction: Initial axial induced velocity
                estimate, nondimensionalized by ``abs(omega) * radius``.
                Larger values help avoid the nonphysical zero-inflow branch in
                static and stalled cases.
            bracketing_iterations: Fixed number of bisection fallback iterations.
                This catches the physical sign-changing residual root if Newton
                lands on a rough stalled branch.
            bracket_upper_psi: Upper inflow angle [rad] used for the bracketing
                fallback.
            minimum_psi: Lower inflow angle [rad].
            residual_tolerance: Residual threshold used only for diagnostics.
            include_tip_loss: Use Drela's modified Prandtl tip factor.
            include_hub_loss: Apply an analogous smooth root/hub factor at the
                inner lifting radius.
            include_root_loss: Alias for ``include_hub_loss``. If supplied, this
                overrides ``include_hub_loss``.
            external_axial_induced_velocity: QPROP ``u_a`` term [m/s].
            external_tangential_induced_velocity: QPROP ``u_t`` term [m/s].
            section_aerodynamics: Optional override callable for sectional aero.
                It should accept keyword arguments ``airfoil``, ``alpha``, ``Re``,
                ``mach``, and ``r_over_R`` and return at least ``CL`` and ``CD``.
                If omitted, NeuralFoil is used through the section airfoil object.
            include_post_stall_confidence_blending: If True, smoothly blends
                low-confidence, high-angle NeuralFoil outputs into a Viterna-style
                empirical post-stall model. This avoids rough low-confidence
                lift/drag branches while leaving high-confidence NeuralFoil data
                essentially unchanged. Disabled by default; the default section
                model uses NeuralFoil directly, including its post-stall model.
            post_stall_confidence_threshold: NeuralFoil confidence where the
                empirical post-stall fallback reaches 50% weight, before the
                high-angle gate is applied.
            post_stall_confidence_width: Width of the confidence blending
                transition.
            post_stall_alpha_start: Angle of attack [deg] where the high-angle
                gate reaches 50% weight.
            post_stall_alpha_width: Width [deg] of the high-angle gate.
            post_stall_alpha_stall: Stall anchor angle [deg] for the Viterna-style
                empirical model. By default, the model samples NeuralFoil at
                this angle and anchors the empirical branch to that local polar.
            post_stall_CL_stall: Optional lift coefficient at the stall anchor.
                If omitted, NeuralFoil is used as the anchor.
            post_stall_CD_stall: Optional drag coefficient at the stall anchor.
                If omitted, NeuralFoil is used as the anchor.
            post_stall_CD90: Drag coefficient at 90 degrees angle of attack.
        """
        super().__init__()

        if op_point is None:
            op_point = OperatingPoint()

        if (rpm is None) == (omega is None):
            raise ValueError("Specify exactly one of `rpm` or `omega`.")

        if omega is None:
            omega = rpm * 2 * np.pi / 60
        if rpm is None:
            rpm = omega * 60 / (2 * np.pi)

        self.propeller = propeller
        self.op_point = op_point
        self.rpm = rpm
        self.omega = omega
        self.pitch_offset = pitch_offset
        self.radial_resolution = radial_resolution
        self.radial_spacing_function = radial_spacing_function
        self.model_size = model_size
        self.n_crit = n_crit
        self.xtr_upper = xtr_upper
        self.xtr_lower = xtr_lower
        self.include_360_deg_effects = include_360_deg_effects
        self.newton_iterations = newton_iterations
        self.newton_step_limit = newton_step_limit
        self.finite_difference_step = finite_difference_step
        self.induced_velocity_initial_guess_fraction = (
            induced_velocity_initial_guess_fraction
        )
        self.bracketing_iterations = bracketing_iterations
        self.bracket_upper_psi = bracket_upper_psi
        self.minimum_psi = minimum_psi
        self.residual_tolerance = residual_tolerance
        if include_root_loss is not None:
            include_hub_loss = include_root_loss
        self.include_tip_loss = include_tip_loss
        self.include_hub_loss = include_hub_loss
        self.include_root_loss = include_hub_loss
        self.external_axial_induced_velocity = external_axial_induced_velocity
        self.external_tangential_induced_velocity = external_tangential_induced_velocity
        self.section_aerodynamics = section_aerodynamics
        self.include_post_stall_confidence_blending = (
            include_post_stall_confidence_blending
        )
        self.post_stall_confidence_threshold = post_stall_confidence_threshold
        self.post_stall_confidence_width = post_stall_confidence_width
        self.post_stall_alpha_start = post_stall_alpha_start
        self.post_stall_alpha_width = post_stall_alpha_width
        self.post_stall_alpha_stall = post_stall_alpha_stall
        self.post_stall_CL_stall = post_stall_CL_stall
        self.post_stall_CD_stall = post_stall_CD_stall
        self.post_stall_CD90 = post_stall_CD90
        self.verbose = verbose

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(\n\t"
            + "\n\t".join(
                [
                    f"propeller={self.propeller}",
                    f"op_point={self.op_point}",
                    f"rpm={self.rpm}",
                ]
            )
            + "\n)"
        )

    def __getitem__(self, item):
        try:
            return self.output[item]
        except AttributeError:
            raise AttributeError(
                "This PropellerAnalysis object has no saved output yet. "
                "Call `.run()` before using dictionary-style indexing."
            )

    def _get_neuralfoil_aero(
        self,
        airfoil,
        alpha,
        Re,
        mach,
    ) -> Dict[str, Any]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in exp",
                category=RuntimeWarning,
                module=r"neuralfoil\..*",
            )
            return airfoil.get_aero_from_neuralfoil(
                alpha=alpha,
                Re=Re,
                mach=mach,
                n_crit=self.n_crit,
                xtr_upper=self.xtr_upper,
                xtr_lower=self.xtr_lower,
                model_size=self.model_size,
                include_360_deg_effects=self.include_360_deg_effects,
            )

    def _section_aero(
        self,
        airfoil,
        alpha,
        Re,
        mach,
        r_over_R,
    ) -> Dict[str, Any]:
        if self.section_aerodynamics is not None:
            return self.section_aerodynamics(
                airfoil=airfoil,
                alpha=alpha,
                Re=Re,
                mach=mach,
                r_over_R=r_over_R,
            )

        aero = self._get_neuralfoil_aero(
            airfoil=airfoil,
            alpha=alpha,
            Re=Re,
            mach=mach,
        )

        CL_neuralfoil = np.reshape(np.array(aero["CL"]), -1)[0]
        CD_neuralfoil = np.reshape(np.array(aero["CD"]), -1)[0]
        CM_neuralfoil = np.reshape(np.array(aero.get("CM", 0.0)), -1)[0]
        analysis_confidence = np.reshape(
            np.array(aero.get("analysis_confidence", 1.0)), -1
        )[0]

        if self.include_post_stall_confidence_blending:
            alpha_abs_deg = (alpha**2 + 0.25**2) ** 0.5
            smooth_sign = alpha / alpha_abs_deg
            alpha_stall_anchor = smooth_sign * self.post_stall_alpha_stall

            anchor_aero = self._get_neuralfoil_aero(
                airfoil=airfoil,
                alpha=alpha_stall_anchor,
                Re=Re,
                mach=mach,
            )
            CL_stall = (
                np.reshape(np.array(anchor_aero["CL"]), -1)[0]
                if self.post_stall_CL_stall is None
                else smooth_sign * self.post_stall_CL_stall
            )
            CD_stall = (
                np.reshape(np.array(anchor_aero["CD"]), -1)[0]
                if self.post_stall_CD_stall is None
                else self.post_stall_CD_stall
            )

            post_stall = self._viterna_post_stall_aero(
                alpha=alpha,
                CL_stall=CL_stall,
                CD_stall=CD_stall,
            )
            post_stall_blend_fraction = self._post_stall_blend_fraction(
                alpha=alpha,
                analysis_confidence=analysis_confidence,
            )
        else:
            CL_stall = CL_neuralfoil
            CD_stall = CD_neuralfoil
            post_stall = {
                "CL": CL_neuralfoil,
                "CD": CD_neuralfoil,
            }
            post_stall_blend_fraction = 0.0

        CL = (
            (1 - post_stall_blend_fraction) * CL_neuralfoil
            + post_stall_blend_fraction * post_stall["CL"]
        )
        CD = (
            (1 - post_stall_blend_fraction) * CD_neuralfoil
            + post_stall_blend_fraction * post_stall["CD"]
        )
        CM = (1 - post_stall_blend_fraction) * CM_neuralfoil

        return {
            "CL": CL,
            "CD": CD,
            "CM": CM,
            "analysis_confidence": analysis_confidence,
            "CL_neuralfoil": CL_neuralfoil,
            "CD_neuralfoil": CD_neuralfoil,
            "CM_neuralfoil": CM_neuralfoil,
            "CL_post_stall": post_stall["CL"],
            "CD_post_stall": post_stall["CD"],
            "CL_post_stall_anchor": CL_stall,
            "CD_post_stall_anchor": CD_stall,
            "post_stall_blend_fraction": post_stall_blend_fraction,
        }

    def _viterna_post_stall_aero(
        self,
        alpha,
        CL_stall,
        CD_stall,
    ) -> Dict[str, Any]:
        """
        NeuralFoil-anchored Viterna-style post-stall lift/drag curve.

        The model matches the supplied local lift and drag at the stall anchor,
        then approaches zero lift and a flat-plate drag coefficient at 90 degrees.
        It is intended as a low-confidence, high-angle fallback for NeuralFoil
        rather than a replacement for attached or mildly separated NeuralFoil
        predictions.
        """
        alpha_abs_deg = (alpha**2 + 0.25**2) ** 0.5
        smooth_sign = alpha / alpha_abs_deg
        alpha_viterna_abs_deg = np.minimum(
            np.softmax(
                alpha_abs_deg,
                self.post_stall_alpha_stall,
                softness=0.5,
            ),
            90.0,
        )
        alpha_viterna = np.radians(smooth_sign * alpha_viterna_abs_deg)
        alpha_stall = np.radians(smooth_sign * self.post_stall_alpha_stall)

        sin_alpha = np.sin(alpha_viterna)
        cos_alpha = np.cos(alpha_viterna)
        sin_stall = np.sin(alpha_stall)
        cos_stall = np.cos(alpha_stall)

        CD90 = self.post_stall_CD90
        CD_stall = np.softmax(CD_stall, 1e-4, softness=1e-4)

        A = (
            (CL_stall - CD90 * sin_stall * cos_stall)
            * sin_stall
            / (cos_stall**2 + 1e-12)
        )
        B = (
            (CD_stall - CD90 * sin_stall**2)
            / (cos_stall + 1e-12)
        )

        CL = CD90 * sin_alpha * cos_alpha + A * cos_alpha**2 / (
            sin_alpha + 1e-12
        )
        CD = CD90 * sin_alpha**2 + B * cos_alpha

        return {
            "CL": CL,
            "CD": CD,
        }

    def _post_stall_blend_fraction(self, alpha, analysis_confidence):
        if not self.include_post_stall_confidence_blending:
            return 0.0

        low_confidence_weight = np.sigmoid(
            (self.post_stall_confidence_threshold - analysis_confidence)
            / self.post_stall_confidence_width
        )
        alpha_abs_deg = (alpha**2 + 0.25**2) ** 0.5
        high_alpha_weight = np.sigmoid(
            (alpha_abs_deg - self.post_stall_alpha_start)
            / self.post_stall_alpha_width
        )

        return low_confidence_weight * high_alpha_weight

    @staticmethod
    def _prandtl_loss_factor(f):
        """
        Smooth Prandtl finite-blade loss factor from a nondimensional wake-gap
        parameter ``f``.
        """
        f = np.softmax(f, 1e-8, softness=1e-8)
        return 2 / np.pi * np.arccos(np.exp(-f))

    def run(self) -> Dict[str, Any]:
        """
        Runs the propeller analysis.

        Returns:
            A dictionary containing integrated loads and coefficients, plus
            spanwise distributions. Key integrated outputs include ``thrust``,
            ``torque``, ``power``, ``efficiency``, ``advance_ratio``, ``C_T``,
            ``C_Q``, and ``C_P``.
        """
        propeller = self.propeller
        op_point = self.op_point

        rho = op_point.atmosphere.density()
        mu = op_point.atmosphere.dynamic_viscosity()
        speed_of_sound = op_point.atmosphere.speed_of_sound()
        V = op_point.velocity
        omega = self.omega

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

        Ua = V + self.external_axial_induced_velocity
        Ut = omega * r - self.external_tangential_induced_velocity
        U = (Ua**2 + Ut**2) ** 0.5

        induced_velocity_guess = (
            self.induced_velocity_initial_guess_fraction
            * np.abs(omega)
            * propeller.radius
        )
        psi = np.arctan2(Ua + induced_velocity_guess, Ut)
        psi = np.maximum(psi, self.minimum_psi)

        def evaluate_at_psi(psi_eval):
            Wa = 0.5 * Ua + 0.5 * U * np.sin(psi_eval)
            Wt = 0.5 * Ut + 0.5 * U * np.cos(psi_eval)
            vt = Ut - Wt

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
                    np.reshape(np.array(aero.get("CM_neuralfoil", aero.get("CM", 0.0))), -1)[0]
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

            Gamma_blade_element = 0.5 * W * chord * CL

            lambda_w = r_over_R * Wa / Wt
            lambda_w_magnitude = (lambda_w**2 + 1e-16) ** 0.5

            if self.include_tip_loss:
                f_tip = (
                    propeller.blade_count
                    / 2
                    * (1 - r_over_R)
                    / lambda_w_magnitude
                )
                tip_loss_factor = self._prandtl_loss_factor(f_tip)
            else:
                tip_loss_factor = np.ones_like(r)

            if self.include_hub_loss:
                hub_over_R = propeller.hub_radius / propeller.radius
                f_hub = (
                    propeller.blade_count
                    / 2
                    * (r_over_R - hub_over_R)
                    / lambda_w_magnitude
                )
                root_loss_factor = self._prandtl_loss_factor(f_hub)
            else:
                root_loss_factor = np.ones_like(r)

            finite_blade_loss_factor = tip_loss_factor * root_loss_factor

            Gamma_vortex = (
                vt
                * 4
                * np.pi
                * r
                / propeller.blade_count
                * finite_blade_loss_factor
                * (
                    1
                    + (
                        (4 * lambda_w_magnitude * propeller.radius)
                        / (np.pi * propeller.blade_count * r)
                    )
                    ** 2
                )
                ** 0.5
            )

            residual = Gamma_vortex - Gamma_blade_element

            return {
                "residual": residual,
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
                "Gamma": Gamma_vortex,
                "tip_loss_factor": tip_loss_factor,
                "root_loss_factor": root_loss_factor,
                "hub_loss_factor": root_loss_factor,
                "finite_blade_loss_factor": finite_blade_loss_factor,
            }

        for _ in range(self.newton_iterations):
            evaluated = evaluate_at_psi(psi)
            residual = evaluated["residual"]
            residual_plus = evaluate_at_psi(psi + self.finite_difference_step)[
                "residual"
            ]
            residual_minus = evaluate_at_psi(psi - self.finite_difference_step)[
                "residual"
            ]
            d_residual_d_psi = (
                residual_plus - residual_minus
            ) / (2 * self.finite_difference_step)

            step = -residual / (d_residual_d_psi + 1e-30)
            step = self.newton_step_limit * np.tanh(step / self.newton_step_limit)
            psi = psi + step
            psi = np.minimum(np.maximum(psi, self.minimum_psi), np.pi - self.minimum_psi)

        if self.bracketing_iterations > 0:
            bracket_low = self.minimum_psi * np.ones_like(psi)
            bracket_high = np.minimum(
                self.bracket_upper_psi,
                np.pi - self.minimum_psi,
            ) * np.ones_like(psi)
            residual_low = evaluate_at_psi(bracket_low)["residual"]
            residual_high = evaluate_at_psi(bracket_high)["residual"]
            has_bracket = residual_low * residual_high <= 0

            for _ in range(self.bracketing_iterations):
                bracket_mid = 0.5 * (bracket_low + bracket_high)
                residual_mid = evaluate_at_psi(bracket_mid)["residual"]
                low_to_mid_has_same_sign = residual_low * residual_mid > 0
                bracket_low = np.where(
                    low_to_mid_has_same_sign,
                    bracket_mid,
                    bracket_low,
                )
                residual_low = np.where(
                    low_to_mid_has_same_sign,
                    residual_mid,
                    residual_low,
                )
                bracket_high = np.where(
                    low_to_mid_has_same_sign,
                    bracket_high,
                    bracket_mid,
                )

            bracketed_psi = 0.5 * (bracket_low + bracket_high)
            current_residual = evaluate_at_psi(psi)["residual"]
            bracketed_residual = evaluate_at_psi(bracketed_psi)["residual"]
            bracket_is_better = np.logical_and(
                has_bracket,
                np.abs(bracketed_residual) < np.abs(current_residual),
            )
            psi = np.where(bracket_is_better, bracketed_psi, psi)

        evaluated = evaluate_at_psi(psi)
        max_abs_residual = np.max(np.abs(evaluated["residual"]))

        Wa = evaluated["Wa"]
        Wt = evaluated["Wt"]
        W = evaluated["W"]
        phi = evaluated["phi"]
        alpha = evaluated["alpha"]
        Re = evaluated["Re"]
        mach = evaluated["mach"]
        CL = evaluated["CL"]
        CD = evaluated["CD"]
        CM = evaluated["CM"]
        Gamma = evaluated["Gamma"]

        dynamic_pressure_local = 0.5 * rho * W**2
        dT = (
            propeller.blade_count
            * dynamic_pressure_local
            * (CL * np.cosd(phi) - CD * np.sind(phi))
            * chord
            * dr
        )
        dQ = (
            propeller.blade_count
            * dynamic_pressure_local
            * (CL * np.sind(phi) + CD * np.cosd(phi))
            * chord
            * r
            * dr
        )

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

        thrust_per_radius = dT / dr
        torque_per_radius = dQ / dr

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
            "psi": psi,
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
            "CL_neuralfoil": evaluated["CL_neuralfoil"],
            "CD_neuralfoil": evaluated["CD_neuralfoil"],
            "CM_neuralfoil": evaluated["CM_neuralfoil"],
            "CL_post_stall": evaluated["CL_post_stall"],
            "CD_post_stall": evaluated["CD_post_stall"],
            "post_stall_blend_fraction": evaluated["post_stall_blend_fraction"],
            "Gamma": Gamma,
            "tip_loss_factor": evaluated["tip_loss_factor"],
            "root_loss_factor": evaluated["root_loss_factor"],
            "hub_loss_factor": evaluated["hub_loss_factor"],
            "finite_blade_loss_factor": evaluated["finite_blade_loss_factor"],
            "analysis_confidence": evaluated["analysis_confidence"],
            "dT": dT,
            "dQ": dQ,
            "thrust_per_radius": thrust_per_radius,
            "torque_per_radius": torque_per_radius,
            "residual": evaluated["residual"],
            "max_abs_residual": max_abs_residual,
            "converged": max_abs_residual <= self.residual_tolerance,
        }

        self.output = output
        return output
