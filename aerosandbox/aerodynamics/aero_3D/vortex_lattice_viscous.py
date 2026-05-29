import aerosandbox.numpy as np
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
from aerosandbox.aerodynamics.aero_3D.vortex_lattice_method import (
    VortexLatticeMethod,
    tall,
    wide,
)
from typing import Dict, Any, List, Callable, Union


class VortexLatticeViscous(VortexLatticeMethod):
    """
    A Vortex Lattice Method analysis with a NeuralFoil viscous closure.

    This analysis uses the same panel geometry, wake model, circulation solution, and
    lifting forces as :class:`VortexLatticeMethod`. After the inviscid solve, it reads
    the VLM sectional ``cl`` distribution and evaluates a NeuralFoil polar for each
    strip. NeuralFoil is then used only to look up profile drag at the corresponding
    VLM ``cl``; stall is exposed as a spanwise ``clmax`` vector for use as a
    constraint.
    """

    def __init__(
        self,
        airplane: Airplane,
        op_point: OperatingPoint,
        xyz_ref: List[float] = None,
        model_size: str = "medium",
        n_crit: Union[float, np.ndarray] = 9.0,
        xtr_upper: Union[float, np.ndarray] = 1.0,
        xtr_lower: Union[float, np.ndarray] = 1.0,
        include_360_deg_effects: bool = True,
        run_symmetric_if_possible: bool = False,
        verbose: bool = False,
        spanwise_resolution: int = 10,
        spanwise_spacing_function: Callable[
            [float, float, float], np.ndarray
        ] = np.cosspace,
        chordwise_resolution: int = 10,
        chordwise_spacing_function: Callable[
            [float, float, float], np.ndarray
        ] = np.cosspace,
        vortex_core_radius: float = 1e-8,
        align_trailing_vortices_with_wind: bool = False,
    ):
        super().__init__(
            airplane=airplane,
            op_point=op_point,
            xyz_ref=xyz_ref,
            run_symmetric_if_possible=run_symmetric_if_possible,
            verbose=verbose,
            spanwise_resolution=spanwise_resolution,
            spanwise_spacing_function=spanwise_spacing_function,
            chordwise_resolution=chordwise_resolution,
            chordwise_spacing_function=chordwise_spacing_function,
            vortex_core_radius=vortex_core_radius,
            align_trailing_vortices_with_wind=align_trailing_vortices_with_wind,
        )

        self.model_size = model_size
        self.n_crit = n_crit
        self.xtr_upper = xtr_upper
        self.xtr_lower = xtr_lower
        self.include_360_deg_effects = include_360_deg_effects

    def run(self) -> Dict[str, Any]:
        """
        Computes the aerodynamic forces.

        Returns the standard AeroSandbox 3D aerodynamic output keys. Lift and induced
        effects are the unmodified VLM result; NeuralFoil contributes profile drag
        evaluated at the corresponding VLM sectional lift coefficient.
        """

        if self.verbose:
            print("Running inviscid VLM foundation...")
        inviscid_output = super().run()
        forces_inviscid_geometry = self.forces_geometry
        moments_inviscid_geometry = self.moments_geometry
        force_inviscid_geometry = self.force_geometry
        moment_inviscid_geometry = self.moment_geometry
        force_inviscid_wind = self.force_wind

        if self.verbose:
            print("Building viscous strip metadata...")
        (
            strip_indices,
            strip_airfoils,
            strip_control_surfaces,
            strip_spanwise_index,
        ) = self._strip_metadata()

        panel_velocities = self.get_velocity_at_points(self.vortex_centers)
        panel_areas = self.areas
        q = self.op_point.dynamic_pressure()
        qS = q * self.airplane.s_ref
        c_ref = self.airplane.c_ref
        b_ref = self.airplane.b_ref

        strip_centers = []
        strip_velocities = []
        strip_span_directions = []
        strip_chords = []

        for indices in strip_indices:
            weights = np.array([panel_areas[i] for i in indices])
            total_weight = np.sum(weights)

            strip_centers.append(
                np.sum(
                    np.array([self.vortex_centers[i, :] * panel_areas[i] for i in indices]),
                    axis=0,
                )
                / total_weight
            )
            strip_velocities.append(
                np.sum(
                    np.array([panel_velocities[i, :] * panel_areas[i] for i in indices]),
                    axis=0,
                )
                / total_weight
            )

            bound_leg = np.mean(
                np.array([self.vortex_bound_leg[i, :] for i in indices]), axis=0
            )
            strip_span_directions.append(bound_leg / np.linalg.norm(bound_leg))

            strip_chords.append(np.sum(np.array([self.panel_chords[i] for i in indices])))

        strip_centers = np.array(strip_centers)
        strip_velocities = np.array(strip_velocities)
        strip_span_directions = np.array(strip_span_directions)
        strip_chords = np.array(strip_chords)
        strip_velocity_magnitudes = np.linalg.norm(strip_velocities, axis=1)
        span_components = np.sum(strip_velocities * strip_span_directions, axis=1)
        strip_velocities_2D = strip_velocities - (
            tall(span_components) * strip_span_directions
        )
        strip_velocity_magnitudes_2D = np.linalg.norm(strip_velocities_2D, axis=1)
        strip_Re = (
            strip_velocity_magnitudes_2D
            * strip_chords
            / self.op_point.atmosphere.kinematic_viscosity()
        )
        strip_mach = self.op_point.mach() * (
            strip_velocity_magnitudes_2D / self.op_point.velocity
        )

        if self.verbose:
            print("Evaluating NeuralFoil profile drag at VLM strip cl...")
        strip_aeros = [
            self._section_aero_at_cl(
                airfoil=airfoil,
                cl=inviscid_output["spanwise_cl"][i],
                Re=strip_Re[i],
                mach=strip_mach[i],
                control_surfaces=strip_control_surfaces[i],
            )
            for i, airfoil in enumerate(strip_airfoils)
        ]

        strip_alphas = np.array([aero["alpha"] for aero in strip_aeros])
        strip_CDs = np.array([aero["CD"] for aero in strip_aeros])
        strip_CMs = np.array([aero["CM"] for aero in strip_aeros])
        strip_clmax = np.array([aero["clmax"] for aero in strip_aeros])
        strip_clmin = np.array([aero["clmin"] for aero in strip_aeros])
        strip_analysis_confidence = np.array(
            [aero["analysis_confidence"] for aero in strip_aeros]
        )

        strip_reference_areas = self.spanwise_chord * self.spanwise_dy
        strip_profile_drags = q * strip_reference_areas * strip_CDs

        forces_profile_wind = np.column_stack(
            (
                -strip_profile_drags,
                np.zeros_like(strip_profile_drags),
                np.zeros_like(strip_profile_drags),
            )
        )
        forces_profile_body = self.op_point.convert_axes(
            forces_profile_wind[:, 0],
            forces_profile_wind[:, 1],
            forces_profile_wind[:, 2],
            from_axes="wind",
            to_axes="body",
        )
        forces_profile_geometry = np.array(
            self.op_point.convert_axes(
                forces_profile_body[0],
                forces_profile_body[1],
                forces_profile_body[2],
                from_axes="body",
                to_axes="geometry",
            )
        ).T

        moment_arms = np.add(strip_centers, -wide(np.array(self.xyz_ref)))
        moments_profile_geometry = np.cross(moment_arms, forces_profile_geometry)

        force_profile_geometry = np.sum(forces_profile_geometry, axis=0)
        moment_profile_geometry = np.sum(moments_profile_geometry, axis=0)
        force_profile_wind = np.sum(forces_profile_wind, axis=0)

        force_geometry = force_inviscid_geometry + force_profile_geometry
        moment_geometry = moment_inviscid_geometry + moment_profile_geometry

        force_wind = force_inviscid_wind + force_profile_wind

        force_body = self.op_point.convert_axes(
            force_wind[0],
            force_wind[1],
            force_wind[2],
            from_axes="wind",
            to_axes="body",
        )
        moment_body = self.op_point.convert_axes(
            moment_geometry[0],
            moment_geometry[1],
            moment_geometry[2],
            from_axes="geometry",
            to_axes="body",
        )
        moment_wind = self.op_point.convert_axes(
            moment_body[0],
            moment_body[1],
            moment_body[2],
            from_axes="body",
            to_axes="wind",
        )

        L = -force_wind[2]
        D = -force_wind[0]
        Y = force_wind[1]
        l_b = moment_body[0]
        m_b = moment_body[1]
        n_b = moment_body[2]

        spanwise_lift = inviscid_output["spanwise_lift"]
        spanwise_lift_per_y = inviscid_output["spanwise_lift_per_y"]
        spanwise_cl = inviscid_output["spanwise_cl"]
        spanwise_clc_over_cref = inviscid_output["spanwise_clc_over_cref"]

        wing_indices = [
            self._get_ordered_wing_indices(
                wing_index=wing_index,
                spanwise_wing_index=self.spanwise_wing_index,
                spanwise_side=self.spanwise_side,
                spanwise_spanwise_index=strip_spanwise_index,
            )
            for wing_index in range(len(self.airplane.wings))
        ]
        y = [np.array([self.spanwise_y[i] for i in indices]) for indices in wing_indices]
        dy = [np.array([self.spanwise_dy[i] for i in indices]) for indices in wing_indices]
        chord = [
            np.array([self.spanwise_chord[i] for i in indices]) for indices in wing_indices
        ]
        lift = [np.array([spanwise_lift[i] for i in indices]) for indices in wing_indices]
        lift_per_y = [
            np.array([spanwise_lift_per_y[i] for i in indices])
            for indices in wing_indices
        ]
        cl = [np.array([spanwise_cl[i] for i in indices]) for indices in wing_indices]
        clc_over_cref = [
            np.array([spanwise_clc_over_cref[i] for i in indices])
            for indices in wing_indices
        ]

        self.panel_velocities = panel_velocities
        self.forces_inviscid_geometry = forces_inviscid_geometry
        self.moments_inviscid_geometry = moments_inviscid_geometry
        self.force_inviscid_geometry = force_inviscid_geometry
        self.moment_inviscid_geometry = moment_inviscid_geometry
        self.strip_centers = strip_centers
        self.strip_velocities = strip_velocities
        self.strip_velocity_magnitudes = strip_velocity_magnitudes
        self.strip_velocities_2D = strip_velocities_2D
        self.strip_velocity_magnitudes_2D = strip_velocity_magnitudes_2D
        self.strip_airfoils = strip_airfoils
        self.strip_control_surfaces = strip_control_surfaces
        self.strip_aeros = strip_aeros
        self.strip_profile_drags = strip_profile_drags
        self.forces_lift_geometry = forces_inviscid_geometry
        self.moments_lift_geometry = moments_inviscid_geometry
        self.forces_profile_geometry = forces_profile_geometry
        self.moments_profile_geometry = moments_profile_geometry
        self.force_profile_geometry = force_profile_geometry
        self.moment_profile_geometry = moment_profile_geometry
        self.forces_geometry = forces_inviscid_geometry
        self.moments_geometry = moments_inviscid_geometry
        self.force_geometry = force_geometry
        self.force_body = force_body
        self.force_wind = force_wind
        self.moment_geometry = moment_geometry
        self.moment_body = moment_body
        self.moment_wind = moment_wind
        self.spanwise_lift = spanwise_lift
        self.spanwise_lift_per_y = spanwise_lift_per_y
        self.spanwise_cl = spanwise_cl
        self.spanwise_clc_over_cref = spanwise_clc_over_cref
        self.y = y
        self.dy = dy
        self.chord = chord
        self.lift = lift
        self.lift_per_y = lift_per_y
        self.cl = cl
        self.clc_over_cref = clc_over_cref

        output = dict(inviscid_output)
        output.update(
            {
                "F_g": force_geometry,
                "F_b": force_body,
                "F_w": force_wind,
                "M_g": moment_geometry,
                "M_b": moment_body,
                "M_w": moment_wind,
                "L": L,
                "D": D,
                "Y": Y,
                "l_b": l_b,
                "m_b": m_b,
                "n_b": n_b,
                "CL": L / qS,
                "CD": D / qS,
                "CY": Y / qS,
                "Cl": l_b / qS / b_ref,
                "Cm": m_b / qS / c_ref,
                "Cn": n_b / qS / b_ref,
                "spanwise_lift": spanwise_lift,
                "spanwise_lift_per_y": spanwise_lift_per_y,
                "spanwise_cl": spanwise_cl,
                "spanwise_clc_over_cref": spanwise_clc_over_cref,
                "y": y,
                "dy": dy,
                "chord": chord,
                "lift": lift,
                "lift_per_y": lift_per_y,
                "cl": cl,
                "clc_over_cref": clc_over_cref,
                "spanwise_alpha": strip_alphas,
                "spanwise_Re": strip_Re,
                "spanwise_mach": strip_mach,
                "spanwise_cl_inviscid": inviscid_output["spanwise_cl"],
                "spanwise_cl_viscous": inviscid_output["spanwise_cl"],
                "spanwise_cd_profile": strip_CDs,
                "spanwise_cm": strip_CMs,
                "spanwise_clmax": strip_clmax,
                "spanwise_clmin": strip_clmin,
                "spanwise_analysis_confidence": strip_analysis_confidence,
                "CL_inviscid": inviscid_output["CL"],
                "CD_inviscid": inviscid_output["CD"],
                "Cm_inviscid": inviscid_output["Cm"],
                "CD_profile": np.sum(strip_profile_drags) / qS,
            }
        )

        self.output = output
        return output

    def _section_aero_at_cl(
        self,
        airfoil: Airfoil,
        cl: float,
        Re: float,
        mach: float,
        control_surfaces: List[ControlSurface],
    ) -> Dict[str, Any]:
        alpha_grid = np.linspace(-20, 20, 161)
        polar = airfoil.get_aero_from_neuralfoil(
            alpha=alpha_grid,
            Re=Re,
            mach=mach,
            n_crit=self.n_crit,
            xtr_upper=self.xtr_upper,
            xtr_lower=self.xtr_lower,
            model_size=self.model_size,
            control_surfaces=control_surfaces,
            include_360_deg_effects=self.include_360_deg_effects,
        )

        cl_grid = np.array(polar["CL"])
        cd_grid = np.array(polar["CD"])
        cm_grid = np.array(polar["CM"])
        confidence_grid = np.array(polar["analysis_confidence"])

        i_clmax = int(np.argmax(cl_grid))
        attached_indices = np.arange(i_clmax + 1)
        cl_attached = cl_grid[attached_indices]

        order = np.argsort(cl_attached)
        cl_sorted = cl_attached[order]
        alpha_sorted = alpha_grid[attached_indices][order]
        cd_sorted = cd_grid[attached_indices][order]
        cm_sorted = cm_grid[attached_indices][order]
        confidence_sorted = confidence_grid[attached_indices][order]

        cl_unique, unique_indices = np.unique(cl_sorted, return_index=True)
        alpha_unique = alpha_sorted[unique_indices]
        cd_unique = cd_sorted[unique_indices]
        cm_unique = cm_sorted[unique_indices]
        confidence_unique = confidence_sorted[unique_indices]

        return {
            "alpha": np.interp(cl, cl_unique, alpha_unique),
            "CL": cl,
            "CD": np.interp(cl, cl_unique, cd_unique),
            "CM": np.interp(cl, cl_unique, cm_unique),
            "clmax": np.max(cl_grid),
            "clmin": np.min(cl_grid),
            "analysis_confidence": np.interp(
                cl,
                cl_unique,
                confidence_unique,
            ),
        }

    def _strip_metadata(self):
        strip_indices = []
        strip_airfoils = []
        strip_control_surfaces = []
        strip_spanwise_index = []
        seen_spanwise_keys = set()

        airfoil_by_key = {}
        control_surface_by_key = {}

        for wing_index, wing in enumerate(self.airplane.wings):
            if self.spanwise_resolution > 1:
                wing = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function,
                )

            wing_airfoils = []
            wing_control_surfaces = []
            for xsec_a, xsec_b in zip(wing.xsecs[:-1], wing.xsecs[1:]):
                wing_airfoils.append(
                    xsec_a.airfoil.blend_with_another_airfoil(
                        airfoil=xsec_b.airfoil,
                        blend_fraction=0.5,
                    )
                )
                wing_control_surfaces.append(xsec_a.control_surfaces)

            for spanwise_index, airfoil in enumerate(wing_airfoils):
                key = (wing_index, 1, spanwise_index)
                airfoil_by_key[key] = airfoil
                control_surface_by_key[key] = wing_control_surfaces[spanwise_index]

            if wing.symmetric:

                def mirror_control_surface(surf: ControlSurface) -> ControlSurface:
                    if surf.symmetric:
                        return surf
                    else:
                        mirrored_surf = surf.copy()
                        mirrored_surf.deflection *= -1
                        return mirrored_surf

                for spanwise_index, airfoil in enumerate(wing_airfoils):
                    key = (wing_index, -1, spanwise_index)
                    airfoil_by_key[key] = airfoil
                    control_surface_by_key[key] = [
                        mirror_control_surface(surf)
                        for surf in wing_control_surfaces[spanwise_index]
                    ]

        panel_metadata = list(
            zip(
                self.panel_wing_indices,
                self.panel_side_indices,
                self.panel_spanwise_indices,
            )
        )

        for wing_index, side, spanwise_index in panel_metadata:
            key = (int(wing_index), int(side), int(spanwise_index))
            if key in seen_spanwise_keys:
                continue
            seen_spanwise_keys.add(key)

            indices = [
                i
                for i, metadata in enumerate(panel_metadata)
                if (int(metadata[0]), int(metadata[1]), int(metadata[2])) == key
            ]
            strip_indices.append(indices)
            strip_airfoils.append(airfoil_by_key[key])
            strip_control_surfaces.append(control_surface_by_key[key])
            strip_spanwise_index.append(key[2])

        return (
            strip_indices,
            strip_airfoils,
            strip_control_surfaces,
            np.array(strip_spanwise_index),
        )

    @staticmethod
    def _get_ordered_wing_indices(
        wing_index,
        spanwise_wing_index,
        spanwise_side,
        spanwise_spanwise_index,
    ):
        indices = []
        for side in [-1, 1]:
            side_indices = [
                i
                for i, (this_wing_index, this_side) in enumerate(
                    zip(spanwise_wing_index, spanwise_side)
                )
                if this_wing_index == wing_index and this_side == side
            ]
            side_indices = sorted(
                side_indices,
                key=lambda i: spanwise_spanwise_index[i],
                reverse=side == -1,
            )
            indices.extend(side_indices)
        return indices
