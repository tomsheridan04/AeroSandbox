from typing import Any, Callable, Dict, Optional

import numpy as _onp


def _cumulative_trapezoid_from_root(y: _onp.ndarray, x: _onp.ndarray) -> _onp.ndarray:
    result = _onp.zeros_like(y, dtype=float)
    if len(y) <= 1:
        return result
    segments = 0.5 * (y[:-1] + y[1:]) * _onp.diff(x)
    result[1:] = _onp.cumsum(segments)
    return result


def _cumulative_trapezoid_from_tip(y: _onp.ndarray, x: _onp.ndarray) -> _onp.ndarray:
    result = _onp.zeros_like(y, dtype=float)
    if len(y) <= 1:
        return result
    segments = 0.5 * (y[:-1] + y[1:]) * _onp.diff(x)
    result[:-1] = _onp.cumsum(segments[::-1])[::-1]
    return result


def _tip_integral_of_load_times_arm(
    load_per_length: _onp.ndarray,
    x: _onp.ndarray,
) -> _onp.ndarray:
    return _cumulative_trapezoid_from_tip(
        load_per_length * x,
        x,
    ) - x * _cumulative_trapezoid_from_tip(load_per_length, x)


def polygon_section_properties(coordinates: _onp.ndarray) -> Dict[str, float]:
    """
    Computes area properties of a closed polygon.

    Coordinates are interpreted as ``[x, y]`` in section axes, where ``x`` is
    chordwise and ``y`` is thickness-normal. Returns centroidal second moments
    about the chordwise x-axis and thickness y-axis.
    """
    coordinates = _onp.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("`coordinates` must be an Nx2 array.")
    if len(coordinates) < 3:
        raise ValueError("At least three polygon points are required.")
    if _onp.linalg.norm(coordinates[0] - coordinates[-1]) > 1e-12:
        coordinates = _onp.vstack([coordinates, coordinates[0]])

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    signed_area = 0.5 * _onp.sum(cross)
    if abs(signed_area) < 1e-18:
        raise ValueError("Polygon has near-zero area.")
    if signed_area < 0:
        return polygon_section_properties(coordinates[::-1])

    area = signed_area
    centroid_x = _onp.sum((x[:-1] + x[1:]) * cross) / (6 * area)
    centroid_y = _onp.sum((y[:-1] + y[1:]) * cross) / (6 * area)

    I_x_origin = _onp.sum(
        (y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2) * cross
    ) / 12
    I_y_origin = _onp.sum(
        (x[:-1] ** 2 + x[:-1] * x[1:] + x[1:] ** 2) * cross
    ) / 12
    I_xy_origin = _onp.sum(
        (
            2 * x[:-1] * y[:-1]
            + x[:-1] * y[1:]
            + x[1:] * y[:-1]
            + 2 * x[1:] * y[1:]
        )
        * cross
    ) / 24

    x_centroidal = x[:-1] - centroid_x
    y_centroidal = y[:-1] - centroid_y
    I_x = I_x_origin - area * centroid_y**2
    I_y = I_y_origin - area * centroid_x**2
    I_xy = I_xy_origin - area * centroid_x * centroid_y

    return {
        "area": area,
        "centroid_chordwise": centroid_x,
        "centroid_thickness": centroid_y,
        "I_chord_axis": I_x,
        "I_thickness_axis": I_y,
        "I_chord_thickness_product": I_xy,
        "max_chordwise_distance": float(_onp.max(_onp.abs(x_centroidal))),
        "max_thickness_distance": float(_onp.max(_onp.abs(y_centroidal))),
        "max_width": float(_onp.max(x[:-1]) - _onp.min(x[:-1])),
        "max_height": float(_onp.max(y[:-1]) - _onp.min(y[:-1])),
    }


def _equivalent_ellipse_torsion_constant(
    area: _onp.ndarray,
    I_chord_axis: _onp.ndarray,
    I_thickness_axis: _onp.ndarray,
) -> _onp.ndarray:
    semi_chord = _onp.sqrt(
        _onp.maximum(4 * I_thickness_axis / _onp.maximum(area, 1e-18), 1e-24)
    )
    semi_thickness = _onp.sqrt(
        _onp.maximum(4 * I_chord_axis / _onp.maximum(area, 1e-18), 1e-24)
    )
    return (
        _onp.pi
        * semi_chord**3
        * semi_thickness**3
        / _onp.maximum(semi_chord**2 + semi_thickness**2, 1e-24)
    )


def airfoil_section_properties(
    airfoil,
    chord: float,
    target_area: Optional[float] = None,
    n_coordinates_per_side: int = 160,
) -> Dict[str, float]:
    """
    Computes solid-section properties from the actual closed airfoil polygon.

    If ``target_area`` is supplied, the airfoil polygon inertias and extreme
    distances are uniformly area-normalized about their centroid. This preserves
    the airfoil shape while matching an externally supplied material area, such
    as APC's ``CROSS-SECTION`` field.
    """
    coordinates = _onp.asarray(
        airfoil.to_airfoil(
            n_coordinates_per_side=n_coordinates_per_side
        ).coordinates,
        dtype=float,
    ).copy()
    coordinates *= chord
    properties = polygon_section_properties(coordinates)
    raw_area = properties["area"]

    if target_area is None:
        area_scale = 1.0
        target_area = raw_area
    else:
        area_scale = float((target_area / raw_area) ** 0.5)

    for key in [
        "I_chord_axis",
        "I_thickness_axis",
        "I_chord_thickness_product",
    ]:
        properties[key] *= area_scale**4
    for key in [
        "max_chordwise_distance",
        "max_thickness_distance",
        "max_width",
        "max_height",
    ]:
        properties[key] *= area_scale

    properties["raw_airfoil_area"] = raw_area
    properties["area"] = target_area
    properties["section_area"] = target_area
    properties["section_centroid_chordwise"] = properties["centroid_chordwise"]
    properties["section_centroid_thickness"] = properties["centroid_thickness"]
    properties["equivalent_width"] = properties["max_width"]
    properties["equivalent_height"] = properties["max_height"]
    properties["area_scale_factor"] = area_scale
    properties["torsion_constant"] = float(
        _equivalent_ellipse_torsion_constant(
            area=_onp.asarray(target_area),
            I_chord_axis=_onp.asarray(properties["I_chord_axis"]),
            I_thickness_axis=_onp.asarray(properties["I_thickness_axis"]),
        )
    )
    return properties


def airfoil_section_properties_over_span(
    r_over_R: _onp.ndarray,
    chord: _onp.ndarray,
    area: _onp.ndarray,
    airfoil_function: Callable[[float], Any],
    n_coordinates_per_side: int = 160,
) -> Dict[str, _onp.ndarray]:
    rows = [
        airfoil_section_properties(
            airfoil=airfoil_function(float(station)),
            chord=float(chord_i),
            target_area=float(area_i),
            n_coordinates_per_side=n_coordinates_per_side,
        )
        for station, chord_i, area_i in zip(r_over_R, chord, area)
    ]
    return {
        key: _onp.asarray([row[key] for row in rows], dtype=float)
        for key in rows[0].keys()
    }


def _equivalent_rectangle_properties(
    area: _onp.ndarray,
    chord: _onp.ndarray,
    max_thickness: _onp.ndarray,
) -> Dict[str, _onp.ndarray]:
    aspect_ratio = _onp.maximum(chord / _onp.maximum(max_thickness, 1e-12), 1.0)
    height = (area / aspect_ratio) ** 0.5
    width = area / _onp.maximum(height, 1e-18)

    I_chord_axis = width * height**3 / 12
    I_thickness_axis = height * width**3 / 12

    long_side = _onp.maximum(width, height)
    short_side = _onp.minimum(width, height)
    side_ratio = short_side / _onp.maximum(long_side, 1e-18)
    torsion_constant = (
        long_side
        * short_side**3
        / 3
        * (1 - 0.63 * side_ratio + 0.052 * side_ratio**5)
    )

    return {
        "equivalent_width": width,
        "equivalent_height": height,
        "section_area": area,
        "section_centroid_chordwise": 0.5 * chord,
        "section_centroid_thickness": 0.0 * chord,
        "I_chord_axis": I_chord_axis,
        "I_thickness_axis": I_thickness_axis,
        "I_chord_thickness_product": 0.0 * I_chord_axis,
        "torsion_constant": torsion_constant,
        "max_chordwise_distance": 0.5 * width,
        "max_thickness_distance": 0.5 * height,
        "raw_airfoil_area": area,
        "area_scale_factor": 1.0 + 0.0 * area,
    }


class PropellerBeamStructuralAnalysis:
    """
    1D cantilever beam postprocessor for a propeller blade.

    This is intended to run alongside ``PropellerAnalysis``. The span coordinate
    starts at the structural root and ends at the blade tip. Section inertias may
    be supplied directly, or computed externally from airfoil polygons using
    ``airfoil_section_properties_over_span``.
    """

    def __init__(
        self,
        radius_stations: _onp.ndarray,
        chord: _onp.ndarray,
        twist: _onp.ndarray,
        area: _onp.ndarray,
        max_thickness: _onp.ndarray,
        cg_chordwise: _onp.ndarray,
        cg_thickness: _onp.ndarray,
        density: float,
        elastic_modulus: float,
        shear_modulus: float,
        aerodynamic_output: Dict[str, Any],
        air_density: Optional[float] = None,
        aerodynamic_center_chord_fraction: float = 0.25,
        aerodynamic_center_thickness: float = 0.0,
        blade_count: int = 2,
        radial_resolution: int = 241,
        section_properties: Optional[Dict[str, _onp.ndarray]] = None,
        use_section_centroid_as_torsion_axis: bool = True,
    ):
        self.radius_stations = _onp.asarray(radius_stations, dtype=float)
        self.chord_input = _onp.asarray(chord, dtype=float)
        self.twist_input = _onp.asarray(twist, dtype=float)
        self.area_input = _onp.asarray(area, dtype=float)
        self.max_thickness_input = _onp.asarray(max_thickness, dtype=float)
        self.cg_chordwise_input = _onp.asarray(cg_chordwise, dtype=float)
        self.cg_thickness_input = _onp.asarray(cg_thickness, dtype=float)
        self.density = float(density)
        self.elastic_modulus = float(elastic_modulus)
        self.shear_modulus = float(shear_modulus)
        self.aerodynamic_output = aerodynamic_output
        self.air_density = None if air_density is None else float(air_density)
        self.aerodynamic_center_chord_fraction = aerodynamic_center_chord_fraction
        self.aerodynamic_center_thickness = aerodynamic_center_thickness
        self.blade_count = blade_count
        self.radial_resolution = radial_resolution
        self.section_properties_input = section_properties
        self.use_section_centroid_as_torsion_axis = use_section_centroid_as_torsion_axis

        if not (
            len(self.radius_stations)
            == len(self.chord_input)
            == len(self.twist_input)
            == len(self.area_input)
            == len(self.max_thickness_input)
            == len(self.cg_chordwise_input)
            == len(self.cg_thickness_input)
        ):
            raise ValueError("All sectional property arrays must have the same length.")
        if section_properties is not None:
            for key, value in section_properties.items():
                if len(_onp.asarray(value).reshape(-1)) != len(self.radius_stations):
                    raise ValueError(
                        f"Section property `{key}` must have one value per station."
                    )
        if _onp.any(_onp.diff(self.radius_stations) <= 0):
            raise ValueError("`radius_stations` must be strictly increasing.")
        if radial_resolution < 3:
            raise ValueError("`radial_resolution` must be at least 3.")

    @staticmethod
    def _array(output: Dict[str, Any], key: str) -> _onp.ndarray:
        return _onp.asarray(output[key], dtype=float).reshape(-1)

    def run(self) -> Dict[str, Any]:
        r = _onp.linspace(
            float(self.radius_stations[0]),
            float(self.radius_stations[-1]),
            self.radial_resolution,
        )
        r_over_R = r / float(self.radius_stations[-1])

        chord = _onp.interp(r, self.radius_stations, self.chord_input)
        twist = _onp.interp(r, self.radius_stations, self.twist_input)
        area = _onp.interp(r, self.radius_stations, self.area_input)
        max_thickness = _onp.interp(
            r,
            self.radius_stations,
            self.max_thickness_input,
        )
        cg_chordwise = _onp.interp(r, self.radius_stations, self.cg_chordwise_input)
        cg_thickness = _onp.interp(r, self.radius_stations, self.cg_thickness_input)

        if self.section_properties_input is None:
            section_properties = _equivalent_rectangle_properties(
                area=area,
                chord=chord,
                max_thickness=max_thickness,
            )
        else:
            section_properties = {}
            for key, value in self.section_properties_input.items():
                section_properties[key] = _onp.interp(
                    r,
                    self.radius_stations,
                    _onp.asarray(value, dtype=float).reshape(-1),
                )
            for key, fallback in _equivalent_rectangle_properties(
                area=area,
                chord=chord,
                max_thickness=max_thickness,
            ).items():
                section_properties.setdefault(key, fallback)

        equivalent_width = section_properties["equivalent_width"]
        equivalent_height = section_properties["equivalent_height"]
        section_area = section_properties["section_area"]
        section_centroid_chordwise = section_properties["section_centroid_chordwise"]
        section_centroid_thickness = section_properties["section_centroid_thickness"]
        I_chord_axis = section_properties["I_chord_axis"]
        I_thickness_axis = section_properties["I_thickness_axis"]
        I_chord_thickness_product = section_properties[
            "I_chord_thickness_product"
        ]
        torsion_constant = section_properties["torsion_constant"]
        max_chordwise_distance = section_properties["max_chordwise_distance"]
        max_thickness_distance = section_properties["max_thickness_distance"]

        mass_per_length = self.density * area
        omega = float(self.aerodynamic_output["omega"])
        q_centrifugal = mass_per_length * omega**2 * r
        axial_force = _cumulative_trapezoid_from_tip(q_centrifugal, r)

        aero_r = self._array(self.aerodynamic_output, "r")
        thrust_per_radius = self._array(
            self.aerodynamic_output,
            "thrust_per_radius",
        )
        torque_per_radius = self._array(
            self.aerodynamic_output,
            "torque_per_radius",
        )
        q_thrust = _onp.interp(r, aero_r, thrust_per_radius, left=0.0, right=0.0)
        q_tangential = _onp.interp(
            r,
            aero_r,
            torque_per_radius / _onp.maximum(aero_r, 1e-9),
            left=0.0,
            right=0.0,
        )

        beta = _onp.radians(twist)
        q_chordwise = q_tangential * _onp.cos(beta) + q_thrust * _onp.sin(beta)
        q_thickness = -q_tangential * _onp.sin(beta) + q_thrust * _onp.cos(beta)

        shear_chordwise = _cumulative_trapezoid_from_tip(q_chordwise, r)
        shear_thickness = _cumulative_trapezoid_from_tip(q_thickness, r)
        moment_about_thickness_axis = _tip_integral_of_load_times_arm(
            q_chordwise,
            r,
        )
        moment_about_chord_axis = _tip_integral_of_load_times_arm(q_thickness, r)

        aerodynamic_center_chordwise = (
            self.aerodynamic_center_chord_fraction * chord
        )
        aerodynamic_center_thickness = (
            self.aerodynamic_center_thickness * _onp.ones_like(r)
        )
        if self.use_section_centroid_as_torsion_axis:
            torsion_axis_chordwise = section_centroid_chordwise
            torsion_axis_thickness = section_centroid_thickness
        else:
            torsion_axis_chordwise = cg_chordwise
            torsion_axis_thickness = cg_thickness
        lever_chordwise = aerodynamic_center_chordwise - torsion_axis_chordwise
        lever_thickness = aerodynamic_center_thickness - torsion_axis_thickness

        torsion_per_length = (
            lever_chordwise * q_thickness - lever_thickness * q_chordwise
        )
        if self.air_density is not None and "CM" in self.aerodynamic_output:
            W = _onp.interp(r, aero_r, self._array(self.aerodynamic_output, "W"))
            CM = _onp.interp(r, aero_r, self._array(self.aerodynamic_output, "CM"))
            torsion_per_length = (
                torsion_per_length
                + 0.5 * self.air_density * W**2 * chord**2 * CM
            )
        torque_about_span = _cumulative_trapezoid_from_tip(torsion_per_length, r)

        EI_chord_axis = self.elastic_modulus * I_chord_axis
        EI_thickness_axis = self.elastic_modulus * I_thickness_axis
        GJ = self.shear_modulus * torsion_constant

        curvature_thickness = moment_about_chord_axis / _onp.maximum(
            EI_chord_axis,
            1e-18,
        )
        curvature_chordwise = moment_about_thickness_axis / _onp.maximum(
            EI_thickness_axis,
            1e-18,
        )
        twist_rate = torque_about_span / _onp.maximum(GJ, 1e-18)

        slope_thickness = _cumulative_trapezoid_from_root(curvature_thickness, r)
        deflection_thickness = _cumulative_trapezoid_from_root(slope_thickness, r)
        slope_chordwise = _cumulative_trapezoid_from_root(curvature_chordwise, r)
        deflection_chordwise = _cumulative_trapezoid_from_root(slope_chordwise, r)
        torsion_angle = _cumulative_trapezoid_from_root(twist_rate, r)

        axial_stress = axial_force / _onp.maximum(area, 1e-18)
        bending_stress = (
            _onp.abs(moment_about_chord_axis)
            * max_thickness_distance
            / _onp.maximum(I_chord_axis, 1e-24)
            + _onp.abs(moment_about_thickness_axis)
            * max_chordwise_distance
            / _onp.maximum(I_thickness_axis, 1e-24)
        )

        output = {
            "r": r,
            "r_over_R": r_over_R,
            "chord": chord,
            "twist": twist,
            "area": area,
            "max_thickness": max_thickness,
            "cg_chordwise": cg_chordwise,
            "cg_thickness": cg_thickness,
            "equivalent_width": equivalent_width,
            "equivalent_height": equivalent_height,
            "section_area": section_area,
            "section_centroid_chordwise": section_centroid_chordwise,
            "section_centroid_thickness": section_centroid_thickness,
            "I_chord_axis": I_chord_axis,
            "I_thickness_axis": I_thickness_axis,
            "I_chord_thickness_product": I_chord_thickness_product,
            "torsion_constant": torsion_constant,
            "max_chordwise_distance": max_chordwise_distance,
            "max_thickness_distance": max_thickness_distance,
            "raw_airfoil_area": section_properties["raw_airfoil_area"],
            "area_scale_factor": section_properties["area_scale_factor"],
            "mass_per_length": mass_per_length,
            "q_centrifugal": q_centrifugal,
            "axial_force": axial_force,
            "q_thrust": q_thrust,
            "q_tangential": q_tangential,
            "q_chordwise": q_chordwise,
            "q_thickness": q_thickness,
            "shear_chordwise": shear_chordwise,
            "shear_thickness": shear_thickness,
            "moment_about_chord_axis": moment_about_chord_axis,
            "moment_about_thickness_axis": moment_about_thickness_axis,
            "lever_chordwise": lever_chordwise,
            "lever_thickness": lever_thickness,
            "torsion_axis_chordwise": torsion_axis_chordwise,
            "torsion_axis_thickness": torsion_axis_thickness,
            "torsion_per_length": torsion_per_length,
            "torque_about_span": torque_about_span,
            "EI_chord_axis": EI_chord_axis,
            "EI_thickness_axis": EI_thickness_axis,
            "GJ": GJ,
            "curvature_thickness": curvature_thickness,
            "curvature_chordwise": curvature_chordwise,
            "twist_rate": twist_rate,
            "deflection_thickness": deflection_thickness,
            "deflection_chordwise": deflection_chordwise,
            "torsion_angle": torsion_angle,
            "axial_stress": axial_stress,
            "bending_stress": bending_stress,
            "combined_tensile_stress": axial_stress + bending_stress,
            "combined_compressive_stress": axial_stress - bending_stress,
            "mass_per_blade": _onp.trapezoid(mass_per_length, r),
            "polar_inertia_per_blade": _onp.trapezoid(mass_per_length * r**2, r),
        }

        self.output = output
        return output
