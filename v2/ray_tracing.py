"""Basic travel-time relations for the new weld model.

Coordinates are two-dimensional ``(x, y)`` positions in metres. Speeds are
in metres per second and times are in seconds. No v2 geometry or material
properties are assumed here; those require the new model files.
"""

import numpy as np


def elastic_wave_speeds(youngs_modulus, poisson_ratio, density):
    """Calculate bulk P- and S-wave speeds in an isotropic elastic solid.

    Parameters
    ----------
    youngs_modulus : float or array_like
        Young's modulus in pascals.
    poisson_ratio : float or array_like
        Poisson's ratio, between -1 and 0.5.
    density : float or array_like
        Mass density in kilograms per cubic metre.

    Returns
    -------
    p_speed : float or ndarray
        Compressional-wave speed in metres per second.
    s_speed : float or ndarray
        Shear-wave speed in metres per second.

    Notes
    -----
    These are the bulk-wave speeds of a linear, isotropic, homogeneous solid.
    They do not describe guided waves or direction-dependent elasticity.
    """
    youngs_modulus = np.asarray(youngs_modulus, dtype=float)
    poisson_ratio = np.asarray(poisson_ratio, dtype=float)
    density = np.asarray(density, dtype=float)

    if np.any(youngs_modulus <= 0) or np.any(density <= 0):
        raise ValueError("Young's modulus and density must be positive.")
    if np.any((poisson_ratio <= -1) | (poisson_ratio >= 0.5)):
        raise ValueError("Poisson's ratio must be between -1 and 0.5.")

    shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))
    p_speed = np.sqrt(
        youngs_modulus * (1 - poisson_ratio)
        / (density * (1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    )
    s_speed = np.sqrt(shear_modulus / density)
    return p_speed, s_speed


def ray_length(start, end):
    """Find the length of a straight segment in the model cross section.

    Parameters
    ----------
    start, end : array_like
        Positions ``(x, y)`` in metres. Arrays of positions may be used;
        their leading dimensions follow NumPy broadcasting rules.

    Returns
    -------
    float or ndarray
        Segment length in metres.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape[-1:] != (2,) or end.shape[-1:] != (2,):
        raise ValueError("Positions must have an (x, y) coordinate pair.")
    return np.linalg.norm(end - start, axis=-1)


def straight_ray_time(start, end, speed):
    """Calculate travel time along one unobstructed straight ray.

    Parameters
    ----------
    start, end : array_like
        Ray endpoints ``(x, y)`` in metres.
    speed : float or array_like
        Constant P- or S-wave speed along the ray, in metres per second.

    Returns
    -------
    float or ndarray
        Travel time in seconds.

    Notes
    -----
    This relation applies only when the entire segment has the given speed
    and does not cross a void or reflecting boundary.
    """
    speed = np.asarray(speed, dtype=float)
    if np.any(speed <= 0):
        raise ValueError("Wave speed must be positive.")
    return ray_length(start, end) / speed


def scattered_ray_time(source, scatterer, receiver, incident_speed, return_speed):
    """Add the source-to-scatterer and scatterer-to-receiver travel times.

    Parameters
    ----------
    source, scatterer, receiver : array_like
        Positions ``(x, y)`` in metres.
    incident_speed : float or array_like
        Speed on the source-to-scatterer leg, in metres per second.
    return_speed : float or array_like
        Speed on the scatterer-to-receiver leg, in metres per second. This
        may differ from ``incident_speed`` when a wave changes mode.

    Returns
    -------
    float or ndarray
        Total travel time in seconds.

    Notes
    -----
    Both legs are assumed straight, unobstructed, and constant-speed.
    """
    return (
        straight_ray_time(source, scatterer, incident_speed)
        + straight_ray_time(scatterer, receiver, return_speed)
    )


def snell_transmitted_angle(incident_angle, incident_speed, transmitted_speed):
    """Find the refracted ray angle at a flat material boundary.

    Parameters
    ----------
    incident_angle : float or array_like
        Signed angle of incidence from the boundary normal, in radians.
        Its magnitude must not exceed pi/2.
    incident_speed : float or array_like
        Wave speed before the boundary, in metres per second.
    transmitted_speed : float or array_like
        Wave speed after the boundary, in metres per second.

    Returns
    -------
    float or ndarray
        Signed transmitted angle from the boundary normal, in radians.

    Raises
    ------
    ValueError
        If speeds or angles are invalid, or if no propagating transmitted
        ray exists because of total internal reflection.

    Notes
    -----
    Snell's law for one fixed wave mode is
    ``sin(theta_1) / c_1 = sin(theta_2) / c_2``. Mode conversion and
    transmission amplitudes are not calculated.
    """
    incident_angle = np.asarray(incident_angle, dtype=float)
    incident_speed = np.asarray(incident_speed, dtype=float)
    transmitted_speed = np.asarray(transmitted_speed, dtype=float)
    if np.any(np.abs(incident_angle) > np.pi / 2):
        raise ValueError("Incident angle must lie between -pi/2 and pi/2.")
    if np.any(incident_speed <= 0) or np.any(transmitted_speed <= 0):
        raise ValueError("Wave speeds must be positive.")

    transmitted_sine = (
        transmitted_speed / incident_speed * np.sin(incident_angle)
    )
    if np.any(np.abs(transmitted_sine) > 1 + 1e-14):
        raise ValueError("No propagating transmitted ray exists.")
    return np.arcsin(np.clip(transmitted_sine, -1, 1))
