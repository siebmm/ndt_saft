"""Analyse laser ultrasonic signals from a partial weld.

Notes
-----
Coordinates describe a two-dimensional pipe-wall cross section in metres:
the inner back wall is at ``y=0`` and the outer surface at ``y=19.8 mm``.
The COMSOL receiver traces contain vertical surface velocity in metres per
second. SAFT delays follow a fixed laser source, a candidate image pixel, and
each receiver through parent steel and the weld/heat-affected zone.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, correlate, correlation_lags, hilbert, sosfiltfilt

DATA_DIRECTORY = Path(__file__).resolve().parent / 'data' / 'receivers'
RECEIVER_DATASETS = {
    'defect': {
        'left': DATA_DIRECTORY / 'hybrid_8MHz_receiver_left.txt',
        'right': DATA_DIRECTORY / 'hybrid_8MHz_receiver_right.txt',
        'weld': DATA_DIRECTORY / 'hybrid_8MHz_receiver_weld.txt',
    },
    'reference': {
        'left': DATA_DIRECTORY / 'hybrid_8MHz_receiver_left_nodefect.txt',
        'right': DATA_DIRECTORY / 'hybrid_8MHz_receiver_right_nodefect.txt',
        'weld': DATA_DIRECTORY / 'hybrid_8MHz_receiver_weld_nodefect.txt',
    },
}
# Geometry coordinates and wave speeds below use SI units and approximate the
# COMSOL pipe-wall cross section for travel-time calculations.
WELD_SURFACE_Y = 0.006339303874144179
GROOVE_ROOT_HALF_WIDTH = 0.0028007853386639377
GEOMETRY_TOLERANCE = 1e-12
DEFAULT_LASER_SOURCE = (0.0, WELD_SURFACE_Y)
PARENT_P_SPEED = 5943.603925367657
WELD_P_SPEED = 5843.507732692268
PARENT_S_SPEED = 3237.967635087099
WELD_S_SPEED = 3116.9681976898137


@dataclass(frozen=True)
class ReceiverData:
    """Selected and combined receiver data in SI units.

    Attributes
    ----------
    x_coords : numpy.ndarray
        Receiver x coordinates in metres.
    y_coords : numpy.ndarray
        Receiver y coordinates in metres.
    times : numpy.ndarray
        Signal sample times in seconds.
    time_series_matrix : numpy.ndarray
        Vertical-velocity signals with shape ``(receivers, samples)``.
    line_names : numpy.ndarray
        Source receiver-line name for every signal row.
    point_indices : numpy.ndarray
        Original zero-based point index for every signal row.
    """

    x_coords: np.ndarray
    y_coords: np.ndarray
    times: np.ndarray
    time_series_matrix: np.ndarray
    line_names: np.ndarray
    point_indices: np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    """Matched defect/reference data and their calibrated difference.

    Attributes
    ----------
    defective : ReceiverData
        Filtered defective-model data.
    reference : ReceiverData
        Filtered, aligned, and scaled reference data.
    difference : ReceiverData
        Trace-by-trace defective-minus-reference residual data.
    shifts_samples : numpy.ndarray
        Integer shift applied to each reference trace.
    reference_scale_factors : numpy.ndarray
        Multiplicative scale applied to each reference trace.
    """

    defective: ReceiverData
    reference: ReceiverData
    difference: ReceiverData
    shifts_samples: np.ndarray
    reference_scale_factors: np.ndarray


@dataclass(frozen=True)
class ReferenceArrivalCalibration:
    """Reference-block calibration of a direct-arrival timing model.

    Attributes
    ----------
    data : ReceiverData
        Reference receiver data used for calibration.
    filtered_signals : numpy.ndarray
        Band-pass-filtered reference signals.
    envelope : numpy.ndarray
        Analytic-signal magnitude of the filtered signals.
    wave_type : {'P', 'S'}
        Wave mode used to calculate direct travel times.
    predicted_times : numpy.ndarray
        Geometry-model direct travel times before adding the fitted offset.
    picked_times : numpy.ndarray
        Local envelope-maximum times near the calibrated arrival curve.
    time_offset : float
        Fitted global signal delay in seconds.
    score : float
        Mean trace-normalized envelope sampled along the fitted curve.
    residual_rms : float
        RMS picked-arrival residual around the fitted curve in seconds.
    """

    data: ReceiverData
    filtered_signals: np.ndarray
    envelope: np.ndarray
    wave_type: str
    predicted_times: np.ndarray
    picked_times: np.ndarray
    time_offset: float
    score: float
    residual_rms: float


@dataclass(frozen=True)
class ArrivalComparison:
    """Reference-picked pulse amplitudes in matched COMSOL simulations.

    Attributes
    ----------
    data : ReceiverData
        Receiver coordinates and time axis for the paired simulations.
    picked_times : numpy.ndarray
        Time of the strongest reference pulse in the search window, in seconds.
    amplitude_ratios : numpy.ndarray
        Defect/reference RMS velocity ratio around each picked pulse.
    correlation : numpy.ndarray
        Normalized defect/reference waveform correlation in the same window.

    Notes
    -----
    A small ratio indicates loss of transmitted surface motion. It does not
    identify the propagation mode or locate the scattering feature by itself.
    """

    data: ReceiverData
    picked_times: np.ndarray
    amplitude_ratios: np.ndarray
    correlation: np.ndarray


def _normalise_receiver_indices(receiver_indices, num_receivers):
    """Validate and normalize a receiver-point selection.

    Parameters
    ----------
    receiver_indices : int, slice, array-like, or None
        Zero-based point selection. Negative integer indices and Boolean masks
        follow standard NumPy indexing conventions. ``None`` selects all
        receiver points.
    num_receivers : int
        Number of receiver points available on the line.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of normalized integer indices.

    Raises
    ------
    TypeError
        If the selection does not contain integer or Boolean values.
    ValueError
        If the selection is empty, multidimensional, duplicated, or has an
        incorrectly sized Boolean mask.
    IndexError
        If an index falls outside the receiver line.
    """
    if receiver_indices is None:
        return np.arange(num_receivers, dtype=int)

    if isinstance(receiver_indices, slice):
        indices = np.arange(num_receivers, dtype=int)[receiver_indices]
    else:
        indices = np.asarray(receiver_indices)
        if indices.ndim == 0:
            indices = indices.reshape(1)
        elif indices.ndim != 1:
            raise ValueError("receiver_indices must be one-dimensional.")

        if indices.size == 0:
            raise ValueError("Select at least one receiver point.")

        if np.issubdtype(indices.dtype, np.bool_):
            if indices.size != num_receivers:
                raise ValueError(
                    "A Boolean receiver mask must have one value per receiver "
                    f"({num_receivers} values required)."
                )
            indices = np.flatnonzero(indices)
        elif not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("receiver_indices must contain integer point indices.")
        else:
            indices = indices.astype(int, copy=False)

    if indices.size == 0:
        raise ValueError("Select at least one receiver point.")

    # Support normal Python negative indexing while still validating the bounds.
    indices = np.where(indices < 0, indices + num_receivers, indices)
    invalid = indices[(indices < 0) | (indices >= num_receivers)]
    if invalid.size:
        raise IndexError(
            f"Receiver index {invalid[0]} is out of bounds for a line with "
            f"{num_receivers} points."
        )

    if np.unique(indices).size != indices.size:
        raise ValueError("receiver_indices contains duplicate point indices.")

    return indices


def load_receiver_data(
    receiver_lines='left',
    receiver_indices=None,
    dataset='defect',
):
    """Load, select, and combine any of the left, right, and weld lines.

    Parameters
    ----------
    receiver_lines : str or sequence of str
        A line name, ``'sides'`` for left plus right, or an explicit sequence
        such as ``['left', 'right', 'weld']``.
    receiver_indices : int, slice, array-like, mapping, or None
        A point selector applied to every chosen line, or a mapping such as
        ``{'left': [0, 5], 'right': [10, 15]}`` for per-line selections.
    dataset : str or mapping
        Either ``'defect'`` or ``'reference'``. A custom mapping from line
        names to file paths can also be supplied.

    Returns
    -------
    ReceiverData
        Coordinates in metres, times in seconds, velocity signals, and the
        source line/name metadata for each selected matrix row.

    Raises
    ------
    KeyError
        If a per-line index mapping omits a selected receiver line.
    ValueError
        If the dataset, receiver lines, file contents, or time axes are
        invalid or incompatible.
    """
    if isinstance(dataset, Mapping):
        receiver_files = {
            str(line).lower(): Path(file_path)
            for line, file_path in dataset.items()
        }
    else:
        dataset_name = str(dataset).lower()
        if dataset_name not in RECEIVER_DATASETS:
            valid_datasets = ", ".join(RECEIVER_DATASETS)
            raise ValueError(
                f"Unknown dataset '{dataset}'. Choose from {valid_datasets}."
            )
        receiver_files = RECEIVER_DATASETS[dataset_name]

    if isinstance(receiver_lines, str):
        receiver_selection = receiver_lines.lower()
        if receiver_selection == 'sides':
            selected_lines = ['left', 'right']
        elif receiver_selection == 'all':
            raise ValueError(
                "The 'all' shortcut is disabled. Inspect individual lines or "
                "use 'sides'; pass ['left', 'right', 'weld'] explicitly only "
                "after validating each line."
            )
        else:
            selected_lines = [receiver_selection]
    else:
        selected_lines = [str(line).lower() for line in receiver_lines]

    if not selected_lines:
        raise ValueError("Select at least one receiver line.")
    if len(set(selected_lines)) != len(selected_lines):
        raise ValueError("receiver_lines contains duplicate line names.")

    unknown_lines = [line for line in selected_lines if line not in receiver_files]
    if unknown_lines:
        valid = ", ".join(receiver_files)
        raise ValueError(
            f"Unknown receiver line(s): {unknown_lines}. Choose from {valid}, "
            "use 'sides', or pass an explicit sequence."
        )

    x_parts = []
    y_parts = []
    signal_parts = []
    line_name_parts = []
    point_index_parts = []
    common_times = None

    for line_name in selected_lines:
        times = []
        data_rows = []
        file_path = receiver_files[line_name]

        with file_path.open('r', encoding='utf-8') as receiver_file:
            for row in receiver_file:
                if row.startswith('%'):
                    if '@ t=' in row:
                        times = [
                            float(value)
                            for value in re.findall(r'@\s*t=([0-9.eE+-]+)', row)
                        ]
                else:
                    values = row.strip().split()
                    if values:
                        data_rows.append([float(value) for value in values])

        if not data_rows or not times:
            raise ValueError(f"No receiver data or time values found in {file_path.name}.")

        line_data = np.asarray(data_rows, dtype=float)
        line_times = np.asarray(times, dtype=float)
        if line_data.shape[1] - 2 != line_times.size:
            raise ValueError(
                f"The time and signal column counts do not match in {file_path.name}."
            )

        if common_times is None:
            common_times = line_times
        elif not np.array_equal(common_times, line_times):
            raise ValueError("Selected receiver lines do not share the same time axis.")

        if isinstance(receiver_indices, Mapping):
            if line_name not in receiver_indices:
                raise KeyError(
                    f"No receiver_indices selection was supplied for '{line_name}'."
                )
            line_selector = receiver_indices[line_name]
        else:
            line_selector = receiver_indices

        selected_indices = _normalise_receiver_indices(line_selector, len(line_data))
        selected_data = line_data[selected_indices]

        # COMSOL coordinates are exported in millimetres; expose SI units.
        x_parts.append(selected_data[:, 0] / 1000.0)
        y_parts.append(selected_data[:, 1] / 1000.0)
        signal_parts.append(selected_data[:, 2:])
        line_name_parts.append(np.full(selected_indices.size, line_name, dtype=object))
        point_index_parts.append(selected_indices)

    return ReceiverData(
        x_coords=np.concatenate(x_parts),
        y_coords=np.concatenate(y_parts),
        times=common_times,
        time_series_matrix=np.concatenate(signal_parts, axis=0),
        line_names=np.concatenate(line_name_parts),
        point_indices=np.concatenate(point_index_parts),
    )


def _with_signals(data, signals):
    """Create receiver data with replacement signals and shared metadata.

    Parameters
    ----------
    data : ReceiverData
        Receiver metadata and time axis to retain.
    signals : array-like
        Replacement signal matrix.

    Returns
    -------
    ReceiverData
        Receiver data containing ``signals`` and the original metadata.
    """
    return ReceiverData(
        x_coords=data.x_coords,
        y_coords=data.y_coords,
        times=data.times,
        time_series_matrix=np.asarray(signals),
        line_names=data.line_names,
        point_indices=data.point_indices,
    )


def _bandpass_signals(signals, times, frequency_band, filter_order):
    """Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    signals : numpy.ndarray
        Signal matrix with time along the final axis.
    times : numpy.ndarray
        Uniform sample times in seconds.
    frequency_band : tuple of float or None
        Lower and upper cutoff frequencies in hertz. ``None`` returns an
        unfiltered copy.
    filter_order : int
        Butterworth filter order.

    Returns
    -------
    numpy.ndarray
        Filtered signal matrix.

    Raises
    ------
    TypeError
        If ``filter_order`` is not an integer.
    ValueError
        If the frequency band, time sampling, or filter order is invalid.
    """
    if frequency_band is None:
        return np.array(signals, copy=True)

    if len(frequency_band) != 2:
        raise ValueError("frequency_band must contain (low_frequency, high_frequency).")
    low_frequency, high_frequency = (float(value) for value in frequency_band)
    time_steps = np.diff(times)
    if not np.allclose(time_steps, time_steps[0], rtol=1e-9, atol=0.0):
        raise ValueError("Band-pass filtering requires a uniformly sampled time axis.")
    sampling_frequency = 1.0 / time_steps[0]
    nyquist_frequency = 0.5 * sampling_frequency
    if not 0.0 < low_frequency < high_frequency < nyquist_frequency:
        raise ValueError(
            "frequency_band must satisfy 0 < low < high < Nyquist "
            f"({nyquist_frequency / 1e6:.3f} MHz)."
        )
    if isinstance(filter_order, bool) or not isinstance(
        filter_order, (int, np.integer)
    ):
        raise TypeError("filter_order must be an integer.")
    if filter_order < 1:
        raise ValueError("filter_order must be at least 1.")

    filter_sections = butter(
        filter_order,
        [low_frequency, high_frequency],
        btype='bandpass',
        fs=sampling_frequency,
        output='sos',
    )
    return sosfiltfilt(filter_sections, signals, axis=1)


def _calibration_window_mask(times, calibration_window, line_name):
    """Resolve a shared or per-line calibration time window.

    Parameters
    ----------
    times : numpy.ndarray
        Signal sample times in seconds.
    calibration_window : tuple of float or mapping
        Shared ``(start, end)`` time window or mapping from line names to
        individual windows.
    line_name : str
        Receiver-line name for which to resolve the window.

    Returns
    -------
    numpy.ndarray
        Boolean mask selecting samples inside the resolved window.

    Raises
    ------
    KeyError
        If a window mapping omits ``line_name``.
    ValueError
        If the window is missing, reversed, or contains fewer than three
        samples.
    """
    if isinstance(calibration_window, Mapping):
        if line_name not in calibration_window:
            raise KeyError(f"No calibration window was supplied for '{line_name}'.")
        line_window = calibration_window[line_name]
    else:
        line_window = calibration_window

    if line_window is None or len(line_window) != 2:
        raise ValueError(
            "Alignment/scaling requires a (start_time, end_time) calibration "
            "window, either shared or specified per receiver line."
        )
    window_start, window_end = (float(value) for value in line_window)
    mask = (times >= window_start) & (times <= window_end)
    if window_end <= window_start or np.count_nonzero(mask) < 3:
        raise ValueError(f"Invalid calibration window for '{line_name}'.")
    return mask


def _shift_signal_with_zeros(signal, shift_samples):
    """Shift a signal without circular wrapping.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional signal to shift.
    shift_samples : int
        Signed integer shift. Positive values delay the signal.

    Returns
    -------
    numpy.ndarray
        Shifted signal with newly exposed samples filled with zeros.
    """
    shifted = np.zeros_like(signal)
    if shift_samples > 0:
        shifted[shift_samples:] = signal[:-shift_samples]
    elif shift_samples < 0:
        shifted[:shift_samples] = signal[-shift_samples:]
    else:
        shifted[:] = signal
    return shifted










def load_calibrated_receiver_data(
    receiver_lines='left',
    receiver_indices=None,
    defect_dataset='defect',
    reference_dataset='reference',
    frequency_band=None,
    filter_order=4,
    align_reference=False,
    scale_reference=False,
    calibration_window=None,
    max_shift_samples=10,
):
    """Load matched defect/reference data and calculate calibrated residuals.

    Parameters
    ----------
    receiver_lines : str or sequence of str, default='left'
        Receiver line or lines to load.
    receiver_indices : int, slice, array-like, mapping, or None
        Shared or per-line receiver-point selection.
    defect_dataset : str or mapping, default='defect'
        Defective dataset name or custom file mapping.
    reference_dataset : str or mapping, default='reference'
        Defect-free dataset name or custom file mapping.
    frequency_band : tuple of float or None
        Optional band-pass cutoff frequencies in hertz.
    filter_order : int, default=4
        Butterworth filter order.
    align_reference : bool, default=False
        Whether to estimate and apply an integer shift to each reference trace.
    scale_reference : bool, default=False
        Whether to least-squares scale each aligned reference trace.
    calibration_window : tuple of float, mapping, or None
        Time window in seconds used for alignment and scaling. A mapping may
        provide different windows for each receiver line.
    max_shift_samples : int, default=10
        Maximum absolute lag considered during reference alignment.

    Returns
    -------
    CalibrationResult
        Processed defective data, calibrated reference data, residual data,
        shifts, and scale factors.

    Raises
    ------
    TypeError
        If ``max_shift_samples`` is not an integer.
    ValueError
        If paired data are incompatible or a calibration setting is invalid.

    Notes
    -----
    Filtering is applied identically to both datasets. Alignment shifts the
    reference toward the defective trace. Scaling multiplies the aligned
    reference before trace-by-trace subtraction. Alignment and scaling require
    an explicit calibration window to avoid silently removing defect response.
    """
    defective = load_receiver_data(
        receiver_lines, receiver_indices, dataset=defect_dataset
    )
    reference = load_receiver_data(
        receiver_lines, receiver_indices, dataset=reference_dataset
    )

    matching_arrays = (
        np.array_equal(defective.times, reference.times),
        np.array_equal(defective.line_names, reference.line_names),
        np.array_equal(defective.point_indices, reference.point_indices),
        np.allclose(defective.x_coords, reference.x_coords, rtol=0.0, atol=1e-12),
        np.allclose(defective.y_coords, reference.y_coords, rtol=0.0, atol=1e-12),
    )
    if not all(matching_arrays):
        raise ValueError(
            "Defect and reference data must have identical times, receiver "
            "lines, point indices, and coordinates."
        )

    defect_signals = _bandpass_signals(
        defective.time_series_matrix,
        defective.times,
        frequency_band,
        filter_order,
    )
    reference_signals = _bandpass_signals(
        reference.time_series_matrix,
        reference.times,
        frequency_band,
        filter_order,
    )

    if isinstance(max_shift_samples, bool) or not isinstance(
        max_shift_samples, (int, np.integer)
    ):
        raise TypeError("max_shift_samples must be an integer.")
    if max_shift_samples < 0:
        raise ValueError("max_shift_samples cannot be negative.")

    shifts = np.zeros(len(defect_signals), dtype=int)
    scale_factors = np.ones(len(defect_signals), dtype=float)
    calibrated_reference = np.empty_like(reference_signals)

    for receiver_number, (defect_signal, reference_signal, line_name) in enumerate(
        zip(defect_signals, reference_signals, defective.line_names)
    ):
        if align_reference or scale_reference:
            window_mask = _calibration_window_mask(
                defective.times, calibration_window, line_name
            )
        else:
            window_mask = slice(None)

        if align_reference:
            defect_window = defect_signal[window_mask]
            reference_window = reference_signal[window_mask]
            correlation = correlate(
                defect_window - np.mean(defect_window),
                reference_window - np.mean(reference_window),
                mode='full',
            )
            lags = correlation_lags(
                defect_window.size, reference_window.size, mode='full'
            )
            allowed = np.abs(lags) <= max_shift_samples
            shift_samples = int(lags[allowed][np.argmax(correlation[allowed])])
            shifts[receiver_number] = shift_samples
            reference_signal = _shift_signal_with_zeros(
                reference_signal, shift_samples
            )

        if scale_reference:
            reference_window = reference_signal[window_mask]
            defect_window = defect_signal[window_mask]
            denominator = np.dot(reference_window, reference_window)
            if denominator > np.finfo(float).eps:
                scale_factors[receiver_number] = (
                    np.dot(defect_window, reference_window) / denominator
                )

        calibrated_reference[receiver_number] = (
            scale_factors[receiver_number] * reference_signal
        )

    processed_defective = _with_signals(defective, defect_signals)
    processed_reference = _with_signals(reference, calibrated_reference)
    difference = _with_signals(
        defective, defect_signals - calibrated_reference
    )
    return CalibrationResult(
        defective=processed_defective,
        reference=processed_reference,
        difference=difference,
        shifts_samples=shifts,
        reference_scale_factors=scale_factors,
    )


def compare_reference_arrivals(
    receiver_lines='sides',
    receiver_indices=None,
    search_window=(4e-6, 5.5e-6),
    pulse_half_width=0.2e-6,
    frequency_band=(2e6, 8e6),
):
    """Measure changes to an early pulse using picks from the reference model.

    Parameters
    ----------
    receiver_lines : str or sequence of str, default='sides'
        Receiver lines to compare.
    receiver_indices : int, slice, array-like, mapping, or None
        Receiver-point selection on each line.
    search_window : tuple of float, default=(4e-6, 5.5e-6)
        Time interval in seconds containing the strong early side-line pulse.
    pulse_half_width : float, default=0.2e-6
        Half-width in seconds of the amplitude comparison around each pick.
    frequency_band : tuple of float or None, default=(2e6, 8e6)
        Band-pass range in hertz, applied equally to both simulations.

    Returns
    -------
    ArrivalComparison
        Reference pulse times, defect/reference RMS ratios, and correlations.

    Raises
    ------
    ValueError
        If the time windows do not contain sufficient samples.

    Notes
    -----
    Picks use only defect-free traces. The default interval follows the strong
    4.5--5.1 microsecond arrivals in the supplied side-line exports. These
    arrivals can include mode conversion at the groove and are not assigned a
    P or S label. A ratio below one measures transmission loss, not defect
    depth. The left side provides a useful control for the right-side flaw.
    """
    if len(search_window) != 2:
        raise ValueError('search_window must contain two times.')
    search_start, search_stop = (float(value) for value in search_window)
    pulse_half_width = float(pulse_half_width)
    if not np.isfinite([search_start, search_stop, pulse_half_width]).all():
        raise ValueError('Arrival windows must contain finite times.')
    if search_stop <= search_start or pulse_half_width <= 0:
        raise ValueError('Arrival windows must have positive width.')

    paired = load_calibrated_receiver_data(
        receiver_lines=receiver_lines,
        receiver_indices=receiver_indices,
        frequency_band=frequency_band,
    )
    times = paired.reference.times
    search = (times >= search_start) & (times <= search_stop)
    if np.count_nonzero(search) < 3:
        raise ValueError('search_window contains fewer than three samples.')

    reference = paired.reference.time_series_matrix
    defective = paired.defective.time_series_matrix
    envelope = np.abs(hilbert(reference, axis=1))
    search_indices = np.flatnonzero(search)
    picks = search_indices[np.argmax(envelope[:, search], axis=1)]
    picked_times = times[picks]
    amplitude_ratios = np.full(len(picks), np.nan)
    correlation = np.full(len(picks), np.nan)
    for row, pick_time in enumerate(picked_times):
        pulse = np.abs(times - pick_time) <= pulse_half_width
        if np.count_nonzero(pulse) < 3:
            raise ValueError('pulse_half_width contains fewer than three samples.')
        reference_pulse = reference[row, pulse]
        defective_pulse = defective[row, pulse]
        reference_norm = np.linalg.norm(reference_pulse)
        defective_norm = np.linalg.norm(defective_pulse)
        if reference_norm > 0:
            amplitude_ratios[row] = defective_norm / reference_norm
        if reference_norm > 0 and defective_norm > 0:
            correlation[row] = (
                np.dot(defective_pulse, reference_pulse)
                / (defective_norm * reference_norm)
            )

    return ArrivalComparison(
        data=paired.reference,
        picked_times=picked_times,
        amplitude_ratios=amplitude_ratios,
        correlation=correlation,
    )








def saft_geometry(x, y):
    """Evaluate the segmented SAFT material geometry.

    Parameters
    ----------
    x : array-like
        Horizontal coordinates in metres.
    y : array-like
        Vertical coordinates in metres. Values are broadcast with ``x``.

    Returns
    -------
    region : numpy.ndarray
        Material identifiers: 0 for groove/outside, 1 for parent steel, and 2
        for weld plus heat-affected zone.
    cP : numpy.ndarray
        P-wave speed in metres per second, or NaN outside solid material.
    cS : numpy.ndarray
        S-wave speed in metres per second, or NaN outside solid material.

    Notes
    -----
    The origin is at the bottom centre of the plate and y increases upward.
    Parent steel represents 293.15 K and weld/HAZ represents 573.15 K. This is
    a simplified SAFT geometry rather than the exact COMSOL mesh; the known
    defect is not removed from the travel-time geometry.
    """
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    slope = 0.05240777928304121  # tan(3 degrees)
    fill_y = WELD_SURFACE_Y
    fill_half_width = GROOVE_ROOT_HALF_WIDTH

    plate = (np.abs(x) <= 0.025) & (y >= 0) & (y <= 0.0198)
    groove_width = fill_half_width + slope * (y - fill_y)
    # COMSOL text export can move nominal boundary coordinates by a few ULPs.
    # Keep points within a picometre of the weld surface on the solid side.
    groove = (
        (y > fill_y + GEOMETRY_TOLERANCE)
        & (np.abs(x) < groove_width - GEOMETRY_TOLERANCE)
    )
    solid = plate & ~groove

    # Lower hot envelope: INTERSECTION of the two outer HAZ circles.
    radius = 0.006200000000000001
    centre_y = 0.0045000000000000005
    circle_left = (x - 0.0005)**2 + (y - centre_y)**2 <= radius**2
    circle_right = (x + 0.0005)**2 + (y - centre_y)**2 <= radius**2
    root_lens = circle_left & circle_right

    # Upper hot envelope: tangent trapezoid, unioned with the root lens.
    tangent_y = 0.004175517071293749
    half_width = 0.005691503115478358 + slope * (y - tangent_y)
    trapezoid = (y >= tangent_y) & (y <= 0.0198) & (np.abs(x) <= half_width)
    hot = solid & (y >= 0) & (y <= 0.008) & (root_lens | trapezoid)

    region = np.where(solid, np.where(hot, 2, 1), 0).astype(np.uint8)
    cP = np.where(solid, np.where(hot, WELD_P_SPEED, PARENT_P_SPEED), np.nan)
    cS = np.where(solid, np.where(hot, WELD_S_SPEED, PARENT_S_SPEED), np.nan)
    return region, cP, cS


def _clip_linear_geq(interval_start, interval_end, value_start, value_delta):
    """Find the part of a ray inside one straight material boundary.

    Parameters
    ----------
    interval_start : numpy.ndarray
        Current lower ray-parameter bounds.
    interval_end : numpy.ndarray
        Current upper ray-parameter bounds.
    value_start : numpy.ndarray
        Linear expression value at ray parameter zero.
    value_delta : numpy.ndarray
        Change in the expression between ray parameters zero and one.

    Returns
    -------
    clipped_start : numpy.ndarray
        Updated lower bounds satisfying ``value_start + t*value_delta >= 0``.
    clipped_end : numpy.ndarray
        Updated upper bounds satisfying the same inequality.

    Notes
    -----
    The weld sidewalls and the hot-zone trapezoid are straight boundaries.
    Their intersection with each candidate acoustic ray is described by the
    ray fraction ``t`` between zero and one.
    """
    tolerance = 1e-12
    parallel = np.abs(value_delta) <= tolerance
    invalid_parallel = parallel & (value_start < -tolerance)
    crossing = np.divide(
        -value_start,
        value_delta,
        out=np.zeros_like(value_delta, dtype=float),
        where=~parallel,
    )

    interval_start = np.where(
        value_delta > tolerance,
        np.maximum(interval_start, crossing),
        interval_start,
    )
    interval_end = np.where(
        value_delta < -tolerance,
        np.minimum(interval_end, crossing),
        interval_end,
    )
    interval_start = np.where(invalid_parallel, 1.0, interval_start)
    interval_end = np.where(invalid_parallel, 0.0, interval_end)
    return interval_start, interval_end


def _clip_circle(
    interval_start,
    interval_end,
    x_start,
    y_start,
    dx,
    dy,
    centre_x,
    centre_y,
    radius,
):
    """Find the part of a ray inside a circular heat-affected zone.

    Parameters
    ----------
    interval_start : numpy.ndarray
        Current lower ray-parameter bounds.
    interval_end : numpy.ndarray
        Current upper ray-parameter bounds.
    x_start : numpy.ndarray
        Ray starting x coordinates in metres.
    y_start : numpy.ndarray
        Ray starting y coordinates in metres.
    dx : numpy.ndarray
        Ray x displacements in metres.
    dy : numpy.ndarray
        Ray y displacements in metres.
    centre_x : float
        Circle-centre x coordinate in metres.
    centre_y : float
        Circle-centre y coordinate in metres.
    radius : float
        Circle radius in metres.

    Returns
    -------
    clipped_start : numpy.ndarray
        Updated lower ray-parameter bounds.
    clipped_end : numpy.ndarray
        Updated upper ray-parameter bounds.

    Notes
    -----
    The lower hot zone is bounded by two circles around the weld root.
    Intersecting their ray intervals gives the distance travelled in that
    zone before the wave reaches an image pixel or receiver.
    """
    offset_x = x_start - centre_x
    offset_y = y_start - centre_y
    quadratic = dx**2 + dy**2
    linear = 2.0 * (offset_x * dx + offset_y * dy)
    constant = offset_x**2 + offset_y**2 - radius**2
    discriminant = linear**2 - 4.0 * quadratic * constant

    stationary = quadratic <= np.finfo(float).eps
    valid_stationary = stationary & (constant <= 0.0)
    valid_moving = (~stationary) & (discriminant >= 0.0)
    square_root = np.sqrt(np.maximum(discriminant, 0.0))
    denominator = np.where(stationary, 1.0, 2.0 * quadratic)
    root_1 = (-linear - square_root) / denominator
    root_2 = (-linear + square_root) / denominator

    interval_start = np.where(
        valid_moving, np.maximum(interval_start, root_1), interval_start
    )
    interval_end = np.where(
        valid_moving, np.minimum(interval_end, root_2), interval_end
    )
    invalid = ~(valid_moving | valid_stationary)
    interval_start = np.where(invalid, 1.0, interval_start)
    interval_end = np.where(invalid, 0.0, interval_end)
    return interval_start, interval_end


def _straight_ray_region_fractions(x_start, y_start, x_end, y_end):
    """Classify material traversal along straight rays.

    Parameters
    ----------
    x_start : array-like
        Ray starting x coordinates in metres.
    y_start : array-like
        Ray starting y coordinates in metres.
    x_end : array-like
        Ray ending x coordinates in metres.
    y_end : array-like
        Ray ending y coordinates in metres.

    Returns
    -------
    weld_fraction : numpy.ndarray
        Fraction of each ray length inside the weld/HAZ region.
    crosses_groove : numpy.ndarray
        Boolean mask identifying rays that enter the open groove.
    """
    x_start, y_start, x_end, y_end = np.broadcast_arrays(
        np.asarray(x_start, dtype=float),
        np.asarray(y_start, dtype=float),
        np.asarray(x_end, dtype=float),
        np.asarray(y_end, dtype=float),
    )
    dx = x_end - x_start
    dy = y_end - y_start
    zeros = np.zeros(x_start.shape, dtype=float)
    ones = np.ones(x_start.shape, dtype=float)

    # Root-lens interval: intersection of both HAZ circles below y=8 mm.
    root_start, root_end = zeros.copy(), ones.copy()
    root_start, root_end = _clip_circle(
        root_start, root_end, x_start, y_start, dx, dy,
        0.0005, 0.0045, 0.0062,
    )
    root_start, root_end = _clip_circle(
        root_start, root_end, x_start, y_start, dx, dy,
        -0.0005, 0.0045, 0.0062,
    )
    root_start, root_end = _clip_linear_geq(
        root_start, root_end, 0.008 - y_start, -dy
    )

    # Upper HAZ trapezoid, clipped at y=8 mm as in saft_geometry().
    slope = 0.05240777928304121
    tangent_y = 0.004175517071293749
    trapezoid_base = 0.005691503115478358
    trapezoid_start, trapezoid_end = zeros.copy(), ones.copy()
    trapezoid_start, trapezoid_end = _clip_linear_geq(
        trapezoid_start, trapezoid_end, y_start - tangent_y, dy
    )
    trapezoid_start, trapezoid_end = _clip_linear_geq(
        trapezoid_start, trapezoid_end, 0.008 - y_start, -dy
    )
    trapezoid_start, trapezoid_end = _clip_linear_geq(
        trapezoid_start,
        trapezoid_end,
        trapezoid_base + slope * (y_start - tangent_y) - x_start,
        slope * dy - dx,
    )
    trapezoid_start, trapezoid_end = _clip_linear_geq(
        trapezoid_start,
        trapezoid_end,
        trapezoid_base + slope * (y_start - tangent_y) + x_start,
        slope * dy + dx,
    )

    root_length = np.maximum(0.0, root_end - root_start)
    trapezoid_length = np.maximum(0.0, trapezoid_end - trapezoid_start)
    overlap_length = np.maximum(
        0.0,
        np.minimum(root_end, trapezoid_end)
        - np.maximum(root_start, trapezoid_start),
    )
    weld_fraction = np.clip(
        root_length + trapezoid_length - overlap_length, 0.0, 1.0
    )

    # Detect rays that enter the open groove above the weld surface.
    groove_base = GROOVE_ROOT_HALF_WIDTH
    groove_start, groove_end = zeros.copy(), ones.copy()
    groove_start, groove_end = _clip_linear_geq(
        groove_start, groove_end, y_start - WELD_SURFACE_Y, dy
    )
    groove_start, groove_end = _clip_linear_geq(
        groove_start,
        groove_end,
        groove_base + slope * (y_start - WELD_SURFACE_Y) - x_start,
        slope * dy - dx,
    )
    groove_start, groove_end = _clip_linear_geq(
        groove_start,
        groove_end,
        groove_base + slope * (y_start - WELD_SURFACE_Y) + x_start,
        slope * dy + dx,
    )
    groove_length = np.maximum(0.0, groove_end - groove_start)
    groove_midpoint = 0.5 * (groove_start + groove_end)
    midpoint_x = x_start + groove_midpoint * dx
    midpoint_y = y_start + groove_midpoint * dy
    midpoint_width = groove_base + slope * (midpoint_y - WELD_SURFACE_Y)
    crosses_groove = (
        (groove_length > GEOMETRY_TOLERANCE)
        & (midpoint_y > WELD_SURFACE_Y + GEOMETRY_TOLERANCE)
        & (
            np.abs(midpoint_x)
            < midpoint_width - GEOMETRY_TOLERANCE
        )
    )
    return weld_fraction, crosses_groove


def integrate_straight_ray_time(
    x_start,
    y_start,
    x_end,
    y_end,
    wave_type='S',
):
    """Integrate slowness along straight rays through parent steel and weld/HAZ.

    Parameters
    ----------
    x_start : array-like
        Ray starting x coordinates in metres.
    y_start : array-like
        Ray starting y coordinates in metres.
    x_end : array-like
        Ray ending x coordinates in metres.
    y_end : array-like
        Ray ending y coordinates in metres.
    wave_type : {'P', 'S'}, default='S'
        Wave mode used to select parent and weld/HAZ speeds.

    Returns
    -------
    numpy.ndarray
        Integrated travel times in seconds. Invalid rays are NaN.

    Raises
    ------
    ValueError
        If ``wave_type`` is neither ``'P'`` nor ``'S'``.

    Notes
    -----
    Inputs are broadcast, allowing a scalar source or receiver to be evaluated
    against a complete image grid. Rays crossing the groove or leaving solid
    material are rejected.
    """
    wave_type = wave_type.upper()
    if wave_type == 'S':
        parent_speed, weld_speed = PARENT_S_SPEED, WELD_S_SPEED
    elif wave_type == 'P':
        parent_speed, weld_speed = PARENT_P_SPEED, WELD_P_SPEED
    else:
        raise ValueError("wave_type must be 'P' or 'S'.")

    x_start, y_start, x_end, y_end = np.broadcast_arrays(
        np.asarray(x_start, dtype=float),
        np.asarray(y_start, dtype=float),
        np.asarray(x_end, dtype=float),
        np.asarray(y_end, dtype=float),
    )
    path_length = np.hypot(x_end - x_start, y_end - y_start)
    weld_fraction, crosses_groove = _straight_ray_region_fractions(
        x_start, y_start, x_end, y_end
    )
    start_region = saft_geometry(x_start, y_start)[0]
    end_region = saft_geometry(x_end, y_end)[0]
    valid_path = (start_region > 0) & (end_region > 0) & ~crosses_groove

    travel_time = path_length * (
        weld_fraction / weld_speed + (1.0 - weld_fraction) / parent_speed
    )
    return np.where(valid_path, travel_time, np.nan)


def integrate_material_path_time(
    x_start,
    y_start,
    x_end,
    y_end,
    wave_type='S',
):
    """Return the shortest integrated path that remains inside the material.

    Parameters
    ----------
    x_start : array-like
        Path starting x coordinates in metres.
    y_start : array-like
        Path starting y coordinates in metres.
    x_end : array-like
        Path ending x coordinates in metres.
    y_end : array-like
        Path ending y coordinates in metres.
    wave_type : {'P', 'S'}, default='S'
        Wave mode used for travel-time integration.

    Returns
    -------
    numpy.ndarray
        Minimum valid material-path travel times in seconds. Paths with an
        endpoint outside solid material are NaN.

    Raises
    ------
    ValueError
        If ``wave_type`` is neither ``'P'`` nor ``'S'``.

    Notes
    -----
    A direct straight ray is used where possible. If the open groove blocks
    it, edge-diffracted paths around both lower groove corners are evaluated
    and the shorter valid time is selected.
    """
    x_start, y_start, x_end, y_end = np.broadcast_arrays(
        np.asarray(x_start, dtype=float),
        np.asarray(y_start, dtype=float),
        np.asarray(x_end, dtype=float),
        np.asarray(y_end, dtype=float),
    )
    direct_time = integrate_straight_ray_time(
        x_start, y_start, x_end, y_end, wave_type
    )
    result = np.array(direct_time, copy=True)
    blocked = ~np.isfinite(result)
    if not np.any(blocked):
        return result

    result_flat = result.reshape(-1)
    blocked_flat = blocked.reshape(-1)
    x_start_blocked = x_start.reshape(-1)[blocked_flat]
    y_start_blocked = y_start.reshape(-1)[blocked_flat]
    x_end_blocked = x_end.reshape(-1)[blocked_flat]
    y_end_blocked = y_end.reshape(-1)[blocked_flat]

    corner_times = []
    for corner_x in (-GROOVE_ROOT_HALF_WIDTH, GROOVE_ROOT_HALF_WIDTH):
        start_to_corner = integrate_straight_ray_time(
            x_start_blocked,
            y_start_blocked,
            corner_x,
            WELD_SURFACE_Y,
            wave_type,
        )
        corner_to_end = integrate_straight_ray_time(
            corner_x,
            WELD_SURFACE_Y,
            x_end_blocked,
            y_end_blocked,
            wave_type,
        )
        corner_times.append(start_to_corner + corner_to_end)

    result_flat[blocked_flat] = np.fmin(corner_times[0], corner_times[1])
    return result


def estimate_reference_time_offset(
    receiver_lines='sides',
    receiver_indices=None,
    wave_type='P',
    reference_dataset='reference',
    frequency_band=(4e6, 12e6),
    filter_order=4,
    offset_range=(-2e-6, 4e-6),
    offset_step=None,
    pick_half_width=0.25e-6,
    source_position=DEFAULT_LASER_SOURCE,
):
    """Estimate a global direct-arrival delay from the defect-free block.

    Parameters
    ----------
    receiver_lines : str or sequence of str, default='sides'
        Receiver line or lines used for reference calibration.
    receiver_indices : int, slice, array-like, mapping, or None
        Shared or per-line receiver-point selection.
    wave_type : {'P', 'S'}, default='P'
        Wave mode used to predict source-to-receiver travel times.
    reference_dataset : str or mapping, default='reference'
        Defect-free dataset name or custom line-to-file mapping.
    frequency_band : tuple of float, default=(4e6, 12e6)
        Band-pass cutoff frequencies in hertz.
    filter_order : int, default=4
        Butterworth filter order.
    offset_range : tuple of float, default=(-2e-6, 4e-6)
        Inclusive global-delay search interval in seconds.
    offset_step : float or None
        Search increment in seconds. ``None`` uses the signal sample interval.
    pick_half_width : float, default=0.25e-6
        Half-width around the fitted curve used for local envelope picks.
    source_position : tuple of float, default=DEFAULT_LASER_SOURCE
        Fixed laser ``(x, y)`` coordinates in metres.

    Returns
    -------
    ReferenceArrivalCalibration
        Fitted delay, curve score, local arrival picks, and residual error.

    Notes
    -----
    Each trace is RMS-normalized before scoring so a few high-amplitude
    receivers cannot determine the fit. Only defect-free signals and known
    source/receiver geometry are used. The score follows the strongest
    compatible envelope ridge. In the supplied side-line traces, the fitted
    P-wave ridge may include mode-converted or S-wave energy. Inspect the
    picks and physical path before applying its offset to an image.
    """
    data = load_receiver_data(
        receiver_lines, receiver_indices, dataset=reference_dataset
    )
    filtered_signals = _bandpass_signals(
        data.time_series_matrix,
        data.times,
        frequency_band,
        filter_order,
    )
    envelope = np.abs(hilbert(filtered_signals, axis=-1))
    trace_rms = np.sqrt(np.mean(filtered_signals**2, axis=1))
    normalized_envelope = np.divide(
        envelope,
        trace_rms[:, None],
        out=np.zeros_like(envelope),
        where=trace_rms[:, None] > np.finfo(float).eps,
    )

    source_position = np.asarray(source_position, dtype=float)
    if source_position.shape != (2,) or not np.isfinite(source_position).all():
        raise ValueError("source_position must contain finite (x, y) coordinates.")
    predicted_times = integrate_material_path_time(
        source_position[0],
        source_position[1],
        data.x_coords,
        data.y_coords,
        wave_type,
    )
    valid_predictions = np.isfinite(predicted_times)
    if not np.any(valid_predictions):
        raise ValueError("No valid direct paths exist for the selected receivers.")

    if len(offset_range) != 2:
        raise ValueError("offset_range must contain exactly two values.")
    offset_start, offset_stop = (float(value) for value in offset_range)
    if offset_stop < offset_start:
        raise ValueError("offset_range must be increasing.")
    sample_interval = float(np.median(np.diff(data.times)))
    if offset_step is None:
        offset_step = sample_interval
    offset_step = float(offset_step)
    if not np.isfinite(offset_step) or offset_step <= 0:
        raise ValueError("offset_step must be finite and greater than zero.")

    offsets = np.arange(
        offset_start,
        offset_stop + 0.5 * offset_step,
        offset_step,
    )
    scores = np.zeros(offsets.size)
    valid_rows = np.flatnonzero(valid_predictions)
    for offset_index, offset in enumerate(offsets):
        samples = [
            np.interp(
                predicted_times[row] + offset,
                data.times,
                normalized_envelope[row],
                left=0.0,
                right=0.0,
            )
            for row in valid_rows
        ]
        scores[offset_index] = np.mean(samples)

    best_index = int(np.argmax(scores))
    time_offset = float(offsets[best_index])
    fitted_times = predicted_times + time_offset
    picked_times = np.full(predicted_times.shape, np.nan)
    for row in valid_rows:
        local = (
            (data.times >= fitted_times[row] - pick_half_width)
            & (data.times <= fitted_times[row] + pick_half_width)
        )
        if np.any(local):
            local_indices = np.flatnonzero(local)
            picked_times[row] = data.times[
                local_indices[np.argmax(envelope[row, local])]
            ]

    residuals = picked_times - fitted_times
    residual_rms = float(np.sqrt(np.nanmean(residuals**2)))
    return ReferenceArrivalCalibration(
        data=data,
        filtered_signals=filtered_signals,
        envelope=envelope,
        wave_type=str(wave_type).upper(),
        predicted_times=predicted_times,
        picked_times=picked_times,
        time_offset=time_offset,
        score=float(scores[best_index]),
        residual_rms=residual_rms,
    )




def _make_spatial_axis(coordinate_range, pixel_size, axis_name):
    """Create a spatial axis with a bounded sample spacing.

    Parameters
    ----------
    coordinate_range : tuple of float
        Inclusive ``(start, stop)`` coordinates in metres.
    pixel_size : float
        Maximum permitted spacing in metres.
    axis_name : str
        Axis label used in validation messages.

    Returns
    -------
    numpy.ndarray
        Uniform coordinate axis including both range endpoints.

    Raises
    ------
    ValueError
        If the range is malformed, non-finite, or non-increasing.
    """
    if len(coordinate_range) != 2:
        raise ValueError(f"{axis_name}_range must contain exactly two values.")

    start, stop = (float(value) for value in coordinate_range)
    if not np.isfinite([start, stop]).all() or stop <= start:
        raise ValueError(
            f"{axis_name}_range must contain finite values in increasing order."
        )

    span = stop - start
    interval_ratio = span / pixel_size
    nearest_integer = round(interval_ratio)
    if np.isclose(interval_ratio, nearest_integer, rtol=1e-12, atol=1e-12):
        num_intervals = max(1, nearest_integer)
    else:
        num_intervals = max(1, int(np.ceil(interval_ratio)))

    return np.linspace(start, stop, num_intervals + 1)


def create_imaging_grid(
    x_range=(-0.025, 0.025),
    y_range=(0.0, 0.0198),
    pixel_size=0.0001,
    grid_res=None,
):
    """Create a SAFT grid using square physical pixels by default.

    Parameters
    ----------
    x_range : tuple of float, default=(-0.025, 0.025)
        Inclusive horizontal imaging range in metres.
    y_range : tuple of float, default=(0.0, 0.0198)
        Inclusive vertical imaging range in metres.
    pixel_size : float, default=0.0001
        Maximum physical spacing in metres on both axes.
    grid_res : int or None
        Legacy equal-sample-count resolution. When supplied, it overrides
        ``pixel_size`` and uses the same number of samples on both axes.

    Returns
    -------
    GX : numpy.ndarray
        Two-dimensional grid of x coordinates in metres.
    GY : numpy.ndarray
        Two-dimensional grid of y coordinates in metres.

    Raises
    ------
    TypeError
        If ``grid_res`` is not an integer or ``pixel_size`` is not scalar.
    ValueError
        If the requested resolution, pixel size, or coordinate ranges are
        invalid.
    """
    if grid_res is not None:
        if isinstance(grid_res, bool) or not isinstance(grid_res, (int, np.integer)):
            raise TypeError("grid_res must be an integer or None.")
        if grid_res < 2:
            raise ValueError("grid_res must be at least 2.")
        gx = np.linspace(x_range[0], x_range[1], grid_res)
        gy = np.linspace(y_range[0], y_range[1], grid_res)
    else:
        if isinstance(pixel_size, bool) or not np.isscalar(pixel_size):
            raise TypeError("pixel_size must be one number specified in metres.")
        pixel_size = float(pixel_size)
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError("pixel_size must be a finite value greater than zero.")
        gx = _make_spatial_axis(x_range, pixel_size, 'x')
        gy = _make_spatial_axis(y_range, pixel_size, 'y')

    return np.meshgrid(gx, gy)


def _finalise_saft_component(
    coherent_sum,
    incoherent_energy,
    contribution_count,
    image_mode,
    minimum_contributions,
    normalise_by_aperture,
):
    """Turn delayed receiver sums into one SAFT image component.

    Parameters
    ----------
    coherent_sum : numpy.ndarray
        Sum of real or analytic delayed receiver samples.
    incoherent_energy : numpy.ndarray or None
        Sum of squared analytic-signal magnitudes for coherence weighting.
    contribution_count : numpy.ndarray or None
        Number of valid delayed samples contributing to each pixel.
    image_mode : {'signed', 'envelope', 'coherence'}
        Requested image representation.
    minimum_contributions : int
        Minimum number of valid delayed samples required at a pixel.
    normalise_by_aperture : bool
        Whether to divide coherent amplitude by its valid contribution count.

    Returns
    -------
    numpy.ndarray
        Signed amplitude, analytic envelope, or coherence-weighted envelope.

    Notes
    -----
    The coherence factor is ``|sum(s)|**2 / (N * sum(|s|**2))``, where ``N``
    is the number of receivers with a valid travel time at the pixel.
    """
    if normalise_by_aperture:
        coherent_amplitude = np.divide(
            coherent_sum,
            contribution_count,
            out=np.zeros_like(coherent_sum),
            where=contribution_count > 0,
        )
    else:
        coherent_amplitude = coherent_sum

    if image_mode == 'signed':
        image = np.asarray(coherent_amplitude.real)
        return np.where(contribution_count >= minimum_contributions, image, 0.0)

    envelope = np.abs(coherent_amplitude)
    if image_mode == 'envelope':
        return np.where(
            contribution_count >= minimum_contributions, envelope, 0.0
        )

    denominator = contribution_count * incoherent_energy
    coherence = np.divide(
        np.abs(coherent_sum) ** 2,
        denominator,
        out=np.zeros_like(incoherent_energy),
        where=denominator > np.finfo(float).eps,
    )
    image = envelope * np.clip(coherence, 0.0, 1.0)
    return np.where(
        contribution_count >= minimum_contributions, image, 0.0
    )


def perform_segmented_saft(
    receiver_lines='left',
    x_range=(-0.025, 0.025),
    y_range=(0.0, 0.0198),
    grid_res=None,
    wave_type='S',
    receiver_indices=None,
    pixel_size=0.0001,
    source_position=DEFAULT_LASER_SOURCE,
    dataset='defect',
    reference_dataset=None,
    frequency_band=None,
    filter_order=4,
    align_reference=False,
    scale_reference=False,
    calibration_window=None,
    max_shift_samples=10,
    return_components=False,
    image_mode='signed',
    minimum_aperture_fraction=0.5,
    normalise_by_aperture=True,
    time_offset=0.0,
    source_wave_type=None,
    receiver_wave_type=None,
):
    """Run segmented SAFT for any receiver-line and point combination.

    Parameters
    ----------
    receiver_lines : str or sequence of str, default='left'
        Receiver line or lines used in the reconstruction.
    x_range : tuple of float, default=(-0.025, 0.025)
        Inclusive horizontal imaging range in metres.
    y_range : tuple of float, default=(0.0, 0.0198)
        Inclusive vertical imaging range in metres.
    grid_res : int or None
        Legacy number of samples per axis. When supplied, it overrides
        ``pixel_size``.
    wave_type : {'P', 'S'}, default='S'
        Default wave mode for both propagation legs. Leg-specific arguments
        override it when modelling mode conversion.
    receiver_indices : int, slice, array-like, mapping, or None
        Shared or per-line receiver-point selection.
    pixel_size : float, default=0.0001
        Maximum x/y grid spacing in metres when ``grid_res`` is ``None``.
    source_position : tuple of float, default=DEFAULT_LASER_SOURCE
        Fixed laser ``(x, y)`` coordinates in metres.
    dataset : str or mapping, default='defect'
        Signal dataset name or custom line-to-file mapping. For calibrated
        reconstruction this is the defective dataset.
    reference_dataset : str, mapping, or None
        Optional defect-free dataset. When supplied, SAFT uses calibrated
        defective-minus-reference signals.
    frequency_band : tuple of float or None
        Optional band-pass cutoff frequencies in hertz.
    filter_order : int, default=4
        Butterworth filter order.
    align_reference : bool, default=False
        Whether to align reference traces before subtraction.
    scale_reference : bool, default=False
        Whether to scale reference traces before subtraction.
    calibration_window : tuple of float, mapping, or None
        Shared or per-line time window used for reference alignment/scaling.
    max_shift_samples : int, default=10
        Maximum absolute sample lag considered during alignment.
    return_components : bool, default=False
        If ``True``, return a dictionary containing the direct, receiver-leg
        backwall, and combined images instead of only the combined image.
    image_mode : {'signed', 'envelope', 'coherence'}, default='signed'
        Image representation. ``'signed'`` reproduces real delay-and-sum,
        ``'envelope'`` takes the magnitude of the coherently summed analytic
        signals, and ``'coherence'`` additionally applies the conventional
        coherence factor across receiver contributions.
    minimum_aperture_fraction : float, default=0.5
        Minimum fraction of selected receivers that must provide an in-range,
        physically valid travel-time sample at a pixel. Pixels below this
        coverage are set to zero.
    normalise_by_aperture : bool, default=True
        Divide each coherent sum by its valid receiver count. This prevents
        receiver count and spatially varying path coverage from controlling
        image brightness.
    time_offset : float, default=0.0
        Global delay in seconds added to every modelled propagation time.
        This can be estimated from the defect-free block with
        :func:`estimate_reference_time_offset`.
    source_wave_type : {'P', 'S'} or None
        Wave mode on the laser-to-pixel leg. ``None`` uses ``wave_type``.
    receiver_wave_type : {'P', 'S'} or None
        Wave mode on the pixel-to-receiver leg, including the optional
        backwall reflection. ``None`` uses ``wave_type``.
    Returns
    -------
    GX : numpy.ndarray
        Two-dimensional image-grid x coordinates in metres.
    GY : numpy.ndarray
        Two-dimensional image-grid y coordinates in metres.
    saft_image : numpy.ndarray or dict of numpy.ndarray
    Combined image in the selected ``image_mode`` when ``return_components``
    is ``False``. Otherwise, a dictionary with ``'direct'``, ``'backwall'``,
    and ``'combined'`` images.

    Raises
    ------
    ValueError
        If the source, wave type, grid, receiver selection, datasets, or
        calibration configuration is invalid.

    Notes
    -----
    The direct delay is the source-to-pixel time plus the pixel-to-receiver
    time and ``time_offset``. The back-wall delay replaces the receiver leg
    with pixel-to-bounce plus bounce-to-receiver travel times. Each leg may
    have a different P or S mode. Slowness is integrated through parent and
    weld/HAZ material; a groove-blocked path goes around the shorter lower
    groove corner. Refraction and diffraction amplitude losses are not
    modelled.
    """
    source_position = np.asarray(source_position, dtype=float)
    if source_position.shape != (2,) or not np.isfinite(source_position).all():
        raise ValueError("source_position must contain finite (x, y) coordinates.")
    source_x, source_y = source_position
    image_mode = str(image_mode).lower()
    valid_image_modes = {'signed', 'envelope', 'coherence'}
    if image_mode not in valid_image_modes:
        raise ValueError(
            "image_mode must be 'signed', 'envelope', or 'coherence'."
        )
    minimum_aperture_fraction = float(minimum_aperture_fraction)
    if not 0.0 <= minimum_aperture_fraction <= 1.0:
        raise ValueError(
            "minimum_aperture_fraction must lie between 0 and 1."
        )
    time_offset = float(time_offset)
    if not np.isfinite(time_offset):
        raise ValueError("time_offset must be finite.")
    wave_type = str(wave_type).upper()
    source_wave_type = (
        wave_type if source_wave_type is None else str(source_wave_type).upper()
    )
    receiver_wave_type = (
        wave_type
        if receiver_wave_type is None
        else str(receiver_wave_type).upper()
    )
    for leg_name, leg_wave_type in (
        ('source_wave_type', source_wave_type),
        ('receiver_wave_type', receiver_wave_type),
    ):
        if leg_wave_type not in {'P', 'S'}:
            raise ValueError(f"{leg_name} must be 'P', 'S', or None.")

    if reference_dataset is None:
        if align_reference or scale_reference:
            raise ValueError(
                "align_reference and scale_reference require reference_dataset."
            )
        data = load_receiver_data(
            receiver_lines, receiver_indices, dataset=dataset
        )
        if frequency_band is not None:
            data = _with_signals(
                data,
                _bandpass_signals(
                    data.time_series_matrix,
                    data.times,
                    frequency_band,
                    filter_order,
                ),
            )
    else:
        calibration_result = load_calibrated_receiver_data(
            receiver_lines=receiver_lines,
            receiver_indices=receiver_indices,
            defect_dataset=dataset,
            reference_dataset=reference_dataset,
            frequency_band=frequency_band,
            filter_order=filter_order,
            align_reference=align_reference,
            scale_reference=scale_reference,
            calibration_window=calibration_window,
            max_shift_samples=max_shift_samples,
        )
        data = calibration_result.difference
    x_coords = data.x_coords
    y_coords = data.y_coords
    analytic_mode = image_mode in {'envelope', 'coherence'}
    interpolation_signals = (
        hilbert(data.time_series_matrix, axis=-1)
        if analytic_mode
        else data.time_series_matrix
    )
    interp_funcs = [
        interp1d(data.times, signal, bounds_error=False, fill_value=0.0)
        for signal in interpolation_signals
    ]

    # The spatial grid represents possible reflector locations.
    GX, GY = create_imaging_grid(x_range, y_range, pixel_size, grid_res)

    accumulator_dtype = complex if analytic_mode else float
    direct_sum = np.zeros(GX.shape, dtype=accumulator_dtype)
    backwall_sum = np.zeros(GX.shape, dtype=accumulator_dtype)
    direct_count = np.zeros(GX.shape, dtype=np.uint16)
    backwall_count = np.zeros(GX.shape, dtype=np.uint16)
    if image_mode == 'coherence':
        direct_energy = np.zeros(GX.shape)
        backwall_energy = np.zeros(GX.shape)
    else:
        direct_energy = backwall_energy = None
    num_receivers = len(x_coords)
    
    # The source-to-pixel travel time is shared by every receiver.
    source_times = integrate_material_path_time(
        source_x, source_y, GX, GY, source_wave_type
    )

    # Sum each receiver signal at the predicted source-pixel-receiver delay.
    for r in range(num_receivers):
        rx, ry = x_coords[r], y_coords[r]

        # Direct bistatic path: laser source -> pixel -> receiver.
        receiver_times = integrate_material_path_time(
            GX, GY, rx, ry, receiver_wave_type
        )
        direct_times = source_times + receiver_times + time_offset
        valid_direct = (
            np.isfinite(direct_times)
            & (direct_times >= data.times[0])
            & (direct_times <= data.times[-1])
        )
        if np.any(valid_direct):
            delayed_samples = interp_funcs[r](direct_times[valid_direct])
            direct_sum[valid_direct] += delayed_samples
            direct_count[valid_direct] += 1
            if image_mode == 'coherence':
                direct_energy[valid_direct] += np.abs(delayed_samples) ** 2

        # Back-wall path on the receiver leg. The straight line from the pixel
        # to the mirrored receiver intersects y=0 at the physical bounce point.
        bounce_fraction = np.divide(
            GY,
            GY + ry,
            out=np.zeros_like(GY),
            where=np.abs(GY + ry) > np.finfo(float).eps,
        )
        bounce_x = GX + bounce_fraction * (rx - GX)
        bounce_y = np.zeros_like(bounce_x)
        pixel_to_bounce_times = integrate_material_path_time(
            GX, GY, bounce_x, bounce_y, receiver_wave_type
        )
        bounce_to_receiver_times = integrate_material_path_time(
            bounce_x, bounce_y, rx, ry, receiver_wave_type
        )
        bounce_times = (
            source_times + pixel_to_bounce_times + bounce_to_receiver_times
            + time_offset
        )
        valid_bounce = (
            np.isfinite(bounce_times)
            & (bounce_times >= data.times[0])
            & (bounce_times <= data.times[-1])
        )
        if np.any(valid_bounce):
            delayed_samples = interp_funcs[r](bounce_times[valid_bounce])
            backwall_sum[valid_bounce] += delayed_samples
            backwall_count[valid_bounce] += 1
            if image_mode == 'coherence':
                backwall_energy[valid_bounce] += np.abs(delayed_samples) ** 2

    minimum_contributions = int(
        np.ceil(minimum_aperture_fraction * num_receivers)
    )

    direct_image = _finalise_saft_component(
        direct_sum,
        direct_energy,
        direct_count,
        image_mode,
        minimum_contributions,
        normalise_by_aperture,
    )
    backwall_image = _finalise_saft_component(
        backwall_sum,
        backwall_energy,
        backwall_count,
        image_mode,
        minimum_contributions,
        normalise_by_aperture,
    )
    saft_image = _finalise_saft_component(
        direct_sum + backwall_sum,
        None if direct_energy is None else direct_energy + backwall_energy,
        direct_count + backwall_count,
        image_mode,
        minimum_contributions,
        normalise_by_aperture,
    )

    if return_components:
        return GX, GY, {
            'direct': direct_image,
            'backwall': backwall_image,
            'combined': saft_image,
        }
    return GX, GY, saft_image


def compare_saft_apertures(
    receiver_line='right',
    mode='PP',
    component='direct',
    x_range=(-0.008, 0.008),
    y_range=(0.0, 0.009),
    pixel_size=0.00025,
    time_offset=0.0,
    frequency_band=(2e6, 8e6),
):
    """Check whether a SAFT focus persists in independent receiver subsets.

    Parameters
    ----------
    receiver_line : {'left', 'right', 'weld'}, default='right'
        One COMSOL receiver line to reconstruct.
    mode : {'PP', 'PS', 'SP', 'SS'}, default='PP'
        Propagation mode on the source and receiver legs, respectively.
    component : {'direct', 'backwall'}, default='direct'
        Scattering path to inspect without mixing two path hypotheses.
    x_range : tuple of float, default=(-0.008, 0.008)
        Horizontal image bounds in metres.
    y_range : tuple of float, default=(0.0, 0.009)
        Vertical image bounds in metres.
    pixel_size : float, default=0.00025
        Maximum image-pixel spacing in metres.
    time_offset : float, default=0.0
        Explicit arrival-time shift in seconds. The strongest side-line pulse
        must not be used automatically as a P-wave timing calibration.
    frequency_band : tuple of float or None, default=(2e6, 8e6)
        Common filtering band for all three reconstructions.

    Returns
    -------
    gx, gy : numpy.ndarray
        Image coordinates in metres.
    images : dict of str to numpy.ndarray
        Full, even-index, and odd-index SAFT envelope images.
    peaks : dict of str to tuple of float
        Coordinates of each strongest pixel in metres, or NaN for a zero image.
    split_separation : float
        Distance between even and odd image peaks in metres.

    Raises
    ------
    ValueError
        If the line, wave mode, or scattering component is unsupported.

    Notes
    -----
    The split tests focus stability under a change in receiver aperture. It
    cannot establish that a peak is a defect: groove scattering, an incorrect
    path model, and transmission shadow can also persist in both halves.
    """
    if receiver_line not in ('left', 'right', 'weld'):
        raise ValueError("receiver_line must be 'left', 'right', or 'weld'.")
    mode = str(mode).upper()
    if mode not in ('PP', 'PS', 'SP', 'SS'):
        raise ValueError("mode must be 'PP', 'PS', 'SP', or 'SS'.")
    if component not in ('direct', 'backwall'):
        raise ValueError("component must be 'direct' or 'backwall'.")

    receiver_count = len(load_receiver_data(receiver_line).x_coords)
    if receiver_count < 4:
        raise ValueError('At least four receivers are required for an aperture split.')
    selections = {
        'all': np.arange(receiver_count),
        'even': np.arange(0, receiver_count, 2),
        'odd': np.arange(1, receiver_count, 2),
    }
    images = {}
    peaks = {}
    gx = gy = None
    for name, indices in selections.items():
        gx, gy, components = perform_segmented_saft(
            receiver_lines=receiver_line,
            receiver_indices=indices,
            x_range=x_range,
            y_range=y_range,
            pixel_size=pixel_size,
            source_wave_type=mode[0],
            receiver_wave_type=mode[1],
            dataset='defect',
            reference_dataset='reference',
            frequency_band=frequency_band,
            time_offset=time_offset,
            image_mode='envelope',
            return_components=True,
        )
        images[name] = components[component]
        if np.max(images[name]) > 0:
            row, column = np.unravel_index(np.argmax(images[name]), gx.shape)
            peaks[name] = (float(gx[row, column]), float(gy[row, column]))
        else:
            peaks[name] = (np.nan, np.nan)

    split_separation = float(np.hypot(
        peaks['even'][0] - peaks['odd'][0],
        peaks['even'][1] - peaks['odd'][1],
    ))
    return gx, gy, images, peaks, split_separation


def main():
    """Run numerical SAFT on the defect-minus-reference receiver signals.

    Notes
    -----
    The command uses the chosen source and receiver wave modes to calculate
    travel times, then reports the strongest pixel in each SAFT component.
    ``--validate`` first compares reference-picked early pulses on both sides,
    then tests whether a selected SAFT focus survives even/odd receiver
    splitting. ``--output`` saves the chosen grid and images.
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--line', choices=('left', 'right', 'weld', 'sides'), default='right')
    parser.add_argument('--mode', choices=('PP', 'PS', 'SP', 'SS'), default='SS')
    parser.add_argument('--pixel-size-mm', type=float, default=0.25)
    parser.add_argument('--validate', action='store_true', help='Compare early arrivals and split-aperture SAFT')
    parser.add_argument('--component', choices=('direct', 'backwall'), default='direct')
    parser.add_argument('--time-offset-us', type=float, default=0.0)
    parser.add_argument('--output', type=Path, help='Optional NPZ file for the grid and SAFT components')
    args = parser.parse_args()
    if args.pixel_size_mm <= 0:
        parser.error('--pixel-size-mm must be positive')
    if args.validate:
        if args.line == 'sides':
            parser.error('--validate needs one receiver line for aperture splitting')
        comparison = compare_reference_arrivals('sides')
        for line_name in ('left', 'right'):
            selected = comparison.data.line_names == line_name
            ratios = comparison.amplitude_ratios[selected]
            print(
                f'{line_name} early pulse: median defect/reference RMS '
                f'={np.nanmedian(ratios):.3f}, '
                f'10--90% range={np.nanpercentile(ratios, 10):.3f}--'
                f'{np.nanpercentile(ratios, 90):.3f}'
            )
        gx, gy, images, peaks, separation = compare_saft_apertures(
            receiver_line=args.line,
            mode=args.mode,
            component=args.component,
            pixel_size=args.pixel_size_mm / 1000,
            time_offset=args.time_offset_us * 1e-6,
        )
        print(f'{args.mode} {args.component} split-aperture peak separation: {separation * 1000:.3f} mm')
        for name, position in peaks.items():
            print(f'{name}: x={position[0] * 1000:.3f} mm, y={position[1] * 1000:.3f} mm')
        if args.output:
            np.savez_compressed(
                args.output, x=gx, y=gy, **images,
                mode=args.mode, component=args.component,
                time_offset_s=args.time_offset_us * 1e-6,
                left_ratios=comparison.amplitude_ratios[
                    comparison.data.line_names == 'left'
                ],
                right_ratios=comparison.amplitude_ratios[
                    comparison.data.line_names == 'right'
                ],
                left_receiver_x=comparison.data.x_coords[
                    comparison.data.line_names == 'left'
                ],
                right_receiver_x=comparison.data.x_coords[
                    comparison.data.line_names == 'right'
                ],
            )
            print(f'Saved {args.output}')
        return

    calibration = load_calibrated_receiver_data(args.line)
    residual_rms = np.sqrt(np.mean(calibration.difference.time_series_matrix ** 2))
    print(
        f'{args.line}: {len(calibration.difference.x_coords)} receivers, '
        f'{calibration.difference.times.size} samples, '
        f'residual RMS={residual_rms:.6e} m/s'
    )
    print(f'{args.mode[0]}->{args.mode[1]} paths; pixel size {args.pixel_size_mm:g} mm')
    gx, gy, components = perform_segmented_saft(
        receiver_lines=args.line,
        reference_dataset='reference',
        source_wave_type=args.mode[0],
        receiver_wave_type=args.mode[1],
        pixel_size=args.pixel_size_mm / 1000,
        image_mode='envelope',
        return_components=True,
    )
    for name, image in components.items():
        row, column = np.unravel_index(np.argmax(image), image.shape)
        print(
            f'{name}: peak={image[row, column]:.6e} at '
            f'x={gx[row, column] * 1000:.3f} mm, '
            f'y={gy[row, column] * 1000:.3f} mm'
        )
    if args.output:
        np.savez_compressed(args.output, x=gx, y=gy, **components)
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
