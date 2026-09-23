"""Plot measured waveforms and SAFT images for the partial weld.

Notes
-----
A-scans show velocity versus time at one receiver. B-scans show arrivals
across a receiver line. SAFT panels show how direct and back-wall paths focus
the defect-minus-reference signals in the pipe-wall cross section. Their
dashed cyan outline shows the known defect from the saved COMSOL geometry.
"""

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyse import (
    DEFAULT_LASER_SOURCE,
    compare_reference_arrivals,
    compare_saft_apertures,
    load_calibrated_receiver_data,
    load_receiver_data,
    perform_segmented_saft,
    saft_geometry,
)


DEFECT_MODEL = Path(__file__).resolve().parent / 'weld_defect_v1.mph'


def load_comsol_defect_outline(model_path=DEFECT_MODEL):
    """Read the built lack-of-fusion polygon from the defective COMSOL model.

    Parameters
    ----------
    model_path : pathlib.Path or str, default=DEFECT_MODEL
        Saved defective COMSOL model. Its geometry must have been built.

    Returns
    -------
    numpy.ndarray
        Closed polygon vertices with shape ``(vertices, 2)`` in metres.

    Raises
    ------
    ValueError
        If the saved model lacks the built defect polygon or uses an
        unsupported geometry length unit.

    Notes
    -----
    COMSOL stores the evaluated polygon vertices in ``dmodel.xml`` inside the
    MPH archive. Reading those vertices keeps the display tied to the actual
    saved geometry. The outline is only drawn after reconstruction and is
    never supplied to the SAFT travel-time model.
    """
    with zipfile.ZipFile(model_path) as archive:
        model = ET.fromstring(archive.read('dmodel.xml'))

    geometry = next(
        (item for item in model.iter('GeomSequence') if item.get('tag') == 'geom1'),
        None,
    )
    if geometry is None:
        raise ValueError('The COMSOL model has no geom1 geometry.')
    unit = geometry.findtext('lengthUnit')
    if unit not in ('mm', 'm'):
        raise ValueError(f'Unsupported COMSOL geometry length unit: {unit!r}.')

    feature = next(
        (item for item in geometry.iter('GeomFeature')
         if item.get('name') == 'Right sidewall lack of fusion'),
        None,
    )
    if feature is None:
        raise ValueError('The COMSOL model has no right sidewall defect feature.')
    if feature.findtext('buildStatus') != 'BUILT':
        raise ValueError('Build and save the COMSOL defect geometry before plotting.')
    polygon = next(
        (item for item in feature.findall('propertyValue')
         if item.get('name') == 'p:segvtxvalid'),
        None,
    )
    if polygon is None:
        raise ValueError('Build and save the COMSOL defect geometry before plotting.')
    coordinates = re.findall(
        r"\|2,'([^']+)','([^']+)'", polygon.get('valueMatrix', '')
    )
    if len(coordinates) < 4:
        raise ValueError('The COMSOL defect polygon has too few vertices.')
    vertices = np.asarray(coordinates, dtype=float)
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack((vertices, vertices[0]))
    return vertices * (0.001 if unit == 'mm' else 1.0)


def _draw_comsol_defect(ax, outline, scale=1.0):
    """Mark the known COMSOL defect on a localization axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Image axis using x and y positions from the pipe-wall cross section.
    outline : numpy.ndarray
        Closed defect polygon in metres.
    scale : float, default=1.0
        Conversion factor from metres to the displayed axis units.
    """
    ax.plot(
        outline[:, 0] * scale,
        outline[:, 1] * scale,
        color='cyan',
        linestyle='--',
        linewidth=2,
        label='COMSOL defect (known)',
    )
    ax.legend(loc='upper right', fontsize='small')


def _symmetric_colour_limit(signals, percentile):
    """Calculate a robust symmetric color limit for signed signal data.

    Parameters
    ----------
    signals : array-like
        Signed values displayed in an image.
    percentile : float or None
        Percentile of absolute amplitude used as the limit. ``None`` uses the
        absolute maximum.

    Returns
    -------
    float
        Positive color limit suitable for ``vmin=-limit`` and ``vmax=limit``.

    Raises
    ------
    ValueError
        If ``percentile`` is outside the interval ``(0, 100]``.
    """
    absolute_signals = np.abs(np.asarray(signals))
    if percentile is None:
        colour_limit = np.max(absolute_signals)
    else:
        percentile = float(percentile)
        if not 0.0 < percentile <= 100.0:
            raise ValueError("colour_percentile must be in the interval (0, 100].")
        colour_limit = np.percentile(absolute_signals, percentile)
    return max(float(colour_limit), np.finfo(float).eps)


def _format_index_ranges(indices):
    """Format integer indices as compact inclusive ranges.

    Parameters
    ----------
    indices : array-like
        Integer point indices to format.

    Returns
    -------
    str
        Comma-separated indices and ranges, such as ``'0-4, 8, 10-12'``.
    """
    sorted_indices = np.sort(np.asarray(indices, dtype=int))
    ranges = []
    range_start = range_end = int(sorted_indices[0])
    for index in sorted_indices[1:]:
        index = int(index)
        if index == range_end + 1:
            range_end = index
            continue
        ranges.append(
            str(range_start)
            if range_start == range_end
            else f'{range_start}-{range_end}'
        )
        range_start = range_end = index
    ranges.append(
        str(range_start)
        if range_start == range_end
        else f'{range_start}-{range_end}'
    )
    return ', '.join(ranges)


def _receiver_selection_label(data):
    """Describe receiver lines and original point indices in loaded data.

    Parameters
    ----------
    data : ReceiverData
        Receiver data whose selection should be described.

    Returns
    -------
    str
        Semicolon-separated line and compact index descriptions.
    """
    labels = []
    for line_name in dict.fromkeys(data.line_names):
        line_indices = data.point_indices[data.line_names == line_name]
        labels.append(
            f'{line_name} indices {_format_index_ranges(line_indices)}'
        )
    return '; '.join(labels)


def _label_bscan_receiver_axis(ax, data, max_ticks=9):
    """Label B-scan rows with receiver identity and horizontal position.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        B-scan axes to label.
    data : ReceiverData
        Receiver data represented by the image rows.
    max_ticks : int, default=9
        Approximate maximum number of regularly spaced labels. The first and
        last receiver of every selected line are always included.
    """
    num_rows = len(data.point_indices)
    if num_rows == 0:
        return

    if num_rows <= max_ticks:
        tick_rows = np.arange(num_rows)
    else:
        tick_rows = np.rint(
            np.linspace(0, num_rows - 1, max_ticks)
        ).astype(int)
        line_starts = np.r_[
            0,
            np.flatnonzero(data.line_names[1:] != data.line_names[:-1]) + 1,
        ]
        line_ends = np.r_[line_starts[1:] - 1, num_rows - 1]
        tick_rows = np.unique(np.r_[tick_rows, line_starts, line_ends])

    tick_labels = [
        f'{data.line_names[row]} #{data.point_indices[row]}\n'
        f'x={data.x_coords[row] * 1e3:.2f} mm'
        for row in tick_rows
    ]
    ax.set_yticks(tick_rows, labels=tick_labels)
    ax.set_ylabel('Receiver line/index and x position')


def plot_comsol_ascan(
    receiver_lines='left',
    receiver_indices=0,
    scale_to_microseconds=True,
    dataset='defect',
):
    """Plot selected A-scans from one or more receiver lines.

    Parameters
    ----------
    receiver_lines : str or sequence of str, default='left'
        Receiver line or lines to plot.
    receiver_indices : int, slice, array-like, mapping, or None, default=0
        Shared or per-line receiver-point selection.
    scale_to_microseconds : bool, default=True
        Display time in microseconds instead of seconds.
    dataset : str or mapping, default='defect'
        Dataset name or custom line-to-file mapping.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    ax : matplotlib.axes.Axes
        A-scan axes.
    """
    data = load_receiver_data(receiver_lines, receiver_indices, dataset=dataset)

    if scale_to_microseconds:
        plot_times = data.times * 1e6
        xlabel = 'Time [$\\mu$s]'
    else:
        plot_times = data.times
        xlabel = 'Time [s]'

    fig, ax = plt.subplots(figsize=(10, 4))
    for signal, line_name, point_index in zip(
        data.time_series_matrix, data.line_names, data.point_indices
    ):
        ax.plot(
            plot_times,
            signal,
            linewidth=1,
            label=f'{line_name} #{point_index}',
        )

    ax.set_title(f'A-Scan Response — {_receiver_selection_label(data)}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Velocity y-component [m/s]')
    ax.grid(True, linestyle='--', alpha=0.6)
    if len(data.point_indices) <= 15:
        ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_comsol_bscan(
    receiver_lines='left',
    receiver_indices=None,
    scale_to_microseconds=True,
    cmap='seismic',
    dataset='defect',
    colour_percentile=99.5,
):
    """Plot a B-scan from selected receiver lines and points.

    Parameters
    ----------
    receiver_lines : str or sequence of str, default='left'
        Receiver line or lines to plot.
    receiver_indices : int, slice, array-like, mapping, or None
        Shared or per-line receiver-point selection. ``None`` selects all.
    scale_to_microseconds : bool, default=True
        Display time in microseconds instead of seconds.
    cmap : str, default='seismic'
        Matplotlib colormap name.
    dataset : str or mapping, default='defect'
        Dataset name or custom line-to-file mapping.
    colour_percentile : float or None, default=99.5
        Absolute-amplitude percentile used for symmetric color limits. ``None``
        uses the full amplitude range.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    ax : matplotlib.axes.Axes
        B-scan axes.
    """
    data = load_receiver_data(receiver_lines, receiver_indices, dataset=dataset)

    if scale_to_microseconds:
        plot_times = data.times * 1e6
        xlabel = 'Time [$\\mu$s]'
    else:
        plot_times = data.times
        xlabel = 'Time [s]'

    colour_limit = _symmetric_colour_limit(
        data.time_series_matrix, colour_percentile
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(
        data.time_series_matrix,
        aspect='auto',
        extent=[plot_times[0], plot_times[-1], len(data.point_indices) - 0.5, -0.5],
        cmap=cmap,
        vmin=-colour_limit,
        vmax=colour_limit,
    )

    # Mark boundaries when several receiver lines are stacked into one B-scan.
    line_changes = np.flatnonzero(data.line_names[1:] != data.line_names[:-1]) + 0.5
    for boundary in line_changes:
        ax.axhline(boundary, color='black', linewidth=0.8)

    fig.colorbar(image, ax=ax, label='Velocity y-component [m/s]')
    ax.set_title(f'B-Scan — {_receiver_selection_label(data)}')
    ax.set_xlabel(xlabel)
    _label_bscan_receiver_axis(ax, data)
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_calibration_bscans(
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
    scale_to_microseconds=True,
    cmap='seismic',
    colour_percentile=99.5,
):
    """Plot defective, calibrated-reference, and difference B-scans.

    Parameters
    ----------
    receiver_lines : str or sequence of str, default='left'
        Receiver line or lines to plot.
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
        Whether to align each reference trace by cross-correlation.
    scale_reference : bool, default=False
        Whether to least-squares scale each reference trace.
    calibration_window : tuple of float, mapping, or None
        Shared or per-line calibration time window in seconds.
    max_shift_samples : int, default=10
        Maximum absolute lag considered during alignment.
    scale_to_microseconds : bool, default=True
        Display time in microseconds instead of seconds.
    cmap : str, default='seismic'
        Matplotlib colormap name.
    colour_percentile : float or None, default=99.5
        Absolute-amplitude percentile used for symmetric color limits. Raw
        scans share one limit; the difference scan has its own. ``None`` uses
        full amplitude ranges.

    Returns
    -------
    result : CalibrationResult
        Calibrated receiver data used by the plots.
    fig : matplotlib.figure.Figure
        Created figure.
    axes : numpy.ndarray
        Axes for defective, reference, and difference B-scans.
    """
    result = load_calibrated_receiver_data(
        receiver_lines=receiver_lines,
        receiver_indices=receiver_indices,
        defect_dataset=defect_dataset,
        reference_dataset=reference_dataset,
        frequency_band=frequency_band,
        filter_order=filter_order,
        align_reference=align_reference,
        scale_reference=scale_reference,
        calibration_window=calibration_window,
        max_shift_samples=max_shift_samples,
    )
    if scale_to_microseconds:
        plot_times = result.difference.times * 1e6
        xlabel = 'Time [$\\mu$s]'
    else:
        plot_times = result.difference.times
        xlabel = 'Time [s]'

    datasets = (
        ('Defective', result.defective),
        ('Reference', result.reference),
        ('Defect - reference', result.difference),
    )
    raw_limit = _symmetric_colour_limit(
        np.concatenate(
            (
                result.defective.time_series_matrix.ravel(),
                result.reference.time_series_matrix.ravel(),
            )
        ),
        colour_percentile,
    )
    difference_limit = _symmetric_colour_limit(
        result.difference.time_series_matrix,
        colour_percentile,
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for axis, (title, data) in zip(axes, datasets):
        colour_limit = difference_limit if data is result.difference else raw_limit
        image = axis.imshow(
            data.time_series_matrix,
            aspect='auto',
            extent=[plot_times[0], plot_times[-1], len(data.point_indices) - 0.5, -0.5],
            cmap=cmap,
            vmin=-colour_limit,
            vmax=colour_limit,
        )
        line_changes = np.flatnonzero(data.line_names[1:] != data.line_names[:-1]) + 0.5
        for boundary in line_changes:
            axis.axhline(boundary, color='black', linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        fig.colorbar(image, ax=axis, label='Velocity y-component [m/s]')

    _label_bscan_receiver_axis(axes[0], result.difference)
    fig.suptitle(f'Receiver selection: {_receiver_selection_label(result.difference)}')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    plt.show()
    return result, fig, axes


def plot_reference_arrival_calibration(
    calibration,
    scale_to_microseconds=True,
    cmap='seismic',
    colour_percentile=99.5,
):
    """Plot a reference B-scan with calibrated direct-arrival predictions.

    Parameters
    ----------
    calibration : ReferenceArrivalCalibration
        Calibration result to visualize.
    scale_to_microseconds : bool, default=True
        Display time in microseconds instead of seconds.
    cmap : str, default='seismic'
        Matplotlib colormap name.
    colour_percentile : float or None, default=99.5
        Absolute-amplitude percentile used for symmetric color limits.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    ax : matplotlib.axes.Axes
        Calibration B-scan axes.
    """
    data = calibration.data
    time_scale = 1e6 if scale_to_microseconds else 1.0
    time_unit = '$\\mu$s' if scale_to_microseconds else 's'
    plot_times = data.times * time_scale
    fitted_times = (
        calibration.predicted_times + calibration.time_offset
    ) * time_scale
    picked_times = calibration.picked_times * time_scale
    colour_limit = _symmetric_colour_limit(
        calibration.filtered_signals, colour_percentile
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(
        calibration.filtered_signals,
        aspect='auto',
        extent=[plot_times[0], plot_times[-1], len(data.point_indices) - 0.5, -0.5],
        cmap=cmap,
        vmin=-colour_limit,
        vmax=colour_limit,
    )
    receiver_rows = np.arange(len(data.point_indices))
    ax.plot(
        fitted_times,
        receiver_rows,
        color='lime',
        linewidth=1.5,
        label=f'{calibration.wave_type} prediction + offset',
    )
    ax.scatter(
        picked_times,
        receiver_rows,
        s=8,
        color='yellow',
        label='Local envelope picks',
    )
    _label_bscan_receiver_axis(ax, data)
    ax.set_xlabel(f'Time [{time_unit}]')
    ax.set_title(
        f'Reference direct-arrival calibration — '
        f'{_receiver_selection_label(data)}\n'
        f'{calibration.wave_type}-wave offset='
        f'{calibration.time_offset * 1e6:.3f} us, '
        f'pick RMS={calibration.residual_rms * 1e9:.1f} ns'
    )
    ax.legend(loc='upper right')
    fig.colorbar(image, ax=ax, label='Velocity y-component [m/s]')
    fig.tight_layout()
    plt.show()
    return fig, ax


def compare_saft_modes(
    receiver_line='right',
    receiver_indices=None,
    modes=('PP', 'PS', 'SP', 'SS'),
    component='direct',
    common_colour_scale=False,
    colour_percentile=99.5,
    defect_model=DEFECT_MODEL,
    **saft_options,
):
    """Reconstruct propagation modes separately for one receiver line.

    Parameters
    ----------
    receiver_line : {'left', 'right', 'weld'}, default='right'
        Single receiver line to inspect. This function intentionally rejects
        combined lines so that their evidence is assessed before fusion.
    receiver_indices : int, slice, array-like, or None
        Zero-based receiver points to use. ``None`` selects every point on the
        chosen line.
    modes : sequence of str, default=('PP', 'PS', 'SP', 'SS')
        Two-letter propagation modes. The first letter is the source-to-pixel
        mode and the second is the pixel-to-receiver mode.
    component : {'direct', 'backwall', 'combined'}, default='direct'
        SAFT component displayed in the comparison figure.
    common_colour_scale : bool, default=False
        Whether all mode panels share one amplitude scale. Independent scales
        emphasize spatial focusing; a common scale permits amplitude
        comparison.
    colour_percentile : float or None, default=99.5
        Absolute-amplitude percentile used for the image colour limits.
    defect_model : pathlib.Path or str, default=DEFECT_MODEL
        Matching COMSOL model used to draw the known defect outline.
    **saft_options : dict
        Additional keyword arguments forwarded to
        :func:`perform_segmented_saft`. Mode selection, receiver selection,
        plotting, and component return settings are controlled by this
        function and cannot be supplied here.

    Returns
    -------
    GX : numpy.ndarray
        Two-dimensional image-grid x coordinates in metres.
    GY : numpy.ndarray
        Two-dimensional image-grid y coordinates in metres.
    mode_images : dict of str to numpy.ndarray
        Selected SAFT component for every requested propagation mode.
    fig : matplotlib.figure.Figure
        Mode-comparison figure.
    axes : numpy.ndarray
        Axes containing the individual mode images.

    Raises
    ------
    TypeError
        If ``receiver_line`` is not a single string.
    ValueError
        If a receiver line, propagation mode, component, or reserved SAFT
        option is invalid.

    Notes
    -----
    This routine does not combine modes and does not select a preferred mode.
    Candidate indications should be checked for spatial consistency across
    independent receiver subsets before any line or mode fusion. The dashed
    outline marks the known COMSOL defect for comparison only.
    """
    if not isinstance(receiver_line, str):
        raise TypeError("receiver_line must be one line name, not a sequence.")
    receiver_line = receiver_line.lower()
    if receiver_line not in {'left', 'right', 'weld'}:
        raise ValueError("receiver_line must be 'left', 'right', or 'weld'.")

    component = str(component).lower()
    if component not in {'direct', 'backwall', 'combined'}:
        raise ValueError(
            "component must be 'direct', 'backwall', or 'combined'."
        )

    reserved_options = {
        'receiver_lines',
        'receiver_indices',
        'source_wave_type',
        'receiver_wave_type',
        'return_components',
    }
    conflicting_options = reserved_options.intersection(saft_options)
    if conflicting_options:
        names = ', '.join(sorted(conflicting_options))
        raise ValueError(f'Reserved compare_saft_modes option(s): {names}.')

    normalised_modes = []
    for mode in modes:
        mode = str(mode).upper().replace('->', '')
        if len(mode) != 2 or any(letter not in {'P', 'S'} for letter in mode):
            raise ValueError(
                "Each mode must be one of 'PP', 'PS', 'SP', or 'SS'."
            )
        if mode not in normalised_modes:
            normalised_modes.append(mode)
    if not normalised_modes:
        raise ValueError('At least one propagation mode is required.')

    mode_images = {}
    GX = GY = None
    for mode in normalised_modes:
        mode_gx, mode_gy, components = perform_segmented_saft(
            receiver_lines=receiver_line,
            receiver_indices=receiver_indices,
            source_wave_type=mode[0],
            receiver_wave_type=mode[1],
            return_components=True,
            **saft_options,
        )
        if GX is None:
            GX, GY = mode_gx, mode_gy
        mode_images[mode] = components[component]

    num_columns = 2 if len(normalised_modes) > 1 else 1
    num_rows = int(np.ceil(len(normalised_modes) / num_columns))
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(7.0 * num_columns, 5.0 * num_rows),
        squeeze=False,
        constrained_layout=True,
    )
    image_mode = str(saft_options.get('image_mode', 'signed')).lower()
    signed_image = image_mode == 'signed'
    if common_colour_scale:
        common_limit = _symmetric_colour_limit(
            np.concatenate([image.ravel() for image in mode_images.values()]),
            colour_percentile,
        )
    else:
        common_limit = None

    plot_extent = [GX.min(), GX.max(), GY.min(), GY.max()]
    source_position = np.asarray(
        saft_options.get('source_position', DEFAULT_LASER_SOURCE), dtype=float
    )
    defect_outline = load_comsol_defect_outline(defect_model)
    for axis, mode in zip(axes.flat, normalised_modes):
        image = mode_images[mode]
        colour_limit = (
            common_limit
            if common_limit is not None
            else _symmetric_colour_limit(image, colour_percentile)
        )
        plotted_image = axis.imshow(
            image,
            extent=plot_extent,
            cmap='seismic' if signed_image else 'inferno',
            origin='lower',
            vmin=-colour_limit if signed_image else 0.0,
            vmax=colour_limit,
        )
        axis.plot(
            source_position[0],
            source_position[1],
            marker='*',
            color='lime',
            markersize=10,
        )
        _draw_comsol_defect(axis, defect_outline)
        axis.set_title(f'{mode[0]}->{mode[1]} {component} SAFT')
        axis.set_xlabel('X Coordinate [m]')
        axis.set_ylabel('Y Coordinate [m]')
        fig.colorbar(plotted_image, ax=axis, label='SAFT amplitude')

    for axis in axes.flat[len(normalised_modes):]:
        axis.set_visible(False)

    selection_data = load_receiver_data(
        receiver_line,
        receiver_indices,
        dataset=saft_options.get('dataset', 'defect'),
    )
    scale_label = 'common scale' if common_colour_scale else 'individual scales'
    fig.suptitle(
        f'Mode comparison | {_receiver_selection_label(selection_data)} | '
        f'{component} component | {scale_label}'
    )
    plt.show()
    return GX, GY, mode_images, fig, axes


def plot_segmented_saft(
    receiver_lines='right', *, colour_percentile=99.5,
    defect_model=DEFECT_MODEL, **saft_options
):
    """Plot direct, back-wall, and combined SAFT images for one aperture.

    Parameters
    ----------
    receiver_lines : str or sequence of str, default='right'
        Receiver line or lines used to form the images.
    colour_percentile : float or None, default=99.5
        Percentile of absolute image amplitude used for the colour scale.
    defect_model : pathlib.Path or str, default=DEFECT_MODEL
        Matching COMSOL model used to draw the known defect outline.
    **saft_options : dict
        Physical and signal-processing options passed to
        :func:`analyse.perform_segmented_saft`.

    Returns
    -------
    gx, gy : numpy.ndarray
        Image pixel coordinates in metres.
    components : dict of str to numpy.ndarray
        Direct, back-wall, and combined SAFT images.
    fig : matplotlib.figure.Figure
        Figure containing the wave-speed map and SAFT panels.
    axes : numpy.ndarray
        Axes for the four panels.

    Notes
    -----
    The speed map and image pixels share the same cross-section coordinates.
    The green star marks the laser source position. The dashed outline marks
    the known COMSOL defect; it is not used to form the SAFT image.
    """
    if 'return_components' in saft_options:
        raise ValueError('return_components is controlled by plot_segmented_saft.')
    gx, gy, components = perform_segmented_saft(
        receiver_lines=receiver_lines,
        return_components=True,
        **saft_options,
    )
    wave_type = str(saft_options.get('wave_type', 'S')).upper()
    source_mode = str(saft_options.get('source_wave_type') or wave_type).upper()
    receiver_mode = str(saft_options.get('receiver_wave_type') or wave_type).upper()
    image_mode = str(saft_options.get('image_mode', 'signed')).lower()
    source_x, source_y = saft_options.get('source_position', DEFAULT_LASER_SOURCE)
    defect_outline = load_comsol_defect_outline(defect_model)
    _, cp, cs = saft_geometry(gx, gy)
    speed_map = cs if receiver_mode == 'S' else cp
    selection = load_receiver_data(
        receiver_lines,
        saft_options.get('receiver_indices'),
        dataset=saft_options.get('dataset', 'defect'),
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    dataset_label = str(saft_options.get('dataset', 'defect'))
    if saft_options.get('reference_dataset') is not None:
        dataset_label += f" - {saft_options['reference_dataset']}"
    fig.suptitle(
        f'{_receiver_selection_label(selection)} | '
        f'{source_mode}->{receiver_mode} | {dataset_label} | {image_mode}'
    )

    extent = [gx.min(), gx.max(), gy.min(), gy.max()]
    speed_plot = axes[0, 0].imshow(
        speed_map, extent=extent, cmap='coolwarm', origin='lower'
    )
    axes[0, 0].set_title(f'Segmented Receiver-Leg Speed Map ({receiver_mode}-wave)')
    fig.colorbar(speed_plot, ax=axes[0, 0], label='Velocity [m/s]')

    component_images = (
        ('Direct path', components['direct']),
        ('Receiver-leg backwall path', components['backwall']),
        ('Direct + backwall', components['combined']),
    )
    colour_limit = _symmetric_colour_limit(
        np.concatenate([image.ravel() for _, image in component_images]),
        colour_percentile,
    )
    signed_image = image_mode == 'signed'
    for axis, (title, image) in zip(axes.flat[1:], component_images):
        plotted = axis.imshow(
            image,
            extent=extent,
            cmap='seismic' if signed_image else 'inferno',
            origin='lower',
            vmin=-colour_limit if signed_image else 0.0,
            vmax=colour_limit,
        )
        axis.set_title(f'{title} SAFT')
        amplitude_label = (
            'Signed SAFT amplitude'
            if signed_image else f'{image_mode.capitalize()} SAFT amplitude'
        )
        fig.colorbar(plotted, ax=axis, label=amplitude_label)
    for axis in axes.flat:
        axis.plot(source_x, source_y, marker='*', color='lime', markersize=10)
        _draw_comsol_defect(axis, defect_outline)
        axis.set_xlabel('X Coordinate [m]')
        axis.set_ylabel('Y Coordinate [m]')
    plt.show()
    return gx, gy, components, fig, axes


def plot_defect_evidence(
    receiver_line='right',
    mode='PP',
    component='direct',
    pixel_size=0.00025,
    time_offset=0.0,
    defect_model=DEFECT_MODEL,
):
    """Show early-pulse transmission loss and independent SAFT apertures.

    Parameters
    ----------
    receiver_line : {'left', 'right', 'weld'}, default='right'
        Receiver line used for the aperture comparison.
    mode : {'PP', 'PS', 'SP', 'SS'}, default='PP'
        Source and receiver propagation modes.
    component : {'direct', 'backwall'}, default='direct'
        Scattering path shown in the SAFT panels.
    pixel_size : float, default=0.00025
        Maximum image-pixel spacing in metres.
    time_offset : float, default=0.0
        Explicit delay added to modelled SAFT arrival times, in seconds.
    defect_model : pathlib.Path or str, default=DEFECT_MODEL
        Matching COMSOL model used to draw the known defect outline.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing measured pulse ratios and split-aperture images.
    axes : numpy.ndarray
        Two-by-three array of axes in the figure.

    Notes
    -----
    The pulse ratios use reference-picked early arrivals and do not depend on
    the SAFT travel-time model. All SAFT panels use defect-minus-reference
    traces. The dashed COMSOL defect outline is shown only for comparison.
    A split-stable focus can still be a groove or wall artefact.
    """
    arrival = compare_reference_arrivals('sides')
    gx, gy, images, peaks, separation = compare_saft_apertures(
        receiver_line=receiver_line,
        mode=mode,
        component=component,
        pixel_size=pixel_size,
        time_offset=time_offset,
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for line, colour in (('left', 'tab:blue'), ('right', 'tab:orange')):
        selected = arrival.data.line_names == line
        x_mm = arrival.data.x_coords[selected] * 1000
        axes[0, 0].plot(
            x_mm, arrival.amplitude_ratios[selected], '.-', color=colour,
            label=f'{line} (median {np.nanmedian(arrival.amplitude_ratios[selected]):.2f})',
        )
        axes[0, 1].plot(
            x_mm, arrival.correlation[selected], '.-', color=colour,
            label=line,
        )
    axes[0, 0].axhline(1.0, color='black', linestyle=':', linewidth=1)
    axes[0, 0].set_title('Early pulse: defect / reference RMS')
    axes[0, 0].set_ylabel('Amplitude ratio')
    axes[0, 1].set_title('Early pulse waveform correlation')
    axes[0, 1].set_ylabel('Correlation')
    for axis in axes[0, :2]:
        axis.set_xlabel('Receiver x [mm]')
        axis.legend()
        axis.grid(alpha=0.2)

    axes[0, 2].axis('off')
    axes[0, 2].text(
        0.02, 0.95,
        f'{receiver_line} receiver line\n'
        f'{mode} {component} SAFT\n'
        f'Timing shift: {time_offset * 1e6:.2f} us\n'
        f'Even/odd peak separation: {separation * 1000:.2f} mm\n\n'
        'A repeatable image peak is a candidate,\n'
        'not a confirmed defect location.',
        va='top', transform=axes[0, 2].transAxes,
    )

    colour_limit = max(float(np.max(image)) for image in images.values())
    extent_mm = np.array([gx.min(), gx.max(), gy.min(), gy.max()]) * 1000
    defect_outline = load_comsol_defect_outline(defect_model)
    for axis, (name, image) in zip(axes[1], images.items()):
        plotted = axis.imshow(
            image, extent=extent_mm, origin='lower', cmap='inferno',
            vmin=0, vmax=colour_limit or 1,
        )
        peak_x, peak_y = peaks[name]
        if np.isfinite(peak_x):
            axis.plot(peak_x * 1000, peak_y * 1000, 'cx', markersize=9)
        _draw_comsol_defect(axis, defect_outline, scale=1000)
        axis.set_title(f'{name} receivers')
        axis.set_xlabel('x [mm]')
        axis.set_ylabel('y [mm]')
        axis.set_aspect('equal')
    fig.colorbar(plotted, ax=axes[1, :], label='SAFT envelope [m/s]')
    plt.show()
    return fig, axes


def main():
    """Plot a residual SAFT reconstruction or receiver diagnostics.

    Notes
    -----
    The default figure uses defect-minus-reference velocity with S waves on
    both propagation legs. ``--diagnostics`` shows the raw receiver evidence
    before SAFT reconstruction. ``--validate`` compares early side-line
    transmission and independent receiver-aperture images. ``--output`` saves
    either SAFT figure with the known COMSOL defect outline.
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--line', choices=('left', 'right', 'weld', 'sides'), default='right')
    parser.add_argument('--mode', choices=('PP', 'PS', 'SP', 'SS'), default='SS')
    parser.add_argument('--pixel-size-mm', type=float, default=0.25)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--component', choices=('direct', 'backwall'), default='direct')
    parser.add_argument('--time-offset-us', type=float, default=0.0)
    parser.add_argument('--defect-model', type=Path, default=DEFECT_MODEL)
    parser.add_argument('--output', type=Path, help='Save the SAFT figure to this image file')
    args = parser.parse_args()
    if args.pixel_size_mm <= 0:
        parser.error('--pixel-size-mm must be positive')
    if args.diagnostics and args.output:
        parser.error('--output is for SAFT figures, not --diagnostics')
    if args.validate:
        if args.line == 'sides':
            parser.error('--validate needs one receiver line for aperture splitting')
        fig, _ = plot_defect_evidence(
            receiver_line=args.line,
            mode=args.mode,
            component=args.component,
            pixel_size=args.pixel_size_mm / 1000,
            time_offset=args.time_offset_us * 1e-6,
            defect_model=args.defect_model,
        )
    elif args.diagnostics:
        plot_calibration_bscans(args.line)
        plot_comsol_ascan(args.line)
    else:
        _, _, _, fig, _ = plot_segmented_saft(
            receiver_lines=args.line,
            reference_dataset='reference',
            source_wave_type=args.mode[0],
            receiver_wave_type=args.mode[1],
            pixel_size=args.pixel_size_mm / 1000,
            image_mode='envelope',
            defect_model=args.defect_model,
        )
    if args.output:
        fig.savefig(args.output, dpi=180)


if __name__ == '__main__':
    main()
