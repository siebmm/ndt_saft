# LUT weld SAFT analysis

This project compares two COMSOL simulations of a partially welded pipe wall:
`weld_defect_v1.mph` has a lack-of-fusion defect, and
`weld_nodefect_v1.mph` is the reference. The receiver exports still name
these simulations `*_v4.mph`; the project owner confirmed that the current
`v1` archives are the matching models. Exported receiver traces are in
`data/receivers/`, and the laser source table is in `data/sources/`.

## COMSOL cross section

The models use a two-dimensional, 50 mm wide by 19.8 mm thick steel wall.
Coordinates are measured from the inner back wall (`y=0`); the outer surface
is at `y=19.8 mm`. An open groove sits above a deposited weld whose top is
at `y=6.339 mm`. The groove walls lean by 3 degrees, the root radius is
3.2 mm, and the HAZ extends approximately 3 mm around the weld. COMSOL uses
temperature-dependent elastic properties with nominal temperatures of
293.15 K in the parent steel and 573.15 K in the hot weld/HAZ.

The source is a prescribed vertical-velocity pulse centered at `x=0` on the
weld top. Its Gaussian spatial radius is 0.5 mm. The saved model parameter
`f_src` is 4 MHz; the `8MHz` receiver filenames describe the exports and
should not be read as that source parameter. COMSOL solves the full elastic
wave equation in time, so the signals include scattering, reflection,
diffraction, and mode conversion permitted by its geometry and mesh.

The defect is a curved lack-of-fusion gap along the right root sidewall. In
the saved, built geometry it is a polygon approximating a 2.5 mm arc with a
0.1 mm opening. Its outline spans approximately `x=1.02..2.62 mm` and
`y=1.66..3.62 mm`. The plotting code reads the polygon vertices from the
COMSOL model each time it draws a localization image. The outline is a
comparison marker and is never used to calculate SAFT delays or select an
image peak.

## Receiver lines and exported signals

Each `.txt` file contains receiver `x, y` coordinates in millimetres followed
by vertical surface velocity `vy` in metres per second. Every trace has
2001 samples from 0 to 20 microseconds at 10 ns intervals. Each line has a
defective-model file and a matching `_nodefect` reference file.

| Line | Points | Receiver x range | Receiver y | What it shows |
| --- | ---: | ---: | ---: | --- |
| `left` | 45 | -15 to -4 mm | 19.8 mm | Outer surface on the left of the open groove; a useful control for the right-side defect. |
| `right` | 45 | 4 to 15 mm | 19.8 mm | Outer surface on the defect side; the early transmitted pulse loses amplitude. |
| `weld` | 21 | -2.5 to 2.5 mm | 6.339 mm | Top of the deposited weld around the laser source; includes a strong local source response. |

Points on each line are spaced by 0.25 mm. `--line sides` loads the left and
right lines together; the weld line is inspected separately because it is on
a different surface and is close to the source.

## SAFT reconstruction

`analyse.py` loads and compares the traces, calculates travel times, and
reconstructs SAFT images. `plotting.py` makes A-scan, B-scan, calibration, and
SAFT figures. For each candidate reflector, the direct model follows source
to pixel to receiver. The back-wall component adds a reflection from the
inner surface on the receiver leg. PP, PS, SP, and SS specify wave modes on
the source and receiver legs. The nominal background includes parent steel,
the hot weld/HAZ, and the open groove. The known defect is absent from the
travel-time model.

SAFT uses approximate rays and piecewise P/S speeds. It does not reproduce
COMSOL's full elastic diffraction, refraction, mode conversion at the groove,
or wave amplitudes. Direct and back-wall results should therefore be checked
separately before interpreting a focus as a defect.

```powershell
python analyse.py --line right --mode SS --pixel-size-mm 0.25 --output saft_right.npz
python plotting.py --line right --mode SS --pixel-size-mm 0.25
python plotting.py --line right --diagnostics
```

The NPZ file contains the grid (`x`, `y`) and the `direct`, `backwall`, and
`combined` images. Omit `--output` to print image peaks without saving. The
SAFT figures draw the saved COMSOL defect outline as a dashed cyan polygon.
For a later matching model, pass `--defect-model path/to/model.mph` to
`plotting.py`; build and save its geometry first so the outline is current.

## Raw and envelope comparison near the root

The [laser-ultrasonic T-SAFT study](https://www.mdpi.com/1424-8220/23/19/8036)
compares images formed from raw traces and Hilbert-transformed traces. Run
the same comparison on the weld root with one receiver line and one P/S mode
at a time:

```powershell
python plotting.py --compare-raw-envelope --line right --mode PP --pixel-size-mm 0.10 --output figures/root_raw_envelope_PP.png
python plotting.py --compare-raw-envelope --line right --mode SS --pixel-size-mm 0.10 --output figures/root_raw_envelope_SS.png
```

Current images: [PP comparison](figures/root_raw_envelope_PP.png) and
[SS comparison](figures/root_raw_envelope_SS.png).

The four panels show raw and Hilbert-envelope reconstructions for the direct
and receiver-leg back-wall paths. Raw means the absolute value of the signed
delay-and-sum image; envelope means the magnitude of the coherently summed
analytic traces. Both methods use the same defect-minus-reference signals,
2-8 MHz filter, travel times, and receiver aperture. All four panels share
one amplitude scale and show the known COMSOL defect outline. The root view
covers `x=-4..8 mm` and `y=0..7 mm`. A 0.10 mm grid samples the image more
densely than the usual 0.25 mm command-line grid; it does not add measured
spatial resolution or fix an incorrect travel-time model.

The saved PP and SS examples show different raw/envelope behavior. Neither
produces a narrow response that follows the full curved defect outline, so
the comparison is a processing diagnostic rather than a validated defect
location.

## Defect evidence and localization check

```powershell
python analyse.py --validate --line right --mode PP --component direct
python plotting.py --validate --line right --mode PP --component direct --output figures/defect_evidence.png
```

An example from the current data is saved as `figures/defect_evidence.png`.

The early-pulse comparison picks each arrival only from the defect-free
reference in a 4.0-5.5 microsecond window. It compares defect/reference RMS
velocity within 0.2 microseconds of each pick, using the same 2-8 MHz band
for both simulations. These pulses are not assigned a P or S label. In the
current exports, the median amplitude ratio is about 0.99 on the left and
0.26 on the right. This shows a strong change in right-side transmission but
does not specify the defect depth.

The validation image reconstructs the full selected receiver line and its
even and odd receiver subsets. A peak that moves with the aperture is
unreliable; a stable peak can still come from the wall or groove. The
current PP direct example peaks at about `x=2.5 mm, y=0 mm` in both halves,
on the back-wall edge and outside the COMSOL defect outline. It is not a
validated defect location.

Add `--output evidence.npz` to the analysis command to save the split images,
side-line ratios, and receiver coordinates. Use `--mode` and `--component` to
inspect wave and path hypotheses separately. Apply `--time-offset-us` only
with a physically justified offset: the strongest side-line reference pulse
gives an apparent PP offset near 1.7 microseconds, but may contain converted
or S-wave energy. The weld-line source response peaks near 0.04 microseconds.

All internal distances and times use metres and seconds. Command-line pixel
sizes are in millimetres; `--time-offset-us` is in microseconds.
