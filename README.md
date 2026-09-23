# LUT weld SAFT analysis

The `weld_defect_*.mph` and `weld_nodefect_*.mph` COMSOL models provide the
defective and reference simulations. Exported vertical-velocity traces are in
`data/receivers/`; the laser source table is in `data/sources/`.

`analyse.py` contains data loading, reference calibration, travel-time models,
and SAFT reconstruction.

The imaging model treats each pixel as a possible reflector. A direct arrival
travels from the fixed laser source to the pixel and then to a receiver; the
back-wall component adds a reflection from the inner pipe surface. Travel
times use separate P- and S-wave speeds in parent steel and the weld/HAZ. The
geometry is a simplified two-dimensional approximation of the COMSOL model.

Run it for a numerical residual reconstruction:

```powershell
python analyse.py --line right --mode SS --pixel-size-mm 0.25 --output saft_right.npz
```

The NPZ file contains the imaging grid (`x`, `y`) and the `direct`, `backwall`,
and `combined` SAFT images. Omit `--output` to print the residual RMS and image
peaks without saving a file.

`plotting.py` contains the A-scan, B-scan, calibration, and SAFT figures. Run it
to view a residual reconstruction or receiver diagnostics:

```powershell
python plotting.py --line right --mode SS --pixel-size-mm 0.25
python plotting.py --line right --diagnostics
```

The `--line` option accepts `left`, `right`, `weld`, or `sides`. `--mode` selects
the source and receiver wave modes (`PP`, `PS`, `SP`, or `SS`). Both scripts use
metres and seconds internally; the command-line pixel size is in millimetres.
