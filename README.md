# Weld SAFT analysis

The new v2 model has uniform temperature, a straight lack-of-fusion defect,
and a more filled weld. Its geometry and receiver exports will be added when
they are available. The current source exports are in `v2/data/sources/`.

`v2/ray_tracing.py` contains basic elastic-wave speed, straight-ray travel-time,
and Snell-law functions. It uses SI units and has no model-specific constants.
These relations are building blocks; they do not yet trace rays through the
new weld geometry or reconstruct a SAFT image.

The previous model, its data, scripts, figures, and documentation are archived
in `v1/`. Each version keeps its own data directory.
