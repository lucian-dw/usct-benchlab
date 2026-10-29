# Property-map conversion boundaries

Property-map conversion produces a matched-model travel-time surrogate, not
pressure measurements. Use `data.waveforms` for imported RF or complex pressure.

`convert_speed_mat_volume` takes an explicit sample axis (default 2), in-plane
axis order (`yx`, default, or `xy`), sound-speed units (`m/s`, default, or
`km/s`), and output pixel spacing in meters. The speed-volume CLI exposes these
as `--sample-axis`, `--image-axes`, `--sound-speed-units` and `--spacing-m`.
These are declared conversion assumptions, not acquisition properties inferred
from a filename. Inspect the source format before selecting them. The remaining
legacy class-selection helper is being replaced with explicit manifests in P4.

Sound speed and labels sample the same physical field of view. Continuous maps
use bilinear interpolation; labels use nearest neighbors. Coordinates describe
cell centers; `GridSpec.origin_m` describes the outer cell edge. Missing k-Wave
`xi_orig` / `yi_orig` coordinates now fail instead of inventing a 1 mm grid.
NBPslices2D uses its format's km/s sound speeds, 0.1 mm pixels and `type` field.
An empty label map records `roi_status=missing_label`, not a fabricated full ROI.

Property-map surrogate geometry is marked `synthetic_ring`. Pressure import uses
recorded TX/RX geometry. The reference speed is a preparation input, not the
median of the true image. Selected output files can be replaced on conversion;
unrelated files in the output directory are retained.
