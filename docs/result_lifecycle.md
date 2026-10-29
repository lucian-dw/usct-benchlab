# Numerical results and optional reports

`run_algorithm_case` separates preparation, solve, save, evaluation and preview.
The solver receives shared observations/geometry with an empty ground-truth object.
Ground truth remains available only to post-reconstruction evaluation.

`result.h5` is written before scoring or plotting. `metadata.yaml` and
`metrics.json` record each stage's status and the actual exception message when a
stage fails. A failed evaluation or preview does not overwrite the numerical HDF5
or change a completed solver result into a numerical failure. A failed numerical
save is a failed run; it is not replaced with an empty successful-looking file.

The run time measures preparation and solving, including failed attempts. The
`process_peak_rss_mib` field measures the current process's cumulative RSS peak,
not incremental per-case allocation; macOS bytes and Linux KiB are converted
explicitly. Do not use it as a per-case GPU-memory measurement.

Image SSIM is the local-window metric over complete valid windows, up to 7 pixels.
If none fit, the score is null with `ssim_status=unavailable_no_complete_window`.
`global_ssim` remains a separately named statistic, never a silent substitute.
Perfect-reconstruction PSNR is null with `psnr_status=perfect_reconstruction`.
These image metric records can be serialized as standard JSON without infinity.
