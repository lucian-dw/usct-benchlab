# Examples

These examples are intentionally small and write all generated data to `/tmp`.
They are meant for installation checks and for learning the CLI workflow before
running dataset-specific benchmarks.

## Synthetic Quickstart

```bash
bash examples/synthetic_quickstart.sh
```

The script creates two deterministic synthetic cases, runs the synthetic demo
benchmark, and prints the run directory. Generated artifacts are written under
`/tmp/usctbench_examples`.

An explicit output root is accepted: `bash examples/synthetic_quickstart.sh /tmp/my-usct`.

## Dataset benchmarks

Prepare a dataset with the public `usct data` commands, then choose the matching
suite explicitly. No automatic replacement dataset is selected:

```bash
usct bench --suite configs/benchmarks/openbreastus_demo.yaml
usct bench --suite configs/benchmarks/nbpslice2d_demo.yaml
```

## WUST FWI

Supply a pressure case, configuration, maintained WUST checkout and output
directory. MATLAB must be on PATH or set explicitly by the deployment configuration.

```bash
bash examples/fwi_quickstart.sh /path/to/pressure_case.h5 \
  configs/algorithms/fwi_wust.yaml /path/to/WaveformInversionUST /tmp/usct-fwi
```

## k-Wave data tooling

Simulation is separate from reconstruction. It requires an explicit property-map
case and external k-Wave/MATLAB installation. Outputs stay below the selected root:

```bash
python scripts/simulate_kwave.py prepare --case /path/to/property.h5 --out /tmp/kwave-pair
python scripts/simulate_kwave.py simulate --out /tmp/kwave-pair --kwave-path /path/to/k-Wave --matlab /path/to/matlab
python scripts/simulate_kwave.py extract --out /tmp/kwave-pair
```

`pressure_case.h5` keeps raw pressure, a water reference, explicit-frequency DTFT
and ToF features. `manifest.json` describes the simulation; `simulation_qc.json`
reports the performed checks, not clinical validity. Reconstruct through `usct run`
or `usct bench`, not through the data tool.

## Saved images

```bash
python scripts/render_examples.py --cases-dir /path/to/cases --run-dir /path/to/run \
  --case-ids my_case --algorithms straight_cgls bent_ray_gn --out /tmp/comparison.png
```

Only selected GT/reconstruction images and scalar metrics are read. Imported
physical axes are preserved; missing results are listed in the companion CSV.
