# Architecture

`usct-benchlab` uses one benchmark contract for every reconstruction method:

```text
USCTCase -> Algorithm.run(...) -> ReconstructionResult
```

This keeps classical ray methods, attenuation baselines, optional MATLAB adapters, and later tiny FWI experiments callable through the same harness.

## Core Package

- `usctbench.schema` defines the shared Pydantic models and preserves physical units in field names.
- `usctbench.io.hdf5` writes and reads `USCTCase` and `ReconstructionResult` files using a stable HDF5 layout.
- `usctbench.registry` stores algorithm factories by CLI-safe names.
- `usctbench.cli` exposes the planned v0.1 command surface.

## Data Contract

`USCTCase` contains:

- image grid metadata in meters;
- transmitter and receiver coordinates in meters;
- frequency-domain, time-domain, or extracted feature measurements;
- optional sound-speed and attenuation ground truth;
- free-form metadata for source dataset details.

`ReconstructionResult` contains:

- one or more reconstructed image fields;
- metrics and artifact references;
- runtime and status;
- an explicit failure reason for failed or skipped runs.

## Runtime Boundaries

The local Mac is suitable for package editing, schema tests, documentation, and small synthetic fixtures. The A100 environment remains authoritative for OpenBreastUS indexing, CUDA-dependent checks, MATLAB/external adapters, and benchmark-scale numerical runs.

Large data, generated runs, external repositories, and checkpoints are ignored by Git and should remain under the workspace-level `data/`, `runs/`, `external/`, and `checkpoints/` directories.

