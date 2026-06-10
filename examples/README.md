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
