# Development

1. Install Python 3.10+ and run pip install -e ".[dev,viz]".
2. Read [AGENTS.md](../AGENTS.md), the affected implementation, callers and tests.
3. Change the complete path: input, numerical behavior, output, docs and relevant tests.
4. Format and run the corresponding verification:

~~~bash
black src tests scripts
ruff check src tests scripts
pytest -q
~~~

5. Record actual verification and the next unit, inspect git diff --check, then commit.

Default CI runs this CPU pytest suite once. No MATLAB/GPU or external dataset is
required. Public numerical references live with operators/solvers, not source-text
or historical-output certification scripts. See [test index](physics_validation.md).

Manual cross-repository acceptance:

~~~bash
python scripts/validate_wust_integration.py --runtime /path/to/WaveformInversionUST \
  --matlab /path/to/matlab --backend cpu --out /absolute/new/acceptance-directory
~~~

Use --backend gpu for the CUDA MEX gate. Fixture generation is separate from
one formal reconstruct invocation. WUST native simulation has its own explicit
small MATLAB/native comparison in Runtime/tests/test_native_pipeline.m.

No data, runtime scratch, compiled binaries or research outputs belong in Git.
Historical material is indexed in [research_index.md](research_index.md).
