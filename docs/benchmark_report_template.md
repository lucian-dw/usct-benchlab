# Benchmark Report Template

Use this template when summarizing benchmark runs for a paper, issue, or release note.

## Run Metadata

- Suite:
- Run id:
- Git commit:
- Machine:
- Python:
- Data root:
- Sample root:
- Run root:

## Summary

- Records:
- Passed:
- Failed:
- Runtime total seconds:
- Runtime max seconds:
- Peak memory max MB:

## Dataset

- Source dataset:
- Index path:
- Cases:
- Density classes:
- Known schema limitations:

## Algorithm Results

| Algorithm | Case | Status | Pass | Runtime s | Key metrics | Reasons |
|---|---|---|---|---:|---|---|
|  |  |  |  |  |  |  |

## Failures

For each failure, link to `failure_report.md` and include:

- error type;
- symptom;
- likely causes;
- first actions to try;
- logs;
- plots.

## Interpretation

- What passed:
- What failed:
- What is a surrogate or synthetic-only result:
- What should be run next:

