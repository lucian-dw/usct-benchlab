"""Small tables of measured outcomes; no inferred failure causes or rankings."""

import csv
from pathlib import Path

FIELDS = (
    "experiment_id",
    "algorithm",
    "variant",
    "case_id",
    "measurement_provenance",
    "forward_model",
    "observation_domain",
    "observation_units",
    "status",
    "stop_reason",
    "runtime_s",
    "primary_image_region",
    "rmse_mps",
    "psnr_db",
    "ssim",
    "full_image_rmse_mps",
    "full_image_psnr_db",
    "full_image_ssim",
    "data_relative_residual",
    "failure_stage",
    "message",
)


def write_summary(records, directory):
    directory = Path(directory)
    csv_path = directory / "benchmark_summary.csv"
    report_path = directory / "benchmark_report.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# Benchmark results",
        "",
        "Rows retain their observation provenance and forward model. ToF and pressure",
        "residuals are different observables, not a shared ranking. Hold-out scores",
        "used for selection are validation, not independent test scores.",
        "",
        "| Experiment | Algorithm | Variant | Case | Observation (units) | Status | Stop | Time (s) | Region | RMSE (m/s) | PSNR (dB) | SSIM |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    columns = (
        "experiment_id",
        "algorithm",
        "variant",
        "case_id",
        "observation",
        "status",
        "stop_reason",
        "runtime_s",
        "primary_image_region",
        "rmse_mps",
        "psnr_db",
        "ssim",
    )

    def cell(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    for row in records:
        row = {
            **row,
            "observation": f"{row.get('observation_domain', 'unknown')} ({row.get('observation_units', 'unknown')})",
        }
        lines.append("| " + " | ".join(cell(row.get(key)) for key in columns) + " |")
        if row.get("message"):
            lines.append(
                f"\n{cell(row.get('experiment_id'))} / {row['case_id']} / {row['algorithm']}: {cell(row['failure_stage'])}: {cell(row['message'])}\n"
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary_csv": str(csv_path), "report_md": str(report_path)}
