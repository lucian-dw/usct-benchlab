from __future__ import annotations

from pathlib import Path


def test_required_v01_docs_exist():
    required = [
        "docs/architecture.md",
        "docs/OPENBREASTUS_DATA_PROTOCOL.md",
        "docs/EVALUATION_ACCEPTANCE_PROTOCOL.md",
        "docs/algorithm_taxonomy.md",
        "docs/EXTERNAL_SOURCES_AND_LICENSES.md",
        "docs/benchmark_report_template.md",
        "docs/ALGORITHM_SETTINGS_TROUBLESHOOTING.md",
        "docs/references.bib",
        "docs/V0_1_READINESS_CHECKLIST.md",
        "docs/algorithm_cards/straight_ray_sart.md",
        "docs/algorithm_cards/straight_ray_sirt.md",
        "docs/algorithm_cards/straight_ray_cgls.md",
        "docs/algorithm_cards/attenuation_tomography.md",
        "docs/algorithm_cards/bent_ray_gn.md",
        "docs/algorithm_cards/rwave_ray_born.md",
        "docs/algorithm_cards/fwi_tiny.md",
    ]
    missing = [path for path in required if not Path(path).exists()]

    assert not missing

