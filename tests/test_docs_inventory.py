from __future__ import annotations

from pathlib import Path

REQUIRED_CARD_SECTIONS = [
    "Physical Assumption",
    "Input Requirements",
    "Default Settings",
    "Expected Failure Modes",
    "What To Adjust First",
    "Acceptance Tests",
    "References and Related Code",
]

ALGORITHM_CARDS = [
    "docs/algorithm_cards/straight_ray_sart.md",
    "docs/algorithm_cards/straight_ray_sirt.md",
    "docs/algorithm_cards/straight_ray_cgls.md",
    "docs/algorithm_cards/attenuation_tomography.md",
    "docs/algorithm_cards/bent_ray_gn.md",
    "docs/algorithm_cards/rwave_ray_born.md",
    "docs/algorithm_cards/fwi_tiny.md",
]


def test_required_v01_docs_exist():
    required = [
        "docs/architecture.md",
        "docs/A100_SERVER_SETUP.md",
        "docs/OPENBREASTUS_DATA_PROTOCOL.md",
        "docs/EVALUATION_ACCEPTANCE_PROTOCOL.md",
        "docs/algorithm_taxonomy.md",
        "docs/EXTERNAL_SOURCES_AND_LICENSES.md",
        "docs/benchmark_report_template.md",
        "docs/ALGORITHM_SETTINGS_TROUBLESHOOTING.md",
        "docs/references.bib",
        "docs/V0_1_READINESS_CHECKLIST.md",
        *ALGORITHM_CARDS,
    ]
    missing = [path for path in required if not Path(path).exists()]

    assert not missing


def test_algorithm_cards_have_required_v01_sections():
    missing: dict[str, list[str]] = {}
    for rel_path in ALGORITHM_CARDS:
        path = Path(rel_path)
        headings = {
            line.removeprefix("## ").strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("## ")
        }
        card_missing = [section for section in REQUIRED_CARD_SECTIONS if section.lower() not in headings]
        if card_missing:
            missing[rel_path] = card_missing

    assert not missing
