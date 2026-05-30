from __future__ import annotations

from usctbench.cli import main
from usctbench.registry import list_algorithms


def test_cli_registers_builtin_algorithms(capsys):
    exit_code = main(["list-algorithms"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "straight_sart" in output
    assert "attenuation_sirt" in output
    assert {entry.name for entry in list_algorithms()} >= {"straight_sart", "straight_sirt", "straight_cgls", "attenuation_sirt"}

