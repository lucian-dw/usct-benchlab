"""Versioned capability and subprocess tests without MATLAB or CUDA."""

import os

import pytest

from usctbench.algorithms.fwi.parameters import WUSTParameters
from usctbench.algorithms.fwi.runtime import (
    Deadline,
    RuntimeFailure,
    WUSTClient,
)


def capabilities():
    return {
        "schema": "wust.capabilities",
        "schema_version": 2,
        "runtime_version": "0.2.0-dev.1",
        "operations": ["reconstruct"],
        "input_schemas": {"reconstruct": ["wust.frequency_input", "wust.measurements"]},
        "profiles": {"cpu": "cpu_float64", "gpu": "gpu_complex64"},
    }


@pytest.fixture
def client(tmp_path):
    (tmp_path / "Runtime/python").mkdir(parents=True)
    return WUSTClient(
        WUSTParameters(runtime_root=str(tmp_path)), Deadline(10), tmp_path
    )


@pytest.mark.parametrize(
    "field,value",
    [("schema_version", 1), ("operations", []), ("profiles", {"cpu": "cpu_float64"})],
)
def test_required_capability_rejection(client, monkeypatch, field, value):
    info = capabilities()
    info[field] = value
    monkeypatch.setattr(client, "command", lambda args: info)
    with pytest.raises(RuntimeFailure, match="v2 reconstruction capability"):
        client.admit()


def test_describe_does_not_probe_matlab(client, monkeypatch):
    calls = []

    def command(arguments):
        calls.append(arguments)
        return capabilities()

    monkeypatch.setattr(client, "command", command)
    assert client.admit()["schema_version"] == 2
    assert calls == [["describe", "--json"]]


def test_runtime_failure_and_unclean_stdout(client):
    client.launcher.write_text(
        'import sys\nsys.stderr.write(\'{"schema":"wust.execution_failure","reason":"numerical_failure","message":"bad factor"}\\n\')\nsys.exit(3)\n'
    )
    with pytest.raises(RuntimeFailure) as error:
        client.command([])
    assert error.value.reason == "numerical_failure"
    client.launcher.write_text('print("startup banner")\nprint("{}")\n')
    with pytest.raises(RuntimeFailure, match="clean JSON"):
        client.command([])


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
def test_deadline_reaps_launcher(client):
    pidfile = client.folder / "child.pid"
    client.launcher.write_text(
        f"import os,time\nfrom pathlib import Path\nPath({str(pidfile)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n"
    )
    client.deadline = Deadline(0.5)
    with pytest.raises(RuntimeFailure) as error:
        client.command([])
    assert error.value.reason == "time_budget"
    assert pidfile.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
