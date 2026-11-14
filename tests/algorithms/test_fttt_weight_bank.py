"""Dataset selection, missing-model guidance and artifact integrity."""

import hashlib
import json

import pytest

from usctbench.algorithms.fttt.weight_bank import inspect_manifest, resolve_manifest
from usctbench.core.config import prepare_config


def manifest(tmp_path, *, status="untrained", backend="score_sde_pytorch"):
    root = tmp_path / "diffusion" / "weights" / "custom"
    root.mkdir(parents=True)
    payload = {
        "dataset_id": "custom",
        "backend": backend,
        "status": status,
        "training": {"repository": "https://github.com/yang-song/score_sde_pytorch"},
    }
    if status != "untrained":
        (root / "model.pt").write_bytes(b"artifact")
        payload.update(
            checkpoint="model.pt", sha256=hashlib.sha256(b"artifact").hexdigest()
        )
    path = root / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_missing_dataset_weights_guide_training(tmp_path):
    path = manifest(tmp_path)
    record = inspect_manifest(path)
    assert record["status"] == "untrained"
    assert "yang-song/score_sde" in record["score_sde_repository"]
    with pytest.raises(
        ValueError, match="No trained diffusion weights.*score_sde_pytorch"
    ):
        resolve_manifest(path)


def test_native_path_and_sha_are_checked(tmp_path):
    path = manifest(tmp_path, status="ready", backend="native_tiny_unet")
    checkpoint, record = resolve_manifest(path)
    assert checkpoint.endswith("model.pt")
    assert record["dataset_id"] == "custom"
    path.with_name("model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        resolve_manifest(path)
    payload = json.loads(path.read_text())
    payload["checkpoint"] = "../outside.pt"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inside"):
        resolve_manifest(path)


def test_registered_external_weights_are_not_misloaded_as_epsilon(tmp_path):
    path = manifest(tmp_path, status="registered")
    with pytest.raises(ValueError, match="score/epsilon conversion"):
        resolve_manifest(path)


def test_ambiguous_weight_selection_rejected():
    with pytest.raises(ValueError, match="checkpoint or prior_manifest"):
        prepare_config(
            "fttt",
            {
                "parameters": {
                    "variant": "dps",
                    "checkpoint": "a.pt",
                    "prior_manifest": "manifest.json",
                }
            },
        )
