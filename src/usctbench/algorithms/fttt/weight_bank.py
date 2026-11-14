"""Dataset-specific prior registration and training guidance; no torch import."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCORE_SDE_URL = "https://github.com/yang-song/score_sde"
SCORE_SDE_PYTORCH_URL = "https://github.com/yang-song/score_sde_pytorch"


class WeightManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["usct.diffusion.weights.v1"] = "usct.diffusion.weights.v1"
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    backend: Literal["native_tiny_unet", "score_sde_pytorch"]
    status: Literal["untrained", "registered", "ready"]
    checkpoint: str | None = None
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    training: dict
    notes: str = ""

    @model_validator(mode="after")
    def artifact_state(self):
        if self.status in ("registered", "ready") and not (
            self.checkpoint and self.sha256
        ):
            raise ValueError("registered weights require checkpoint and SHA256")
        if self.status == "ready" and self.backend != "native_tiny_unet":
            raise ValueError(
                "Score SDE weights may be registered; runtime adaptation must be validated before ready"
            )
        return self


def inspect_manifest(path):
    path = Path(path).expanduser().resolve()
    manifest = WeightManifest.model_validate_json(path.read_text(encoding="utf-8"))
    result = manifest.model_dump(mode="json")
    result.update(
        manifest=str(path),
        score_sde_repository=SCORE_SDE_URL,
        recommended_pytorch_repository=SCORE_SDE_PYTORCH_URL,
    )
    if manifest.checkpoint:
        checkpoint = (path.parent / manifest.checkpoint).resolve()
        if not checkpoint.is_relative_to(path.parent):
            raise ValueError(
                "weight bank checkpoint must stay inside the dataset directory"
            )
        result["checkpoint_path"] = str(checkpoint)
        result["checkpoint_exists"] = checkpoint.is_file()
    return result


def resolve_manifest(path):
    record = inspect_manifest(path)
    guidance = f"See {Path(path).resolve().parents[2] / 'README.md'}; train at {SCORE_SDE_PYTORCH_URL}"
    if record["status"] == "untrained" or not record.get("checkpoint_exists"):
        raise ValueError(
            f"No trained diffusion weights for dataset '{record['dataset_id']}'. {guidance}"
        )
    if record["backend"] != "native_tiny_unet" or record["status"] != "ready":
        raise ValueError(
            f"Dataset '{record['dataset_id']}' has registered external Score SDE weights; "
            f"its network, score/epsilon conversion and SDE schedule require a validated runtime adapter. {guidance}"
        )
    digest = hashlib.sha256()
    with Path(record["checkpoint_path"]).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != record["sha256"]:
        raise ValueError("diffusion weight bank checkpoint SHA256 mismatch")
    return record["checkpoint_path"], record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("diffusion"))
    parser.add_argument(
        "--dataset", help="Dataset ID; omit to list registered datasets"
    )
    args = parser.parse_args()
    if args.dataset is not None:
        if not args.dataset or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.dataset
        ):
            raise ValueError("invalid dataset ID")
        records = [
            inspect_manifest(args.root / "weights" / args.dataset / "manifest.json")
        ]
    else:
        records = [
            inspect_manifest(p)
            for p in sorted((args.root / "weights").glob("*/manifest.json"))
        ]
    print(json.dumps(records, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
