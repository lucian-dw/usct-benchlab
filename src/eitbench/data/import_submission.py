"""Import data from the supplied ZIP without executing its source or instructions."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def import_submission(archive, output):
    archive, output = Path(archive), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("data output must be empty")
    members = {}
    with ZipFile(archive) as z:
        for name in z.namelist():
            parts = PurePosixPath(name).parts
            if "data" not in parts or ".." in parts or name.endswith("/"):
                continue
            relative = Path(*parts[parts.index("data") + 1 :])
            if (
                relative.suffix.lower() not in (".npz", ".json", ".mat", ".jpg")
                or relative.is_absolute()
            ):
                continue
            if str(relative) in members:
                raise ValueError("duplicate archive data member")
            members[str(relative)] = z.read(name)
        if "manifest.json" not in members:
            raise ValueError("EIT data manifest missing")
        manifest = json.loads(members["manifest.json"])
        for case in manifest["cases"]:
            if case["input"] not in members:
                raise ValueError("EIT case missing: " + case["input"])
        output.mkdir(parents=True, exist_ok=True)
        for relative, payload in members.items():
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    (output / "import.json").write_text(
        json.dumps(
            dict(
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                files={
                    name: hashlib.sha256(payload).hexdigest()
                    for name, payload in members.items()
                },
            ),
            indent=2,
        )
        + "\n"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(import_submission(args.archive, args.out))
