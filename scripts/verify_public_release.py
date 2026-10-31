"""Verify checksummed artifacts included in the public ARF release.

Run from any working directory with:
    python scripts/verify_public_release.py

The raw ARF data and the generated outer-fold file are intentionally excluded
from version control. Their checksum records are therefore reported as skips,
not failures, until an authorized user supplies the data and generates splits.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_DIR = ROOT / "data" / "checksums"
OPTIONAL_PREFIXES = (
    "data/splits/",
    "git_ignore_folder/",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    verified = 0
    skipped = 0

    for manifest in sorted(CHECKSUM_DIR.iterdir()):
        if manifest.suffix not in {".sha256", ".txt"}:
            continue
        for raw_line in manifest.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            expected, relative_path = line.split(maxsplit=1)
            relative_path = relative_path.strip()
            target = ROOT / relative_path
            if not target.is_file():
                if relative_path.startswith(OPTIONAL_PREFIXES):
                    skipped += 1
                    print(f"SKIP {relative_path} (intentionally excluded from public release)")
                else:
                    failures.append(f"MISSING {relative_path}")
                continue
            actual = sha256(target)
            if actual != expected:
                failures.append(f"MISMATCH {relative_path}")
            else:
                verified += 1
                print(f"OK   {relative_path}")

    if failures:
        print("\nVerification failed:")
        print("\n".join(failures))
        return 1

    print(f"\nPASS: verified {verified} public artifacts; skipped {skipped} excluded artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
