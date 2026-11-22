"""Check downloaded Colab outputs against the manifest.json written next to them.

Usage: python scripts/verify_manifest.py <folder with manifest.json and the files>
Exits with status 1 if any listed file is missing or differs in bytes or SHA-256.
"""
import hashlib
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
manifest = json.loads((folder / "manifest.json").read_text())
bad = 0
for name, want in manifest.items():
    path = folder / name
    if not path.exists():
        print("MISSING ", name)
        bad += 1
        continue
    data = path.read_bytes()
    ok = len(data) == want["bytes"] and hashlib.sha256(data).hexdigest() == want["sha256"]
    print("ok      " if ok else "DIFFERS ", name)
    bad += not ok
print(f"{len(manifest) - bad}/{len(manifest)} files match")
sys.exit(1 if bad else 0)
