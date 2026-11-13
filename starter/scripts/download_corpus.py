"""Download the OpenStax PDF declared in src/config.py and verify its SHA-256."""
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CORPUS, REPO_ROOT  # noqa: E402

for spec in CORPUS:
    target = REPO_ROOT / spec["local_path"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {spec['pdf_url']}")
        urllib.request.urlretrieve(spec["pdf_url"], target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest().upper()
    if digest != spec["sha256"]:
        sys.exit(f"{target}: SHA-256 {digest} does not match manifest {spec['sha256']}; "
                 "OpenStax may have published a new PDF version")
    print(f"{target.name}: OK ({target.stat().st_size} bytes, sha256 verified)")
