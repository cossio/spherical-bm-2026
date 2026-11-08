# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Refresh release hashes only after an intentional change to data or figures."""
from pathlib import Path
import hashlib
import json
root=Path(__file__).resolve().parents[1]
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['data','figures'] for p in sorted((root/folder).rglob('*'))
        if p.is_file() and p.name!='sha256.json'}
(root/'data/sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
