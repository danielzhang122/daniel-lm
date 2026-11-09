"""Verify exported files without installing PyTorch or using the network."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root): raise ValueError('Invalid manifest path')
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected: raise ValueError('Checksum mismatch: '+name)
    return len(manifest['sha256'])


if __name__=='__main__':
    print(f'Verified {verify(Path(__file__).parent)} files. Checksums detect changes, not source authenticity.')
