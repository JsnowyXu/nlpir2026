#!/usr/bin/env python
"""Package only the standalone main experiment, excluding data and checkpoints."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'main_experiment'
out = ROOT / 'artifacts/releases'
out.mkdir(parents=True, exist_ok=True)
files = [p for p in sorted(source.rglob('*')) if p.is_file()
         and not any(part in {'__pycache__', '.pytest_cache', 'dist', 'build', 'data', 'results'}
                     or part.endswith('.egg-info') for part in p.relative_to(source).parts)
         and p.suffix not in {'.pyc', '.pt', '.npz', '.npy'}
         and p.name != 'RELEASE_MANIFEST.json']
manifest = {'package': 'patentflow-main', 'version': '0.1.0',
            'files': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'scope': 'Main training/evaluation only. Prepared artifacts are external inputs.'}
path = source / 'RELEASE_MANIFEST.json'
path.write_text(json.dumps(manifest, indent=2))
files.append(path)
archive = out / 'patentflow-main-20260919.tar.gz'
with tarfile.open(archive, 'w:gz') as bundle:
    for path in files:
        info = bundle.gettarinfo(str(path), arcname='main_experiment/' + str(path.relative_to(source)))
        info.uid = info.gid = 0
        info.uname = info.gname = ''
        with path.open('rb') as handle:
            bundle.addfile(info, handle)
checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix(archive.suffix + '.sha256').write_text(checksum + '  ' + archive.name + '\n')
print(str(archive), 'files', len(files), 'bytes', archive.stat().st_size, 'sha256', checksum)
