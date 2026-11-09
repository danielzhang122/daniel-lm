"""Export inference-only weights and a portable, self-contained source bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from contextlib import closing
import torch

ROOT=Path(__file__).parent


def export(checkpoint,tokenizer,out,library=None):
    if out.exists() and any(out.iterdir()): raise ValueError('Use an empty output directory')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    digest=hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    if digest!=ck['data']['tokenizer_sha256']: raise ValueError('Tokenizer mismatch')
    out.mkdir(parents=True,exist_ok=True)
    slim={key:ck[key] for key in ('model','config','data','step')}
    slim['phase']=ck.get('phase','pretrain')
    torch.save(slim,out/'model.pt')
    shutil.copyfile(tokenizer,out/'tokenizer.json')
    for file in ['model.py','generate.py','dialogue.py','portable_server.py','verify_bundle.py','library.py']:
        shutil.copyfile(ROOT/file,out/file)
    if library:
        (out/'data').mkdir()
        with closing(sqlite3.connect(f'{Path(library).resolve().as_uri()}?mode=ro',uri=True)) as source, closing(sqlite3.connect(out/'data'/'library.sqlite')) as dest:
            source.backup(dest)
    (out/'requirements.txt').write_text('torch>=2.6,<3\ntokenizers>=0.21,<0.23\n',encoding='utf-8')
    card=dict(name='daniellm',step=ck['step'],phase=slim['phase'],config=ck['config'],
              source_checkpoint=str(checkpoint.resolve()),tokenizer_sha256=digest,
              limitations='Experimental small model. May repeat or invent facts. Not independently validated for reliability.',
              reference_source='https://huggingface.co/datasets/wikimedia/wikipedia',
              library_included=bool(library))
    (out/'model_card.json').write_text(json.dumps(card,indent=2),encoding='utf-8')
    (out/'README.md').write_text('''# daniellm portable snapshot

This is an intermediate model snapshot, not a finished reliable assistant.
Weights, tokenizer, source code, and file checksums are included. Python and dependency wheels are NOT included.
No Wikipedia training binaries or optimizer state are needed.

On the Linux PC, while internet is available (Python 3.11 or later):

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install 'tokenizers>=0.21,<0.23'
python verify_bundle.py
python portable_server.py
```

Open http://127.0.0.1:8765. Use `--host 0.0.0.0` only on a trusted home network;
this basic server has no authentication and is not for public internet hosting.
CPU speed and compatibility must be tested on the actual Dell.

To prepare recovery packages on that same Linux architecture and Python version:

```sh
pip freeze > requirements-linux.txt
pip download --dest wheels -r requirements-linux.txt --extra-index-url https://download.pytorch.org/whl/cpu
# On a matching machine, offline:
pip install --no-index --find-links wheels -r requirements-linux.txt
```

Back up this directory, the Linux wheel folder, and Python/OS installation media separately.
Test restoration and generation with networking disabled before relying on the backup.
The checksums do not authenticate an untrusted download. Only load your own trusted model files.

If included, `data/library.sqlite` is a partial offline library of cleaned Wikipedia text,
with titles and source URLs preserved. It is not a complete Wikipedia mirror. Source license information:
https://huggingface.co/datasets/wikimedia/wikipedia . Keep attribution and source licensing information with redistributed content.
''',encoding='utf-8')
    files={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out/'manifest.json').write_text(json.dumps({'sha256':files},indent=2))
    print(json.dumps(dict(out=str(out),step=ck['step'],weight_bytes=(out/'model.pt').stat().st_size,files=len(files))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=Path('runs/full/latest.pt'))
    p.add_argument('--tokenizer',type=Path,default=Path('data/clean-100k/tokenizer.json'))
    p.add_argument('--out',type=Path,default=Path('exports/daniellm-snapshot'))
    p.add_argument('--library',type=Path)
    a=p.parse_args();export(a.checkpoint,a.tokenizer,a.out,a.library)
