"""Join prepared dataset folders end to end into one, for training on a mix (e.g. FineWeb-Edu + Wikipedia).

All inputs must use the same tokenizer (checked by hash). train.bin and val.bin are the inputs' files
joined in the order given; the trainer's block shuffle mixes them, so the order does not matter for training.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def combine(inputs, out):
    metas = [json.loads((d / 'metadata.json').read_text(encoding='utf-8')) for d in inputs]
    hashes = {m['tokenizer_sha256'] for m in metas}
    if len(hashes) != 1:
        raise SystemExit('Inputs use different tokenizers; they cannot be combined.')
    for d, m in zip(inputs, metas):
        if hashlib.sha256((d / 'tokenizer.json').read_bytes()).hexdigest() != m['tokenizer_sha256']:
            raise SystemExit(f'{d}: tokenizer.json does not match its metadata.')
        for split in ('train', 'val'):
            if (d / f'{split}.bin').stat().st_size != 2 * m['tokens'][split]:
                raise SystemExit(f'{d}: {split}.bin does not match its metadata.')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit('Output must be empty: use a new --out directory.')
    shutil.copyfile(inputs[0] / 'tokenizer.json', out / 'tokenizer.json')
    for split in ('train', 'val'):
        with (out / f'{split}.bin').open('wb') as dst:
            for d in inputs:
                with (d / f'{split}.bin').open('rb') as src:
                    shutil.copyfileobj(src, dst, 64 << 20)
    meta = dict(
        dataset='combined',
        sources=[dict(path=str(d), dataset=m.get('dataset'), tokens=m['tokens'], license=m.get('license'))
                 for d, m in zip(inputs, metas)],
        tokens={s: sum(m['tokens'][s] for m in metas) for s in ('train', 'val')},
        vocab_size=metas[0]['vocab_size'],
        tokenizer_sha256=hashes.pop(),
        dtype='little-endian uint16',
    )
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', type=Path, nargs='+', help='Prepared dataset folders to join')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(combine(a.inputs, a.out), indent=2))


if __name__ == '__main__':
    main()
