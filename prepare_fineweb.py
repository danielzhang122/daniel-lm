"""Stream FineWeb-Edu and pack it with an existing tokenizer, for a later training stage."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path(__file__).parent / '.cache' / 'huggingface'))
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer

DATASET, CONFIG = 'HuggingFaceFW/fineweb-edu', 'sample-10BT'


def validation(doc_id):
    return int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 100 == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', type=Path, required=True, help='tokenizer.json of the model being continued')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=3_000_000_000, help='Stop after about this many tokens')
    p.add_argument('--min-chars', type=int, default=200)
    a = p.parse_args()
    if a.tokens < 1:
        p.error('--tokens must be positive')
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise SystemExit('Output must be empty: use a new --out directory.')
    shutil.copyfile(a.tokenizer, a.out / 'tokenizer.json')  # byte-identical so transfer checks pass
    tok = Tokenizer.from_file(str(a.out / 'tokenizer.json'))
    eos = tok.token_to_id('<eos>')
    if eos is None or tok.get_vocab_size() > 65536:
        raise SystemExit('Tokenizer must fit uint16 and include <eos>.')
    revision = HfApi().dataset_info(DATASET).sha
    ds = load_dataset(DATASET, CONFIG, revision=revision, split='train', streaming=True)
    counts, docs = {'train': 0, 'val': 0}, {'train': 0, 'val': 0}
    batch = []

    def flush(train, val):
        texts = [d['text'] for d in batch]
        for d, enc in zip(batch, tok.encode_batch(texts)):
            split = 'val' if validation(d['id']) else 'train'
            ids = enc.ids + [eos]
            np.asarray(ids, dtype='<u2').tofile(val if split == 'val' else train)
            counts[split] += len(ids)
            docs[split] += 1
        batch.clear()

    with (a.out / 'train.bin').open('wb') as train, (a.out / 'val.bin').open('wb') as val:
        for d in ds:
            if len(d['text']) < a.min_chars:
                continue
            batch.append(d)
            if len(batch) == 1000:
                flush(train, val)
                print(f'{sum(docs.values()):,} documents; {sum(counts.values()):,} tokens', flush=True)
                if sum(counts.values()) >= a.tokens:
                    break
        if batch:
            flush(train, val)
    meta = dict(dataset=DATASET, config=CONFIG, revision=revision, tokens=counts, documents=docs,
                vocab_size=tok.get_vocab_size(), min_chars=a.min_chars, requested_tokens=a.tokens,
                tokenizer_sha256=hashlib.sha256((a.out / 'tokenizer.json').read_bytes()).hexdigest(),
                dtype='little-endian uint16', validation='sha256(document id) first 8 hex digits modulo 100 == 0')
    (a.out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
