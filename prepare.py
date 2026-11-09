"""Stream Wikipedia, train a tokenizer, and pack article-disjoint splits."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path(__file__).parent / '.cache' / 'huggingface'))
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from cleaning import clean_article


def validation(article):
    return int(hashlib.sha256(article['id'].encode()).hexdigest()[:8], 16) % 100 == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--articles', type=int, default=10000, help='0 = all English articles')
    parser.add_argument('--tokenizer-articles', type=int, default=10000)
    parser.add_argument('--out', type=Path, default=Path('data/pilot'))
    parser.add_argument('--tokenizer', type=Path, help='Reuse a previously trained tokenizer')
    parser.add_argument('--min-chars', type=int, default=200)
    args = parser.parse_args()
    if args.articles < 0 or args.tokenizer_articles < 1 or args.min_chars < 0:
        parser.error('Article counts must be nonnegative; tokenizer count must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise SystemExit('Output must be empty: use a new --out directory.')
    revision = HfApi().dataset_info('wikimedia/wikipedia').sha

    def source():
        ds = load_dataset('wikimedia/wikipedia', '20231101.en', revision=revision,
                          split='train', streaming=True)
        ds = ds.shuffle(seed=42, buffer_size=1000)
        scanned = itertools.islice(ds, args.articles) if args.articles else iter(ds)
        for article in scanned:
            article = dict(article)
            article['text'] = clean_article(article['text'])
            if len(article['text']) >= args.min_chars:
                yield article

    if args.tokenizer:
        tok = Tokenizer.from_file(str(args.tokenizer))
    else:
        tok = Tokenizer(models.BPE(unk_token='<unk>'))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=16384, special_tokens=['<unk>', '<eos>'],
                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        texts = (a['title'] + '\n\n' + a['text'] for a in source() if not validation(a))
        tok.train_from_iterator(itertools.islice(texts, args.tokenizer_articles), trainer)
    if tok.get_vocab_size() > 65536 or tok.token_to_id('<eos>') is None:
        raise SystemExit('Tokenizer must fit uint16 and include <eos>.')
    tok.save(str(args.out / 'tokenizer.json'))
    counts = {'train': 0, 'val': 0}
    articles = {'train': 0, 'val': 0}
    with (args.out / 'train.bin').open('wb') as train, (args.out / 'val.bin').open('wb') as val, (args.out / 'sources.jsonl').open('w', encoding='utf-8') as provenance:
        for i, a in enumerate(source(), 1):
            split = 'val' if validation(a) else 'train'
            ids = tok.encode(a['title'] + '\n\n' + a['text']).ids + [tok.token_to_id('<eos>')]
            np.asarray(ids, dtype='<u2').tofile(val if split == 'val' else train)
            counts[split] += len(ids)
            articles[split] += 1
            provenance.write(json.dumps({k: a[k] for k in ('id', 'url', 'title')}, ensure_ascii=False) + '\n')
            if i % 1000 == 0:
                print(f'{i:,} articles; {sum(counts.values()):,} tokens', flush=True)
    metadata = dict(dataset='wikimedia/wikipedia', config='20231101.en', revision=revision,
                    tokens=counts, articles=articles, vocab_size=tok.get_vocab_size(),
                    cleaning='terminal-headings-v1', min_chars=args.min_chars,
                    requested_articles=args.articles,
                    tokenizer_sha256=hashlib.sha256((args.out / 'tokenizer.json').read_bytes()).hexdigest(),
                    dtype='little-endian uint16', validation='sha256(article id) first 8 hex digits modulo 100 == 0')
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
