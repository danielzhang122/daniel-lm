"""Turn an official Wikipedia XML dump into a dataset folder like prepare.py makes.

Used to add articles the Hugging Face snapshot lacks (for example "Paris"): pass
--skip-ids with the existing sources.jsonl and only articles not already there are kept.
Output: train.bin, val.bin, sources.jsonl, tokenizer.json, metadata.json.
"""
import argparse
import bz2
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import re
from urllib.parse import quote
import xml.etree.ElementTree as ET

import numpy as np

from cleaning import clean_article

DROP_TAGS = {'ref', 'gallery', 'math', 'chem', 'score', 'timeline', 'graph', 'mapframe', 'imagemap', 'syntaxhighlight', 'templatedata'}
DROP_LINKS = ('file:', 'image:', 'category:', 'media:')


def wikitext_to_text(wikitext):
    """Plain article text: templates, references, tables and file links removed; headings kept as lines."""
    import mwparserfromhell
    code = mwparserfromhell.parse(wikitext)
    for node in list(code.ifilter_templates(recursive=False)):
        _remove(code, node)
    for node in list(code.ifilter_tags(recursive=True)):
        if str(node.tag).strip().lower() in DROP_TAGS or str(node.tag).strip().lower() == 'table':
            _remove(code, node)
    for node in list(code.ifilter_wikilinks(recursive=True)):
        if str(node.title).strip().lower().startswith(DROP_LINKS):
            _remove(code, node)
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.S)           # leftover wiki tables
    text = re.sub(r"'{2,}", '', text)                                 # bold/italic quotes
    text = re.sub(r'\s*\((?:[\s,;:]|&nbsp;)*\)', '', text)            # brackets emptied by template removal
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _remove(code, node):
    try:
        code.remove(node)
    except ValueError:
        pass  # already removed with its parent


def convert(page):
    page_id, title, wikitext = page
    try:
        text = clean_article(wikitext_to_text(wikitext))
    except Exception:  # a malformed page should not stop a 7M-page run
        return None
    return dict(id=page_id, url='https://en.wikipedia.org/wiki/' + quote(title), title=title, text=text)


def pages(dump, skip):
    """Yield (id, title, wikitext) for main-namespace, non-redirect pages not in `skip`."""
    with bz2.open(dump, 'rb') as f:
        context = ET.iterparse(f, events=('end',))
        ns = None
        for _, elem in context:
            tag = elem.tag.rsplit('}', 1)[-1]
            if tag != 'page':
                continue
            if ns is None:
                ns = elem.tag[:-len('page')]
            title = elem.findtext(ns + 'title')
            page_id = elem.findtext(ns + 'id')
            redirect = elem.find(ns + 'redirect') is not None
            text = elem.findtext(f'{ns}revision/{ns}text') or ''
            if elem.findtext(ns + 'ns') == '0' and not redirect and page_id not in skip and not text.lstrip().lower().startswith('#redirect'):
                yield page_id, title, text
            elem.clear()


def validation(page_id):
    return int(hashlib.sha256(page_id.encode()).hexdigest()[:8], 16) % 100 == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dump', type=Path, required=True, help='enwiki-*-pages-articles*.xml.bz2')
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--skip-ids', type=Path, help='sources.jsonl of a dataset whose articles to leave out')
    p.add_argument('--min-chars', type=int, default=200)
    p.add_argument('--limit', type=int, default=0, help='Stop after this many kept articles (0 = all), for testing')
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    if a.limit < 0 or a.workers < 1 or a.min_chars < 0:
        p.error('limit, workers and min-chars must be nonnegative (workers at least 1)')
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise SystemExit('Output must be empty: use a new --out directory.')
    from tokenizers import Tokenizer
    (a.out / 'tokenizer.json').write_bytes(a.tokenizer.read_bytes())  # byte-identical for transfer checks
    tok = Tokenizer.from_file(str(a.out / 'tokenizer.json'))
    eos = tok.token_to_id('<eos>')
    skip = set()
    if a.skip_ids:
        with a.skip_ids.open('rb') as f:
            skip = {json.loads(line)['id'] for line in f}
        print(f'Leaving out {len(skip):,} articles already in {a.skip_ids}', flush=True)
    counts, articles, short = {'train': 0, 'val': 0}, {'train': 0, 'val': 0}, 0
    with Pool(a.workers) as pool, (a.out / 'train.bin').open('wb') as train, \
            (a.out / 'val.bin').open('wb') as val, (a.out / 'sources.jsonl').open('w', encoding='utf-8') as sources:
        for done, art in enumerate(pool.imap(convert, pages(a.dump, skip), chunksize=16), 1):
            if art is None or len(art['text']) < a.min_chars:
                short += 1
            else:
                split = 'val' if validation(art['id']) else 'train'
                ids = tok.encode(art['title'] + '\n\n' + art['text']).ids + [eos]
                np.asarray(ids, dtype='<u2').tofile(val if split == 'val' else train)
                counts[split] += len(ids)
                articles[split] += 1
                sources.write(json.dumps({k: art[k] for k in ('id', 'url', 'title')}, ensure_ascii=False) + '\n')
            if done % 10000 == 0:
                print(f'{done:,} pages read; {sum(articles.values()):,} kept; {sum(counts.values()):,} tokens', flush=True)
            if a.limit and sum(articles.values()) >= a.limit:
                pool.terminate()
                break
    meta = dict(dataset='wikipedia-dump', dump=a.dump.name, tokens=counts, articles=articles, skipped_short=short,
                skip_ids=str(a.skip_ids) if a.skip_ids else None, vocab_size=tok.get_vocab_size(),
                cleaning='mwparserfromhell-strip-v1+terminal-headings-v1', min_chars=a.min_chars,
                tokenizer_sha256=hashlib.sha256((a.out / 'tokenizer.json').read_bytes()).hexdigest(),
                dtype='little-endian uint16', validation='sha256(article id) first 8 hex digits modulo 100 == 0',
                license='CC BY-SA 4.0 (Wikipedia text)')
    (a.out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
