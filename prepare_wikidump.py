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


RANGE_WORDS = {'-', '–', 'to', 'and', 'or', 'by', 'x', '×', 'to(-)', '+/-', '±'}
UNIT_NAMES = {'C': '°C', 'F': '°F', 'K': 'K', 'storey': 'storey', 'story': 'story'}
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']
KEEP_FIRST = {'nowrap', 'nobr', 'small', 'big', 'abbr', 'not a typo', 'sic', 'visible anchor', 'em',
              'strong', 'smallcaps', 'sc', 'math', 'mvar', 'var', 'nobold', 'noitalic', 'proper name'}
KEEP_LAST = {'lang', 'langx', 'transl', 'transliteration', 'script'}
SYMBOLS = {'nbsp': ' ', 'snd': ' – ', 'spaced ndash': ' – ', 'ndash': '–', 'mdash': '—', 'spaced en dash': ' – ',
           '\'': "'", '\'"': '\'"', 'thinsp': ' ', '=': '=', '!': '|', 'pi': 'π', 'deg': '°', 'degree': '°'}
DATE_TEMPLATES = {'start date', 'end date', 'birth date', 'death date', 'start date and age', 'birth date and age',
                  'death date and age', 'dts', 'date', 'film date', 'released'}
# Every template name _template_text can turn into text (plus "lang-xx"); all others are dropped unread.
KEPT_TEMPLATES = (set(SYMBOLS) | KEEP_FIRST | KEEP_LAST | DATE_TEMPLATES |
                  {'convert', 'cvt', 'val', 'frac', 'fraction', 'sfrac', 'circa', 'c.', 'as of', 'us$', 'usd'})


def _template_text(template):
    """Readable text for the templates that carry article content (values, units, dates, foreign words), else None."""
    name = str(template.name).strip().replace('_', ' ')
    low = name.lower()
    pos = [p.value.strip_code().strip() for p in template.params if not p.showkey]
    named = {str(p.name).strip(): p.value.strip_code().strip() for p in template.params if p.showkey}
    if low in SYMBOLS:
        return SYMBOLS[low]
    if low in ('convert', 'cvt') and pos:
        if named.get('disp') in ('out', 'output only', 'output number only'):
            return None  # shows only the converted value, which usually repeats the one written just before
        if len(pos) >= 4 and pos[1] in RANGE_WORDS:
            unit = UNIT_NAMES.get(pos[3], pos[3])
            return f'{pos[0]} {pos[1]} {pos[2]} {unit}'.replace(' - ', '–')
        unit = UNIT_NAMES.get(pos[1], pos[1]) if len(pos) > 1 else ''
        return f'{pos[0]} {unit}'.strip()
    if low == 'val' and pos:
        return f"{pos[0]} {named.get('u', named.get('ul', ''))}".strip()
    if low.startswith('lang-') and pos:
        return pos[0]
    if low in KEEP_LAST and len(pos) >= 2:
        return pos[1]
    if low in KEEP_FIRST and pos:
        return pos[0]
    if low in ('frac', 'fraction', 'sfrac') and pos:
        return pos[0] if len(pos) == 1 else (f'{pos[0]} {pos[1]}/{pos[2]}' if len(pos) >= 3 else f'{pos[0]}/{pos[1]}')
    if low in ('circa', 'c.') and pos:
        return 'c. ' + pos[0]
    if low == 'as of' and pos:
        return ('as of ' if named.get('lc') else 'As of ') + pos[0]
    if low in ('us$', 'usd') and pos:
        return 'US$' + pos[0]
    if low in DATE_TEMPLATES:
        nums = [p for p in pos if p.isdigit()]
        if nums and len(nums[0]) == 4:
            year = nums[0]
            month = MONTHS[int(nums[1]) - 1] if len(nums) > 1 and 1 <= int(nums[1]) <= 12 else ''
            day = str(int(nums[2])) if len(nums) > 2 and 1 <= int(nums[2]) <= 31 else ''
            return ' '.join(x for x in (day, month, year) if x)
    return None


def wikitext_to_text(wikitext):
    """Plain article text: references, tables, file links and layout templates removed; values written with
    templates (measurements, dates, foreign-language words) kept as text; headings kept as lines."""
    import mwparserfromhell
    code = mwparserfromhell.parse(wikitext)
    _clean_nodes(code, mwparserfromhell.parse)
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.S)           # leftover wiki tables
    text = re.sub(r"'{2,}", '', text)                                 # bold/italic quotes
    text = re.sub(r'\s*\((?:[\s,;:]|&nbsp;)*\)', '', text)            # brackets emptied by template removal
    text = re.sub(r'\(\s*(?:[,;:]\s*)+', '(', text)                       # "( ; Tour Eiffel )" -> "(Tour Eiffel )"
    text = re.sub(r'(?:\s*[,;:])+\s*\)', ')', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _clean_nodes(code, parse):
    """One pass over the parse tree, rebuilding each node list in place: dropped templates, tags and file
    links are skipped without looking inside; kept templates have their insides cleaned first, so in
    {{nowrap|{{convert|5|m}}}} the convert becomes "5 m" before nowrap keeps its first parameter.
    (Calling code.replace/remove per node instead searched the whole page each time: 1.5 s per big article.)"""
    from mwparserfromhell.nodes import Tag, Template, Wikilink
    kept = []
    for node in code.nodes:
        if isinstance(node, Template):
            low = str(node.name).strip().replace('_', ' ').lower()
            if low not in KEPT_TEMPLATES and not low.startswith('lang-'):
                continue
            for child in node.__children__():
                _clean_nodes(child, parse)
            try:
                text = _template_text(node)
            except (ValueError, IndexError):
                text = None
            if text:
                kept.extend(parse(text).nodes)
            continue
        if isinstance(node, Tag) and str(node.tag).strip().lower() in DROP_TAGS | {'table'}:
            continue
        if isinstance(node, Wikilink) and str(node.title).strip().lower().startswith(DROP_LINKS):
            continue
        for child in node.__children__():
            _clean_nodes(child, parse)
        kept.append(node)
    code.nodes = kept


_worker = {}


def _init_worker(tokenizer_path, min_chars):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    _worker.update(tok=tok, eos=tok.token_to_id('<eos>'), min_chars=min_chars)


def convert(page):
    """Clean and tokenize one page in a worker process (tokenizing in the main process made it the
    bottleneck). Returns None for a malformed or too-short page."""
    page_id, title, wikitext = page
    try:
        text = clean_article(wikitext_to_text(wikitext))
    except Exception:  # a malformed page should not stop a 7M-page run
        return None
    if len(text) < _worker['min_chars']:
        return None
    ids = _worker['tok'].encode(title + '\n\n' + text).ids + [_worker['eos']]
    return dict(id=page_id, url='https://en.wikipedia.org/wiki/' + quote(title), title=title,
                ids=np.asarray(ids, dtype='<u2').tobytes())


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
    skip = set()
    if a.skip_ids:
        with a.skip_ids.open('rb') as f:
            skip = {json.loads(line)['id'] for line in f}
        print(f'Leaving out {len(skip):,} articles already in {a.skip_ids}', flush=True)
    counts, articles, short = {'train': 0, 'val': 0}, {'train': 0, 'val': 0}, 0
    with Pool(a.workers, initializer=_init_worker, initargs=(a.out / 'tokenizer.json', a.min_chars)) as pool, \
            (a.out / 'train.bin').open('wb') as train, (a.out / 'val.bin').open('wb') as val, \
            (a.out / 'sources.jsonl').open('w', encoding='utf-8') as sources:
        for done, art in enumerate(pool.imap(convert, pages(a.dump, skip), chunksize=16), 1):
            if art is None:
                short += 1
            else:
                split = 'val' if validation(art['id']) else 'train'
                (val if split == 'val' else train).write(art['ids'])
                counts[split] += len(art['ids']) // 2
                articles[split] += 1
                sources.write(json.dumps({k: art[k] for k in ('id', 'url', 'title')}, ensure_ascii=False) + '\n')
            if done % 10000 == 0:
                print(f'{done:,} pages read; {sum(articles.values()):,} kept; {sum(counts.values()):,} tokens', flush=True)
            if a.limit and sum(articles.values()) >= a.limit:
                pool.terminate()
                break
    meta = dict(dataset='wikipedia-dump', dump=a.dump.name, tokens=counts, articles=articles, skipped_short=short,
                skip_ids=str(a.skip_ids) if a.skip_ids else None, vocab_size=tok.get_vocab_size(),
                cleaning='mwparserfromhell-strip-v3-nested-value-templates+terminal-headings-v1', min_chars=a.min_chars,
                tokenizer_sha256=hashlib.sha256((a.out / 'tokenizer.json').read_bytes()).hexdigest(),
                dtype='little-endian uint16', validation='sha256(article id) first 8 hex digits modulo 100 == 0',
                license='CC BY-SA 4.0 (Wikipedia text)')
    (a.out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
