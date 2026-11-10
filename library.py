"""Offline search over reconstructed, cleaned Wikipedia articles."""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

DEFAULT_DB = Path(os.environ.get('DANIELLM_LIBRARY',Path(__file__).parent / 'data' / 'library.sqlite'))


def connect(path=DEFAULT_DB):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''CREATE TABLE IF NOT EXISTS articles(id INTEGER PRIMARY KEY, wiki_id TEXT UNIQUE, title TEXT, url TEXT, body TEXT);
    CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(title, body, content=articles, content_rowid=id);
    CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);
    CREATE INDEX IF NOT EXISTS articles_title ON articles(title COLLATE NOCASE);''')
    return db


def search(query, path=DEFAULT_DB):
    if not Path(path).exists():
        return []
    words = re.findall(r'\w+', query, flags=re.UNICODE)[:10]
    if not words:
        return []
    match = ' AND '.join('"'+w+'"' for w in words)
    with closing(sqlite3.connect(path, timeout=5)) as db:
        rows = db.execute('''SELECT a.id,a.title,a.url,snippet(search,1,'','', ' ... ',45)
          FROM search JOIN articles a ON a.id=search.rowid WHERE search MATCH ?
          ORDER BY bm25(search,5,1) LIMIT 12''', (match,)).fetchall()
    return [dict(id=r[0], title=r[1], url=r[2], excerpt=r[3]) for r in rows]


STOPWORDS = set('''a an and are as at be been but by can did do does for from had has have how i in is it its
of on or that the their them there these they this to was were what when where which who whom whose why will
with would you your about into than then tell me please much many'''.split())


def _terms(text):
    return [w for w in re.findall(r'\w+', text.lower(), flags=re.UNICODE) if w not in STOPWORDS]


def _stems(text):
    """Crude stems for overlap scoring: "painted" and "painting" both become "paint"."""
    return {w[:5] for w in _terms(text)}


def passages(question, k=3, chars=700, path=DEFAULT_DB, seconds=1.5):
    """Best paragraph from each of the k most relevant articles, for answering `question`.

    Candidates: articles whose title exactly equals a phrase from the question, plus
    full-text hits. Each candidate is scored by its best paragraph's share of question
    words, with a bonus for an exact title match (plain BM25 favours tiny articles).
    Ties go to the earlier paragraph (the opening one is usually the summary). Each
    search stage stops after about `seconds`, keeping what it found by then.
    """
    terms = list(dict.fromkeys(_terms(question)))[:8]
    if not terms or not Path(path).exists():
        return []
    candidates = {}
    with closing(sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True, timeout=5)) as db:
        deadline = [0.0]
        db.set_progress_handler(lambda: time.monotonic() > deadline[0], 10000)

        def query(match, order, limit):
            try:
                return db.execute(f'''SELECT a.id,a.title,a.url FROM search JOIN articles a ON a.id=search.rowid
                  WHERE search MATCH ? ORDER BY {order} LIMIT ?''', (match, limit)).fetchall()
            except sqlite3.OperationalError as exc:
                if 'interrupt' not in str(exc):
                    raise
                return []  # too slow (very common words); keep the candidates found so far
        deadline[0] = time.monotonic() + seconds
        # Exact titles, longest phrase first: "world war ii" before "war". Original capitalisation
        # and punctuation come from the question, so the phrase is re-read from its raw words.
        raw = re.findall(r"[\w'.-]+", question, flags=re.UNICODE)
        indexed = db.execute("SELECT 1 FROM sqlite_master WHERE name='articles_title'").fetchone()
        covered = set()  # word positions already inside a longer matched title
        for n in range(min(5, len(raw)), 0, -1):
            for i in range(len(raw) - n + 1):
                phrase = ' '.join(raw[i:i+n]).strip(".'-")
                span = set(range(i, i + n))
                edge_filler = raw[i].lower() in STOPWORDS or raw[i + n - 1].lower() in STOPWORDS
                if not _terms(phrase) or not indexed or span <= covered or edge_filler:
                    continue
                for r in db.execute('SELECT id,title,url FROM articles WHERE title=? COLLATE NOCASE LIMIT 3', (phrase,)):
                    candidates[r[0]] = (r, 1.0 + 0.1 * n if n > 1 else 0.4)
                    covered |= span
        deadline[0] = time.monotonic() + seconds
        for joiner in (' AND ', ' OR '):
            hits = query(joiner.join('"' + t + '"' for t in terms), 'bm25(search,5,1)', 20)
            for r in hits:
                candidates.setdefault(r[0], (r, 0.0))
            if len(hits) >= k:
                break
        db.set_progress_handler(None, 0)
        scored = []
        for (aid, title, url), bonus in candidates.values():
            body = db.execute('SELECT body FROM articles WHERE id=?', (aid,)).fetchone()[0]
            if title.endswith('(disambiguation)') or re.search(r'\b(may|commonly) refers? to\b', body[:400]):
                continue  # a list of other articles, not an answer
            paragraphs = [p.strip() for p in body.split('\n\n')[1:] if len(p.strip()) >= 80] or [body]
            wanted = _stems(' '.join(terms))
            best, _, text = max((len(wanted & _stems(p)) / len(wanted), -i, p) for i, p in enumerate(paragraphs))
            title_share = len(wanted & _stems(title)) / len(wanted)
            scored.append((best + bonus + 0.5 * title_share, -aid, dict(id=aid, title=title, url=url, text=_clip(text, chars))))
    return [s[2] for s in sorted(scored, key=lambda s: s[:2], reverse=True)[:k]]


def _clip(text, chars):
    text = re.sub(r'\s*\((?:[\s,;:]|or)*\)', '', ' '.join(text.split()))  # brackets left empty by markup removal
    return text if len(text) <= chars else text[:chars].rsplit(' ', 1)[0] + ' ...'


def article(article_id, path=DEFAULT_DB):
    if not Path(path).exists():
        return None
    with closing(sqlite3.connect(path)) as db:
        r = db.execute('SELECT title,url,body FROM articles WHERE id=?', (article_id,)).fetchone()
    return dict(title=r[0], url=r[1], body=r[2]) if r else None


def status(path=DEFAULT_DB):
    """Article count across every dataset indexed into this library."""
    if not Path(path).exists():
        return dict(articles=0, complete=False)
    with closing(sqlite3.connect(path)) as db:
        saved = [json.loads(v) for (v,) in db.execute("SELECT value FROM state WHERE key LIKE 'progress%' ORDER BY key")]
    if not saved:
        return dict(articles=0, complete=False)
    return dict(saved[0], articles=sum(s['articles'] for s in saved), complete=all(s['complete'] for s in saved),
                datasets=len(saved))


def build(data, path, limit):
    with closing(connect(path)) as db:
        _build(data, db, limit)


def _build(data, db, limit):
    import numpy as np
    from tokenizers import Tokenizer
    meta = json.loads((data / 'metadata.json').read_text())
    tok = Tokenizer.from_file(str(data / 'tokenizer.json'))
    if hashlib.sha256((data / 'tokenizer.json').read_bytes()).hexdigest() != meta['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum mismatch')
    eos = tok.token_to_id('<eos>')
    arrays = {s: np.memmap(data / (s+'.bin'), dtype='<u2', mode='r') for s in ('train','val')}
    fingerprint = hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    first = db.execute("SELECT value FROM state WHERE key='progress'").fetchone()
    # The first dataset uses the original keys; each added dataset gets keys of its own.
    suffix = '' if not first or json.loads(first[0])['fingerprint'] == fingerprint else ':' + fingerprint[:16]
    db.execute(f"INSERT OR REPLACE INTO state VALUES('source{suffix}',?)",(json.dumps(dict(metadata=meta,
        source_url='https://dumps.wikimedia.org/' if meta.get('dataset') == 'wikipedia-dump' else 'https://huggingface.co/datasets/wikimedia/wikipedia',
        note='Cleaned article text reconstructed from token files; original URLs retained. See source dataset licensing.')),))
    old = db.execute("SELECT value FROM state WHERE key=?", ('progress' + suffix,)).fetchone()
    state = json.loads(old[0]) if old else dict(articles=0, source_offset=0, train=0, val=0, fingerprint=fingerprint, complete=False)
    with (data / 'sources.jsonl').open('rb') as sources:
        sources.seek(state['source_offset'])
        while not limit or state['articles'] < limit:
            line = sources.readline()
            if not line:
                state['complete'] = True
                break
            source = json.loads(line)
            split = 'val' if int(hashlib.sha256(source['id'].encode()).hexdigest()[:8],16)%100==0 else 'train'
            start = state[split]
            a = arrays[split]
            end = start
            while end < len(a):
                found = np.flatnonzero(a[end:end+4096] == eos)
                if len(found):
                    end += int(found[0])
                    break
                end += min(4096, len(a)-end)
            if end >= len(a):
                raise ValueError('Missing article separator; source/token alignment failed')
            body = tok.decode(a[start:end].tolist())
            if not body.startswith(source['title'] + '\n'):
                raise ValueError('Article title/token alignment failed; refusing to misattribute text')
            if db.execute('SELECT 1 FROM articles WHERE wiki_id=?', (source['id'],)).fetchone():
                raise ValueError(f"Article {source['id']} is already in the library; datasets must not overlap")
            cur = db.execute('INSERT INTO articles(wiki_id,title,url,body) VALUES(?,?,?,?)',
                             (source['id'],source['title'],source['url'],body))
            db.execute('INSERT INTO search(rowid,title,body) VALUES(?,?,?)',(cur.lastrowid,source['title'],body))
            state[split] = end+1
            state['source_offset'] = sources.tell()
            state['articles'] += 1
            if state['articles'] % 250 == 0:
                db.execute("INSERT OR REPLACE INTO state VALUES(?,?)",('progress'+suffix,json.dumps(state)))
                db.commit()
                print(f"Indexed {state['articles']:,} articles",flush=True)
        db.execute("INSERT OR REPLACE INTO state VALUES(?,?)",('progress'+suffix,json.dumps(state)))
        db.commit()
    print(json.dumps(state,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('E:/wiki-llm/data/full'))
    p.add_argument('--db',type=Path,default=DEFAULT_DB)
    p.add_argument('--articles',type=int,default=10000,help='Total target; 0 indexes all. Resumes automatically.')
    args=p.parse_args()
    if args.articles < 0: p.error('articles must be nonnegative')
    build(args.data,args.db,args.articles)
