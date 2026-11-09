"""Offline search over reconstructed, cleaned Wikipedia articles."""
import argparse
import hashlib
import json
import os
import re
import sqlite3
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
    CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);''')
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


def article(article_id, path=DEFAULT_DB):
    if not Path(path).exists():
        return None
    with closing(sqlite3.connect(path)) as db:
        r = db.execute('SELECT title,url,body FROM articles WHERE id=?', (article_id,)).fetchone()
    return dict(title=r[0], url=r[1], body=r[2]) if r else None


def status(path=DEFAULT_DB):
    if not Path(path).exists():
        return dict(articles=0, complete=False)
    with closing(sqlite3.connect(path)) as db:
        saved = db.execute("SELECT value FROM state WHERE key='progress'").fetchone()
    return json.loads(saved[0]) if saved else dict(articles=0, complete=False)


def build(data, path, limit):
    import numpy as np
    from tokenizers import Tokenizer
    meta = json.loads((data / 'metadata.json').read_text())
    tok = Tokenizer.from_file(str(data / 'tokenizer.json'))
    if hashlib.sha256((data / 'tokenizer.json').read_bytes()).hexdigest() != meta['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum mismatch')
    eos = tok.token_to_id('<eos>')
    arrays = {s: np.memmap(data / (s+'.bin'), dtype='<u2', mode='r') for s in ('train','val')}
    fingerprint = hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    db = connect(path)
    db.execute("INSERT OR REPLACE INTO state VALUES('source',?)",(json.dumps(dict(metadata=meta,
        source_url='https://huggingface.co/datasets/wikimedia/wikipedia',
        note='Cleaned article text reconstructed from token files; original URLs retained. See source dataset licensing.')),))
    old = db.execute("SELECT value FROM state WHERE key='progress'").fetchone()
    state = json.loads(old[0]) if old else dict(articles=0, source_offset=0, train=0, val=0, fingerprint=fingerprint, complete=False)
    if state['fingerprint'] != fingerprint:
        raise ValueError('Library belongs to a different dataset; choose another --db')
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
            cur = db.execute('INSERT INTO articles(wiki_id,title,url,body) VALUES(?,?,?,?)',
                             (source['id'],source['title'],source['url'],body))
            db.execute('INSERT INTO search(rowid,title,body) VALUES(?,?,?)',(cur.lastrowid,source['title'],body))
            state[split] = end+1
            state['source_offset'] = sources.tell()
            state['articles'] += 1
            if state['articles'] % 250 == 0:
                db.execute("INSERT OR REPLACE INTO state VALUES('progress',?)",(json.dumps(state),))
                db.commit()
                print(f"Indexed {state['articles']:,} articles",flush=True)
        db.execute("INSERT OR REPLACE INTO state VALUES('progress',?)",(json.dumps(state),))
        db.commit()
    db.close()
    print(json.dumps(state,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('E:/wiki-llm/data/full'))
    p.add_argument('--db',type=Path,default=DEFAULT_DB)
    p.add_argument('--articles',type=int,default=10000,help='Total target; 0 indexes all. Resumes automatically.')
    args=p.parse_args()
    if args.articles < 0: p.error('articles must be nonnegative')
    build(args.data,args.db,args.articles)
