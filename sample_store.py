"""Persist text samples, not large model checkpoints."""
import datetime
import json
from pathlib import Path
import sqlite3
from contextlib import closing

DB = Path(__file__).parent / 'data' / 'samples.sqlite'


def connection(path=DB):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path,timeout=10)
    db.execute('CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY, created TEXT, payload TEXT)')
    return db


def save(run,prompt,result,temperature=.8,requested_tokens=80,path=DB):
    item=dict(run=run,prompt=prompt,result=result,temperature=temperature,requested_tokens=requested_tokens,seed=42)
    with closing(connection(path)) as db, db:
        cursor=db.execute('INSERT INTO samples(created,payload) VALUES(?,?)',
                          (datetime.datetime.now(datetime.timezone.utc).isoformat(),json.dumps(item)))
        return cursor.lastrowid


def recent(path=DB):
    with closing(connection(path)) as db:
        rows=db.execute('SELECT id,created,payload FROM samples ORDER BY id DESC LIMIT 200').fetchall()
    return [dict(id=r[0],created=r[1],**json.loads(r[2])) for r in rows]
