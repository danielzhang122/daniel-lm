"""Prepare short, complete conversations with assistant-only training labels.

Two sources, shuffled together:
- smol-smoltalk: general chat (Apache-2.0).
- SQuAD 2.0: library passages + question -> answer taken from a passage (CC BY-SA 4.0).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
os.environ.setdefault('HF_HOME',str(Path(__file__).parent/'.cache'/'huggingface'))
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer
from dialogue import encode_messages, rag_user_message, RAG_NOT_FOUND

CHAT = 'HuggingFaceTB/smol-smoltalk'
SQUAD = 'rajpurkar/squad_v2'
DISTRACTOR_CHARS = 700  # about the size of a passage the dashboard retrieves
MAX_PASSAGE_CHARS = 2000


def clip(text, chars):
    """Shorten at a word boundary."""
    text = ' '.join(text.split())
    if len(text) <= chars:
        return text
    return text[:chars].rsplit(' ', 1)[0] + ' ...'


def squad_conversations(rows, count, rng):
    """Build `count` retrieval conversations: the real paragraph plus 2 distractors from other articles."""
    contexts = {}
    rows = [{**r, 'title': r['title'].replace('_', ' ')} for r in rows]  # match library titles
    for r in rows:
        contexts.setdefault(r['context'], r['title'])
    pool = list(contexts.items())
    order = list(range(len(rows)))
    rng.shuffle(order)
    made = 0
    for i in order:
        if made >= count:
            return
        r = rows[i]
        if len(r['context']) > MAX_PASSAGE_CHARS:
            continue
        distractors = []
        while len(distractors) < 2:
            text, title = pool[rng.randrange(len(pool))]
            if title != r['title'] and text not in (d['text'] for d in distractors):
                distractors.append(dict(title=title, text=clip(text, DISTRACTOR_CHARS)))
        position = rng.randrange(3)
        passages = distractors[:position] + [dict(title=r['title'], text=r['context'])] + distractors[position:]
        answers = r['answers']['text']
        reply = f'{answers[0].strip()} [{position + 1}].' if answers else RAG_NOT_FOUND
        made += 1
        yield [dict(role='user', content=rag_user_message(passages, r['question'])),
               dict(role='assistant', content=reply)], dict(source='squad_v2', squad_id=r['id'],
                                                             answerable=bool(answers), answers=answers)


def chat_conversations(revision):
    ds = load_dataset(CHAT, split='train', revision=revision, streaming=True).shuffle(seed=42, buffer_size=1000)
    for row in ds:
        yield row['messages'], dict(source=row.get('source'))


def encode(tok, messages, context):
    """Token ids and labels, or None when the conversation doesn't fit whole (never cut an answer)."""
    try:
        ids, labels = encode_messages(tok, messages)
    except (ValueError, KeyError, TypeError):
        return None
    if len(ids) > context + 1 or len(ids) < 2 or not any(x != -100 for x in labels[1:]) or messages[-1]['role'] != 'assistant':
        return None
    return ids, labels


def prepare(tokenizer, out, examples, context, rag_examples=0, rag_val=0):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    tok=Tokenizer.from_file(str(tokenizer))
    shutil.copyfile(tokenizer,out/'tokenizer.json')
    records={'train':[],'val':[]}
    counts={}
    sources=[]
    seen=set()

    def add(messages, extra, split, tally):
        tally['scanned']+=1
        encoded=encode(tok,messages,context)
        if encoded is None:
            tally['skipped']+=1
            return False
        prompts=json.dumps([m for m in messages if m['role']!='assistant'],sort_keys=True,ensure_ascii=False)
        key=hashlib.sha256(prompts.encode()).hexdigest()
        if key in seen:
            tally['skipped']+=1
            return False
        seen.add(key)
        split=split or ('val' if int(key[:8],16)%50==0 else 'train')
        records[split].append(dict(ids=encoded[0],labels=encoded[1],messages=messages,prompt_hash=key,**extra))
        tally[split]+=1
        return True

    if examples:
        revision=HfApi().dataset_info(CHAT).sha
        card=hf_hub_download(CHAT,'README.md',repo_type='dataset',revision=revision)
        shutil.copyfile(card,out/'SOURCE_DATASET_CARD.md')
        tally=counts['smol-smoltalk']=dict(train=0,val=0,skipped=0,scanned=0)
        for messages,extra in chat_conversations(revision):
            add(messages,extra,None,tally)
            if tally['train']+tally['val']>=examples: break
        sources.append(dict(dataset=CHAT,revision=revision,source_url='https://huggingface.co/datasets/'+CHAT,
                            declared_license='apache-2.0 (see retained source card and upstream component terms)',synthetic=True))
    if rag_examples:
        revision=HfApi().dataset_info(SQUAD).sha
        card=hf_hub_download(SQUAD,'README.md',repo_type='dataset',revision=revision)
        shutil.copyfile(card,out/'SQUAD_DATASET_CARD.md')
        squad=load_dataset(SQUAD,revision=revision)
        rng=random.Random(42)
        tally=counts['squad_v2']=dict(train=0,val=0,skipped=0,scanned=0)
        # SQuAD's own validation articles become our val set, so no val article is ever trained on.
        for split,source,wanted in (('train','train',rag_examples),('val','validation',rag_val)):
            target=tally[split]+wanted
            for messages,extra in squad_conversations(squad[source],len(squad[source]),rng):
                add(messages,extra,split,tally)
                if tally[split]>=target: break
        sources.append(dict(dataset=SQUAD,revision=revision,source_url='https://huggingface.co/datasets/'+SQUAD,
                            declared_license='cc-by-sa-4.0 (share-alike: keep this license if you publish the prepared data)',
                            synthetic=False))
    if not records['train'] or not records['val']: raise ValueError('Too few examples for both splits')
    shuffle=random.Random(42)
    for split,rows in records.items():
        shuffle.shuffle(rows)
        with (out/(split+'.jsonl')).open('w',encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    meta=dict(sources=sources,context=context,format='plain-roles-eos-v1',rag_format='rag-passages-v1',counts=counts,
              examples={s:len(r) for s,r in records.items()},
              files_sha256={s:hashlib.sha256((out/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','val')},
              tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tokenizer',type=Path,default=Path('data/clean-100k/tokenizer.json'))
    p.add_argument('--out',type=Path,default=Path('data/dialogue'))
    p.add_argument('--examples',type=int,default=5000,help='General chat conversations (smol-smoltalk); 0 for none')
    p.add_argument('--rag-examples',type=int,default=0,help='Passage + question conversations from SQuAD 2.0 train')
    p.add_argument('--rag-val',type=int,help='SQuAD 2.0 validation conversations for the val set (default: rag-examples / 30, at least 100)')
    p.add_argument('--context',type=int,default=512)
    a=p.parse_args()
    if a.rag_val is None: a.rag_val=max(100,a.rag_examples//30) if a.rag_examples else 0
    if a.examples<0 or a.rag_examples<0 or a.rag_val<0 or a.examples+a.rag_examples<100 or a.context<16:
        p.error('Use at least 100 examples in total and context 16')
    prepare(a.tokenizer,a.out,a.examples,a.context,a.rag_examples,a.rag_val)
