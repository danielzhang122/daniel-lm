"""Prepare short, complete conversations with assistant-only training labels."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
os.environ.setdefault('HF_HOME',str(Path(__file__).parent/'.cache'/'huggingface'))
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer
from dialogue import encode_messages


def prepare(tokenizer, out, examples, context):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    tok=Tokenizer.from_file(str(tokenizer))
    name='HuggingFaceTB/smol-smoltalk'
    revision=HfApi().dataset_info(name).sha
    card=hf_hub_download(name,'README.md',repo_type='dataset',revision=revision)
    shutil.copyfile(card,out/'SOURCE_DATASET_CARD.md')
    shutil.copyfile(tokenizer,out/'tokenizer.json')
    counts={'train':0,'val':0,'skipped':0,'scanned':0}
    ds=load_dataset(name,split='train',revision=revision,streaming=True).shuffle(seed=42,buffer_size=1000)
    seen=set()
    with (out/'train.jsonl').open('w',encoding='utf-8') as train, (out/'val.jsonl').open('w',encoding='utf-8') as val:
        for row in ds:
            counts['scanned']+=1
            messages=row['messages']
            # Keep complete short conversations. Never cut off an answer to make it fit.
            try: ids,labels=encode_messages(tok,messages)
            except (ValueError,KeyError,TypeError): counts['skipped']+=1; continue
            if len(ids)>context+1 or len(ids)<2 or not any(x!=-100 for x in labels[1:]) or messages[-1]['role']!='assistant':
                counts['skipped']+=1
                continue
            prompts=json.dumps([m for m in messages if m['role']!='assistant'],sort_keys=True,ensure_ascii=False)
            key=hashlib.sha256(prompts.encode()).hexdigest()
            if key in seen: counts['skipped']+=1; continue
            seen.add(key)
            split='val' if int(key[:8],16)%50==0 else 'train'
            record=dict(ids=ids,labels=labels,messages=messages,source=row.get('source'),prompt_hash=key)
            (val if split=='val' else train).write(json.dumps(record,ensure_ascii=False)+'\n')
            counts[split]+=1
            if (counts['train']+counts['val'])%500==0: print(counts,flush=True)
            if counts['train']+counts['val']>=examples: break
    if not counts['train'] or not counts['val']: raise ValueError('Too few examples for both splits')
    meta=dict(dataset=name,revision=revision,source_url='https://huggingface.co/datasets/'+name,
              declared_license='apache-2.0 (see retained source card and upstream component terms)',
              synthetic=True,context=context,format='plain-roles-eos-v1',counts=counts,
              files_sha256={s:hashlib.sha256((out/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','val')},
              tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tokenizer',type=Path,default=Path('data/clean-100k/tokenizer.json'))
    p.add_argument('--out',type=Path,default=Path('data/dialogue'))
    p.add_argument('--examples',type=int,default=5000)
    p.add_argument('--context',type=int,default=512)
    a=p.parse_args()
    if a.examples<100 or a.context<16: p.error('Use at least 100 examples and context 16')
    prepare(a.tokenizer,a.out,a.examples,a.context)
