"""Separate supervised dialogue stage. Run after pretraining, not concurrently."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import GPT
from training_utils import learning_rate,replace_checkpoint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data/dialogue'))
    p.add_argument('--out',type=Path,default=Path('runs/dialogue'))
    p.add_argument('--init-from',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--schedule-steps',type=int,default=1000)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--accum',type=int,default=4)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--eval-every',type=int,default=100)
    a=p.parse_args()
    if a.resume==bool(a.init_from): p.error('Choose exactly one of --init-from or --resume')
    if min(a.steps,a.schedule_steps,a.batch,a.accum,a.eval_every)<1: p.error('Counts must be positive')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    if a.device=='cpu': torch.set_num_threads(2)
    meta=json.loads((a.data/'metadata.json').read_text())
    digest=hashlib.sha256((a.data/'tokenizer.json').read_bytes()).hexdigest()
    if digest!=meta['tokenizer_sha256']: raise ValueError('Tokenizer mismatch')
    for split,expected in meta.get('files_sha256',{}).items():
        if split not in ('train','val') or hashlib.sha256((a.data/(split+'.jsonl')).read_bytes()).hexdigest()!=expected:
            raise ValueError('Dialogue data checksum mismatch')
    rows={s:[json.loads(l) for l in (a.data/(s+'.jsonl')).read_text(encoding='utf-8').splitlines()] for s in ['train','val']}
    if not all(rows.values()): raise ValueError('Both splits must contain examples')
    a.out.mkdir(parents=True,exist_ok=True)
    dest=a.out/'latest.pt'
    if dest.exists() and not a.resume: raise ValueError('Output exists; use a new folder or --resume')
    ck=torch.load(dest if a.resume else a.init_from,map_location='cpu',weights_only=False)
    if ck['data']['tokenizer_sha256']!=digest: raise ValueError('Source checkpoint uses a different tokenizer')
    if meta['context']!=ck['config']['context']: raise ValueError('Context mismatch')
    settings=dict(batch=a.batch,accum=a.accum,horizon=a.schedule_steps,lr=5e-5)
    if a.resume and (ck.get('dialogue_metadata')!=meta or ck.get('sft_settings')!=settings):
        raise ValueError('Resume data/settings mismatch')
    torch.manual_seed(42)
    model=GPT(**ck['config']).to(a.device)
    model.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01)
    rng=np.random.default_rng(42)
    start=0
    if a.resume:
        opt.load_state_dict(ck['optimizer']); start=ck['step'];rng.bit_generator.state=ck['rng']
    source=ck.get('transfer') if a.resume else dict(checkpoint=str(a.init_from.resolve()),source_step=ck['step'])
    data=dict(tokenizer_sha256=digest,vocab_size=model.config['vocab_size'],phase='dialogue')
    (a.out/'run.json').write_text(json.dumps(dict(data=str(a.data.resolve()),phase='dialogue',target_steps=a.steps,
        parameters=sum(x.numel() for x in model.parameters()),transfer=source,sft_settings=settings),indent=2))
    def loss(split,generator):
        batch=[rows[split][i] for i in generator.integers(len(rows[split]),size=a.batch)]
        width=max(len(r['ids']) for r in batch)
        x=torch.zeros((a.batch,width),dtype=torch.long,device=a.device)
        labels=torch.full_like(x,-100)
        for i,r in enumerate(batch):
            n=len(r['ids']); x[i,:n]=torch.tensor(r['ids'],device=a.device);labels[i,:n]=torch.tensor(r['labels'],device=a.device)
        logits=model(x[:,:-1])
        return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100)
    with (a.out/'metrics.jsonl').open('a') as log:
        for step in range(start+1,a.steps+1):
            model.train();opt.zero_grad(set_to_none=True)
            lr=learning_rate(step,5e-5,5e-6,min(50,a.schedule_steps-1),a.schedule_steps)
            for group in opt.param_groups: group['lr']=lr
            total=0
            for _ in range(a.accum):
                value=loss('train',rng)
                if not torch.isfinite(value): raise ValueError('Nonfinite loss')
                total+=value.item()/a.accum;(value/a.accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
            if step%10==0 or step==a.steps:
                row=dict(step=step,train_loss=total,lr=lr);log.write(json.dumps(row)+'\n');log.flush();print(row,flush=True)
            if step%a.eval_every==0 or step==a.steps:
                model.eval();val_rng=np.random.default_rng(123)
                with torch.no_grad(): val=sum(loss('val',val_rng).item() for _ in range(10))/10
                log.write(json.dumps(dict(step=step,val_loss=val))+'\n');log.flush()
                temp=dest.with_suffix('.tmp')
                torch.save(dict(model=model.state_dict(),config=model.config,data=data,step=step,phase='dialogue',
                    optimizer=opt.state_dict(),rng=rng.bit_generator.state,sft_settings=settings,
                    dialogue_metadata=meta,transfer=source),temp)
                replace_checkpoint(temp,dest)
                print(f'Validation loss: {val:.4f}',flush=True)


if __name__=='__main__': main()
