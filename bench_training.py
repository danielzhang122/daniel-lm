"""Benchmark candidate microbatches without changing any training checkpoint."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import GPT
from training_utils import BlockSampler


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('E:/wiki-llm/data/full'))
    p.add_argument('--checkpoint',type=Path,default=Path('runs/full/latest.pt'))
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--warmup',type=int,default=3)
    a=p.parse_args()
    if min(a.steps,a.warmup)<1:p.error('Counts must be positive')
    if not torch.cuda.is_available():p.error('CUDA is required')
    torch.manual_seed(42)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=GPT(**ck['config']).cuda();model.load_state_dict(ck['model']);del ck
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.1)
    arr=np.memmap(a.data/'train.bin',dtype='<u2',mode='r')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    context=model.config['context']
    # Interleave baseline measurements with alternatives while the existing job runs.
    for batch in [2,4,2,8,2,16,2]:
        accum=16//batch
        sampler=BlockSampler(len(arr),context,position=0)
        try:
            torch.cuda.reset_peak_memory_stats()
            def step():
                opt.zero_grad(set_to_none=True)
                total=torch.zeros((),device='cuda')
                for _ in range(accum):
                    ids=arr[sampler.next_indices(batch)].astype(np.int64)
                    x=torch.from_numpy(ids).cuda()
                    with torch.autocast('cuda',dtype=dtype):
                        logits=model(x[:,:-1])
                        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),x[:,1:].reshape(-1))
                    total+=loss.detach()/accum
                    scaler.scale(loss/accum).backward()
                if not torch.isfinite(total):raise RuntimeError('Nonfinite loss')
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt);scaler.update()
            for _ in range(a.warmup):step()
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(a.steps):step()
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            print(json.dumps(dict(batch=batch,accum=accum,steps=a.steps,seconds=elapsed,
                tokens_per_second=a.steps*16*context/elapsed,
                peak_vram_gb=torch.cuda.max_memory_allocated()/1024**3)),flush=True)
        except torch.cuda.OutOfMemoryError:
            print(json.dumps(dict(batch=batch,error='CUDA out of memory')),flush=True)
            opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()


if __name__=='__main__':main()
