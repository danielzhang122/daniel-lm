"""Bounded single-GPU training with validation and resumable checkpoints."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from model import GPT
from training_utils import BlockSampler, learning_rate, validate_transfer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/pilot'))
    p.add_argument('--out', type=Path, default=Path('runs/pilot'))
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--init-from', type=Path, help='Transfer weights to new data; reset optimizer, steps and schedule')
    p.add_argument('--schedule-steps', type=int, default=10000)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min-lr', type=float, default=3e-5)
    args = p.parse_args()
    if args.resume and args.init_from:
        p.error('--resume and --init-from are mutually exclusive')
    if min(args.steps, args.batch, args.accum, args.context, args.eval_every) < 1:
        p.error('Numeric arguments must be positive')
    if not (0 <= args.warmup < args.schedule_steps and 0 <= args.min_lr <= args.lr and args.lr > 0):
        p.error('Require 0 <= warmup < schedule-steps and 0 <= min-lr <= lr, with lr > 0')
    training = dict(sampler='blocks-v1', batch=args.batch, accum=args.accum,
                    horizon=args.schedule_steps, warmup=args.warmup, lr=args.lr, min_lr=args.min_lr)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Install a CUDA PyTorch wheel and check nvidia-smi.')
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    meta = json.loads((args.data / 'metadata.json').read_text())
    if hashlib.sha256((args.data / 'tokenizer.json').read_bytes()).hexdigest() != meta['tokenizer_sha256']:
        raise SystemExit('Dataset tokenizer does not match its metadata.')
    arrays = {s: np.memmap(args.data / f'{s}.bin', dtype='<u2', mode='r') for s in ('train', 'val')}
    if any(len(arrays[s]) != meta['tokens'][s] for s in arrays):
        raise SystemExit('Dataset token files are incomplete or do not match metadata.')
    if any(len(a) <= args.context for a in arrays.values()):
        raise SystemExit('Both splits need more than context tokens. Prepare more articles.')
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / 'latest.pt'
    if checkpoint.exists() and not args.resume:
        raise SystemExit('Checkpoint exists: use --resume or a new --out.')
    model = GPT(vocab_size=meta['vocab_size'], context=args.context).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    start = 0
    transfer = None
    if args.init_from:
        source = torch.load(args.init_from, map_location='cpu', weights_only=False)
        validate_transfer(source, model.config, meta)
        model.load_state_dict(source['model'])
        transfer = dict(checkpoint=str(args.init_from.resolve()), source_step=source['step'])
        print(f'Transferred weights from step {source["step"]}; starting a new data stage at step 0.', flush=True)
        del source
    if args.resume:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)  # own local checkpoint only
        if ck['config'] != model.config or ck['data'] != meta:
            raise SystemExit('Checkpoint model/data mismatch.')
        if 'training' in ck and ck['training'] != training:
            raise SystemExit('Resume requires the original batch, accumulation and schedule settings.')
        if 'training' not in ck:
            training = None  # Preserve original pilot sampling and constant learning rate.
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        start = ck['step']
        transfer = ck.get('transfer')
        rng.bit_generator.state = ck['rng']
        torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state(ck['cuda_rng'])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    if args.resume:
        scaler.load_state_dict(ck['scaler'])
    sampler = BlockSampler(len(arrays['train']), args.context, start * args.batch * args.accum)
    (args.out / 'run.json').write_text(json.dumps(dict(data=str(args.data.resolve()),
        target_steps=args.steps, tokens_per_step=args.batch*args.accum*args.context,
        training=training, transfer=transfer, parameters=sum(p.numel() for p in model.parameters())), indent=2), encoding='utf-8')

    def batch(split, generator):
        a = arrays[split]
        if split == 'train' and training:
            packed = a[sampler.next_indices(args.batch)].astype(np.int64)
        else:
            ix = generator.integers(0, len(a) - args.context, size=args.batch)
            packed = np.stack([a[i:i + args.context + 1].astype(np.int64) for i in ix])
        data = torch.from_numpy(packed).cuda()
        return data[:, :-1], data[:, 1:]

    def loss(split, generator):
        x, y = batch(split, generator)
        with torch.autocast('cuda', dtype=dtype):
            logits = model(x)
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    print(f'{sum(p.numel() for p in model.parameters()):,} parameters on {torch.cuda.get_device_name()}', flush=True)
    tokens_per_step = args.batch * args.accum * args.context
    torch.cuda.synchronize()
    begin = time.perf_counter()
    with (args.out / 'metrics.jsonl').open('a', encoding='utf-8') as log:
        for step in range(start + 1, args.steps + 1):
            if training:
                lr = learning_rate(step, args.lr, args.min_lr, args.warmup, args.schedule_steps)
                for group in opt.param_groups:
                    group['lr'] = lr
            model.train()
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(args.accum):
                value = loss('train', rng)
                if not torch.isfinite(value):
                    raise RuntimeError('Nonfinite loss')
                total += value.detach().item() / args.accum
                scaler.scale(value / args.accum).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if step % 10 == 0 or step == args.steps:
                torch.cuda.synchronize()
                row = dict(step=step, train_loss=total, lr=opt.param_groups[0]['lr'],
                           tokens_seen=step*tokens_per_step,
                           tokens_per_second=(step-start)*tokens_per_step/(time.perf_counter()-begin),
                           peak_vram_gb=torch.cuda.max_memory_allocated()/1024**3)
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(json.dumps(row), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                val_rng = np.random.default_rng(123)
                with torch.no_grad():
                    val = sum(loss('val', val_rng).item() for _ in range(10))/10
                print(f'validation loss: {val:.4f}', flush=True)
                log.write(json.dumps(dict(step=step, val_loss=val)) + '\n')
                log.flush()
                temp = checkpoint.with_suffix('.tmp')
                state = dict(model=model.state_dict(), optimizer=opt.state_dict(), config=model.config,
                                data=meta, step=step, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state(), scaler=scaler.state_dict(), transfer=transfer)
                if training:
                    state['training'] = training
                torch.save(state, temp)
                temp.replace(checkpoint)


if __name__ == '__main__':
    main()
