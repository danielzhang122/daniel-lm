"""Loss of a checkpoint on a dataset's validation split, read-only.

Uses the same evenly spaced blocks every time, so two checkpoints measured on the same dataset are
compared on exactly the same text (e.g. "did the FineWeb stage make the model forget Wikipedia?").
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT


def val_loss(checkpoint, data, blocks=64, batch=4, device='cpu'):
    meta = json.loads((data / 'metadata.json').read_text(encoding='utf-8'))
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck['data']['tokenizer_sha256'] != meta['tokenizer_sha256']:
        raise SystemExit('This checkpoint and dataset use different tokenizers.')
    model = GPT(**ck['config']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    step, context = ck['step'], model.config['context']
    del ck
    tokens = np.memmap(data / 'val.bin', dtype='<u2', mode='r')
    if len(tokens) <= context + 1:
        raise SystemExit('Validation split is shorter than one block.')
    starts = np.linspace(0, len(tokens) - context - 1, blocks).astype(np.int64)
    total = 0.0
    with torch.inference_mode():
        for i in range(0, blocks, batch):
            chunk = np.stack([tokens[s:s + context + 1] for s in starts[i:i + batch]]).astype(np.int64)
            x = torch.from_numpy(chunk).to(device)
            logits = model(x[:, :-1]).float()
            total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction='sum').item()
    return dict(checkpoint=str(checkpoint), step=step, data=str(data), blocks=blocks, context=context,
                loss=round(total / (blocks * context), 4))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--blocks', type=int, default=64)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    a = p.parse_args()
    if a.blocks < 1:
        p.error('--blocks must be at least 1')
    if a.device == 'cpu':
        torch.set_num_threads(4)
    print(json.dumps(val_loss(a.checkpoint, a.data, a.blocks, device=a.device)))


if __name__ == '__main__':
    main()
