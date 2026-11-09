import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import GPT

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, default=Path('runs/pilot/latest.pt'))
p.add_argument('--tokenizer', type=Path, default=Path('data/pilot/tokenizer.json'))
p.add_argument('--prompt', default='The Solar System')
p.add_argument('--tokens', type=int, default=100)
args = p.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
import hashlib
if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() != ck['data']['tokenizer_sha256']:
    raise SystemExit('Wrong tokenizer for this checkpoint')
model = GPT(**ck['config']).to(device)
model.load_state_dict(ck['model'])
model.eval()
tok = Tokenizer.from_file(str(args.tokenizer))
ids = tok.encode(args.prompt).ids or [tok.token_to_id('<eos>')]
x = torch.tensor([ids], device=device)
with torch.inference_mode():
    for _ in range(args.tokens):
        logits = model(x[:, -ck['config']['context']:])[:, -1, :] / 0.8
        threshold = logits.topk(min(50, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))
        nxt = torch.multinomial(logits.softmax(-1), 1)
        x = torch.cat((x, nxt), dim=1)
        if nxt.item() == tok.token_to_id('<eos>'):
            break
print(tok.decode(x[0].tolist()))
