"""Generate text from a local checkpoint, on CPU or GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from tokenizers import Tokenizer
from model import GPT
from dialogue import chat_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=Path('runs/pilot/latest.pt'))
    p.add_argument('--tokenizer', type=Path, default=Path('data/pilot/tokenizer.json'))
    p.add_argument('--prompt', default='The Solar System')
    p.add_argument('--tokens', type=int, default=100)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--json', action='store_true')
    p.add_argument('--messages-json', help='JSON conversation; requires a dialogue-trained checkpoint')
    args = p.parse_args()
    if not 1 <= args.tokens <= 512 or not 0.1 <= args.temperature <= 2:
        p.error('Require 1–512 tokens and temperature 0.1–2.')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cpu':
        torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() != ck['data']['tokenizer_sha256']:
        raise SystemExit('Wrong tokenizer for this checkpoint')
    model = GPT(**ck['config']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    step = ck['step']
    phase = ck.get('phase','pretrain')
    del ck  # Release the optimizer state; generation only needs weights.
    tok = Tokenizer.from_file(str(args.tokenizer))
    if args.messages_json:
        if phase!='dialogue': raise SystemExit('This checkpoint has not been dialogue-trained yet.')
        ids=chat_prompt(tok,json.loads(args.messages_json),model.config['context']-args.tokens)
    else:
        ids = tok.encode(args.prompt).ids or [tok.token_to_id('<eos>')]
    x = torch.tensor([ids], device=device)
    generated = []
    with torch.inference_mode():
        for _ in range(args.tokens):
            logits = model(x[:, -model.config['context']:])[:, -1, :] / args.temperature
            threshold = logits.topk(min(50, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float('-inf'))
            nxt = torch.multinomial(logits.softmax(-1), 1)
            if nxt.item() == tok.token_to_id('<eos>'):
                break
            generated.append(nxt.item())
            x = torch.cat((x, nxt), dim=1)
    result = dict(text=tok.decode(x[0].tolist()), continuation=tok.decode(generated),
                  step=step, tokens=len(generated), seconds=round(time.perf_counter()-started, 2), device=device)
    print(json.dumps(result, ensure_ascii=True) if args.json else result['text'])


if __name__ == '__main__':
    main()
