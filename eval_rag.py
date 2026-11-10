"""How well a checkpoint answers from passages, measured on the held-out SQuAD 2.0 part of a dialogue-rag dataset.

Two measurements, both on the dataset's val split (SQuAD's own validation articles, never trained on):
- answer loss: average loss on the answer tokens only, for every SQuAD val example. Works for any checkpoint,
  so it can also pick which pretrained checkpoint to start dialogue training from (lower is better).
- answers (dialogue-trained checkpoints): the model writes a reply to each of the first --questions questions
  (greedy: always the most likely token, so runs are repeatable) and replies are scored:
    answer found   - answerable questions whose reply contains a correct answer (SQuAD-style normalized text)
    right passage  - answerable questions whose reply cites the passage number the answer is in
    says not found - unanswerable questions whose reply says the passages don't say
    false not found - answerable questions whose reply wrongly says the passages don't say
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import string
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from dialogue import RAG_NOT_FOUND, chat_prompt
from model import GPT


def normalize(text):
    """SQuAD's answer normalization: lowercase, no punctuation, no articles, single spaces."""
    text = ''.join(c for c in text.lower() if c not in set(string.punctuation) | {'’', '‘'})
    return ' '.join(w for w in text.split() if w not in ('a', 'an', 'the'))


def cited(text):
    return set(re.findall(r'\[(\d+)\]', text))


def score(reply, example):
    """Booleans for one reply, given its val example (messages + answerable + answers)."""
    said_none = normalize(RAG_NOT_FOUND) in normalize(reply)
    if not example['answerable']:
        return dict(answerable=False, says_not_found=said_none)
    reply_n = normalize(reply)
    return dict(answerable=True, false_not_found=said_none,
                answer_found=any(normalize(a) and normalize(a) in reply_n for a in example['answers']),
                right_passage=bool(cited(reply) & cited(example['messages'][-1]['content'])))


def summarize(scores):
    yes = [s for s in scores if s['answerable']]
    no = [s for s in scores if not s['answerable']]
    pct = lambda rows, key: round(100 * sum(s[key] for s in rows) / len(rows), 1) if rows else None
    return dict(answerable=len(yes), unanswerable=len(no), answer_found_pct=pct(yes, 'answer_found'),
                right_passage_pct=pct(yes, 'right_passage'), says_not_found_pct=pct(no, 'says_not_found'),
                false_not_found_pct=pct(yes, 'false_not_found'))


def answer_loss(model, rows, device):
    total, count = 0.0, 0
    with torch.inference_mode():
        for r in rows:
            x = torch.tensor([r['ids']], device=device)
            y = torch.tensor([r['labels']], device=device)
            logits = model(x[:, :-1]).float()
            total += F.cross_entropy(logits[0], y[0, 1:], ignore_index=-100, reduction='sum').item()
            count += int((y[0, 1:] != -100).sum())
    return round(total / count, 4)


def reply(model, tok, messages, tokens, device):
    ids = chat_prompt(tok, messages, model.config['context'] - tokens)
    x = torch.tensor([ids], device=device)
    eos, out = tok.token_to_id('<eos>'), []
    with torch.inference_mode():
        for _ in range(tokens):
            nxt = model(x)[0, -1].argmax().item()
            if nxt == eos:
                break
            out.append(nxt)
            x = torch.cat((x, torch.tensor([[nxt]], device=device)), dim=1)
    return tok.decode(out).strip()


def evaluate(checkpoint, data, questions=300, tokens=40, device='cpu', samples=5, log=print):
    meta = json.loads((data / 'metadata.json').read_text(encoding='utf-8'))
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck['data']['tokenizer_sha256'] != meta['tokenizer_sha256'] or \
            hashlib.sha256((data / 'tokenizer.json').read_bytes()).hexdigest() != meta['tokenizer_sha256']:
        raise SystemExit('This checkpoint and dataset use different tokenizers.')
    model = GPT(**ck['config']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    step, phase = ck['step'], ck.get('phase', 'pretrain')
    del ck
    tok = Tokenizer.from_file(str(data / 'tokenizer.json'))
    rows = [json.loads(line) for line in (data / 'val.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r.get('source') == 'squad_v2']
    if not rows:
        raise SystemExit('No SQuAD examples in this dataset\'s val split (prepare it with --rag-examples).')
    started = time.perf_counter()
    result = dict(checkpoint=str(checkpoint), step=step, phase=phase, data=str(data), val_examples=len(rows),
                  answer_loss=answer_loss(model, rows, device))
    log(f"Answer loss on {len(rows)} SQuAD val examples: {result['answer_loss']} ({time.perf_counter() - started:.0f} s)")
    if questions and phase != 'dialogue':
        log('Skipping answers: this checkpoint has not been dialogue-trained, so it does not reply in turn.')
        questions = 0
    scores, shown = [], []
    for i, r in enumerate(rows[:questions], 1):
        text = reply(model, tok, r['messages'][:-1], tokens, device)
        s = score(text, r)
        scores.append(s)
        if len(shown) < samples:
            shown.append(dict(prompt=r['messages'][0]['content'], expected=r['messages'][-1]['content'], reply=text, **s))
        if i % 50 == 0:
            log(f'{i}/{min(questions, len(rows))} answered ({time.perf_counter() - started:.0f} s)')
    if scores:
        result.update(summarize(scores), questions=len(scores), samples=shown)
    result['seconds'] = round(time.perf_counter() - started, 1)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('E:/wiki-llm/data/dialogue-rag'))
    p.add_argument('--questions', type=int, default=300, help='Questions to answer (0: answer loss only)')
    p.add_argument('--tokens', type=int, default=40, help='Longest reply, in tokens')
    p.add_argument('--samples', type=int, default=5, help='Full conversations to print')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='cuda only when no training is using the GPU')
    p.add_argument('--out', type=Path, help='Also save the result as JSON here')
    a = p.parse_args()
    if a.questions < 0 or not 1 <= a.tokens <= 256 or a.samples < 0:
        p.error('Need questions >= 0, tokens 1-256 and samples >= 0')
    device = a.device
    if device == 'cpu':
        torch.set_num_threads(4)
    result = evaluate(a.checkpoint, a.data, a.questions, a.tokens, device, a.samples)
    for s in result.get('samples', []):
        print('\n' + '-' * 60 + '\n' + s['prompt'] + f"\nExpected: {s['expected']}\nReply:    {s['reply']}")
    summary = {k: v for k, v in result.items() if k != 'samples'}
    print('\n' + json.dumps(summary, indent=2))
    if a.out:
        a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
