"""Describe a trusted local GPT checkpoint and tokenize a prompt on CPU."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


def inspect(checkpoint, tokenizer, prompt):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 500:
        raise ValueError('Enter between 1 and 500 characters.')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest() != ck['data']['tokenizer_sha256']:
        raise ValueError('Tokenizer does not match the checkpoint.')
    cfg, weights = ck['config'], ck['model']
    tokenizer = Tokenizer.from_file(str(tokenizer))
    encoding = tokenizer.encode(prompt)
    total = len(encoding.ids)
    offset = max(0, total - cfg['context'])
    # Each BPE token is represented as stored; byte fragments need not decode alone.
    tokens = [dict(index=i, id=encoding.ids[i], piece=encoding.tokens[i])
              for i in range(offset, min(total, offset + 16))]
    block_parameters = [sum(t.numel() for n, t in weights.items() if n.startswith(f'blocks.{i}.'))
                        for i in range(cfg['layers'])]
    # A small, exact slice of the two dense MLP weight matrices in every block.
    # Each nn.Linear weight is indexed [output feature, input feature].
    sample_width, sample_hidden = min(32, cfg['width']), min(48, 4 * cfg['width'])
    weight_samples = [dict(
        expand=weights[f'blocks.{i}.mlp.0.weight'][:sample_hidden, :sample_width].float().tolist(),
        project=weights[f'blocks.{i}.mlp.2.weight'][:sample_width, :sample_hidden].float().tolist(),
    ) for i in range(cfg['layers'])]
    return dict(config=cfg, step=ck['step'], total_parameters=sum(t.numel() for t in weights.values()),
                token_embedding_parameters=weights['tokens.weight'].numel(),
                position_embedding_parameters=weights['positions.weight'].numel(),
                final_norm_parameters=weights['norm.weight'].numel()+weights['norm.bias'].numel(),
                block_parameters=block_parameters, weight_samples=weight_samples,
                tokens=tokens, total_tokens=total, token_offset=offset)


if __name__ == '__main__':
    try:
        request = json.load(sys.stdin)
        print(json.dumps(inspect(request['checkpoint'], request['tokenizer'], request['prompt'])))
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
