"""Deterministic block coverage and a schedule independent of run length."""
import math
import numpy as np


def validate_transfer(checkpoint, config, metadata):
    if checkpoint['config'] != config:
        raise ValueError('Transfer requires the same model architecture and vocabulary size.')
    source_hash = checkpoint['data'].get('tokenizer_sha256')
    if not source_hash or source_hash != metadata.get('tokenizer_sha256'):
        raise ValueError('Transfer requires the identical tokenizer, not just the same vocabulary size.')


def validate_resume_training(previous, current):
    """Allow regrouping microbatches only when the effective batch is unchanged."""
    if not isinstance(previous, dict) or not isinstance(current, dict):
        raise ValueError('Missing training settings in checkpoint.')
    for key in set(previous) | set(current):
        if key not in ('batch', 'accum') and previous.get(key) != current.get(key):
            raise ValueError('Resume requires the original sampler and schedule settings.')
    if previous.get('sampler') != 'blocks-v1':
        if previous != current:
            raise ValueError('Changing batching is only supported for the block sampler.')
    if any(type(config.get(key)) is not int or config[key] < 1
           for config in (previous, current) for key in ('batch','accum')):
        raise ValueError('Invalid batch or accumulation settings.')
    if previous['batch'] * previous['accum'] != current['batch'] * current['accum']:
        raise ValueError('Resume requires the same effective batch (batch × accumulation).')


def optimizer_state_with_backend(state, backend):
    """Select an AdamW implementation without discarding optimizer moments or steps."""
    if backend is None:
        return state
    if backend not in ('default', 'fused'):
        raise ValueError('Unknown optimizer backend.')
    result = dict(state)
    result['param_groups'] = [dict(group, fused=True if backend == 'fused' else None,
                                   foreach=False if backend == 'fused' else None)
                              for group in state['param_groups']]
    return result


def learning_rate(step, peak, minimum, warmup, horizon):
    if warmup and step <= warmup:
        return peak * step / warmup
    progress = min(1.0, max(0.0, (step - warmup) / (horizon - warmup)))
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2


class BlockSampler:
    """Each epoch visits every block; the final block wraps to the beginning."""
    def __init__(self, tokens, context, position=0, seed=42):
        self.tokens, self.context, self.position, self.seed = tokens, context, position, seed
        self.blocks = math.ceil(tokens / context)
        self.epoch = None

    def next_indices(self, batch):
        result = []
        for _ in range(batch):
            epoch, offset = divmod(self.position, self.blocks)
            if epoch != self.epoch:
                self.order = np.random.default_rng(self.seed + epoch).permutation(self.blocks)
                self.epoch = epoch
            start = int(self.order[offset]) * self.context
            result.append((start + np.arange(self.context + 1)) % self.tokens)
            self.position += 1
        return np.stack(result)
