import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(b, t, c))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size=16384, context=512, width=512, layers=8, heads=8):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, context=context, width=width, layers=layers, heads=heads)
        self.tokens = nn.Embedding(vocab_size, width)
        self.positions = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*(Block(width, heads) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        x = self.tokens(ids) + self.positions(torch.arange(ids.size(1), device=ids.device))
        return F.linear(self.norm(self.blocks(x)), self.tokens.weight)
