import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class AttentionHead(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size, num_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, num_heads, head_size, num_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(
                head_size, num_embed, block_size, dropout
            ) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, num_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([att(x) for att in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by ReLu"""

    def __init__(self, num_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embed, 4 * num_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * num_embed, num_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block. This class will group Multihead and FeedForward layers."""

    def __init__(self, num_heads, block_size, num_embed, dropout):
        super().__init__()
        head_size = block_size // num_heads
        self.sa = MultiHeadAttention(
            num_heads, head_size, num_embed, block_size, dropout
        )
        self.ff = FeedForward(num_embed, dropout)
        self.norm1 = nn.LayerNorm(num_embed)
        self.norm2 = nn.LayerNorm(num_embed)

    def forward(self, x):
        x = x + self.sa(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

