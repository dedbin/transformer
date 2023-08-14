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


class Transformer(nn.Module):
    """Transformer."""

    def __init__(self, **kwargs):
        super().__init__()
        self.vocab_size = kwargs.get("vocab_size", 100)
        self.num_embed = kwargs.get("num_embed", 32)
        self.block_size = kwargs.get("block_size", 8)
        self.num_heads = kwargs.get("num_heads", 4)
        self.num_layers = kwargs.get("num_layers", 4)
        self.dropout = kwargs.get("dropout", 0.2)

        self.token_embedding = nn.Embedding(self.vocab_size, self.num_embed)
        self.pos_embedding = nn.Embedding(self.block_size, self.num_embed)
        self.blocks = nn.Sequential(
            *[TransformerBlock(num_heads=self.num_heads,
                               block_size=self.block_size,
                               num_embed=self.num_embed,
                               dropout=self.dropout,
                               ) for _ in range(self.num_layers)]
        )
        self.ln = nn.LayerNorm(self.num_embed)
        self.lm_head = nn.Linear(self.num_embed, self.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets != None:
            B, T, C = logits.shape
            logits = torch.reshape(logits, (B * T, C))
            targets = torch.reshape(targets, (B * T,))
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -block_size:]
            logits, loss = self.forward(idx_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
