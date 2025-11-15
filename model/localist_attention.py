import torch
import torch.nn as nn
import torch.nn.functional as F

from locality_dial import LocalityDial


class LocalistMultiheadAttention(nn.Module):
    """
    Minimal multi-head attention with a controllable locality dial.

    - dial = 0.0 → standard global attention
    - dial = 1.0 → simple locality-biased attention

    This is a *conceptual* and *educational* implementation. It deliberately uses
    a generic distance-based bias and does NOT represent the full patented method.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Locality dial: safe, minimal version
        self.locality_dial = LocalityDial(initial_value=0.0, trainable=False)

        # Scale factor for dot-product attention
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        Returns:
            output: (batch_size, seq_len, embed_dim)
            attn_weights: (batch_size, num_heads, seq_len, seq_len)
        """

        bsz, seq_len, _ = x.size()

        # ---- Linear projections ----
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape for multi-head attention: (B, H, L, D)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ---- Standard dot-product attention ----
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,L,L)

        # ---- Add simple locality bias (safe, generic) ----
        dial = self.locality_dial.get()  # scalar in [0,1]

        if dial.item() > 0.0:
            # |i - j| distance matrix
            dist = torch.arange(seq_len, device=x.device)
            dist = (dist[None, :] - dist[:, None]).abs().float()  # (L,L)

            # Negative distance bias → closer tokens preferred
            sigma = 4.0
            local_bias = -dist / sigma  # (L,L)

            # Expand to all heads and batches
            local_bias = local_bias.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)

            # Blend based on dial value
            attn_scores = attn_scores + dial * local_bias

        # ---- Softmax ----
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ---- Weighted sum over values ----
        context = torch.matmul(attn_weights, v)  # (B,H,L,D)

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)

        # Final projection
        output = self.out_proj(context)

        return output, attn_weights
