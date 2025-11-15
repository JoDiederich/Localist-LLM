import torch
import torch.nn as nn

from .localist_attention import LocalistMultiheadAttention


class TinyLocalistEncoderLayer(nn.Module):
    """
    A minimal transformer encoder layer using LocalistMultiheadAttention.

    This is a small, educational example only. It is NOT a full Localist LLM
    implementation.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        self.self_attn = LocalistMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention with residual
        attn_out, attn_weights = self.self_attn(x)
        x = x + attn_out
        x = self.norm1(x)

        # MLP with residual
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)

        return x, attn_weights


class TinyLocalistEncoder(nn.Module):
    """
    A tiny encoder-only transformer using a locality-aware attention layer.

    This is intended for concept demonstration:
    - small model
    - random inputs
    - easy to inspect attention patterns
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 1, max_len: int = 128, dropout: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        self.layers = nn.ModuleList([
            TinyLocalistEncoderLayer(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids: torch.Tensor):
        """
        input_ids: (batch_size, seq_len) of token IDs

        Returns:
            hidden_states: (batch_size, seq_len, embed_dim)
            all_attn_weights: list of attention tensors, one per layer
        """
        bsz, seq_len = input_ids.size()
        device = input_ids.device

        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        # Token + position embeddings
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            all_attn_weights.append(attn_weights)

        return x, all_attn_weights
