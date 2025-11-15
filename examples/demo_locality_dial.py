"""
Minimal demo script for the Localist-LLM concept implementation.

This script:
- builds a tiny Localist encoder
- sets different locality dial values
- prints attention matrices so you can see how locality changes the pattern
"""

import os
import sys
import torch

# ------------------------------------------------------------
# Ensure that the repository root is on sys.path
# (so that 'model' can be imported reliably)
# ------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.tiny_transformer import TinyLocalistEncoder


def run_demo(seq_len: int = 10, vocab_size: int = 50):
    """
    Build a tiny encoder, run a random sequence through it with different
    locality dial settings, and print the resulting attention matrices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Build the tiny Localist encoder
    model = TinyLocalistEncoder(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        max_len=seq_len,
        dropout=0.0,
    ).to(device)

    # Random input token IDs
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # Access the only encoder layer
    layer = model.layers[0]
    attn_module = layer.self_attn

    # Try different dial values
    dial_values = [0.0, 0.5, 1.0]

    for dv in dial_values:
        attn_module.locality_dial.set(dv)

        print("\n" + "=" * 50)
        print(f"Locality dial = {dv}")
        print("=" * 50)

        with torch.no_grad():
            _, attn_weights_all = model(input_ids)

        # Only one layer in this tiny model
        attn_weights = attn_weights_all[0]  # (batch, heads, L, L)

        # Print only batch 0, head 0
        attn_matrix = attn_weights[0, 0]  # (L, L)

        attn_np = attn_matrix.cpu().numpy()

        # Pretty-print each row
        for row in attn_np:
            print(" ".join(f"{v:0.2f}" for v in row))


if __name__ == "__main__":
    run_demo()
