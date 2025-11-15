"""
Localist-LLM – Minimal locality demo

This script:
- builds a tiny Localist encoder
- sets different locality dial values (0.0, 0.5, 1.0)
- prints attention matrices (with indices)
- prints a simple 'locality fraction' showing how much attention mass
  lies on neighbouring positions (|i-j| <= 1)
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


def compute_locality_fraction(attn_np, radius: int = 1) -> float:
    """
    Compute the fraction of attention mass that lies within a given
    distance 'radius' from the diagonal, i.e. |i - j| <= radius.

    attn_np: (L, L) numpy array
    radius: neighbourhood size (1 = immediate neighbours)
    """
    L = attn_np.shape[0]
    local_mass = 0.0
    total_mass = 0.0

    for i in range(L):
        for j in range(L):
            v = attn_np[i, j]
            total_mass += v
            if abs(i - j) <= radius:
                local_mass += v

    if total_mass == 0.0:
        return 0.0

    return float(local_mass / total_mass)


def run_demo(seq_len: int = 8, vocab_size: int = 50):
    """
    Build a tiny encoder, run a random sequence through it with different
    locality dial settings, and print the resulting attention matrices and
    locality fractions.
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

    # Access the only encoder layer and its attention module
    layer = model.layers[0]
    attn_module = layer.self_attn

    # Try different dial values
    dial_values = [0.0, 0.5, 1.0]

    for dv in dial_values:
        attn_module.locality_dial.set(dv)

        print("\n" + "=" * 60)
        print(f"Locality dial = {dv}")
        print("=" * 60)

        with torch.no_grad():
            _, attn_weights_all = model(input_ids)

        # Only one layer in this tiny model
        attn_weights = attn_weights_all[0]  # (batch, heads, L, L)

        # Take batch 0, head 0
        attn_matrix = attn_weights[0, 0]  # (L, L)
        attn_np = attn_matrix.cpu().numpy()
        L = attn_np.shape[0]

        # ---- Print locality fraction ----
        loc_frac = compute_locality_fraction(attn_np, radius=1)
        print(f"Locality fraction (|i-j| <= 1): {loc_frac:0.3f}")

        # ---- Print labelled attention matrix ----
        print("\nAttention matrix (batch 0, head 0):\n")

        # Column indices header
        header = "      " + " ".join(f"{j:>6}" for j in range(L))
        print(header)
        print("      " + "-" * (7 * L))

        # Each row with row index
        for i, row in enumerate(attn_np):
            row_str = " ".join(f"{v:0.2f}" for v in row)
            print(f"{i:>3} | {row_str}")


if __name__ == "__main__":
    run_demo()
