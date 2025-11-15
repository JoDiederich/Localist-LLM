"""
Minimal demo script for the Localist-LLM concept implementation.

This script:
- builds a tiny Localist encoder
- sets different locality dial values
- prints attention matrices so you can see how locality changes attention
"""

import torch

from model.tiny_transformer import TinyLocalistEncoder


def run_demo(seq_len: int = 10, vocab_size: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a tiny encoder model
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

    # Access the single encoder layer and its attention module
    layer = model.layers[0]
    attn_module = layer.self_attn

    # Try a few different locality dial settings
    dial_values = [0.0, 0.5, 1.0]

    for dv in dial_values:
        # Set the dial value
        attn_module.locality_dial.set(dv)

        print("\n" + "=" * 40)
        print(f"Locality dial = {dv}")
        print("=" * 40)

        # Forward pass
        with torch.no_grad():
            _, attn_weights_all = model(input_ids)

        # attn_weights_all is a list (one entry per layer); here we have 1 layer
        attn_weights = attn_weights_all[0]  # shape: (batch, heads, L, L)

        # Take batch 0, head 0 for printing
        attn_matrix = attn_weights[0, 0]  # (L, L)

        # Print a rounded version of the matrix
        attn_np = attn_matrix.cpu().numpy()
        # Format each row nicely
        for row in attn_np:
            row_str = " ".join(f"{v:0.2f}" for v in row)
            print(row_str)


if __name__ == "__main__":
    run_demo()
