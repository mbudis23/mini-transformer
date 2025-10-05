# example/sanity_check.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.model import DecoderOnlyTransformer
from src.visualize import visualize_attention
from src.utils import default_rng

def _sanity_check():
    rng_seed = 42
    rng = default_rng(rng_seed)
    # config
    vocab_size = 50
    seq_len = 8
    batch = 2
    d_model = 32
    n_heads = 4
    n_layers = 2

    model = DecoderOnlyTransformer(vocab_size=vocab_size,
                                   max_seq_len=seq_len,
                                   d_model=d_model,
                                   n_heads=n_heads,
                                   n_layers=n_layers,
                                   rng_seed=rng_seed,
                                   tie_weights=True)

    # create random tokens in [0, vocab_size)
    tokens = rng.integers(low=0, high=vocab_size, size=(batch, seq_len), dtype=np.int64)

    logits, probs_last, attn_list = model.forward(tokens, return_attentions=True)

    # Basic checks
    assert logits.shape == (batch, seq_len, vocab_size), f"logits shape mismatch: {logits.shape}"
    assert probs_last.shape == (batch, vocab_size), f"probs_last shape mismatch: {probs_last.shape}"

    print("Tokens:\n", tokens)
    print("Logits shape:", logits.shape)
    print("Probs (last position) shape:", probs_last.shape)
    print("Sum of probs (per example):", np.sum(probs_last, axis=-1))  # should be 1.0

    # Check attention shapes and that rows sum to 1 across allowed positions
    for i, attn in enumerate(attn_list):
        # attn: (batch, n_heads, seq_len, seq_len)
        print(f"Block {i} attention shape: {attn.shape}")
        # Check sums on last axis are 1 (within numerical tolerance)
        row_sums = np.sum(attn, axis=-1)  # (batch, n_heads, seq_len)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            print("Warning: attention rows do not sum to 1 exactly; max dev:", np.max(np.abs(row_sums - 1.0)))
        else:
            print("All attention rows sum to 1 (within tol)")

    # Demonstrate causal mask effect:
    # For the last target position (index seq_len-1), attention should only attend to indices <= seq_len-1 (i.e., all)
    # For e.g., position k, attention[:, :, k, j] should be ~0 for j > k
    block0_attn = attn_list[0]  # first block
    # show attention for first example, first head
    a = block0_attn[0, 0]  # (seq_len, seq_len)
    print("Example attention matrix (block 0, batch 0, head 0):\n", np.round(a, 3))
    print("Note: upper-triangular (future) elements should be ~0 due to causal mask.")
    visualize_attention(np.round(a, 3))

if __name__ == "__main__":
    _sanity_check()