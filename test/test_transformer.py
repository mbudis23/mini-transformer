# test/test_transformer.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.model import DecoderOnlyTransformer

def run_unit_tests():
    print("Running unit tests...")

    rng_seed = 123
    vocab_size = 20
    seq_len = 6
    batch = 2
    d_model = 16
    n_heads = 4
    n_layers = 1

    model = DecoderOnlyTransformer(vocab_size=vocab_size,
                                   max_seq_len=seq_len,
                                   d_model=d_model,
                                   n_heads=n_heads,
                                   n_layers=n_layers,
                                   rng_seed=rng_seed)

    tokens = np.arange(seq_len)[None, :].repeat(batch, axis=0)
    logits, probs_last, attn_list = model.forward(tokens, return_attentions=True)

    # --- Tests ---
    assert logits.shape == (batch, seq_len, vocab_size), "❌ Logits shape mismatch"
    assert probs_last.shape == (batch, vocab_size), "❌ probs_last shape mismatch"
    np.testing.assert_allclose(np.sum(probs_last, axis=-1), 1.0, atol=1e-5)
    print("✅ Logits and probs shape OK, softmax normalized")

    # Check attention masking
    attn = attn_list[0][0, 0]  # (seq_len, seq_len)
    upper_tri = np.triu_indices(seq_len, k=1)
    assert np.allclose(attn[upper_tri], 0, atol=1e-3), "❌ Future positions not masked properly"
    print("✅ Causal mask verified (upper triangle ≈ 0)")

    print("All unit tests passed!\n")


if __name__ == "__main__":
    run_unit_tests()