# transformer_numpy.py
"""
Transformer decoder-only (GPT-style) minimal, from-scratch using NumPy.

Fitur:
- Token Embedding
- Sinusoidal Positional Encoding
- Scaled Dot-Product Attention dengan causal mask
- Multi-Head Attention
- Feed-Forward Network (2-layer + activation)
- Residual connections + LayerNorm (pre-norm)
- Output projection ke vocab dengan opsi weight-tying
- Modular: setiap komponen diimplementasikan sebagai class/fungsi terpisah
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Utility / initializers
# ---------------------------
def default_rng(seed=None):
    return np.random.default_rng(seed)

def xavier_init(rng, shape):
    """Xavier normal initialization."""
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 1:
        fan_in = shape[0]
        fan_out = shape[0]
    else:
        # For simplicity fallback:
        fan_in = np.prod(shape[:-1])
        fan_out = shape[-1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(loc=0.0, scale=std, size=shape).astype(np.float32)


def softmax(x, axis=-1):
    """Stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------
# Components
# ---------------------------

class TokenEmbedding:
    """Embedding layer: token ids -> vector embeddings"""
    def __init__(self, vocab_size, d_model, rng):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rng = rng
        self.W = xavier_init(rng, (vocab_size, d_model))  # shape (V, D)

    def forward(self, tokens):
        # tokens: (batch, seq_len) integers
        return self.W[tokens]  # (batch, seq_len, d_model)

class LearnedPositionalEmbedding:
    """Learned positional embeddings"""
    def __init__(self, max_len, d_model, rng):
        self.W = xavier_init(rng, (max_len, d_model))
    def forward(self, seq_len):
        return self.W[:seq_len]

class SinusoidalPositionalEncoding:
    """Sinusoidal positional encoding (deterministic, no params)"""
    def __init__(self, max_len, d_model):
        self.max_len = max_len
        self.d_model = d_model
        self.P = self._build_pe(max_len, d_model).astype(np.float32)  # (max_len, d_model)

    def _build_pe(self, max_len, d_model):
        pos = np.arange(max_len)[:, None]  # (max_len, 1)
        i = np.arange(d_model)[None, :]    # (1, d_model)
        angle_rates = 1 / (10000 ** (2 * (i // 2) / d_model))
        angle_rads = pos * angle_rates
        pe = np.zeros_like(angle_rads)
        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return pe  # shape (max_len, d_model)

    def forward(self, seq_len):
        assert seq_len <= self.max_len, "requested seq_len > max_len"
        return self.P[:seq_len]  # (seq_len, d_model)


class LayerNorm:
    """Layer Normalization (learnable)"""
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        # initialize scale (gamma) and bias (beta)
        self.gamma = np.ones((d_model,), dtype=np.float32)
        self.beta = np.zeros((d_model,), dtype=np.float32)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        mean = np.mean(x, axis=-1, keepdims=True)  # (batch, seq_len, 1)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta  # broadcasting (d_model,)


class ScaledDotProductAttention:
    """Scaled dot-product attention with causal mask support"""
    def __init__(self):
        pass

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: shapes (batch, n_heads, seq_len, head_dim)
        mask: boolean array broadcastable to (batch, n_heads, seq_len, seq_len)
              True = allowed, False = masked
        returns:
          output: (batch, n_heads, seq_len, head_dim)
          attn: (batch, n_heads, seq_len, seq_len)
        """
        d_k = Q.shape[-1]
        # scores: (batch, n_heads, seq_len, seq_len)
        scores = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(d_k)
        if mask is not None:
            # mask is boolean: True means allowed; False means masked
            # set masked positions to large negative value
            large_neg = -1e9
            scores = np.where(mask, scores, large_neg)
        attn = softmax(scores, axis=-1)
        output = np.matmul(attn, V)  # (batch, n_heads, seq_len, head_dim)
        return output, attn


class MultiHeadAttention:
    """Multi-head attention module"""
    def __init__(self, d_model, n_heads, rng):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rng = rng
        # projection matrices
        self.W_q = xavier_init(rng, (d_model, d_model))
        self.W_k = xavier_init(rng, (d_model, d_model))
        self.W_v = xavier_init(rng, (d_model, d_model))
        self.W_o = xavier_init(rng, (d_model, d_model))
        self.attn = ScaledDotProductAttention()

    def _split_heads(self, x):
        # x: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, head_dim)
        b, seq_len, _ = x.shape
        x = x.reshape(b, seq_len, self.n_heads, self.head_dim)
        return x.transpose(0,2,1,3)

    def _combine_heads(self, x):
        # x: (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        b, n_heads, seq_len, head_dim = x.shape
        x = x.transpose(0,2,1,3)  # (batch, seq_len, n_heads, head_dim)
        return x.reshape(b, seq_len, n_heads * head_dim)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: boolean (1=allowed) broadcastable to (batch, n_heads, seq_len, seq_len)
        returns:
          out: (batch, seq_len, d_model)
          attn: (batch, n_heads, seq_len, seq_len)
        """
        Q = np.matmul(x, self.W_q)  # (b, seq_len, d_model)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)

        Qh = self._split_heads(Q)  # (b, n_heads, seq_len, head_dim)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        attn_out, attn_weights = self.attn.forward(Qh, Kh, Vh, mask=mask)  # out: (b, n_heads, seq_len, head_dim)
        combined = self._combine_heads(attn_out)  # (b, seq_len, d_model)
        out = np.matmul(combined, self.W_o)  # final linear
        return out, attn_weights


class FeedForwardNetwork:
    """Two-layer FFN with ReLU activation"""
    def __init__(self, d_model, d_ff, rng):
        self.W1 = xavier_init(rng, (d_model, d_ff))
        self.b1 = np.zeros((d_ff,), dtype=np.float32)
        self.W2 = xavier_init(rng, (d_ff, d_model))
        self.b2 = np.zeros((d_model,), dtype=np.float32)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        hidden = np.matmul(x, self.W1) + self.b1  # (b, seq_len, d_ff)
        hidden = np.maximum(hidden, 0)  # ReLU
        out = np.matmul(hidden, self.W2) + self.b2  # (b, seq_len, d_model)
        return out


class DecoderBlock:
    """Single decoder block: pre-norm MHA -> add -> pre-norm FFN -> add"""
    def __init__(self, d_model, n_heads, d_ff, rng):
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, rng)

    def forward(self, x, mask=None):
        # Pre-norm MHA
        x_norm = self.ln1.forward(x)
        mha_out, attn = self.mha.forward(x_norm, mask=mask)
        x = x + mha_out  # residual

        # Pre-norm FFN
        x_norm = self.ln2.forward(x)
        ffn_out = self.ffn.forward(x_norm)
        x = x + ffn_out  # residual
        return x, attn


# ---------------------------
# Full Decoder-only Transformer
# ---------------------------
class DecoderOnlyTransformer:
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 d_model=64,
                 n_heads=8,
                 n_layers=2,
                 d_ff=None,
                 rng_seed=0,
                 tie_weights=True,
                 learned_pos_emb=False):
        self.rng = default_rng(rng_seed)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.tie_weights = tie_weights

        # modules
        self.token_emb = TokenEmbedding(vocab_size, d_model, self.rng)
        if learned_pos_emb:
            self.pos_emb = LearnedPositionalEmbedding(max_seq_len, d_model, self.rng)
        else:
            self.pos_emb = SinusoidalPositionalEncoding(max_seq_len, d_model)

        # stack of decoder blocks
        self.blocks = [DecoderBlock(d_model, n_heads, self.d_ff, self.rng) for _ in range(n_layers)]
        # final layer norm (optional; typical in many implementations)
        self.ln_f = LayerNorm(d_model)

        # output projection (if not tying weights)
        if not tie_weights:
            self.W_out = xavier_init(self.rng, (d_model, vocab_size))

    def _causal_mask(self, seq_len, batch_size):
        """
        Returns boolean mask broadcastable to (batch, n_heads, seq_len, seq_len)
        mask[i, j] = True if j <= i (i: target index, j: source index)
        """
        base = np.tril(np.ones((seq_len, seq_len), dtype=bool))  # (seq_len, seq_len)
        mask = base[None, None, :, :]  # (1,1,seq_len,seq_len)
        # broadcast to (batch, n_heads, seq_len, seq_len)
        mask = np.broadcast_to(mask, (batch_size, self.n_heads, seq_len, seq_len))
        return mask

    def forward(self, tokens, return_attentions=False):
        """
        tokens: (batch, seq_len) integer token ids
        returns:
          logits: (batch, seq_len, vocab_size)
          probs_last: (batch, vocab_size) -> softmax over last position
          optionally attentions list: list of attn arrays per block shape (batch, n_heads, seq_len, seq_len)
        """
        batch, seq_len = tokens.shape
        assert seq_len <= self.max_seq_len, "seq_len > max_seq_len"

        # Embedding + positional encoding
        x = self.token_emb.forward(tokens)  # (b, seq_len, d_model)
        pos = self.pos_emb.forward(seq_len)  # (seq_len, d_model)
        x = x + pos[None, :, :]  # broadcast

        # prepare causal mask
        mask = self._causal_mask(seq_len, batch)

        # pass through transformer blocks
        attn_list = []
        for block in self.blocks:
            x, attn = block.forward(x, mask=mask)
            attn_list.append(attn)

        x = self.ln_f.forward(x)  # final layernorm

        # output logits
        if self.tie_weights:
            # weight tying: logits = x @ token_emb.W.T
            logits = np.matmul(x, self.token_emb.W.T)  # (b, seq_len, vocab_size)
        else:
            logits = np.matmul(x, self.W_out)  # (b, seq_len, vocab_size)

        # distribution for the next token = softmax on last position
        probs_last = softmax(logits[:, -1, :], axis=-1)  # (batch, vocab_size)

        if return_attentions:
            return logits, probs_last, attn_list
        return logits, probs_last


# ---------------------------
# Small test / example usage
# ---------------------------
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

    def visualize_attention(attn, title="Attention Heatmap (Block 0, Head 0)"):
        plt.imshow(attn, cmap="viridis", interpolation="nearest")
        plt.title(title)
        plt.xlabel("Key positions")
        plt.ylabel("Query positions")
        plt.colorbar()
        plt.show()
    
    visualize_attention(np.round(a, 3))

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
    _sanity_check()
    run_unit_tests()
