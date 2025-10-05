# src/components.py
import numpy as np
from .utils import xavier_init, softmax

# Token Embedding
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

# Positional Encoding
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

# Scaled Dot-Product Attention dengan softmax
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

# Multi-Head Attention (Q, K, V, concat, dan proyeksi akhir)
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

# Feed-Forward Network (FFN) dua lapisan dengan aktivasi non-linear)
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

# Residual Connection + Layer Normalization (pre-norm)
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