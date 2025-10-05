# src/transformer_layers.py

import numpy as np


# ============================================================
# 🧠 Layer Normalization
# ============================================================

class LayerNorm:
    """
    Normalisasi layer sederhana berbasis NumPy
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones((1, d_model))
        self.beta = np.zeros((1, d_model))

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self.x_centered = x - mean
        self.std_inv = 1. / np.sqrt(var + self.eps)
        out = self.gamma * self.x_centered * self.std_inv + self.beta
        return out


# ============================================================
# ⚡ Feed Forward Network
# ============================================================

class FeedForward:
    """
    FeedForward sederhana (2-layer MLP)
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((1, d_model))

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        out = np.dot(self.a1, self.W2) + self.b2
        return out


# ============================================================
# 🎯 Multi-Head Self-Attention
# ============================================================

class MultiHeadSelfAttention:
    """
    Multi-Head Self-Attention sederhana berbasis NumPy
    -------------------------------------------------
    - Tidak menggunakan masking otomatis
    - Semua komputasi menggunakan NumPy
    """
    def __init__(self, d_model=8, num_heads=2, seed=42):
        assert d_model % num_heads == 0, "d_model harus habis dibagi num_heads"
        np.random.seed(seed)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Bobot linear Q, K, V dan output
        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01
        self.Wo = np.random.randn(d_model, d_model) * 0.01

    def split_heads(self, x):
        """Ubah bentuk menjadi (batch, num_heads, seq_len, d_k)"""
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Perhitungan attention utama"""
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        weights = self.softmax(scores)
        out = np.matmul(weights, V)
        return out, weights

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Proyeksi linear Q, K, V
        Q = np.dot(x, self.Wq)
        K = np.dot(x, self.Wk)
        V = np.dot(x, self.Wv)

        # Split menjadi beberapa head
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Hitung attention per head
        context, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Gabungkan kembali semua head
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Linear projection output
        out = np.dot(context, self.Wo)

        return out, attn_weights


# ============================================================
# 🔁 Transformer Block (gabungan semua)
# ============================================================

class TransformerBlock:
    """
    Satu blok transformer: 
    [Input] → [Multi-Head Attention] → [Add & Norm] → [FeedForward] → [Add & Norm]
    """
    def __init__(self, d_model=8, num_heads=2, d_ff=16):
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Multi-head attention
        attn_out, attn_weights = self.attn.forward(x, mask)
        x = self.norm1.forward(x + attn_out)   # Add & Norm

        # FeedForward
        ff_out = self.ff.forward(x)
        x = self.norm2.forward(x + ff_out)     # Add & Norm

        return x, attn_weights


# ============================================================
# 🧩 Testing cepat
# ============================================================

if __name__ == "__main__":
    print("🚀 Testing cepat Transformer Layers (NumPy)\n")

    batch_size, seq_len, d_model = 1, 5, 8
    dummy_input = np.random.randn(batch_size, seq_len, d_model)

    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)
    out, attn = block.forward(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", out.shape)
    print("Attention shape:", attn.shape)
    print("\n✅ Semua layer bekerja dengan benar!")