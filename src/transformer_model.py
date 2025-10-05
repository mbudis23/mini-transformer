# src/transformer_model.py

import numpy as np
from transformer_layers import TransformerBlock


class MiniTransformer:
    """
    Mini Transformer berbasis NumPy (Decoder-only)
    ----------------------------------------------
    - Embedding kata + positional (learnable)
    - Beberapa TransformerBlock
    - Output: prediksi distribusi kata berikutnya
    """

    def __init__(
        self,
        vocab_size,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_len=20,
        seed=42
    ):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_len = max_len

        # === Embedding layer ===
        self.word_embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.pos_embedding = np.random.randn(max_len, d_model) * 0.01  # learnable positional embedding

        # === Transformer blocks ===
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

        # === Output projection ===
        self.Wo = np.random.randn(d_model, vocab_size) * 0.01
        self.bo = np.zeros((1, vocab_size))

    # ---------------------------------------------------------
    # Helper
    # ---------------------------------------------------------
    def get_positional_embeddings(self, seq_len):
        """Mengambil embedding posisi yang bisa dilatih"""
        return self.pos_embedding[:seq_len]

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    def forward(self, input_ids, mask=None):
        """
        input_ids: array (batch_size, seq_len)
        """
        batch_size, seq_len = input_ids.shape

        # === Token + Positional Embedding ===
        token_emb = self.word_embedding[input_ids]                   # (batch, seq_len, d_model)
        pos_emb = self.get_positional_embeddings(seq_len)            # (seq_len, d_model)
        x = token_emb + pos_emb                                      # broadcast ke batch

        # === Transformer Blocks ===
        for block in self.blocks:
            x, _ = block.forward(x, mask=mask)

        # === Output logits ===
        logits = np.dot(x, self.Wo) + self.bo                        # (batch, seq_len, vocab_size)
        probs = self.softmax(logits)

        return probs, logits

    # ---------------------------------------------------------
    # Simple loss (cross-entropy)
    # ---------------------------------------------------------
    def compute_loss(self, probs, target_ids):
        """
        probs: (batch, seq_len, vocab_size)
        target_ids: (batch, seq_len)
        """
        batch_size, seq_len = target_ids.shape
        losses = []
        for b in range(batch_size):
            for t in range(seq_len):
                target = target_ids[b, t]
                losses.append(-np.log(probs[b, t, target] + 1e-8))
        return np.mean(losses)

    # ---------------------------------------------------------
    # Simple training step (manual gradient, single-step)
    # ---------------------------------------------------------
    def train_step(self, input_ids, target_ids, lr=0.01):
        """
        Placeholder training: hanya forward loss (tanpa backprop penuh).
        Untuk penelitian/eksperimen sederhana.
        """
        probs, _ = self.forward(input_ids)
        loss = self.compute_loss(probs, target_ids)
        return loss

    # ---------------------------------------------------------
    # Text generation (sampling)
    # ---------------------------------------------------------
    def generate(self, start_ids, max_new_tokens=10):
        """
        Autoregressive text generation sederhana
        """
        generated = list(start_ids)
        for _ in range(max_new_tokens):
            input_ids = np.array([generated[-self.max_len:]])  # ambil max_len terakhir
            probs, _ = self.forward(input_ids)
            next_token = np.argmax(probs[0, -1])
            generated.append(next_token)
        return generated


# ============================================================
# 🧪 Testing cepat
# ============================================================
if __name__ == "__main__":
    print("🚀 Testing cepat MiniTransformer NumPy\n")

    vocab_size = 10
    seq_len = 5
    batch_size = 1

    np.random.seed(1)
    dummy_input = np.random.randint(0, vocab_size, (batch_size, seq_len))
    dummy_target = np.random.randint(0, vocab_size, (batch_size, seq_len))

    model = MiniTransformer(vocab_size=vocab_size, d_model=8, num_heads=2, num_layers=2, max_len=10)
    probs, logits = model.forward(dummy_input)
    loss = model.compute_loss(probs, dummy_target)

    print("Input shape :", dummy_input.shape)
    print("Output probs shape:", probs.shape)
    print("Loss:", round(loss, 4))

    print("\nContoh generasi token:")
    generated = model.generate(start_ids=[1, 2, 3], max_new_tokens=5)
    print("Generated token IDs:", generated)

    print("\n✅ Transformer model berjalan normal dengan NumPy!")