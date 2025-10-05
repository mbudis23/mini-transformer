# src/training.py

import numpy as np
from utils import cross_entropy_loss, compute_accuracy, clip_gradients


class Trainer:
    """
    Trainer sederhana berbasis NumPy untuk Mini Transformer atau model lainnya.
    Tidak menggunakan library eksternal selain NumPy.
    """

    def __init__(self, model, lr=0.01, epochs=1000, clip_norm=1.0, print_every=100):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.clip_norm = clip_norm
        self.print_every = print_every

    def step(self, grads):
        """Update parameter menggunakan gradien"""
        grads = clip_gradients(grads, self.clip_norm)
        for name, param in self.model.params.items():
            self.model.params[name] -= self.lr * grads[name]

    def train(self, X, y):
        """
        Loop pelatihan sederhana.
        -----------------------------------------
        X: input (batch, seq_len)
        y: label (batch, seq_len)
        """
        history = {"loss": [], "acc": []}

        for epoch in range(1, self.epochs + 1):
            logits = self.model.forward(X)  # (batch, seq_len, vocab)
            loss, probs = cross_entropy_loss(logits, y)

            # Dummy gradien (karena backward manual Transformer sangat kompleks)
            grads = {name: np.random.randn(*param.shape) * 0.001 for name, param in self.model.params.items()}

            # Update parameter
            self.step(grads)

            # Hitung akurasi
            acc = compute_accuracy(logits, y)

            history["loss"].append(loss)
            history["acc"].append(acc)

            if epoch % self.print_every == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")

        return history


# ======================================================
# 🔹 Quick Test (jalankan langsung untuk test cepat)
# ======================================================
if __name__ == "__main__":
    class DummyModel:
        """
        Model dummy untuk memastikan Trainer bekerja dengan benar.
        Tidak ada perhitungan kompleks — hanya NumPy murni.
        """
        def __init__(self, vocab_size=10, seq_len=5, d_model=8):
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.d_model = d_model
            self.params = {
                "W1": np.random.randn(d_model, d_model) * 0.01,
                "W2": np.random.randn(d_model, vocab_size) * 0.01,
            }

        def forward(self, X):
            batch_size, seq_len = X.shape
            # Representasi sederhana
            hidden = np.tanh(np.random.randn(batch_size, seq_len, self.d_model))
            logits = np.einsum('bsd,dk->bsk', hidden, self.params["W2"])
            return logits

    # Dummy data
    np.random.seed(42)
    X_dummy = np.random.randint(0, 10, (4, 5))  # batch=4, seq_len=5
    y_dummy = np.random.randint(0, 10, (4, 5))

    dummy_model = DummyModel()
    trainer = Trainer(dummy_model, lr=0.05, epochs=10, print_every=2)

    history = trainer.train(X_dummy, y_dummy)

    print("\nTesting selesai ✅")
    print(f"Loss akhir: {history['loss'][-1]:.4f}")
    print(f"Akurasi akhir: {history['acc'][-1]:.4f}")