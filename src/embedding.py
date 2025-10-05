# src/embedding.py

import numpy as np


class WordEmbedding:
    """
    Implementasi Word Embedding (Skip-Gram) sederhana berbasis NumPy.
    ----------------------------------------------------------------
    - Tanpa library eksternal
    - Menggunakan perhitungan manual untuk forward & backward pass
    """

    def __init__(self, vocab_size, embedding_dim=8, lr=0.05, epochs=5000, window_size=2, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.epochs = epochs
        self.window_size = window_size

        # Matriks bobot: W1 (input) dan W2 (output)
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01

    def softmax(self, x):
        """Fungsi Softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, center_idx):
        """Forward pass — menghitung distribusi probabilitas kata konteks"""
        h = self.W1[center_idx]           # (embedding_dim,)
        u = np.dot(h, self.W2)            # (vocab_size,)
        y_pred = self.softmax(u)
        return y_pred, h

    def backward(self, y_pred, context_idx, h, center_idx):
        """Backward pass — update bobot berdasarkan error"""
        y_true = np.zeros(self.vocab_size)
        y_true[context_idx] = 1.0

        # Gradien
        e = y_pred - y_true
        dW2 = np.outer(h, e)              # (embedding_dim, vocab_size)
        dW1 = np.dot(self.W2, e)          # (embedding_dim,)

        # Update parameter
        self.W1[center_idx] -= self.lr * dW1
        self.W2 -= self.lr * dW2

        # Hitung loss
        loss = -np.log(y_pred[context_idx] + 1e-8)
        return loss

    def train(self, training_data, verbose=True):
        """Proses pelatihan skip-gram"""
        for epoch in range(self.epochs):
            total_loss = 0.0
            np.random.shuffle(training_data)
            for center, context in training_data:
                y_pred, h = self.forward(center)
                loss = self.backward(y_pred, context, h, center)
                total_loss += loss
            if verbose and (epoch + 1) % (self.epochs // 10) == 0:
                avg_loss = total_loss / len(training_data)
                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f}")

    def get_embedding_matrix(self):
        """Mengambil matriks embedding hasil pelatihan"""
        return self.W1

    def get_vector(self, word_idx):
        """Mendapatkan vektor embedding untuk kata tertentu"""
        return self.W1[word_idx]


def generate_training_data(tokens, word_to_idx, window_size=2):
    """
    Membuat pasangan (center, context) untuk Skip-Gram.
    Contoh: "ai is transforming the world" → pasangan (ai,is), (ai,transforming), ...
    """
    training_data = []
    for i, word in enumerate(tokens):
        center = word_to_idx[word]
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                context = word_to_idx[tokens[j]]
                training_data.append((center, context))
    return np.array(training_data)


# ==============================
# Contoh penggunaan & testing cepat
# ==============================
if __name__ == "__main__":
    print("Testing cepat WordEmbedding (Skip-Gram NumPy)\n")

    # Mini korpus
    corpus = "saya suka belajar mesin belajar membantu manusia"
    tokens = corpus.lower().split()

    # Buat vocab
    vocab = sorted(set(tokens))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    print(f"Vocab: {vocab}\n")

    # Buat pasangan (center, context)
    training_data = generate_training_data(tokens, word_to_idx, window_size=1)
    print("Contoh pasangan training (center -> context):")
    for i in range(5):
        c, ctx = training_data[i]
        print(f"  {idx_to_word[c]} -> {idx_to_word[ctx]}")

    # Buat dan latih model
    model = WordEmbedding(vocab_size=len(vocab), embedding_dim=4, lr=0.05, epochs=500)
    model.train(training_data, verbose=False)

    print("\nTraining selesai!")
    print("Embedding untuk kata 'belajar':")
    print(model.get_vector(word_to_idx["belajar"]))

    # Coba kemiripan sederhana
    emb = model.get_embedding_matrix()
    sim = np.dot(emb[word_to_idx["belajar"]], emb.T)
    sim_idx = np.argsort(-sim)[:3]
    print("\nKata paling mirip dengan 'belajar':")
    for idx in sim_idx:
        print(f"  {idx_to_word[idx]} (similarity={sim[idx]:.4f})")