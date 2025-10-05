import numpy as np
import re
from collections import Counter
import random

class TextPreprocessor:
    """
    Kelas untuk melakukan preprocessing teks:
    - Tokenisasi sederhana
    - Pembuatan vocabulary
    - Konversi token <-> id
    - Pembuatan pasangan training (Skip-Gram)
    - Pembuatan sequence untuk training Transformer
    """

    def __init__(self, min_freq=1, window_size=2, seed=42):
        """
        Parameters
        ----------
        min_freq : int
            Frekuensi minimum agar kata dimasukkan ke vocab.
        window_size : int
            Ukuran konteks di sekitar target (Skip-Gram).
        seed : int
            Random seed untuk konsistensi hasil.
        """
        self.min_freq = min_freq
        self.window_size = window_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.vocab = None
        self.token2id = {}
        self.id2token = {}

    def tokenize(self, text):
        """Tokenisasi sederhana dengan lowercase dan hapus simbol."""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s\u00C0-\u024F\u1E00-\u1EFF]", "", text)
        tokens = text.strip().split()
        return tokens

    def build_vocab(self, sentences):
        """Bangun vocabulary dari daftar kalimat."""
        tokens = []
        for s in sentences:
            tokens.extend(self.tokenize(s))

        counter = Counter(tokens)
        vocab = [word for word, freq in counter.items() if freq >= self.min_freq]

        # Tambahkan token spesial
        vocab = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + sorted(vocab)

        self.vocab = vocab
        self.token2id = {w: i for i, w in enumerate(vocab)}
        self.id2token = {i: w for w, i in self.token2id.items()}
        return self.vocab

    def encode(self, tokens):
        """Ubah daftar token menjadi daftar ID."""
        return [self.token2id.get(t, self.token2id["<UNK>"]) for t in tokens]

    def decode(self, ids):
        """Ubah daftar ID menjadi daftar token."""
        return [self.id2token.get(i, "<UNK>") for i in ids]

    def generate_skipgram_pairs(self, sentences):
        """
        Membuat pasangan (target, context) untuk Word2Vec Skip-Gram.
        Output:
            pairs: list of tuples (target_id, context_id)
        """
        pairs = []
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            token_ids = self.encode(tokens)
            for i, target_id in enumerate(token_ids):
                window = random.randint(1, self.window_size)
                start = max(0, i - window)
                end = min(len(token_ids), i + window + 1)
                for j in range(start, end):
                    if i != j:
                        pairs.append((target_id, token_ids[j]))
        return np.array(pairs)

    def create_sequences(self, sentences, max_len=10):
        """
        Membuat input sequence untuk training transformer (decoder-only).
        Tiap kalimat diberi token <BOS> dan <EOS>.
        Output:
            inputs: np.ndarray [num_seq, max_len]
            targets: np.ndarray [num_seq, max_len]
        """
        sequences = []
        for s in sentences:
            tokens = ["<BOS>"] + self.tokenize(s) + ["<EOS>"]
            token_ids = self.encode(tokens)
            # Padding
            if len(token_ids) < max_len:
                token_ids += [self.token2id["<PAD>"]] * (max_len - len(token_ids))
            else:
                token_ids = token_ids[:max_len]
            sequences.append(token_ids)

        sequences = np.array(sequences)
        inputs = sequences[:, :-1]   # semua kecuali token terakhir
        targets = sequences[:, 1:]   # semua kecuali token pertama
        return inputs, targets


def load_corpus(file_path):
    """Membaca file teks dan memisahkannya menjadi daftar kalimat."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def prepare_datasets(file_path, max_len=10, window_size=2, min_freq=1):
    """
    Fungsi helper cepat untuk:
    - Membaca corpus
    - Membuat vocab
    - Menghasilkan data untuk Skip-Gram dan Transformer
    """
    pre = TextPreprocessor(min_freq=min_freq, window_size=window_size)
    sentences = load_corpus(file_path)
    vocab = pre.build_vocab(sentences)
    skipgram_pairs = pre.generate_skipgram_pairs(sentences)
    transformer_inputs, transformer_targets = pre.create_sequences(sentences, max_len=max_len)

    return {
        "vocab": vocab,
        "skipgram_pairs": skipgram_pairs,
        "transformer_inputs": transformer_inputs,
        "transformer_targets": transformer_targets,
        "preprocessor": pre
    }


if __name__ == "__main__":
    data = prepare_datasets("data/corpus.txt", max_len=8)
    print("Vocab size:", len(data["vocab"]))
    print("Contoh pasangan Skip-Gram:", data["skipgram_pairs"][:5])
    print("Input Transformer:", data["transformer_inputs"][:2])
    print("Target Transformer:", data["transformer_targets"][:2])