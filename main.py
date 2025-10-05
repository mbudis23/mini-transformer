# main.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import argparse
from src.model import DecoderOnlyTransformer
import matplotlib.pyplot as plt

def visualize_attention(attn, title="Attention Heatmap"):
    """Visualisasi matriks attention (Q-K Softmax)."""
    plt.imshow(attn, cmap="viridis", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.colorbar()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Minimal GPT-style Transformer (NumPy)")

    # --- HYPERPARAMETERS ---
    parser.add_argument("--vocab_size", type=int, default=30, help="Ukuran vocabulary (jumlah token unik)")
    parser.add_argument("--seq_len", type=int, default=8, help="Panjang maksimum sequence")
    parser.add_argument("--d_model", type=int, default=32, help="Dimensi embedding (ukuran vektor per token)")
    parser.add_argument("--n_heads", type=int, default=4, help="Jumlah head dalam Multi-Head Attention")
    parser.add_argument("--n_layers", type=int, default=2, help="Jumlah layer Transformer block")
    parser.add_argument("--positional", type=str, choices=["sinusoidal", "learned"], default="sinusoidal",
                        help="Tipe positional encoding (sinusoidal / learned)")
    parser.add_argument("--seed", type=int, default=42, help="Seed random untuk reproducibility")

    # --- INPUT/OUTPUT OPTIONS ---
    parser.add_argument("--input", type=str, nargs="+", help="Daftar token (contoh: --input 1 2 3 4 5)")
    parser.add_argument("--show_attention", action="store_true", help="Tampilkan heatmap attention pertama")
    parser.add_argument("--save_logits", type=str, help="Simpan logits ke file .npy")

    args = parser.parse_args()

    # ------------------------------
    # 1️⃣ Inisialisasi model
    # ------------------------------
    model = DecoderOnlyTransformer(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        rng_seed=args.seed
    )

    # ------------------------------
    # 2️⃣ Siapkan token input
    # ------------------------------
    if args.input:
        tokens = np.array([list(map(int, args.input))])
        print(f"Input tokens: {tokens}")
    else:
        rng = np.random.default_rng(args.seed)
        tokens = rng.integers(low=0, high=args.vocab_size, size=(1, args.seq_len))
        print(f"[Auto] Generated random tokens: {tokens}")

    if tokens.shape[1] > args.seq_len:
        raise ValueError(f"Sequence length ({tokens.shape[1]}) > max_seq_len ({args.seq_len})")

    # ------------------------------
    # 3️⃣ Forward pass
    # ------------------------------
    logits, probs_last, attn_list = model.forward(tokens, return_attentions=True)

    # ------------------------------
    # 4️⃣ Output
    # ------------------------------
    print("\n=== HASIL TRANSFORMER ===")
    print(f"Logits shape: {logits.shape}  -> [batch, seq_len, vocab_size]")
    print("Distribusi probabilitas token berikutnya (posisi terakhir):")
    print(np.round(probs_last, 4))
    print("Jumlah probabilitas:", np.sum(probs_last))

    # Simpan logits jika diminta
    if args.save_logits:
        np.save(args.save_logits, logits)
        print(f"\nLogits disimpan ke file: {args.save_logits}.npy")

    # Visualisasi attention matrix pertama
    if args.show_attention:
        attn = attn_list[0][0, 0]  # block 0, head 0
        print("\nAttention matrix (Block 0, Head 0):\n", np.round(attn, 3))
        visualize_attention(attn, "Attention Heatmap (Block 0, Head 0)")

if __name__ == "__main__":
    main()
