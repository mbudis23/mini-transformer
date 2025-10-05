# Mini Transformer

## Deskripsi

Kode ini merupakan implementasi arsitektur transformer dari nol menggunakan Numpy tanpa deep learning seperti PyTorch atau tensorflow.

## Instalasi

1. Clone repository

```bash
git clone https://github.com/mbudis23/mini-transformer
cd mini-transformer
```

2. Buat virtual env dan aktifkan

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install depedensi

```bash
pip install numpy matplotlib
```

## Menjalankan Program

Kode untuk menjalankan program:

```bash
python main.py
```

Secara default, program akan membuat token input acak, melakukan forward pass ke model Transformer, dan menampilkan hasil logits dan distribusi probabilitas token berikutnya.

Contoh output:

```lua
[Auto] Generated random tokens: [[ 2 23 19 13 12 25  2 20]]

=== HASIL TRANSFORMER ===
Logits shape: (1, 8, 30) -> [batch, seq_len, vocab_size]

Contoh nilai logits (token terakhir):
[ 0.4647  0.3593 -0.3293 -0.5756 -0.078   0.3178 -0.5636 -1.5524  1.0219
  0.0025  0.786  -0.2663  0.316   0.324  -2.4536 -1.9422  1.3678 -1.1374
  0.5244  1.4412  0.7193 -0.6414 -0.2374 -0.3963 -0.2695 -2.3858  1.3944
  1.1637 -0.8517  2.0631]

Distribusi probabilitas token berikutnya (softmax di posisi terakhir):
[[0.0334 0.03   0.0151 0.0118 0.0194 0.0288 0.0119 0.0044 0.0582 0.021
  0.046  0.0161 0.0288 0.029  0.0018 0.003  0.0823 0.0067 0.0354 0.0886
  0.043  0.011  0.0165 0.0141 0.016  0.0019 0.0845 0.0671 0.0089 0.165 ]]
Jumlah probabilitas: 1.0

Token yang paling mungkin muncul berikutnya: 29 (probabilitas = 0.1650)
```

## Token Input Manual

Guanakan argumen `--input` untuk menentukan token secara eksplisit:

```bash
python main.py --input 2 5 7 9 10 3 12 8
```

## Visualisasi Attention

Gunakan `--show_attention` untuk menampilkan heatmap dari attention matriks pada block dan head pertama.

```bash
python main.py --show_attention
```

## Argumen Lengkap

| Argumen            | Default | Deskripsi                                       |
| ------------------ | ------- | ----------------------------------------------- |
| `--vocab_size`     | 30      | Jumlah token unik (ukuran vocabulary)           |
| `--seq_len`        | 8       | Panjang maksimum urutan token                   |
| `--d_model`        | 32      | Dimensi vektor embedding                        |
| `--n_heads`        | 4       | Jumlah head di multi-head attention             |
| `--n_layers`       | 2       | Jumlah layer Transformer block                  |
| `--seed`           | 42      | Nilai seed untuk random generator               |
| `--input`          | –       | Token input manual (misal: `--input 1 2 3 4 5`) |
| `--show_attention` | `False` | Menampilkan heatmap attention                   |
| `--save_logits`    | –       | Menyimpan hasil logits ke file `.npy`           |

## Komponen yang dibuat dalam Transformer

| Komponen                   | Penjelasan                                                                |
| -------------------------- | ------------------------------------------------------------------------- |
| TokenEmbedding             | Mengubah token ID menjadi vektor embedding berukuran `d_model`            |
| Positional Encoding        | Menambahkan informasi posisi ke embedding                                 |
| Multi-Head Attention       | Mekanisme untuk memperhatikan hubungan antar token dengan _causal mask_   |
| Feed Forward Network (FFN) | Layer non-linear untuk transformasi vektor tiap posisi                    |
| Residual + LayerNorm       | Menstabilkan training dan menjaga aliran gradien                          |
| Output Projection          | Mengubah representasi terakhir menjadi skor logits untuk tiap token vocab |

## Cara Kerja

1. Input Token --> diubah jadi embedding.
2. Positional Encoding ditambahkan pada embedding.
3. Decoder blocks melakukan multi-head attention + FFN.
4. Hasil akhir --> diproyeksikan ke `vocab_size` untuk menghasilkan `logits`.
5. Softmax di posisi terakhir --> menghasilkan distribusi probabilitas token berikutnya.

## Pembuat

Muhammad Budi Setiawan (22/505064/TK/55254)

## Catatan Tambahan

- Model ini tidak melakukan training, hanya forward pass.
- Semua bobot diinisialisasi acak dengan Xavier initialization.
- Distribusi output hanya menunjukkan perilaku inferensi acak berdasarkan bobot awal.
