# src/model.py
import numpy as np
from .utils import default_rng, softmax, xavier_init
from .components import (
    TokenEmbedding, SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding, DecoderBlock, LayerNorm
)

class DecoderOnlyTransformer:
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 d_model=64,
                 n_heads=8,
                 n_layers=2,
                 d_ff=None,
                 rng_seed=0,
                 tie_weights=True):
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
        
        self.pos_emb = SinusoidalPositionalEncoding(max_seq_len, d_model)

        # stack of decoder blocks
        self.blocks = [DecoderBlock(d_model, n_heads, self.d_ff, self.rng) for _ in range(n_layers)]
        # final layer norm (optional; typical in many implementations)
        self.ln_f = LayerNorm(d_model)

        # output projection (if not tying weights)
        if not tie_weights:
            self.W_out = xavier_init(self.rng, (d_model, vocab_size))

    # Causal Masking untuk mencegah akses informasi dari token masa depan
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

    # Output Layer: proyeksi ke ukuran vocab dan distribusi softmax
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