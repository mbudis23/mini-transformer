# src/utils.py

import numpy as np

def softmax(x, axis=-1):
    x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

def cross_entropy_loss(logits, y_true):
    """
    logits: (batch, seq_len, vocab)
    y_true: (batch, seq_len)
    """
    probs = softmax(logits, axis=-1)
    batch, seq_len, vocab = probs.shape
    flat_indices = y_true.flatten()
    selected = probs.reshape(-1, vocab)[np.arange(batch * seq_len), flat_indices]
    loss = -np.mean(np.log(selected + 1e-8))
    return loss, probs

def compute_accuracy(logits, y_true):
    preds = np.argmax(logits, axis=-1)
    return np.mean(preds == y_true)

def clip_gradients(grads, max_norm):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        grads = {k: v * scale for k, v in grads.items()}
    return grads