# src/utils.py
import numpy as np

def default_rng(seed=None):
    return np.random.default_rng(seed)

def xavier_init(rng, shape):
    """Xavier normal initialization."""
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 1:
        fan_in = shape[0]
        fan_out = shape[0]
    else:
        # For simplicity fallback:
        fan_in = np.prod(shape[:-1])
        fan_out = shape[-1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(loc=0.0, scale=std, size=shape).astype(np.float32)


def softmax(x, axis=-1):
    """Stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)
