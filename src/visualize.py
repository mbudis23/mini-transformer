# src/visualize.py
import matplotlib.pyplot as plt

def visualize_attention(attn, title="Attention Heatmap (Block 0, Head 0)"):
    plt.imshow(attn, cmap="viridis", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.colorbar()
    plt.show()