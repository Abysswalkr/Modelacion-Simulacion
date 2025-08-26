# eca.py — Autómata celular elemental (Wolfram)
import numpy as np
import matplotlib.pyplot as plt

def run_eca(rule=90, width=601, steps=601, seed="center", wrap=False):
    """
    rule : int [0..255] (p.ej. 30, 90, 110)
    width: número de celdas por fila
    steps: número de generaciones (filas)
    seed : "center" | "random" | array binario de largo `width`
    wrap : True para frontera periódica, False para fijar bordes a 0
    """
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)

    if isinstance(seed, str):
        if seed == "center":
            state = np.zeros(width, dtype=np.uint8)
            state[width // 2] = 1
        elif seed == "random":
            state = np.random.randint(0, 2, size=width, dtype=np.uint8)
        else:
            raise ValueError("seed debe ser 'center', 'random' o un array binario.")
    else:
        state = np.array(seed, dtype=np.uint8)
        assert state.size == width, "El seed debe tener tamaño `width`"

    grid = np.zeros((steps, width), dtype=np.uint8)
    grid[0] = state

    for t in range(1, steps):
        left  = np.roll(state, 1)
        right = np.roll(state, -1)
        if not wrap:
            left[0] = 0
            right[-1] = 0
        idx = (left << 2) | (state << 1) | right  # patrón LCR como número 0..7
        state = rule_bits[idx]
        grid[t] = state

    return grid

if __name__ == "__main__":
    # Ejemplo: Regla 90 (Sierpinski)
    grid = run_eca(rule=90, width=601, steps=601, seed="center", wrap=False)

    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(grid, cmap="binary", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
