import numpy as np
import matplotlib.pyplot as plt

# ── Parámetros del LCG solicitados ─────────────────────────────────
x0 = 2
a  = 1
c  = 3
m  = 10

def lcg(n, seed=x0):
    """
    Genera n números pseudo‑aleatorios U(0,1) con el LCG:
        x_{k+1} = (a*x_k + c) mod m
    Normalizados dividiendo entre m.
    """
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        yield x / m                 # en (0,1)

# ── Juego de dardos: círculo de radio 1 en el cuadrado [-1,1]² ─────
N = 200          # cantidad de dardos (puedes cambiarlo)
coords = np.fromiter(lcg(2*N), float, count=2*N).reshape(N, 2)
# Mapea U(0,1) → [-1,1]
points = coords * 2.0 - 1.0
x, y = points[:, 0], points[:, 1]

# Clasificación dentro/fuera del círculo de radio 1
inside = x**2 + y**2 <= 1.0
pi_est = inside.mean() * 4          # proporción * área del cuadrado (4)

# ── Gráfica ────────────────────────────────────────────────────────
plt.figure(figsize=(6, 6))
plt.scatter(x[inside],  y[inside],  s=20, marker='o', label="Dentro del círculo")
plt.scatter(x[~inside], y[~inside], s=20, marker='x', label="Fuera del círculo")

# Circunferencia límite
theta = np.linspace(0, 2*np.pi, 400)
plt.plot(np.cos(theta), np.sin(theta), label="Circunferencia")

# Límite del cuadrado
plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1])

plt.gca().set_aspect('equal', 'box')
plt.xlim(-1.05, 1.05)
plt.ylim(-1.05, 1.05)
plt.title(f"Monte Carlo con LCG")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
