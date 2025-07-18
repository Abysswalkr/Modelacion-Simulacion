import numpy as np
import matplotlib.pyplot as plt

# -------- Generador Congruencial Lineal (LCG) ----------------------
def lcg(n, seed=246813579,
        a=1103515245, c=12345, m=2**31):
    """Devuelve n valores en (0,1) usando el LCG clásico de glibc."""
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        yield x / m

# -------- Simulación Monte Carlo: círculo de radio 1 dentro de [-1,1]² ----
N          = 6000               # número de puntos
gen        = lcg(2 * N)         # 2*N valores -> (x,y)
puntos     = np.fromiter(gen, float).reshape(N, 2) * 2 - 1  # escala a [-1,1]
x, y       = puntos[:, 0], puntos[:, 1]
dist_sq    = x**2 + y**2
inside     = dist_sq <= 1.0     # círculo de radio 1
pi_est     = inside.mean() * 4  # proporción*área cuadrado (4)

# -------- Gráfica ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x[inside],  y[inside],  s=12, c="tab:blue",  label="Inside circle")
ax.scatter(x[~inside], y[~inside], s=12, c="tab:red",   label="Outside circle")

# circunferencia límite
theta = np.linspace(0, 2*np.pi, 400)
ax.plot(np.cos(theta), np.sin(theta), color="green", lw=2, label="Boundary")

# cuadro límite
ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="black", lw=1)

ax.set_aspect("equal", "box")
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Monte Carlo: π ≈ {pi_est:.5f}  (N={N})")
ax.legend(loc="upper right")
ax.grid(True)
plt.show()
