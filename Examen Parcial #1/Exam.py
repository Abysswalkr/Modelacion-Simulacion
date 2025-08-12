# -*- coding: utf-8 -*-
"""
Examen Parcial - Modelización y Simulaciones (CC2017)

Reproduce:
- Bootstrapping (10 000 iteraciones) mediante Transformada Inversa
- Histograma de distribución de medias y su desviación estándar
- Segmentación en 5 rangos uniformes y cálculo de probabilidades
- Aceptación-Rechazo con un Generador Congruencial Lineal Mixto (GCLM)
- PMF simplificada (5 categorías) y CDF (teórica vs empírica)

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Datos de entrada
# -------------------------------
vals = np.array([9, 6, 7, 9, 3, 10, 10, 7, 8, 4], dtype=float)
probs = np.array([0.25, 0.08, 0.07, 0.04, 0.17, 0.06, 0.02, 0.03, 0.03, 0.25], dtype=float)
assert abs(probs.sum() - 1.0) < 1e-12, "Las probabilidades deben sumar 1."

# (a) Media original y offset a cero
mean_original = float(vals.mean())
vals_centered = vals - mean_original

# -------------------------------
# Transformada inversa (discreta)
# -------------------------------
cdf = np.cumsum(probs)

def inverse_transform_discrete(u: np.ndarray, support: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(cdf, u, side="left")
    return support[idx]

# -------------------------------
# (d) Bootstrapping de medias
# -------------------------------
N_iter = 10_000
n = len(vals)
rng = np.random.default_rng(42)
U = rng.random((N_iter, n))
samples = inverse_transform_discrete(U, vals, cdf)
means = samples.mean(axis=1)

std_of_means = float(means.std(ddof=1))

plt.figure()
plt.hist(means, bins=50, density=True)
plt.title("Distribución de Medias (Bootstrapping con Transformada Inversa)")
plt.xlabel("Media muestral")
plt.ylabel("Densidad")
plt.tight_layout()
plt.savefig("hist_means.png")
plt.close()

# -------------------------------
# (e) 5 rangos uniformes y probabilidades
# -------------------------------
min_m, max_m = float(means.min()), float(means.max())
edges = np.linspace(min_m, max_m, 6)  # 5 bins
counts, _ = np.histogram(means, bins=edges)
prob_bins = counts / counts.sum()
intervals_text = [f"[{edges[i]:.4f}, {edges[i+1]:.4f})" if i < 4 else f"[{edges[i]:.4f}, {edges[i+1]:.4f}]" for i in range(5)]
midpoints = 0.5 * (edges[:-1] + edges[1:])

df_bins = pd.DataFrame({"Rango": intervals_text, "Centro": midpoints, "Probabilidad": prob_bins})
df_bins.to_csv("prob_por_bin.csv", index=False)

# -------------------------------
# (f) Aceptación-Rechazo con GCLM (mixto)
# -------------------------------
# Soporte objetivo: 5 categorías (los bins), pmf objetivo = prob_bins
# Propuesta q: uniforme (1/5)
# c = 5 * max(prob_bins)

m = 2**32
a = 1664525
c_gclm = 1013904223
state = 123456789

def lcg_uniform():
    global state
    state = (a * state + c_gclm) % m
    return state / m

p = prob_bins.copy()
p_max = float(p.max())
c_rej = 5.0 * p_max

N_ar = 10_000
accepted = []
proposals = 0
while len(accepted) < N_ar:
    proposals += 1
    u1 = lcg_uniform()
    j = min(int(5 * u1), 4)
    u2 = lcg_uniform()
    if u2 <= (p[j] / p_max if p_max > 0 else 0.0):
        accepted.append(j)

accepted = np.array(accepted, dtype=int)
accept_rate = len(accepted) / proposals

counts_emp = np.bincount(accepted, minlength=5).astype(float)
pmf_emp = counts_emp / counts_emp.sum()
cdf_emp = np.cumsum(pmf_emp)

pmf_theo = p
cdf_theo = np.cumsum(pmf_theo)

# PMF teórica (barras)
plt.figure()
plt.bar(range(1, 6), pmf_theo)
plt.xticks(range(1, 6), [f"Bin {i}" for i in range(1, 6)])
plt.title("PMF simplificada (5 categorías)")
plt.xlabel("Categoría (bin)")
plt.ylabel("Probabilidad")
plt.tight_layout()
plt.savefig("pmf_simplificada.png")
plt.close()

# CDF teórica vs empírica (escalones)
x_support = np.arange(1, 6)
plt.figure()
plt.step(x_support, cdf_theo, where='post', label="CDF teórica")
plt.step(x_support, cdf_emp, where='post', label="CDF empírica")
plt.title("CDF simplificada (teórica vs empírica AR)")
plt.xlabel("Categoría (bin)")
plt.ylabel("Probabilidad acumulada")
plt.legend()
plt.tight_layout()
plt.savefig("cdf_simplificada.png")
plt.close()

# -------------------------------
# Resumen y exportes
# -------------------------------
summary = pd.DataFrame({
    "Métrica": [
        "Media original de los 10 valores",
        "Desv. estándar de la distribución de medias (bootstrap)",
        "Mínimo de las medias",
        "Máximo de las medias",
        "Constante de rechazo c",
        "Tasa de aceptación (AR)",
        "Propuestas totales (AR)",
        "Muestras aceptadas (AR)"
    ],
    "Valor": [
        mean_original,
        std_of_means,
        min_m,
        max_m,
        c_rej,
        accept_rate,
        proposals,
        len(accepted)
    ]
})
summary.to_csv("resumen_metricas.csv", index=False)

df_vals = pd.DataFrame({"Valor original": vals, "Valor centrado (offset)": vals_centered})
df_vals.to_csv("valores_y_offset.csv", index=False)

print("Listo. Archivos generados:")
for f in ["hist_means.png", "pmf_simplificada.png", "cdf_simplificada.png",
          "prob_por_bin.csv", "resumen_metricas.csv", "valores_y_offset.csv"]:
    print("-", f)
