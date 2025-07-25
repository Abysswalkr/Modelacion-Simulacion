import random
from collections import Counter

# 1. Definimos la distribución objetivo Pj (valores sin normalizar)
Pj = [6, 6, 6, 6, 6, 15, 13, 14, 15, 13]

# 2. Normalizamos Pj para obtener p_j (sum p_j = 1)
total = sum(Pj)
pj = [p/total for p in Pj]

# 3. Propuesta uniforme q_j = 1/10
n = len(Pj)
qj = [1/n] * n

# 4. Calculamos la constante C >= max(pj/qj)
#    Aquí max(pj/qj) = max(pj) / (1/10) = 10 * max(pj)
C = max(pj_i / qj_i for pj_i, qj_i in zip(pj, qj))

def rejection_sample(num_samples):
    """
    Genera num_samples muestras de la distribución pj
    usando muestreo por rechazo con propuesta uniforme.
    Devuelve una lista de índices en 1..n.
    """
    muestras = []
    while len(muestras) < num_samples:
        # 5. Muestreamos y ~ Cat({1,...,n}, probs=qj) uniformemente
        j = random.randint(0, n-1)      # índice 0..n-1
        u = random.random()             # U ~ Uniform(0,1)
        # 6. Criterio de aceptación: U <= pj[j] / (C * qj[j])
        if u <= pj[j] / (C * qj[j]):
            muestras.append(j+1)        # +1 para que sea 1..n
    return muestras

if __name__ == "__main__":
    N = 10000
    muestras = rejection_sample(N)

    # Mostramos las primeras 20 muestras
    print("Primeras 20 muestras:", muestras[:20])

    # Calculamos frecuencias empíricas
    freq = Counter(muestras)
    dist_emp = {j: freq[j]/N for j in sorted(freq)}
    print("Frecuencia empírica aproximada:")
    for j, fr in dist_emp.items():
        print(f"  P({j}) ≈ {fr:.3f}")

    # Para comparar, mostramos la distribución teórica
    print("\nDistribución teórica p_j:")
    for j, p in enumerate(pj, start=1):
        print(f"  P({j}) = {p:.3f}")
