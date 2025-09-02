# Python 3.9+
# Bandits: 3 restaurantes (R1, R2, R3) con recompensas ~ Normal(mu, sigma)
# Estrategia: ε-greedy (ε fijo o decreciente)

from dataclasses import dataclass
import numpy as np
import random
from typing import List, Tuple

@dataclass
class Restaurante:
    nombre: str
    mu: float
    sigma: float

    def sample(self) -> float:
        # Muestra una recompensa (satisfacción) del restaurante
        return float(np.random.normal(self.mu, self.sigma, size=1)[0])

def epsilon_schedule(t: int, eps0: float = 0.10, eps_min: float = 0.02, k: float = 50.0) -> float:
    """
    ε decreciente: eps(t) = max(eps_min, eps0 / (1 + t/k))
    - eps0: ε inicial
    - eps_min: piso de ε
    - k: controla qué tan rápido cae
    """
    return max(eps_min, eps0 / (1.0 + t / k))

def run_epsilon_greedy(
    restaurantes: List[Restaurante],
    M: int = 300,
    epsilon_fijo: float = None,   # si se da, usa ε fijo; si es None, usa schedule decreciente
    seed: int = 42
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    K = len(restaurantes)
    # Estimaciones Q(a) y conteos N(a)
    Q = [0.0] * K
    N = [0] * K

    total_reward = 0.0
    elecciones = []

    # Inicializa explorando una vez cada brazo (opcional pero recomendado)
    for a in range(K):
        r = restaurantes[a].sample()
        Q[a] = r
        N[a] = 1
        total_reward += r
        elecciones.append(a)

    # Rondas restantes
    for t in range(K, M):
        eps = epsilon_fijo if epsilon_fijo is not None else epsilon_schedule(t)
        # Explora con prob ε; explota con 1-ε
        if random.random() < eps:
            a = random.randrange(K)
        else:
            # argmax con desempate aleatorio
            maxQ = max(Q)
            candidatos = [i for i, q in enumerate(Q) if q == maxQ]
            a = random.choice(candidatos)

        r = restaurantes[a].sample()
        elecciones.append(a)
        total_reward += r

        # Actualización incremental de Q(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

    # Métricas
    mu_opt = max(r.mu for r in restaurantes)
    reward_opt_esperado = mu_opt * M
    regret = reward_opt_esperado - total_reward

    # Ranking estimado por Q
    ranking = sorted([(restaurantes[i].nombre, Q[i]) for i in range(K)],
                     key=lambda x: x[1], reverse=True)

    # Resumen de elecciones
    conteos = [elecciones.count(i) for i in range(K)]
    porcentajes = [100.0 * c / M for c in conteos]

    return {
        "total_reward": total_reward,
        "regret_vs_optimo": regret,
        "Q_estimados": {restaurantes[i].nombre: Q[i] for i in range(K)},
        "conteos": {restaurantes[i].nombre: conteos[i] for i in range(K)},
        "porcentajes": {restaurantes[i].nombre: porcentajes[i] for i in range(K)},
        "ranking_estimado": ranking,
        "elecciones_indices": elecciones,
    }

if __name__ == "__main__":
    # Parámetros según tu pizarra (ajusta si es necesario)
    mus =   [10.0, 8.0, 5.0]
    sigmas = [5.0, 4.0, 2.5]
    restaurantes = [
        Restaurante("R1", mus[0], sigmas[0]),
        Restaurante("R2", mus[1], sigmas[1]),
        Restaurante("R3", mus[2], sigmas[2]),
    ]

    # --- Ejecución con ε decreciente ---
    res_decay = run_epsilon_greedy(restaurantes, M=300, epsilon_fijo=None, seed=123)
    print("=== ε-greedy con ε decreciente ===")
    print("Satisfacción total:", round(res_decay["total_reward"], 2))
    print("Regret vs óptimo (300*10):", round(res_decay["regret_vs_optimo"], 2))
    print("Q estimados:", {k: round(v, 3) for k, v in res_decay["Q_estimados"].items()})
    print("Conteos:", res_decay["conteos"])
    print("Porcentajes (%):", {k: round(v, 1) for k, v in res_decay["porcentajes"].items()})
    print("Ranking estimado:", res_decay["ranking_estimado"])

    # --- Ejecución con ε fijo (por ejemplo 0.1) ---
    res_fixed = run_epsilon_greedy(restaurantes, M=300, epsilon_fijo=0.10, seed=123)
    print("\n=== ε-greedy con ε fijo=0.10 ===")
    print("Satisfacción total:", round(res_fixed["total_reward"], 2))
    print("Regret vs óptimo (300*10):", round(res_fixed["regret_vs_optimo"], 2))
    print("Q estimados:", {k: round(v, 3) for k, v in res_fixed["Q_estimados"].items()})
    print("Conteos:", res_fixed["conteos"])
    print("Porcentajes (%):", {k: round(v, 1) for k, v in res_fixed["porcentajes"].items()})
    print("Ranking estimado:", res_fixed["ranking_estimado"])
