# -*- coding: utf-8 -*-
# Python 3.9+
# Multi-armed bandit con ε-greedy y exploración balanceada.
# Incluye modo con CUOTA fija de exploración (como en la pizarra)
# y cálculo de P teórico.

from dataclasses import dataclass
from typing import List, Dict, Literal
import numpy as np
import random

Policy = Literal["eps_quota", "eps_prob"]
ExploitRule = Literal["Q", "true_mu"]

@dataclass
class Restaurante:
    nombre: str
    mu: float
    sigma: float
    def sample(self) -> float:
        return float(np.random.normal(self.mu, self.sigma, size=1)[0])

# ---------- Métricas teóricas ----------
def st_teorico(eps: float, M: int, mus: List[float]) -> float:
    """ST esperado si exploras εM veces y luego explotas al mejor verdadero."""
    mu_star = max(mus)
    mu_prom = sum(mus) / len(mus)
    E = eps * M
    return E * mu_prom + (M - E) * mu_star

def p_teorico(eps: float, M: int, mus: List[float]) -> float:
    """P = SM - ST teórico."""
    mu_star = max(mus)
    SM = M * mu_star
    ST = st_teorico(eps, M, mus)
    return SM - ST

# ---------- Simulación ----------
def run_bandit(
    restaurantes: List[Restaurante],
    M: int = 300,
    epsilon: float = 0.07,                 # 7%
    policy: Policy = "eps_quota",          # "eps_quota" (cuota fija) o "eps_prob" (probabilístico)
    explore_strategy: str = "balanced",    # "balanced" reparte por igual
    exploit_rule: ExploitRule = "true_mu", # "true_mu" reproduce la pizarra; "Q" usa estimados
    seed: int = 123
) -> Dict:
    random.seed(seed); np.random.seed(seed)
    K = len(restaurantes)

    Q = [0.0] * K           # medias estimadas
    N = [0] * K             # conteos por brazo
    total_reward = 0.0

    elecciones = []

    # --- Modo con CUOTA fija de exploración (primero explora, luego explota) ---
    if policy == "eps_quota":
        E = int(round(epsilon * M))        # pizarra: exploras exactamente εM días
        # Objetivo por brazo (balanceado)
        base = E // K
        per_arm_targets = [base] * K
        for i in range(E % K):             # reparte remanente si no es divisible
            per_arm_targets[i] += 1
        explored = [0] * K

        # Fase de EXPLORACIÓN balanceada (E pasos)
        for t in range(E):
            # elige el brazo con menor exploración respecto a su objetivo
            candidatos = [i for i in range(K) if explored[i] < per_arm_targets[i]]
            if not candidatos:
                a = t % K
            else:
                a = min(candidatos, key=lambda i: (explored[i], i))
            r = restaurantes[a].sample()
            explored[a] += 1
            N[a] += 1
            Q[a] += (r - Q[a]) / N[a]
            total_reward += r
            elecciones.append(a)

        # Fase de EXPLOTACIÓN (M-E pasos)
        for t in range(E, M):
            if exploit_rule == "true_mu":
                a = max(range(K), key=lambda i: restaurantes[i].mu)  # mejor verdadero
            else:
                maxQ = max(Q)
                cand = [i for i, q in enumerate(Q) if q == maxQ]
                a = random.choice(cand)
            r = restaurantes[a].sample()
            N[a] += 1
            Q[a] += (r - Q[a]) / N[a]
            total_reward += r
            elecciones.append(a)

    # --- Modo probabilístico clásico (ε por paso) ---
    else:
        def epsilon_schedule(_t: int) -> float:
            return epsilon  # fijo
        # (opcional) una pasada inicial para no arrancar en cero
        for a in range(K):
            r = restaurantes[a].sample()
            N[a] = 1; Q[a] = r
            total_reward += r
            elecciones.append(a)

        for t in range(K, M):
            eps = epsilon_schedule(t)
            if random.random() < eps:  # EXPLORAR
                if explore_strategy == "balanced":
                    a = min(range(K), key=lambda i: N[i])   # menos muestreado
                else:
                    a = random.randrange(K)
            else:  # EXPLOTAR (por Q)
                maxQ = max(Q)
                cand = [i for i, q in enumerate(Q) if q == maxQ]
                a = random.choice(cand)
            r = restaurantes[a].sample()
            N[a] += 1
            Q[a] += (r - Q[a]) / N[a]
            total_reward += r
            elecciones.append(a)

    # Métricas
    mus = [r.mu for r in restaurantes]
    mu_star = max(mus)
    SM = M * mu_star
    ST = total_reward
    P_sim = SM - ST

    ranking = sorted([(restaurantes[i].nombre, Q[i]) for i in range(K)],
                     key=lambda x: x[1], reverse=True)
    porcentajes = [100.0 * n / M for n in N]

    return {
        "total_reward_ST": ST,
        "SM": SM,
        "P_simulado": P_sim,
        "P_teorico": p_teorico(epsilon, M, mus),
        "Q_estimados": {restaurantes[i].nombre: Q[i] for i in range(K)},
        "conteos": {restaurantes[i].nombre: N[i] for i in range(K)},
        "porcentajes": {restaurantes[i].nombre: porcentajes[i] for i in range(K)},
        "ranking_estimado": ranking,
        "policy": policy,
        "epsilon": epsilon
    }

# ---------- Demo ----------
if __name__ == "__main__":
    # Parámetros de la pizarra
    R1 = Restaurante("R1", 10.0, 5.0)
    R2 = Restaurante("R2", 8.0, 4.0)
    R3 = Restaurante("R3", 5.0, 2.5)
    restaurantes = [R1, R2, R3]

    M = 300
    EPS = 0.07  # <<< 7%

    # Modo pizarra: explora εM balanceado y luego explota al mejor real
    res_quota = run_bandit(
        restaurantes, M=M, epsilon=EPS,
        policy="eps_quota",
        exploit_rule="true_mu",   # produce P_teórico=49 con estos datos
        seed=123
    )
    print("=== ε-greedy (cuota fija, exploración balanceada, explotación por μ verdadero) ===")
    print("ST (simulado):", round(res_quota["total_reward_ST"], 2))
    print("SM:", res_quota["SM"])
    print("P_simulado:", round(res_quota["P_simulado"], 2))
    print("P_teorico:", res_quota["P_teorico"])
    print("Conteos:", res_quota["conteos"])
    print("Porcentajes (%):", {k: round(v, 1) for k, v in res_quota["porcentajes"].items()})
    print("Ranking estimado (Q):", [(n, round(q, 3)) for n, q in res_quota["ranking_estimado"]])

    # Modo ε probabilístico clásico (por si quieres comparar)
    res_prob = run_bandit(
        restaurantes, M=M, epsilon=EPS,
        policy="eps_prob",
        explore_strategy="balanced",
        exploit_rule="Q",
        seed=123
    )
    print("\n=== ε-greedy (probabilístico, exploración balanceada, explotación por Q) ===")
    print("ST (simulado):", round(res_prob["total_reward_ST"], 2))
    print("SM:", res_prob["SM"])
    print("P_simulado:", round(res_prob["P_simulado"], 2))
    print("P_teorico:", res_prob["P_teorico"])
    print("Conteos:", res_prob["conteos"])
    print("Porcentajes (%):", {k: round(v, 1) for k, v in res_prob["porcentajes"].items()})
    print("Ranking estimado (Q):", [(n, round(q, 3)) for n, q in res_prob["ranking_estimado"]])
