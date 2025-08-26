#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic Algorithm to evolve the phrase "to be or not to be" (or any target string).
- Selection: tournament selection
- Crossover: uniform (per-gene 50/50 from each parent)
- Mutation: per-gene random replacement with probability = mutation_rate
- Replacement: generational with elitism
Python 3.9+
"""
from dataclasses import dataclass
import argparse
import random
import string
from typing import List, Tuple

ALPHABET = string.ascii_lowercase + " "  # letters + space


@dataclass
class GAConfig:
    target: str
    pop_size: int = 300
    mutation_rate: float = 0.02  # 2%
    tournament_k: int = 3
    elitism: int = 2
    max_gens: int = 2000
    seed: int = None
    log_path: str = None
    verbose: bool = True


def random_individual(length: int) -> str:
    return "".join(random.choice(ALPHABET) for _ in range(length))


def fitness(individual: str, target: str) -> int:
    # Count of matching characters in the correct positions
    return sum(1 for a, b in zip(individual, target) if a == b)


def tournament_select(population: List[str], target: str, k: int) -> str:
    # Sample k individuals, return the fittest
    sample = random.sample(population, k)
    sample.sort(key=lambda s: fitness(s, target), reverse=True)
    return sample[0]


def uniform_crossover(p1: str, p2: str) -> str:
    # Per-gene 50/50 choice from parents
    child_chars = [random.choice((a, b)) for a, b in zip(p1, p2)]
    return "".join(child_chars)


def mutate(individual: str, mutation_rate: float) -> str:
    # Replace each gene with probability mutation_rate
    out = []
    for ch in individual:
        if random.random() < mutation_rate:
            out.append(random.choice(ALPHABET))
        else:
            out.append(ch)
    return "".join(out)


def evolve(config: GAConfig) -> Tuple[str, int, list]:
    if config.seed is not None:
        random.seed(config.seed)

    target = config.target
    L = len(target)

    # Initialize population
    population = [random_individual(L) for _ in range(config.pop_size)]

    history = []  # (gen, best_fit, avg_fit, best_str)
    for gen in range(1, config.max_gens + 1):
        # Evaluate
        fits = [fitness(ind, target) for ind in population]
        best_idx = max(range(config.pop_size), key=lambda i: fits[i])
        best_fit = fits[best_idx]
        avg_fit = sum(fits) / len(fits)
        best_str = population[best_idx]
        history.append((gen, best_fit, avg_fit, best_str))

        if config.verbose:
            print(f"Gen {gen:4d} | best={best_fit}/{L} | avg={avg_fit:.2f} | {best_str}")

        # Check convergence
        if best_fit == L:
            return best_str, gen, history

        # Elitism: keep top N
        elite_count = max(0, min(config.elitism, config.pop_size))
        # Sort population by fitness desc
        ranked = [s for _, s in sorted(zip(fits, population), key=lambda t: t[0], reverse=True)]
        new_pop = ranked[:elite_count]

        # Fill rest via selection, crossover, mutation
        while len(new_pop) < config.pop_size:
            p1 = tournament_select(population, target, config.tournament_k)
            p2 = tournament_select(population, target, config.tournament_k)
            child = uniform_crossover(p1, p2)
            child = mutate(child, config.mutation_rate)
            new_pop.append(child)

        population = new_pop

    # If we reach here, no perfect convergence within max_gens
    # Return best-so-far
    fits = [fitness(ind, target) for ind in population]
    best_idx = max(range(config.pop_size), key=lambda i: fits[i])
    return population[best_idx], config.max_gens, history


def main():
    parser = argparse.ArgumentParser(description="GA to evolve a target phrase.")
    parser.add_argument("--target", type=str, default="to be or not to be",
                        help="Target phrase to evolve.")
    parser.add_argument("--pop", type=int, default=300, help="Population size.")
    parser.add_argument("--mut", type=float, default=2.0, help="Mutation rate in percent (0-100).")
    parser.add_argument("--k", type=int, default=3, help="Tournament size (selection pressure).")
    parser.add_argument("--elitism", type:int, default=2, help="Number of elites to carry over.")
    parser.add_argument("--max-gens", type=int, default=2000, help="Maximum generations.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--log", type=str, default=None, help="Optional CSV path to log progress.")
    args = parser.parse_args()

    cfg = GAConfig(
        target=args.target,
        pop_size=args.pop,
        mutation_rate=max(0.0, min(1.0, args.mut / 100.0)),
        tournament_k=max(2, args.k),
        elitism=max(0, args.elitism),
        max_gens=max(1, args.max_gens),
        seed=args.seed,
        log_path=args.log,
        verbose=True,
    )

    best, gen, history = evolve(cfg)

    if cfg.log_path:
        import csv
        with open(cfg.log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["generation", "best_fitness", "avg_fitness", "best_string"])
            for g, bf, af, bs in history:
                w.writerow([g, bf, f"{af:.6f}", bs])

    print("\nResult:")
    print(f"- Target   : '{cfg.target}'")
    print(f"- Best     : '{best}'")
    print(f"- Converged: {best == cfg.target}")
    print(f"- Generations used: {gen}")


if __name__ == "__main__":
    main()
