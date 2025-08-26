# shakespeare_ga.py
# Algoritmo Genético para evolucionar: "to be or not to be"
# Individuo: cadena del mismo largo que la frase objetivo

import random
import string
import argparse

# --- Configuración base ---
TARGET = "to be or not to be"
ALPHABET = string.ascii_lowercase + " "  # letras minúsculas + espacio


# --- Functions ---
def compute_fitness(genes: str, target: str = TARGET) -> int:
    """
    Fitness: número de coincidencias exactas por posición con la frase objetivo.
    Máximo = len(target). Fitness perfecto cuando coincide en todas las posiciones.
    """
    return sum(1 for g, t in zip(genes, target) if g == t)


def random_individual(length: int) -> str:
    """Crea un individuo aleatorio (cadena) del alfabeto definido."""
    return "".join(random.choice(ALPHABET) for _ in range(length))


def tournament_selection(population, k: int = 3):
    """
    Selección por torneo:
    - Elige k individuos al azar y devuelve el de mejor fitness.
    - population: lista de dicts con keys 'genes' y 'fitness'
    """
    k = min(k, len(population))
    contenders = random.sample(population, k)
    return max(contenders, key=lambda ind: ind["fitness"])


def uniform_crossover(p1: str, p2: str) -> str:
    """
    Cruce uniforme:
    - Para cada posición, toma el gen de p1 o p2 con prob. 0.5.
    """
    return "".join((c1 if random.random() < 0.5 else c2) for c1, c2 in zip(p1, p2))


def mutate(genes: str, mutation_rate: float = 0.01) -> str:
    """
    Mutación:
    - Recorre cada posición y, con prob. 'mutation_rate', la reemplaza por un gen aleatorio.
    """
    gene_list = list(genes)
    for i in range(len(gene_list)):
        if random.random() < mutation_rate:
            gene_list[i] = random.choice(ALPHABET)
    return "".join(gene_list)


# --- GA principal ---
def genetic_algorithm(
    target: str = TARGET,
    pop_size: int = 200,
    max_generations: int = 10_000000000,
    tournament_k: int = 3,
    mutation_rate: float = 0.3,
    elitism: int = 2,
    seed: int | None = None,
    verbose: bool = True,
    log_every: int = 100,
):
    if seed is not None:
        random.seed(seed)

    L = len(target)
    # Población inicial
    population = [{"genes": random_individual(L)} for _ in range(pop_size)]
    for ind in population:
        ind["fitness"] = compute_fitness(ind["genes"], target)

    best = max(population, key=lambda ind: ind["fitness"])

    for gen in range(1, max_generations + 1):
        # Ordenar por fitness (descendente) y actualizar mejor
        population.sort(key=lambda ind: ind["fitness"], reverse=True)
        if population[0]["fitness"] > best["fitness"]:
            best = population[0]

        # Criterio de parada: fitness perfecto
        if best["fitness"] == L:
            if verbose:
                print(f"Generación {gen}: ¡objetivo alcanzado! -> '{best['genes']}'")
            return best, gen

        # Nueva generación con elitismo
        new_pop = population[:elitism]

        # Rellenar con descendencia
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, k=tournament_k)
            p2 = tournament_selection(population, k=tournament_k)
            child_genes = uniform_crossover(p1["genes"], p2["genes"])
            child_genes = mutate(child_genes, mutation_rate=mutation_rate)
            new_pop.append({
                "genes": child_genes,
                "fitness": compute_fitness(child_genes, target)
            })

        population = new_pop

        if verbose and (gen % log_every == 0):
            pct = (best["fitness"] / L) * 100
            print(f"Gen {gen:5d} | Mejor='{best['genes']}' | Fitness={best['fitness']}/{L} ({pct:.1f}%)")

    if verbose:
        print(f"Tope de generaciones alcanzado. Mejor='{best['genes']}' con fitness {best['fitness']}/{L}")
    return best, max_generations


def main():
    parser = argparse.ArgumentParser(description="GA para evolucionar 'to be or not to be'")
    parser.add_argument("--target", type=str, default=TARGET, help="Frase objetivo")
    parser.add_argument("--pop", type=int, default=200, help="Tamaño de población")
    parser.add_argument("--gens", type=int, default=10_000, help="Tope de generaciones")
    parser.add_argument("--k", type=int, default=3, help="k del torneo (selección)")
    parser.add_argument("--mut", type=float, default=0.01, help="Tasa de mutación [0-1]")
    parser.add_argument("--elit", type=int, default=2, help="N de élites por generación")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria")
    parser.add_argument("--quiet", action="store_true", help="Silenciar progreso")
    parser.add_argument("--log-every", type=int, default=100, help="Frecuencia de logs")
    args = parser.parse_args()

    best, gen = genetic_algorithm(
        target=args.target,
        pop_size=args.pop,
        max_generations=args.gens,
        tournament_k=args.k,
        mutation_rate=args.mut,
        elitism=args.elit,
        seed=args.seed,
        verbose=not args.quiet,
        log_every=args.log_every,
    )
    if not args.quiet:
        L = len(args.target)
        print(f"\nResultado final en gen {gen}: '{best['genes']}' | Fitness {best['fitness']}/{L}")


if __name__ == "__main__":
    main()
