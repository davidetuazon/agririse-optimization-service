import numpy as np
from deap import base, tools, creator, algorithms
import time

from app.core.individual_func import dirichlet_generate_individual
from app.core.fitness_func import evaluate_individual
from app.core.crossover_func import crossover
from app.core.mutate_func import mutate

from app.core.config import AGGREGATED_SOIL_WATER_RETENTION_MM

if not hasattr(creator, "FitnessMulti"):
    creator.create('FitnessMulti', base.Fitness, weights=(-1.0, +1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)

def run_ga(
    canal_input: list[dict],
    total_water_available: float,
    # GA Paremeters
    pop_size=100,
    ngen=2000,
    cxpb=0.6,
    mutpb=0.4,
    stall_generations=200,
    min_improvement=1e-4,
):
    start_time = time.time()

    toolbox = base.Toolbox()
    # Individual Function
    toolbox.register(
        'individual',
        dirichlet_generate_individual,
        canal_input=canal_input,
        total_water_available=total_water_available
    )
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    # Fitness Function
    toolbox.register(
        'evaluate',
        evaluate_individual,
        canal_input=canal_input,
        aggregated_soil_retention_mm=AGGREGATED_SOIL_WATER_RETENTION_MM,
        alpha=1.0
    )

    # mutate and crossover
    toolbox.register('mutate', mutate, sigma=1e6, indpb=0.1, total_water_available=total_water_available)
    toolbox.register('mate', crossover, eta=15, total_water_available=total_water_available)

    # selection
    toolbox.register('select', tools.selNSGA2)

    # statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min, axis=0)
    stats.register('avg', np.mean, axis=0)

    population = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    best_total_deficit = float("inf")
    stall_counter = 0

    for gen in range(ngen):
        # Run one generation using eaMuPlusLambda
        population, logbook = algorithms.eaMuPlusLambda(
            population, toolbox,
            mu=pop_size, lambda_=pop_size,
            cxpb=cxpb, mutpb=mutpb,
            ngen=1,
            stats=stats,
            halloffame=hof,
            verbose=False
        )
        
        current_best = min(ind.fitness.values[0] for ind in population)
        if best_total_deficit - current_best > min_improvement:
            best_total_deficit = current_best
            stall_counter = 0
        else:
            stall_counter += 1

        if stall_counter >= stall_generations:
            print(f"Stopping early at generation {gen+1} due to stall")
            break

        # Optional: print generation summary
        record = logbook[-1] if logbook else None
        if record:
            print(f"Gen {gen+1}: min={record['min']}, avg={record['avg']}")

    # --- Results ---
    print("\nPareto-optimal solutions (hall of fame):")
    for i, ind in enumerate(hof):
        print(f"Solution {i+1}: {np.array(ind)}")
        print("Total allocation:", sum(ind))
        print("Fitness (deficit+waste ↓, fairness ↑):", ind.fitness.values)
        print()

    end_time = time.time()
    print(f"GA run completed in {end_time - start_time:.2f} seconds.")

    return {
        "pareto_front": [
            {
                "allocation": list(ind),
                "fitness": ind.fitness.values,
                "total_allocated": float(sum(ind))
            }
            for ind in hof
        ],
        "generations_run": gen + 1,
        "runtime_seconds": end_time - start_time
    }