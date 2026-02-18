import numpy as np
from deap import base, tools, creator, algorithms
import time

from app.core.individual_func import dirichlet_generate_individual
from app.core.fitness_func import evaluate_individual
from app.core.crossover_func import crossover
from app.core.mutate_func import mutate

from app.core.config import AGGREGATED_SOIL_WATER_RETENTION_MM

# Only if previously defined
if "Individual" in dir(creator):
    del creator.Individual
if "FitnessMin" in dir(creator):
    del creator.FitnessMin

# Then create 4-objective fitness & individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

def run_ga(
    canal_input: list[dict],
    total_water_available: float,
    # GA Paremeters
    pop_size=100,
    ngen=2000,
    cxpb=0.6,
    mutpb=0.4,
    stall_generations=250,
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
    )

    # mutate and crossover
    toolbox.register(
        'mutate',
        mutate,
        canal_input=canal_input,
        sigma=0.05 * total_water_available,
        indpb=0.1,
        total_water_available=total_water_available
    )
    toolbox.register(
        'mate',
        crossover,
        canal_input=canal_input,
        eta=5,
        total_water_available=total_water_available
    )

    # selection
    toolbox.register('select', tools.selNSGA2)

    # statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min, axis=0)
    stats.register('avg', np.mean, axis=0)

    population = toolbox.population(n=pop_size)