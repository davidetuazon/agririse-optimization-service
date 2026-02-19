import numpy as np
from deap import base, tools, creator, algorithms
import time
import httpx
import os
from dotenv import load_dotenv
load_dotenv()

from app.core.formatters import sample_pareto_front, format_pareto_results
from app.core.individual_func import dirichlet_generate_individual
from app.core.fitness_func import evaluate_individual
from app.core.crossover_func import crossover
from app.core.mutate_func import mutate

# Only if previously defined
if "Individual" in dir(creator):
    del creator.Individual
if "FitnessMin" in dir(creator):
    del creator.FitnessMin

# Then create 2-objective fitness & individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

def run_ga(
    canal_input: list[dict],
    total_water_available: float,
    run_id: str,
    # GA Paremeters
    pop_size=100,
    ngen=500,
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
    hof = tools.ParetoFront()
    
    population, logbook = algorithms.eaMuPlusLambda(
        population, toolbox,
        mu=pop_size,
        lambda_=pop_size,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    end_time = time.time()
    execution_time = end_time - start_time

    sampled = sample_pareto_front(hof, n_samples=10)
    formatted = format_pareto_results(sampled, canal_input)

    callback_url = os.getenv('BACKEND_CALLBACK_URL')

    try:
        response = httpx.post(callback_url, json={
            'runId': run_id,
            'status': 'completed',
            'executionTimeSeconds': round(execution_time, 2),
            'paretoSolutions': formatted,
        })
        print(f"Callback status: {response.status_code}")
    except Exception as e:
        print(f'Callback failed: {e}')