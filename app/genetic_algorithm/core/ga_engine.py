import numpy as np
import time
import httpx
import os
import asyncio
from app.genetic_algorithm.executor import ga_executor, release
from dotenv import load_dotenv
load_dotenv()

from app.genetic_algorithm.core.formatters import sample_pareto_front, format_pareto_results
from app.genetic_algorithm.core.individual_func import dirichlet_generate_individual
from app.genetic_algorithm.core.fitness_func import evaluate_individual
from app.genetic_algorithm.core.crossover_func import crossover
from app.genetic_algorithm.core.mutate_func import mutate

async def send_callback_with_retry(payload, max_retries=5):
    callback_url = os.getenv('GA_CALLBACK_URL')
    headers = {'x-api-key': os.getenv('API_SHARED_KEY')}

    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    callback_url,
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )

                if 200 <= response.status_code < 300:
                    print(f'Callback successful on attempt {attempt + 1}.')
                    return True
                
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    print(f"Permanent error {response.status_code}. Stopping retries.")
                    break
                
                print(f"Server error {response.status_code} on attempt {attempt + 1}.")

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                print(f"Network error on attempt {attempt + 1}: {e}")

            # Exponential backoff: 2s, 4s, 8s, 16s, 32s
            wait_time = 2 ** (attempt + 1)
            print(f"Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
            
        print(f"Critical: All {max_retries} attempts failed for run {payload['runId']}.")
        return False


def run_ga_sync(
    canal_input: list[dict],
    total_water_available: float,
    run_id: str,
    # GA Paremeters
    pop_size=100,
    ngen=500,
    cxpb=0.6,
    mutpb=0.4,
):
    from deap import base, tools, creator, algorithms

    # Then create 2-objective fitness & individual
    if "FitnessMin" not in dir(creator):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))
    if "Individual" not in dir(creator):
        creator.create("Individual", list, fitness=creator.FitnessMin)

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
        verbose=False
    )

    end_time = time.time()
    execution_time = end_time - start_time

    sampled = sample_pareto_front(hof, n_samples=20)
    formatted = format_pareto_results(sampled, canal_input)

    success_payload = {
        'runId': run_id,
        'status': 'completed',
        'executionTimeSeconds': round(execution_time, 2),
        'paretoSolutions': formatted,
    }

    return success_payload


async def run_ga(
    canal_input: list[dict],
    total_water_available: float,
    run_id: str,
    # GA Run Paremeters
    pop_size=100,
    ngen=500,
    cxpb=0.6,
    mutpb=0.4,
):
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            ga_executor,
            run_ga_sync,
            canal_input,
            total_water_available,
            run_id,
            pop_size,
            ngen,
            cxpb,
            mutpb
        )
        await send_callback_with_retry(result)

    except Exception as e:
        print(f'GA failed: {e}')

        # send payload with status: failed when GA fails
        failed_payload = {
            'runId': run_id,
            'status': 'failed',
            'error_details': str(e),
        }
        await send_callback_with_retry(failed_payload)

    finally:
        release()