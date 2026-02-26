import numpy as np

def sample_pareto_front(hof, n_samples=20):
    solutions = list(hof)

    if len(solutions) <= n_samples:
        return solutions
    
    solutions.sort(key=lambda ind: ind.fitness.values[0])
    indices =np.linspace(0, len(solutions) - 1, n_samples, dtype=int)

    return [solutions[i] for i in indices]


def format_pareto_results(sampled_solutions, canal_input_list):
    results = []
    for ind in sampled_solutions:
        allocation_vector = []
        for i, canal in enumerate(canal_input_list):
            alloc = {
                'mainLateralId': canal['mainLateralId'],
                'coverage': canal['coverage'],
                'areaHa': canal['tbsByDamHa'],
                'allocatedWaterM3': round(float(ind[i]), 2),
                'effectiveWaterM3': round(float(ind[i]) * (1 - canal['lossFactorPercentage'] / 100), 2),
                'netWaterDemandM3': canal['netWaterDemandM3'],
            }
            allocation_vector.append(alloc)
        
        solution = {
            'allocationVector': allocation_vector,
            'objectiveValues': {
                'deficit': {
                    'value': round(ind.fitness.values[0], 4),
                    'unit': 'ratio',
                    'direction': 'minimize'
                },
                'fairness': {
                    'value': round(ind.fitness.values[1], 4),
                    'unit': 'ratio',
                    'direction': 'maximize'
                },
            }
        }
        results.append(solution)

    return results