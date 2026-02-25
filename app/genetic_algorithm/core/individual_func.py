import numpy as np
from deap import creator, base
import random
from scipy.stats import dirichlet

random.seed()
np.random.seed() 

# Dirichlet distribution
def dirichlet_generate_individual(canal_input, total_water_available, alpha=None, min_fraction=0.05):
    """
    Generates a single individual that represents water allocation to canals
    
    Parameters:
        canal_input: list of canals

        total_water_available: float, total seasonal water supply (m³)

        alpha: weight for adjusting prioritization in allocations
            **increasing alpha makes allocation more "balanced"

    Returns:
        - creator.Individual: list of allocations where sum <= total_water_available
    """

    n = len(canal_input)

    if alpha is None:
        alpha = np.ones(n) * 2.0  # higher alpha -> more balanced allocations
    # base proportions
    proportions = dirichlet.rvs(alpha, size=1)[0]
    allocations = proportions * total_water_available

    min_allocs = [min_fraction * canal['netWaterDemandM3'] for canal in canal_input]
    # constraint to ensure min allocation per canal
    allocations = np.maximum(allocations, min_allocs)
    # noramlization to available water supply
    allocations = allocations / allocations.sum() * total_water_available
    # optional stochastic perturbation
    rand_f = np.random.uniform(0.95, 1.05, n) # within +/- 5% randomly
    allocations = allocations * rand_f
    # final normalization
    allocations = allocations / allocations.sum() * total_water_available

    return creator.Individual(allocations.tolist())