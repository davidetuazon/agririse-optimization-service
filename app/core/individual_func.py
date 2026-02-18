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
        canal_input: list of canals, used only the length

        total_water_available: float, total seasonal water supply (m³)

        alpha: weight for adjusting prioritization in allocations
            **increasing alpha makes allocation more "balanced"

    Returns:
        - creator.Individual: list of allocations where sum <= total_water_available
    """

    n = len(canal_input)

    if alpha is None:
        alpha = alpha = np.ones(n) * 2.0  # higher alpha -> more balanced allocations
    # base proportions
    proportions = dirichlet.rvs(alpha, size=1)[0]
    allocations = proportions * total_water_available

    # constraint to ensure min allocation per canal
    allocations = np.maximum(allocations, min_fraction * total_water_available / n)
    allocations = allocations / allocations.sum() * total_water_available

    # noramlization to available water supply
    allocations = allocations / allocations.sum() * total_water_available

    # optional stochastic perturbation
    rand_f = np.random.uniform(0.95, 1.05, n) # within +/- 5% randomly
    allocations = allocations * rand_f

    # final normalization
    allocations = allocations / allocations.sum() * total_water_available

    return creator.Individual(allocations.tolist())


# Dirichlet-like distribution
def generate_individual(canal_input, total_water_available):
    """
    Generates a single individual that represents water allocation to canals
    
    Parameters:
        canal_input: list of canals, used only the length
        total_water_available: float, total seasonal water supply (m³)

    Returns:
        - creator.Individual: list of allocations where sum <= total_water_available
    """


    n = len(canal_input)

    proportions = np.random.rand(n)
    proportions /= proportions.sum()
    allocations = proportions * total_water_available

    # optional stochastic perturbation
    rand_f = np.random.uniform(0.95, 1.05, n) # within +/- 5% randomly
    allocations = allocations * rand_f
    allocations = np.maximum(allocations, 0)

    # normalization to available water supply
    allocations = allocations / allocations.sum() * total_water_available

    return creator.Individual(allocations.tolist())