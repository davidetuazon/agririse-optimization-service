import numpy as np
from deap import creator, base
import random

random.seed()
np.random.seed()

# Jain's Fairness index
def jain_index(fractions):
    fractions = np.array(fractions)
    numerator = np.sum(fractions) ** 2
    denominator = len(fractions) * np.sum(fractions ** 2)

    return numerator / denominator if denominator > 0 else 0

# score deficit based on area size, so deficit score scales with the area size
def evaluate_individual(individual, canal_input):
    """
    Fitness with 2 objectives:
      1. Weighted normalized deficit (minimize)
      2. Fairness (maximize)
    """
    deficits = []
    fractions_fulfilled = []

    # System-level metrics
    total_service_area = sum(canal['tbsByDamHa'] for canal in canal_input)

    for alloc, canal in zip(individual, canal_input):
        # water after applying loss
        effective_water = alloc * (1 - canal['lossFactorPercentage'] /100)

        # deficit, waste, fractions_fulfilled computations
        deficit = max(0, canal['netWaterDemandM3'] - effective_water)
        fraction = effective_water / canal['netWaterDemandM3']

        deficits.append(deficit)
        fractions_fulfilled.append(fraction)

    # weight each canal by their service area size
    weights = [canal['tbsByDamHa'] / total_service_area for canal in canal_input]
    # normalization
    numerator = sum(weight * deficit for weight, deficit in zip(weights, deficits))
    denominator = sum(weight * canal['netWaterDemandM3'] for weight, canal in zip(weights, canal_input))
    weighted_normalized_deficit = numerator / denominator

    # fairness (Jain's index)
    fairness = jain_index(fractions_fulfilled)

    return weighted_normalized_deficit, float(fairness)


# same as trial one but introduces a uniform penalization for allocations below threshold
def evaluate_individual_trial_one(individual, canal_input, threshold, lam):
    """
    Fitness with 2 objectives:
      1. Penalized deficit (minimize)
      2. Fairness (maximize)
    """
    deficits = []
    fractions_fulfilled = []
    penalty_terms = []

    # System-level metrics
    total_service_area = sum(canal['tbsByDamHa'] for canal in canal_input)

    for alloc, canal in zip(individual, canal_input):
        # water after applying loss
        effective_water = alloc * (1 - canal['lossFactorPercentage'] /100)

        # deficit, waste, fractions_fulfilled computations
        deficit = max(0, canal['netWaterDemandM3'] - effective_water)
        fraction = effective_water / canal['netWaterDemandM3']

        deficits.append(deficit)
        fractions_fulfilled.append(fraction)
        
        # penalty
        # proportional to how far below threshold
        shortfall = max(0, threshold - fraction)
        penalty_terms.append(shortfall * (1 / len(canal_input)))
    
    # weight each canal by their service area size
    weights = [canal['tbsByDamHa'] / total_service_area for canal in canal_input]
    # Normalization
    numerator = sum(weight * deficit for weight, deficit in zip(weights, deficits))
    denominator = sum(weight * canal['netWaterDemandM3'] for weight, canal in zip(weights, canal_input))
    weighted_normalized_deficit = numerator / denominator

    # penalized deficit
    total_penalty = sum(penalty_terms)
    penalized_deficit = weighted_normalized_deficit + lam * total_penalty

    # Fairness (Jain's index)
    fairness = jain_index(fractions_fulfilled)

    return penalized_deficit, float(fairness)