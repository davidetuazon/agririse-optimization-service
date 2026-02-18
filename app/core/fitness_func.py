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


def evaluate_individual(individual, canal_input):
    """
    Fitness with 2 objectives:
      1. Normalized deficit (minimize)
      2. Fairness (maximize, returned as -fairness for NSGA-II)
    """
    deficits = []
    fractions_fulfilled = []

    total_allocated = 0
    for alloc, canal in zip(individual, canal_input):
        # water after applying loss
        effective_water = alloc * (1 - canal['lossFactorPercentage'])

        # deficit, waste, fractions_fulfilled computations
        deficit = max(0, canal['netWaterDemandM3'] - effective_water)
        fraction = effective_water / canal['netWaterDemandM3']

        deficits.append(deficit)
        fractions_fulfilled.append(fraction)
        
        total_allocated += alloc

    # System-level metrics
    total_net_demand = sum(canal['netWaterDemandM3'] for canal in canal_input)
    # Normalization
    normalized_deficit = sum(deficits) / total_net_demand
    # Fairness (Jain's index)
    fairness = jain_index(fractions_fulfilled)

    # print for debugging
    # print(f"Allocation sum: {total_allocated:.2f}, Surplus: {surplus:.2f}, Fairness: {fairness:.4f}")
    # print(f"Normalized: deficit={normalized_deficit:.4f}, waste={normalized_waste:.4f}, surplus={normalized_surplus:.4f}, -fairness={-fairness:.4f}")
    # print("-" * 50)

    return normalized_deficit, float(fairness)