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
      2. Fairness (maximize)
    """
    deficits = []
    fractions_fulfilled = []

    for alloc, canal in zip(individual, canal_input):
        # water after applying loss
        effective_water = alloc * (1 - canal['lossFactorPercentage'])

        # deficit, waste, fractions_fulfilled computations
        deficit = max(0, canal['netWaterDemandM3'] - effective_water)
        fraction = effective_water / canal['netWaterDemandM3']

        deficits.append(deficit)
        fractions_fulfilled.append(fraction)

    # System-level metrics
    total_net_demand = sum(canal['netWaterDemandM3'] for canal in canal_input)
    # Normalization
    normalized_deficit = sum(deficits) / total_net_demand
    # Fairness (Jain's index)
    fairness = jain_index(fractions_fulfilled)

    return normalized_deficit, float(fairness)