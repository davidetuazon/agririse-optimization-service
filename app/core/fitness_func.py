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


def evaluate_individual(individual, canal_input, aggregated_soil_retention_mm, alpha=None):
    """
    Docstring for evaluate_individual
    
    :param individual: Description
    :param canal_input: Description
    :param aggregated_soil_retention_mm: Description
    :param alpha: Description
    """

    deficits = []
    wastes = []
    fractions_fulfilled = []

    for alloc, canal in zip(individual, canal_input):
        # water after applying loss
        effective_water = alloc * (1 - canal['lossFactorPercentage'])
        
        # apply soil retention
        useable_water = effective_water * (aggregated_soil_retention_mm / 1000)

        # deficit and waste computation
        deficit = max(0, canal['netWaterDemandM3'] - useable_water)
        waste = max(0, useable_water - canal['netWaterDemandM3'])

        deficits.append(deficit)
        wastes.append(waste)

        fraction = min(useable_water / canal['netWaterDemandM3'], 1.0)
        fractions_fulfilled.append(fraction)

    # apply jain's fairness index
    fairness = jain_index(fractions_fulfilled)

    # apply normalization
    total_net_demand = sum(canal['netWaterDemandM3'] for canal in canal_input)
    normalized_deficit = sum(deficits) / total_net_demand
    normalized_waste = sum(wastes) / total_net_demand

    ALPHA = alpha
    fitness_value = -(normalized_deficit + normalized_waste) + ALPHA * fairness

    return (normalized_deficit + normalized_waste, -fairness)