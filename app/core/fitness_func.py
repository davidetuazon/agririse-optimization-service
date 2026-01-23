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


def evaluate_individual(individual, canal_input, aggregated_soil_retention_mm, alpha_under=0.5, alpha_over=0.8, max_penalty=5):
    """
    Fitness with four objectives:
      1. Normalized deficit (minimize)
      2. Normalized waste (minimize)
      3. Normalized surplus (minimize)
      4. Fairness (maximize, returned as -fairness for NSGA-II)
    """
    deficits = []
    wastes = []
    fractions_fulfilled = []

    total_allocated = 0
    for alloc, canal in zip(individual, canal_input):
        # water after applying loss
        effective_water = alloc * (1 - canal['lossFactorPercentage'])

        # apply soil retention
        soil_retention_fraction = 1 - (aggregated_soil_retention_mm / 1000) # ~0.95
        useable_water = effective_water * soil_retention_fraction

        # quadratic penalty factor
        deviation = useable_water - canal['netWaterDemandM3']
        # under allocation
        if deviation < 0:
            penalty = min(1 + alpha_under * (deviation / canal['netWaterDemandM3']) ** 2, max_penalty)
        # over allocation
        else:
            penalty = min(1 + alpha_over * (deviation / canal['netWaterDemandM3']) ** 2, max_penalty)

        # deficit, waste, fractions_fulfilled computations
        deficit = max(0, canal['netWaterDemandM3'] - useable_water) * penalty
        waste = max(0, useable_water - canal['netWaterDemandM3']) * penalty
        fraction = min(useable_water / canal['netWaterDemandM3'], 1.0)

        deficits.append(deficit)
        wastes.append(waste)
        fractions_fulfilled.append(fraction)
        
        total_allocated += alloc

    # System-level metrics
    total_net_demand = sum(canal['netWaterDemandM3'] for canal in canal_input)
    surplus = max(0, total_allocated - total_net_demand)
    # Normalization
    normalized_deficit = sum(deficits) / total_net_demand
    normalized_waste = sum(wastes) / total_net_demand
    normalized_surplus = surplus / total_net_demand
    # Fairness (Jain's index)
    fairness = jain_index(fractions_fulfilled)

    # print for debugging
    # print(f"Allocation sum: {total_allocated:.2f}, Surplus: {surplus:.2f}, Fairness: {fairness:.4f}")
    # print(f"Normalized: deficit={normalized_deficit:.4f}, waste={normalized_waste:.4f}, surplus={normalized_surplus:.4f}, -fairness={-fairness:.4f}")
    # print("-" * 50)

    return normalized_deficit, normalized_waste, normalized_surplus, -float(fairness)