import numpy as np
from deap import tools
import random

random.seed()
np.random.seed()

def mutate(individual, canal_input, sigma, indpb, total_water_available):
    tools.mutGaussian(individual, mu=0, sigma=sigma, indpb=indpb)
    # clamp negative allocations to per-canal floor(minimum viable water allocation)
    min_allocs = [0.05 * canal['netWaterDemandM3'] for canal in canal_input]
    individual[:] = np.maximum(individual, min_allocs)
    # re-normalize
    s = sum(individual)
    if s > 0:
        individual[:] = [x / s * total_water_available for x in individual]
    return (individual,)