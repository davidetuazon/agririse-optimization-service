import numpy as np
from deap import tools
import random

random.seed()
np.random.seed()

def mutate(individual, sigma=1e6, indpb=0.1, total_water_available=None):
    tools.mutGaussian(individual, mu=0, sigma=sigma, indpb=indpb)
    # clamp negative allocations to zero
    individual[:] = np.maximum(individual, 0)
    # re-normalize
    if total_water_available is not None:
        s = sum(individual)
        if s > 0:
            individual[:] = [x / s * total_water_available for x in individual]
    return (individual,)