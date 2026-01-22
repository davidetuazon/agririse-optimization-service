import numpy as np
from deap import tools
import random

random.seed()
np.random.seed()

def crossover(ind1, ind2, eta=15, total_water_available=None):
    tools.cxSimulatedBinaryBounded(ind1, ind2, eta=eta, low=0, up=total_water_available)
    # normalize
    for ind in (ind1, ind2):
        s = sum(ind)
        if s > 0:
            ind[:] = [x / s * total_water_available for x in ind]
    return ind1, ind2
