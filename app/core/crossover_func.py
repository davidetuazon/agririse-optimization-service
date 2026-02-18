import numpy as np
from deap import tools
import random

random.seed()
np.random.seed()

def crossover(ind1, ind2, canal_input, eta, total_water_available=None):
    tools.cxSimulatedBinaryBounded(ind1, ind2, eta=eta, low=min(0.05 * canal['netWaterDemandM3'] for canal in canal_input), up=total_water_available/len(ind1))
    # normalize
    for ind in (ind1, ind2):
        s = sum(ind)
        if s > 0:
            ind[:] = [x / s * total_water_available for x in ind]
    return ind1, ind2