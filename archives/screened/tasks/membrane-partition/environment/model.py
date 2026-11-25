from functools import lru_cache
import numpy as np
from scipy.optimize import brentq, least_squares


@lru_cache(maxsize=8192)
def equilibrium(partition, gel_volume, bath_volume, charge, amount_a, amount_b):
    cations=amount_a+amount_b
    mean_charge=(amount_a+2*amount_b)/cations
    amounts=np.array([cations, amount_a+2*amount_b-charge*gel_volume])
    valences=np.array([mean_charge, -1.])
    def state(potential):
        factors=partition*np.exp(valences*potential)
        bath=amounts/(bath_volume+gel_volume*factors)
        return factors*bath, bath
    potential=brentq(lambda p: state(p)[0]@valences-charge, -40., 40., xtol=1e-13)
    pools=np.array(state(potential))
    concentration=np.column_stack((pools[:,0]*amount_a/cations,
                                   pools[:,0]*amount_b/cations,pools[:,1]))
    return concentration


class Model:
    def __init__(self):
        self.partition=None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out=[]
        for e in experiments:
            concentration=equilibrium(self.partition,e['gel_volume'],e['bath_volume'],
                                      e['fixed_charge'],e['amount_a'],e['amount_b'])
            out.append(concentration[0 if e['compartment']=='gel' else 1,e['species']])
        return np.array(out)
