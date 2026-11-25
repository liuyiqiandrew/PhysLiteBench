from functools import lru_cache
import numpy as np
from scipy.optimize import brentq, least_squares


@lru_cache(maxsize=8192)
def binary(partition, gel_volume, bath_volume, charge, amount, valence):
    anion = valence*amount-charge*gel_volume
    def state(potential):
        factors = partition*np.exp(np.array([valence, -1.])*potential)
        bath = np.array([amount, anion])/(bath_volume+gel_volume*factors)
        return factors*bath, bath
    def neutral(potential):
        gel, bath = state(potential)
        return valence*gel[0]-gel[1]-charge
    potential = brentq(neutral, -40., 40., xtol=1e-13)
    return state(potential)


class Model:
    def __init__(self):
        self.partition = None

    def fit(self, records):
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        experiments=[r['input'] for r in records]
        def residual(parameter):
            self.partition=float(parameter[0])
            return (self.predict(experiments)-values)/sigma
        result=least_squares(residual, [.8], bounds=(.2, 2.),
                             ftol=1e-12, xtol=1e-12, gtol=1e-12)
        self.partition=float(result.x[0])
        return self

    def predict(self, experiments):
        result=[]
        for e in experiments:
            amounts=np.array([e['amount_a'], e['amount_b']])
            valences=np.array([1., 2.])
            weights=amounts*valences/(amounts@valences)
            concentration=np.zeros((2, 3))
            for s in range(2):
                if amounts[s] == 0:
                    continue
                gel, bath=binary(self.partition, e['gel_volume'], e['bath_volume'],
                                 e['fixed_charge']*weights[s], amounts[s], valences[s])
                concentration[:, s]=[gel[0], bath[0]]
                concentration[:, 2]+=[gel[1], bath[1]]
            result.append(concentration[0 if e['compartment']=='gel' else 1, e['species']])
        return np.array(result)

