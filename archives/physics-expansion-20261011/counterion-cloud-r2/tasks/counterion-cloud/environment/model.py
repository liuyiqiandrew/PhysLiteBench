import numpy as np


def coefficient(xi,fraction):
    number_per_charge = 1-fraction+fraction/3
    effective_xi = xi/number_per_charge
    return 1-effective_xi/2 if effective_xi<=1 else 1/(2*effective_xi)


def predict_at(experiments,line_density):
    return np.asarray([coefficient(e['bjerrum_length']*e['charge_multiplier']*
                                   line_density,e['trivalent_charge_fraction'])
                       for e in experiments])


class Model:
    def __init__(self):
        self.line_density = None

    def fit(self,records):
        raise NotImplementedError('Fit the rod line-charge density.')

    def predict(self,experiments):
        return predict_at(experiments,self.line_density)
