import numpy as np


def coefficient(xi,fraction):
    number_per_charge = 1-fraction+fraction/3
    if xi<=1/3:
        residual = xi
        free_number = xi*number_per_charge
    elif xi*(1-fraction)<=1/3:
        residual = 1/3
        free_number = xi*(1-fraction)*2/3+1/9
    elif xi*(1-fraction)<=1:
        residual = xi*(1-fraction)
        free_number = residual
    else:
        residual = 1.
        free_number = 1.
    return (free_number-residual*residual/2)/(xi*number_per_charge)


def predict_at(experiments,line_density):
    return np.asarray([coefficient(e['bjerrum_length']*e['charge_multiplier']*
                                   line_density,e['trivalent_charge_fraction'])
                       for e in experiments])


class Model:
    def __init__(self):
        self.line_density = None

    def fit(self,records):
        design = np.asarray([r['input']['bjerrum_length']*
                             r['input']['charge_multiplier']/(2*(1-
                             2*r['input']['trivalent_charge_fraction']/3))/r['sigma']
                             for r in records])
        target = np.asarray([(1-r['value'])/r['sigma'] for r in records])
        self.line_density = float(np.clip(design@target/(design@design),.7,1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.line_density)
