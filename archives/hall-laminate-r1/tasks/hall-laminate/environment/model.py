import numpy as np

CHARGE_DENSITY = 1e4


def conductivity(mobility,field,sign):
    beta = mobility*field
    return CHARGE_DENSITY*mobility/(1+beta*beta)*np.array([[1.,sign*beta],[-sign*beta,1.]])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            first = conductivity(self.mobility,e['field'],1)
            second = conductivity(2*self.mobility,e['field'],-1)
            effective = e['fraction']*first+(1-e['fraction'])*second
            electric = np.linalg.solve(effective,np.array([1.,0.]))
            out.append(electric[0 if e['observable']=='longitudinal_resistivity' else 1])
        return np.array(out)
