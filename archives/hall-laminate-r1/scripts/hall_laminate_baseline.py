import numpy as np

CHARGE_DENSITY = 1e4


def conductivity(mobility,field,sign):
    beta = mobility*field
    return CHARGE_DENSITY*mobility/(1+beta*beta)*np.array([[1.,sign*beta],[-sign*beta,1.]])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        factor = np.array([1. if r['input']['fraction']==1. else 2. for r in records])
        design = 1/(CHARGE_DENSITY*factor)
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        inverse = np.sum(design*values/sigma**2)/np.sum(design**2/sigma**2)
        self.mobility = float(np.clip(1/inverse,.2,1.5))
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            first = conductivity(self.mobility,e['field'],1)
            second = conductivity(2*self.mobility,e['field'],-1)
            effective = e['fraction']*first+(1-e['fraction'])*second
            electric = np.linalg.solve(effective,np.array([1.,0.]))
            out.append(electric[0 if e['observable']=='longitudinal_resistivity' else 1])
        return np.array(out)
