import numpy as np
from scipy.integrate import quad


class Model:
    def __init__(self):
        self.total_energy = None

    def fit(self, records):
        k = np.array([r['input']['quadratic'] for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        design = 1/(3*k)
        self.total_energy = float(np.clip(np.sum(design*values/sigma**2)/np.sum(design**2/sigma**2),.9,3.))
        return self

    def predict(self, experiments):
        out = []
        temperature = self.total_energy/3
        for e in experiments:
            k,b = e['quadratic'],e['quartic']
            if b == 0:
                second = temperature/k
                fourth = 3*second**2
            else:
                def weight(q):
                    return np.exp(-(.5*k*q*q+.25*b*q**4)/temperature)
                norm = quad(weight,0,np.inf,epsabs=1e-11,epsrel=1e-11)[0]
                second = quad(lambda q:q*q*weight(q),0,np.inf,epsabs=1e-11,epsrel=1e-11)[0]/norm
                fourth = quad(lambda q:q**4*weight(q),0,np.inf,epsabs=1e-11,epsrel=1e-11)[0]/norm
            out.append(dict(second_moment=second,fourth_moment=fourth,kurtosis=fourth/second**2)[e['observable']])
        return np.array(out)
