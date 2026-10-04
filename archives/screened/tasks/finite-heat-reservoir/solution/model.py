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
        for e in experiments:
            k,b = e['quadratic'],e['quartic']
            if b == 0 and e['observable']=='second_moment':
                out.append(self.total_energy/(3*k))
                continue
            qmax = np.sqrt(4*self.total_energy/(k+np.sqrt(k*k+4*b*self.total_energy)))
            def weight(q):
                remaining = self.total_energy-.5*k*q*q-.25*b*q**4
                return max(remaining,0.)**1.5
            norm = quad(weight,0,qmax,epsabs=1e-11,epsrel=1e-11)[0]
            second = quad(lambda q:q*q*weight(q),0,qmax,epsabs=1e-11,epsrel=1e-11)[0]/norm
            fourth = quad(lambda q:q**4*weight(q),0,qmax,epsabs=1e-11,epsrel=1e-11)[0]/norm
            out.append(dict(second_moment=second,fourth_moment=fourth,kurtosis=fourth/second**2)[e['observable']])
        return np.array(out)
