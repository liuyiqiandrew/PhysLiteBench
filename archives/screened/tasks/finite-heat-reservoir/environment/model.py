import numpy as np
from scipy.integrate import quad


class Model:
    def __init__(self):
        self.total_energy = None

    def fit(self, records):
        raise NotImplementedError

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
