import numpy as np


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        times = np.array([r['input']['time'] for r in records])
        base = np.array([r['input']['initial_width']**2 for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        design = 2*times
        self.diffusivity = float(np.clip(np.sum(design*(values-base)/sigma**2)/np.sum(design**2/sigma**2),.005,.06))
        return self

    def predict(self, experiments):
        from scipy.special import jn_zeros
        roots = jn_zeros(1,160)
        out = []
        for e in experiments:
            t,U,a = e['time'],e['mean_flow'],e['radius']
            rates = self.diffusivity*roots**2/a**2
            z = rates*t
            integral = np.where(z<1e-4,z*z*(.5-z/6+z*z/24),z+np.expm1(-z))/rates**2
            shear = 2*np.sum(64*U*U/roots**4*integral)
            variance = e['initial_width']**2+2*self.diffusivity*t+shear
            out.append(U*t if e['observable']=='mean_position' else variance)
        return np.array(out)
