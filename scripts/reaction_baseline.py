import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize_scalar

class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        def objective(log_d):
            self.diffusivity = np.exp(log_d)
            predicted = np.array([self._binary(data["t"],data["x"],initial) for initial in data["initial"]])
            return np.sum(((predicted-data["concentration"])/data["sigma"])**2)
        result = minimize_scalar(objective,bounds=(np.log(1e-11),np.log(1e-7)),method="bounded",options={"xatol":1e-11})
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self,t,x,initial):
        t,x,initial=np.asarray(t),np.asarray(x),np.asarray(initial)
        count=len(x)-1
        spatial_rate=4/(x[1]-x[0])**2*np.sin(np.arange(count+1)*np.pi/(2*count))**2
        coefficients=dct(initial,type=1,axis=0)
        rates=spatial_rate[:,None]*self.diffusivity*np.array([4/3,8/3])[None,:]
        return idct(np.exp(-t[:,None,None]*rates[None,:,:])*coefficients[None,:,:],type=1,axis=1)

    def predict(self,t,x,initial):
        return self._binary(t,x,initial)
