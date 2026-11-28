import numpy as np


class Meter:
    def fit(self,rows):
        x=np.array([self.calibration(r["input"]) for r in rows])
        self.diffusivity=float(np.dot(x,[r["value"] for r in rows])/np.dot(x,x))
        return self

    @staticmethod
    def calibration(q):
        k,v,t=q["spring"],q["kick"],q["time"]
        return 2*v*np.exp(-k*t)*(-np.expm1(-k*t))/k

    def measure(self,q):
        survival=1.
        for f in q["flashes"]:
            t,a,b=f
            mean=sum(h*v*np.exp(-k*t) for (v,k,m),h in zip(q["walls"],(a,b)))
            survival*=np.exp(-mean)
        return survival

    def predict(self,inputs):
        return np.asarray([self.diffusivity*self.calibration(q) if "walls" not in q
                           else self.measure(q) for q in inputs],dtype=float)
