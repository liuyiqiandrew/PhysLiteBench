import numpy as np
from scipy.special import ndtr


def windows(q):
    length=q["length"]
    (u,k),(v,h)=q["walls"]
    a=length-q["sizes"][1]
    b=q["sizes"][0]
    rows=[]
    for x,side,lo,hi in q["gates"]:
        low=k*x-u
        high=v-h*(length-x)
        if side=="forward":
            lo,hi=low+lo,low+hi
        else:
            lo,hi=high-hi,high-lo
        rows.append((x,lo,hi))
    return a,b,rows


class Meter:
    def fit(self,rows):
        f=np.array([2*r["input"]["x"]*(1-r["input"]["x"]/r["input"]["length"]) for r in rows])
        self.diffusivity=float(np.dot(f,[r["value"] for r in rows])/np.dot(f,f))
        return self

    def measure(self,q):
        a,b,gates=windows(q)
        (u,k),(v,h)=q["walls"]
        x,lo,hi=gates[1]
        y0=v-h*(q["length"]-a)
        y1=k*b-u
        mean=((b-x)*y0+(x-a)*y1)/(b-a)
        sd=np.sqrt(2*self.diffusivity*(b-x)*(x-a)/(b-a))
        return ndtr((hi-mean)/sd)-ndtr((lo-mean)/sd)

    def predict(self,inputs):
        return np.array([self.diffusivity*2*q["x"]*(1-q["x"]/q["length"])
                         if "x" in q else self.measure(q) for q in inputs],dtype=float)
