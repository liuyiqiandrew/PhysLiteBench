"""Ramsey visibility calibration and a Gaussian-field approximation."""
import numpy as np


class Readout:
    def __init__(self):
        self.visibility = .8

    def fit(self, records):
        x = np.cos([r["input"]["phase"] for r in records])
        y = [r["value"] for r in records]
        self.visibility = float(np.clip(x @ y / (x @ x), .65, .95))
        return self

    def predict(self, inputs):
        return self.visibility*np.array([self.fringe(x) for x in inputs], dtype=float)


def gaussian(q):
    if q.get("blank"):
        return np.cos(q["phase"])
    a,b=q["gate"];u,v=q["mix"];meter=np.array([[1.,u],[v,1.]])
    times=np.array([k[0] for k in q["kicks"]]);areas=np.array([k[1:] for k in q["kicks"]])
    den=0.;num=0j
    for domain in q["domains"]:
        cov=np.zeros((2,2));cross=np.zeros(2);variance=0.
        for (lam,sigma,gamma),vec in zip(domain["pockets"],[(1.,0.),(0.,1.),(1.,domain["ratio"])]):
            direction=meter@vec;h=areas@vec;strength=lam*sigma**2
            cov+=strength*np.outer(direction,direction)
            cross+=strength*direction*(np.exp(-2*gamma*times)@h)
            variance+=strength*(h@np.exp(-2*gamma*abs(times[:,None]-times))@h)
        beta=np.linalg.solve(cov,cross);width=variance-cross@beta
        for target in {(a,b),(a,-b)} if q["unsigned_b"] else {(a,b)}:
            prob=domain["weight"]*np.exp(-.5*np.dot(target,np.linalg.solve(cov,target)))/np.sqrt(np.linalg.det(cov))
            den+=prob;num+=prob*np.exp(-.5*width-1j*np.dot(beta,target))
    return (np.exp(-1j*q["phase"])*num).real/den
