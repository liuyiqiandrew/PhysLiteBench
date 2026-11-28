import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

BOUNDS=(.69,1.31)
X,W=leggauss(36)
S2=np.sqrt(2*np.pi)


def grid(lo,hi):
    return (hi+lo)/2+(hi-lo)*X/2,(hi-lo)*W/2


def density(x,v):
    return np.exp(-x*x/(2*v))/(S2*np.sqrt(v))


def gaussian_band(a,v,lo,hi):
    return np.exp(.5*a*a*v)*max(0.,ndtr((a*v-lo)/np.sqrt(v))-ndtr((a*v-hi)/np.sqrt(v))) if hi>lo else 0.


def answer(e,g,cone=True):
    vf,vb,vc=g*g*np.asarray(e["variances"])
    vp=vb+vc
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(vf+vp)
    vr=vb*vc/vp
    c=np.sqrt(2*e["ratio"])/g
    A=np.array([c*np.asarray(j["fields"])-[j["window"],j["window"],0.] for j in e["settings"]])
    B,L,M=[],[],[]
    row=e["observable"]=="row_echo"
    outside=e["outside"]
    for j,(a,b,f) in enumerate(A):
        x,wx=grid(0.,max(0.,a*vf+12*np.sqrt(vf)))
        limit=-x if cone else np.zeros_like(x)
        bulk=wx@(density(x-a*vf,vf)*ndtr((b*vp-limit)/np.sqrt(vp)))
        atom=bulk*np.exp(.5*(a*a*vf+b*b*vp))
        B.append(atom*np.exp(.5*f*f*vr))
        L.append(gaussian_band(b,vp,0.,np.inf)*np.exp(.5*f*f*vr))
        if row:
            lo,hi=e["pool_bands"][j]
            M.append(gaussian_band(b,vp,max(lo,0.),hi)*gaussian_band(f,vr,*e["contrast_bands"][j]))
        if outside is not None and outside["setting"]==j:
            out=atom*gaussian_band(f,vr,*outside["contrast_band"])
    t0,t1=e["line_window"]
    q,wq=grid(np.exp(-c*t1),np.exp(-c*t0))
    if row:
        den=np.array(B)[:,None]+np.array(L)[:,None]*q
        ratio=M[0]*M[1]/(den[0]*den[1])
        if outside is not None:
            ratio*=out/den[outside["setting"]]
        return float(wq@ratio/(c*S2*np.sqrt(vf)))
    lo,hi=e["bridge_band"]
    lo=max(0.,lo)
    if lo>=hi:
        return 0.
    x,wx=grid(lo,hi)
    r,wr=grid(*e["contrast_band"])
    mix=np.exp((A[:,0]-A[:,1])[:,None,None]*x[None,:,None]+A[:,2,None,None]*r[None,None,:])
    den=np.array(B)[:,None,None,None]+mix[:,:,:,None]*q
    joint=(mix[0]*mix[1])[:,:,None]/(den[0]*den[1])
    if outside is not None:
        joint*=out/den[outside["setting"]]
    return float((wx*density(x,vf)*density(-x,vp))@((joint@wq)@(wr*density(r,vr)))/c)


def _response(experiments,gain):
    return np.array([answer(e,gain,cone=False) for e in experiments],dtype=float)


class Model:
    def fit(self, records):
        inputs = [r["input"] for r in records]
        measured = np.array([r["value"] for r in records])
        sigma = np.array([r["sigma"] for r in records])
        def chi2(parameter):
            residual = (_response(inputs, parameter) - measured) / sigma
            return residual @ residual
        result = minimize_scalar(chi2, bounds=BOUNDS, method="bounded",
                                 options={"xatol": 1e-10})
        if not result.success:
            raise RuntimeError(result.message)
        self.gain = float(result.x)
        return self

    def predict(self, experiments):
        return _response(experiments, self.gain)
