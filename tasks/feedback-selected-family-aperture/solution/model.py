import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

BOUNDS=(.69,1.31)
P,W=leggauss(32)
S2=np.sqrt(2*np.pi)


def rule(lo,hi):
    return (hi+lo)/2+(hi-lo)*P/2,(hi-lo)*W/2


def answer(e,g,finite_root=True):
    vf,vl=g*g*np.array(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c=np.sqrt(2*e["ratio"])/g
    sf,sl=np.sqrt([vf,vl])
    a=np.array([c*np.array(s["fields"])-s["window"] for s in e["settings"]])
    mp,mq=a.T*np.array([[vf],[vl]])
    theta=e["aperture"]
    b,s,m=[theta[k] for k in ("bias","slope","mixing")]
    out=e["outside_band"]

    def extended(j,gate,marked=None):
        mu,mv=mp[j],mq[j]
        lo,hi=mv-12*sl,mv+12*sl
        if marked is not None:
            lo=max(lo,marked[0])
            hi=min(hi,marked[1])
        if hi<=lo:
            return 0.
        cuts=[lo,hi]
        for q in (0.,gate/m if m else 0.,gate/(m-1) if m!=1 else 0.):
            if lo<q<hi:
                cuts.append(q)
        cuts=sorted(set(cuts))
        value=0.
        for x,y in zip(cuts[:-1],cuts[1:]):
            z,w=rule(x,y)
            low=np.maximum(0.,np.maximum(-z,gate-m*z))
            kernel=np.exp(-.5*((z-mv)/sl)**2)/(S2*sl)
            value+=w@(kernel*ndtr((mu-low)/sf))
        return float(value*np.exp(.5*(a[j,0]*mu+a[j,1]*mv)))

    def tagged(j,lo,hi):
        if hi<=lo:
            return 0.
        q=mq[j]
        return np.exp(.5*a[j,1]*q)*(ndtr((q-lo)/sl)-ndtr((q-hi)/sl))

    B0=extended(0,-np.inf)
    L0=tagged(0,0.,np.inf)
    root_l,root_h=e["root_window"]
    if m==0 and b==0 and s==0 and finite_root:
        root_l=max(0.,root_l)
    yl,yh=e["first_band"]
    yl=max(0.,yl)
    if root_h<=root_l or yh<=yl:
        return 0.
    first,wt=rule(root_l,root_h)
    u=np.exp(-c*first)
    siblings=e["sibling_band"]
    cuts=[yl,yh]
    if s:
        for h in [0.,m*siblings[0],m*siblings[1]]:
            y=(h-b)/s
            if yl<y<yh:
                cuts.append(y)
    cuts=sorted(set(cuts))
    result=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        y,wy=rule(lo,hi)
        for yy,w in zip(y,wy):
            h=b+s*yy
            if not m and h>0:
                continue
            limit=max(0.,h/m) if m else 0.
            picked=tagged(1,max(limit,siblings[0]),siblings[1])
            if picked==0:
                continue
            D=extended(1,h)+u*tagged(1,limit,np.inf)
            prob=u*picked/D
            if out is not None:
                prob*=extended(1,h,out)/D
            weight=np.exp(-.5*(yy/sl)**2+a[0,1]*yy)*w/(S2*sl)
            result+=weight*(wt@(prob/(B0+u*L0)))
    return float(result/(S2*sf))


def _response(experiments,gain):
    return np.array([answer(e,gain) for e in experiments],dtype=float)


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
