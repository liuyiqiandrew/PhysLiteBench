import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, roots_legendre

BOUNDS=(.68,1.33)
X,W=roots_legendre(54)
X=(X+1)/2
W=W/2
SQRT_2PI=np.sqrt(2*np.pi)


def gaussian(x,m,s):
    return np.exp(-((x-m)/s)**2/2)/(s*SQRT_2PI)


def band(interval):
    a,b=interval
    return -np.inf if a is None else a, np.inf if b is None else b


def rare(e,i,variances,means,marked,rectangular):
    a,b=e["settings"][i]["aperture"]
    y_bounds,z_bounds=map(band,e["site_bands"][i]) if marked else [(0.,np.inf),(-np.inf,np.inf)]
    y0,y1=y_bounds
    z0,z1=z_bounds
    std_y,std_z=np.sqrt(variances)
    my,mz=means
    lo=max(0.,y0,a-z1)
    hi=min(y1,b-z0,my+13*std_y)
    if rectangular:
        z0=max(0.,z0)
        hi=min(hi,b-z0)
    if lo>=hi or z0>=z1:
        return 0.
    points=sorted(set([lo,hi]+[q for q in (a-z0,b-z1) if lo<q<hi]))
    value=0.
    for left,right in zip(points,points[1:]):
        y=left+(right-left)*X
        zl=np.maximum(z0,a-y)
        zh=np.minimum(z1,b-y)
        prob=np.maximum(0.,ndtr((zh-mz)/std_z)-ndtr((zl-mz)/std_z))
        value+=(right-left)*(W @ (gaussian(y,my,std_y)*prob))
    return value


def ordinary(e,i,variance,mean,rectangular):
    lo,hi=e["settings"][i]["aperture"]
    v,u,w=variance
    m,n,p=mean
    vn=v+u
    total=vn+w
    s=lo+(hi-lo)*X
    a=m+n+vn/total*(s-m-n-p)
    b=np.sqrt(vn*w/total)
    upper=s if rectangular else np.maximum(0.,a+13*b)
    z=upper[:,None]*X
    density=gaussian(z,a[:,None],b)
    x_center=m+v/vn*(z-m-n)
    x_std=np.sqrt(v*u/vn)
    x_prob=ndtr(x_center/x_std)
    if rectangular:
        x_prob-=ndtr((x_center-z)/x_std)
    inside=upper*((density*x_prob)@W)
    return (hi-lo)*(W @ (gaussian(s,m+n+p,np.sqrt(total))*inside))


def answer(e,g,rectangular=False):
    q=np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*g*g*q.sum()
    c=np.sqrt(2*e["ratio"])/g
    v=g*g*q
    B=[]; S=[]; K=[]
    for i,setting in enumerate(e["settings"]):
        powers=c*np.asarray(setting["fields"])-setting["window"]
        center=powers*v
        # Normalize both masses for this setting by the common tilt prefactor
        # within the descendants.
        factor=np.exp(powers[0]*center[0]/2)
        B.append(factor*ordinary(e,i,v,center,rectangular))
        S.append(rare(e,i,v[1:],center[1:],False,rectangular))
        K.append(rare(e,i,v[1:],center[1:],True,rectangular))
    lo,hi=e["root_window"]
    u0,u1=np.exp(-c*hi),np.exp(-c*lo)
    u=u0+(u1-u0)*X
    integrand=K[0]*K[1]/((B[0]+u*S[0])*(B[1]+u*S[1]))
    return float((u1-u0)*(W @ integrand)/(c*SQRT_2PI*np.sqrt(v[0])))


def _response(experiments,gain):
    return np.array([answer(e,gain) for e in experiments])


class Model:
    def fit(self,records):
        data=[row["input"] for row in records]
        y=np.array([row["value"] for row in records])
        sigma=np.array([row["sigma"] for row in records])

        def chi2(g):
            resid=(_response(data,g)-y)/sigma
            return resid@resid

        result=minimize_scalar(chi2,bounds=BOUNDS,method="bounded",options={"xatol":1e-10})
        if not result.success:
            raise RuntimeError(result.message)
        self.gain=float(result.x)
        return self

    def predict(self,experiments):
        return _response(experiments,self.gain)


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
