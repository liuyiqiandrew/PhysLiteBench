import numpy as np
from scipy.linalg import schur, solve_continuous_lyapunov
from scipy.optimize import minimize_scalar

BOUNDS=(.57,1.36)
COUPLED=False
TURN=np.array([[0.,-1.],[1.,0.]])


def normal(stage,gain):
    s,c=np.sin(stage["orientation"]),np.cos(stage["orientation"])
    rotation=np.array([[c,-s],[s,c]])
    K=(rotation*np.array(stage["stiffness"]))@rotation.T
    A=-K+gain*stage["drive"]*TURN
    D=np.diag(stage["temperature"])
    H=np.array(stage["weights"])[:,None]*A
    S=(H+H.T)/2
    F=(H-H.T)/2
    covariance=solve_continuous_lyapunov(A,-2*D)
    return A,D,S,F,covariance,H


def asymptotic(row,h):
    A,D,S,F,C0,H=row
    shifted=A+2*h*D@F
    Q0=h*F.T@A+h*h*F.T@D@F
    ham=np.block([[shifted,2*D],[-(Q0+Q0.T),-shifted.T]])
    triangular,V,count=schur(ham,output="complex",sort=lambda x:x.real<-1e-11)
    if count!=2:
        return None
    R=np.linalg.solve(V[:2,:2].T,V[2:,:2].T).T.real
    R=(R+R.T)/2
    eig=float(np.trace(D@R))
    if COUPLED:
        transformed=shifted+2*D@R
        cov=solve_continuous_lyapunov(transformed,-2*D)
        alpha=R-h*S
    else:
        cov=C0
        alpha=-h*S
    return eig,alpha,np.linalg.inv(cov)+alpha


def read(e,gain):
    rows=[normal(stage,gain) for stage in e["stages"]]
    if e["observable"]=="mean_heat":
        A,D,_,_,C,H=rows[0]
        return float(np.trace(H.T@A@C+H@D))
    start=np.linalg.inv(rows[0][4])
    fractions=np.array([p["fraction"] for p in e["stages"]])
    target=e["target"]
    def values(h):
        results=[asymptotic(row,h) for row in rows]
        if any(a is None for a in results):
            return None
        S=start
        for _,initial,final in results:
            if np.linalg.eigvalsh(S-initial)[0]<=0:
                return None
            S=final
        if np.linalg.eigvalsh(S)[0]<=0:
            return None
        return float(fractions@np.array([r[0] for r in results]))
    def edge(sign):
        good,bad=0.,.08
        while values(sign*bad) is not None:
            good,bad=bad,2*bad
        for _ in range(38):
            mid=(good+bad)/2
            if values(sign*mid) is None:
                bad=mid
            else:
                good=mid
        return sign*good
    def objective(h):
        result=values(h)
        return -np.inf if result is None else h*target-result
    lo,hi=edge(-1),edge(1)
    optimum=minimize_scalar(lambda h:-objective(h),bounds=(lo,hi),method="bounded",
                            options=dict(xatol=2e-10))
    return float(max(0.,-optimum.fun,objective(lo),objective(hi)))


def _response(experiments,gain):
    return np.array([read(e,gain) for e in experiments])


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
