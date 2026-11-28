from functools import lru_cache
import json

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import minimize_scalar

PARAMETER="gain"
TRUE_PARAMETER=.942
CALIBRATION_SEED=28212885
NOISE_FRACTION=.0026


@lru_cache(maxsize=180)
def shortrun(encoded,gain,dt):
    p=json.loads(encoded)
    a,b=p["stiffness"]
    t=p["orientation"]
    c,s=np.cos(t),np.sin(t)
    K=np.array([[a*c*c+b*s*s,(a-b)*s*c],[(a-b)*s*c,a*s*s+b*c*c]])
    omega=gain*p["drive"]
    M=-K+np.array([[0,-omega],[omega,0]])
    T=np.diag(p["temperature"])
    H=np.diag(p["weights"])@M
    C=solve_continuous_lyapunov(M,-2*T)
    U=expm(dt*M)
    Q=C-U@C@U.T
    inv=np.linalg.inv(Q)
    return U.T@inv@U,-U.T@inv,inv,-np.linalg.slogdet(Q)[1]/2, H,C


def square_quadratic(base,theta,dt,steps):
    start,mix,end,c,H,C=base
    # Fold the dimensional normalizing factor 1/(2*pi) into composition.
    side=(H+H.T)/2
    ker=(H-H.T)/2
    start=start.copy()
    end=end.copy()
    mix=mix+theta*ker
    for _ in range(steps):
        M=end+start
        diag=M.diagonal()
        det=diag[0]*diag[1]-M[0,1]*M[1,0]
        if det<=0. or min(diag)<=0.:
            return None
        solved=np.linalg.solve(M,np.c_[mix.T,mix])
        dx,dy=solved[:,:2],solved[:,2:]
        start,end,mix=(start-mix@dx,end-mix.T@dy,-mix@dy)
        next_=2*c-.5*np.log(det)
        difference=next_-c
        c=next_
    return difference/(dt*2**(steps-1)),start+theta*side,end-theta*side,mix


def rate(e,gain,dt):
    rows=e["stages"]
    records=[shortrun(json.dumps(p,sort_keys=True),gain,dt) for p in rows]
    if e["observable"]=="mean_heat":
        p=rows[0]
        k,T=p["stiffness"][0],p["temperature"][0]
        return sum(p["weights"])*T*(gain*p["drive"])**2/k
    N=max(1,int(np.ceil(np.log2(500/dt))))
    init=np.linalg.inv(records[0][-1])
    frac=np.array([p["fraction"] for p in rows])
    target=e["target"]
    def evaluate(theta):
        kernels=[square_quadratic(r,theta,dt,N) for r in records]
        if any(k is None for k in kernels):
            return None
        precision=np.zeros((2*len(rows)+2,2*len(rows)+2))
        precision[:2,:2]=init
        for j,(_,s,t,x) in enumerate(kernels):
            a,b=slice(2*j,2*j+2),slice(2*j+2,2*j+4)
            precision[a,a]+=s
            precision[b,b]+=t
            precision[a,b]=x
            precision[b,a]=x.T
        if np.linalg.eigvalsh(precision)[0]<=0:
            return None
        return float(frac@np.array([k[0] for k in kernels]))
    def interval(sign):
        a,b=0.,.11
        while evaluate(sign*b) is not None:
            a,b=b,2*b
        for _ in range(27):
            x=(a+b)/2
            if evaluate(sign*x) is not None:
                a=x
            else:
                b=x
        return sign*a
    def obj(v):
        score=evaluate(v)
        return -np.inf if score is None else v*target-score
    lo,hi=interval(-1),interval(1)
    r=minimize_scalar(lambda t:-obj(t),bounds=(lo,hi),method="bounded",options={"xatol":1e-9})
    return float(max(0.,-r.fun,obj(lo),obj(hi)))


@lru_cache(maxsize=120)
def result(encoded,gain):
    e=json.loads(encoded)
    if e["observable"]=="mean_heat":
        return rate(e,gain,.003125)
    low=rate(e,gain,.00625)
    high=rate(e,gain,.003125)
    return max(0.,(4*high-low)/3)


def predict(experiments,gain):
    return np.array([result(json.dumps(e,sort_keys=True),gain) for e in experiments])


def stage(k,t,o,w,h,f):
    return dict(stiffness=k,temperature=t,orientation=o,drive=w,weights=h,fraction=f)


def calibration_inputs():
    vals=[]
    for i in range(25):
        k=.58+.067*i
        T=.58+.105*(i%6)
        vals.append(dict(observable="mean_heat",stages=[stage([k,k],[T,T],0.,.34+.1*(i%9),[.83,1.07],1.)]))
    return vals


def experiment(rows,target):
    return dict(stages=rows,target=target,observable="tail_rate")


def hidden_inputs():
    groups={"noncoaxial_traps_and_baths":[], "component_weights_at_common_junctions":[],
            "three_stage_energy_storage":[], "steady_mean_and_subdivision_anchors":[]}
    for i in range(4):
        A=stage([.53+.17*i,1.61-.12*i],[1.38-.14*i,.56+.09*i],.37+.20*i,.42+.10*i,
                [1.28,.53+.11*i],.43)
        B=stage([1.86-.13*i,.83+.09*i],[.47+.063*i,1.21+.12*i],-.68+.14*i,-.34-.08*i,
                [.49+.07*i,1.31],.57)
        C=stage([.77+.14*i,1.24],[.85,1.54-.05*i],.62,-.63+.11*i,[1.52,.46],.22)
        for q in (-12.,-4.8,5.7,14.):
            groups["noncoaxial_traps_and_baths"].append(experiment([A,B],q))
            groups["component_weights_at_common_junctions"].append(experiment([
                dict(A,weights=[.39,1.57],fraction=.28),
                dict(B,weights=[1.53,.39],fraction=.72)],q))
        for q in (-14.,3.4,15.5):
            groups["three_stage_energy_storage"].append(experiment([
                dict(B,fraction=.36),C,dict(A,fraction=.42)],q))
        groups["steady_mean_and_subdivision_anchors"].extend([
            calibration_inputs()[i],experiment([dict(A,fraction=1.)],-6.2),
            experiment([dict(A,fraction=.54),dict(A,fraction=.46)],-6.2)])
    return groups


def checks():
    groups=hidden_inputs()
    anchors=groups["steady_mean_and_subdivision_anchors"]
    delta=max(abs(predict([anchors[i]],TRUE_PARAMETER)[0]-predict([anchors[i+1]],TRUE_PARAMETER)[0])
              for i in range(1,len(anchors)-1,3))
    assert delta<2e-6,delta
    refined=[]
    for e in [groups["noncoaxial_traps_and_baths"][0],groups["three_stage_energy_storage"][6]]:
        val=predict([e],TRUE_PARAMETER)[0]
        q0=rate(e,TRUE_PARAMETER,.003125)
        q1=rate(e,TRUE_PARAMETER,.0015625)
        difference=abs((4*q1-q0)/3-val)
        assert difference<4e-4,difference
        refined.append(difference)
    return dict(unchanged_trap_subdivision_error=delta,reference_time_step_refinement=refined)
