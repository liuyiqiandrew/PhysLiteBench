from functools import lru_cache
from math import sqrt

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

PARAMETER="mass"
TRUE_PARAMETER=.942
CALIBRATION_SEED=27112748
NOISE_FRACTION=.0028
SPIN=np.array([-1.,0.,1.])
B=np.array([[1,-1],[1,0],[1,1.]])


@lru_cache(maxsize=1200)
def thermal_integrals(gap):
    gap=max(gap,0)
    def bose(power):
        return 4/sqrt(np.pi)*quad(lambda x:x**power/np.expm1(x*x+gap),0,9,epsabs=1e-10)[0]
    return bose(4)*2/3,bose(2)


@lru_cache(maxsize=200)
def macrostates(rho,mag,T,quadratic,mass):
    Q=quadratic*SPIN**2/T
    density_scale=(mass*T/(2*np.pi))**1.5
    target=np.array([rho,mag])
    def dual(y):
        eqn=Q-B@y
        pressures,thermals=np.array([thermal_integrals(g) for g in eqn]).T*density_scale
        return pressures.sum()-target@y,B.T@thermals-target
    result=minimize(dual,[-.6+min(0.,quadratic/T),0.],jac=True,method="SLSQP",
                    constraints=[{"type":"ineq","fun":lambda y:Q-B@y,"jac":lambda y:-B}],
                    options=dict(ftol=2e-13,maxiter=150))
    if not result.success and np.min(Q-B@result.x)<-2e-8:
        raise ValueError(result.message)
    gaps=Q-B@result.x
    active=np.flatnonzero(gaps<2e-8)
    if active.size==0:
        return np.zeros(3),np.zeros(3)
    thermal=np.array([thermal_integrals(0. if g<2e-8 else g)[1] for g in gaps])*density_scale
    residual=target-B.T@thermal
    if active.size<3:
        cond=np.zeros(3)
        cond[active]=np.linalg.lstsq(B[active].T,residual,rcond=None)[0]
        assert min(cond)>-1e-6 and np.max(np.abs(B.T@cond-residual))<3e-5
        return cond,cond.copy()
    count,spin=residual
    ends=[np.array([max(-spin,0.),count-abs(spin),max(spin,0.)]),
          np.array([(count-spin)/2,0.,(count+spin)/2])]
    return tuple(ends)


def one(e,mass):
    rho,T=e["density"],e["temperature"]
    start,end=macrostates(rho,e["magnetization"],T,e["quadratic"],mass)
    if e["observable"]=="zero_fraction":
        return start.sum()/rho
    raw_limits=e["window"]
    a=e["pulse_angle"]
    weak=np.array(e["split"])
    differences=(end-start)
    def window_probability(c):
        radius=abs(np.sin(2*a))*2*sqrt(max(0,c[0]*c[2]))/rho
        shift=(c[2]-c[0])*np.cos(2*a)/rho
        if radius<1e-13:
            return float(raw_limits[0]<=shift<=raw_limits[1])
        low,high=[np.arccos(np.clip((x-shift)/radius,-1,1)) for x in raw_limits]
        return (low-high)/np.pi
    def weight(x):
        return np.exp(-x*np.dot(differences,weak)/T)
    def moment(x):
        c=(1-x)*start+x*end
        return weight(x)*window_probability(c)*(c[1]/rho if e["observable"]=="neutral_window_mean" else 1.)
    return quad(moment,0,1,epsabs=2e-8,limit=140,points=np.linspace(0,1,21)[1:-1])[0]/quad(weight,0,1)[0]


def predict(experiments,mass):
    return np.array([one(e,mass) for e in experiments])


def spec(rho,mag,T,Q,angles,split,window,obs):
    return dict(density=rho,magnetization=mag,temperature=T,quadratic=Q,pulse_angle=angles,
                split=split,window=window,observable=obs)


def calibration_inputs():
    return [spec(r,0.,T,0.,.39,[0.,0.,0.],[-.35,.27],"zero_fraction") for r in (.81,.96,1.17,1.41)
            for T in (.46,.59,.66,.81,.96,1.18,1.29)]


def hidden_inputs():
    z,p,m="zero_fraction","window_probability","neutral_window_mean"
    return {
        "thermal_saturation_faces":[spec(r,c,t,Q,.41,[.1,.6,-.4],[-.29,.37],z)
                                    for r,c,t,Q in [
            (.83,.39,1.26,.27),(.82,-.43,1.39,-.32),(.63,.47,1.52,.34),(.70,.04,1.25,.48),
            (.97,.68,1.38,0.),(.71,.31,1.41,-.51),(.65,-.39,1.47,.49),(.15,.029,1.61,.13),
            (.38,.27,1.38,-.29),(.76,.32,1.51,.34),(.44,-.34,1.44,-.27),(.91,.17,1.62,.49)]],
        "two_component_shots":[spec(r,c,t,Q,angle,[.47,-.51,.14],window,obs)
                               for r,c,t,Q,angle in [
                                   (.89,.19,1.04,-.24,.68),(1.22,-.38,1.25,-.43,.55),(.97,.26,1.49,-.17,.83),
                                   (1.01,.43,1.05,.21,.51),(.92,-.25,1.03,.48,.74)]
                               for window,obs in [([-.39,-.08],p),([-.24,.17],p),([.19,.53],p),([-.31,.36],m)]],
        "tilted_constraint_segment":[spec(r,c,t,0.,a,split,window,obs)
                                    for r,c,t,a,split in [
            (1.33,.23,.63,.63,[1.15,-.68,.54]),(1.65,-.26,.91,.77,[-.7,1.25,-.46]),
            (1.29,.08,1.11,.56,[.92,.17,-.84]),(.97,-.34,.58,.71,[.68,-.46,1.18])]
                                    for window,obs in [([-.41,-.16],p),([.12,.44],p),([-.22,.21],m),([.08,.54],m)]],
        "fully_selected_and_normal":[spec(r,c,t,Q,a,[.16,.6,-.4],win,obs)
                                      for r,c,t,Q,a,win in [
                                          (.88,.74,.87,-.33,.52,[.14,.61]),
                                          (.76,-.68,.71,.48,.86,[-.25,.44]),
                                          (.21,.029,1.55,0.,.51,[-.27,.12]),
                                          (.14,-.018,1.48,.24,.65,[.19,.33])]
                                      for obs in [z,p,m]],
    }


def finite_spin_window(n,N,total_fraction,imbalance_fraction,a,window):
    j=int(np.rint(n*(1+imbalance_fraction)/2))
    k=np.arange(n)
    coupling=np.sqrt((k+1)*(n-k))
    H=diags((coupling,coupling),(-1,1),format="csc")
    source=np.zeros(n+1,complex)
    source[j]=1
    after=expm_multiply(-1j*a*H,source,traceA=0.)
    imbalance=(2*np.arange(n+1)-n)/N
    low,high=window
    signal=np.sum(abs(after[(imbalance>=low)&(imbalance<=high)])**2)
    center=total_fraction*imbalance_fraction*np.cos(2*a)
    amplitude=total_fraction*np.sqrt(1-imbalance_fraction**2)*np.sin(2*a)
    target=np.diff(np.arcsin(np.clip((np.array(window)-center)/amplitude,-1,1)))[0]/np.pi
    return float(signal),float(target)


def checks():
    n,e=1600,[-.19,.27]
    measured,expected=finite_spin_window(n,2500,.64,.205,.48,e)
    refined,_=finite_spin_window(2*n,5000,.64,.205,.48,e)
    assert abs(refined-expected)<.009 and abs(refined-measured)<.02
    e=hidden_inputs()["tilted_constraint_segment"][1]
    full={**e,"window":[-1.,1.]}
    left={**e,"window":[-1.,.149]}
    right={**e,"window":[.149,1.]}
    a,l,r=predict([full,left,right],TRUE_PARAMETER)
    assert abs(a-1)<1e-6 and abs(l+r-1)<1e-6
    return dict(finite_spin_1600=measured,finite_spin_3200=refined,thermodynamic_window=expected,
                complementary_windows=l+r)
