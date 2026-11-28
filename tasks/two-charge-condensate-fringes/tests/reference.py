from functools import lru_cache
import json

import numpy as np
from scipy.linalg import expm
from scipy.special import zeta
from scipy.stats import qmc

PARAMETER="mass"
TRUE_PARAMETER=.936
CALIBRATION_SEED=28212895
NOISE_FRACTION=.0018


@lru_cache(maxsize=4)
def sampling(n,seed):
    return qmc.Sobol(6,scramble=True,seed=seed).random_base2(n)


def thermal(e,mass):
    T=e["temperature"]
    x=(mass*T/(2*np.pi))**1.5
    r=np.arange(1,241)
    p=np.array([x*np.sum(np.exp(-r*d/T)/r**1.5) for d in e["gaps"]])
    return e["density"]-5*x*zeta(1.5)-p.sum(),np.array(e["charges"])-p[0]+p[1]


def configurations(stress,left,right):
    a,b=left.T
    z=right-a-b
    return np.array([(a+stress[0])/2,(a-stress[0])/2,
                     (b+stress[1])/2,(b-stress[1])/2,z]).T


def aggregate(e,x):
    H=np.array(e["interaction"])
    value=np.einsum("bi,ij,bj->b",x,H,x)/2+x@e["weak"]
    return np.exp((value.min()-value)/e["temperature"])


@lru_cache(maxsize=145)
def one(sent,mass,n=19,seed=361):
    e=json.loads(sent)
    total,charge=thermal(e,mass)
    if e["observable"]=="line_density":
        return float(total)
    S=sampling(n,seed)
    points=np.sort(S[:,:2],axis=1)
    gap=(total-np.sum(abs(charge)))
    allowed=abs(charge)+gap*np.column_stack([points[:,0],points[:,1]-points[:,0]])
    X=configurations(charge,allowed,total)
    w=aggregate(e,X)
    matrix=np.eye(5,dtype=complex)
    for h in e["pulses"]:
        matrix=expm(-1j*np.array(h))@matrix
    if e["observable"]=="port_density":
        Y=np.average(X,weights=w,axis=0)
        return float(abs(matrix[e["port"]])**2@Y)
    phase=2*np.pi*np.column_stack([S[:,2:],np.zeros(len(S))])
    state=np.sqrt(X)*np.exp(1j*phase)
    I=abs(matrix@state.T).T**2
    factor=np.exp(-np.sum(((I-e["center"])/e["width"])**2,axis=1)/2)
    index=e["ports"]
    return float(np.average((I[:,index[0]]-I[:,index[1]])*factor,weights=w))


def predict(experiments,mass):
    return np.array([one(json.dumps(e,sort_keys=True),mass) for e in experiments])


def calibrate(i):
    return dict(observable="line_density",temperature=.44+.026*i,density=1.82+.041*i,
                charges=[.07,-.12],gaps=[.27+.058*(i%8),.92-.036*(i%6)])
    

def calibration_inputs():
    return [calibrate(i) for i in range(27)]


def stage(i):
    x=np.arange(5)
    mat=np.eye(5)*(.18-.029*i)+.39*np.cos(.58*(x[:,None]+x)+i*.35)
    mat[2,4]=mat[4,2]=-.61+.051*i
    return np.round(mat,11).tolist()


def case(i,obs):
    e=calibrate(i*4)
    e.update(observable=obs,charges=[[.08,-.19],[-.14,.11],[.18,-.09],[.21,.14],[-.26,.10],
                [.06,-.22],[-.21,-.09]][i])
    e["weak"]=[.14+.2*i,-1.2+.31*i,.87-.24*i,.18-.13*i,-.23+.32*i]
    H=np.array(stage(i+3))*2.4
    H[0,1]=H[1,0]=1.64-.33*i
    H[2,3]=H[3,2]=-.84+.34*i
    e["interaction"]=H.tolist()
    e["pulses"]=[stage(i),stage(i+3)]
    e["center"]=[.32+.05*i,.24,.29,.24+.04*i,.15]
    e["width"]=[.19,.16,.18,.16,.22]
    e["ports"]=[i%5,(i+2)%5]
    e["port"]=i%5
    return e


def hidden_inputs():
    groups={"two_charge_thermal_populations":[], "coherent_window_weighted_currents":[],
             "sector_reweighting_under_weak_interaction":[], "calibration_and_unpulsed_anchors":[]}
    for i in range(7):
        e=case(i,"weighted_difference")
        groups["two_charge_thermal_populations"].extend(
            [dict(e,observable="port_density",port=k,pulses=[]) for k in (0,2,4)])
        for ports in (e["ports"],[1,4]):
            groups["coherent_window_weighted_currents"].append(dict(e,ports=ports))
        groups["sector_reweighting_under_weak_interaction"].extend([
            dict(e,interaction=(np.array(e["interaction"])*-.68).tolist()),
            dict(e,weak=[-.22,.52,-.72,.18,.34],pulses=[stage(i+1)],
                 center=[.23,.34,.25,.23,.31],ports=[4,0])])
        groups["calibration_and_unpulsed_anchors"].extend([
            calibrate(3+i),dict(e,observable="port_density",port=1)])
    return groups


def checks():
    errors=[]
    for i in (0,3,6):
        e=case(i,"weighted_difference")
        assert all(thermal(e,m)[0]-np.sum(abs(thermal(e,m)[1]))>.06 for m in [.62,1.34])
        encoded=json.dumps(e,sort_keys=True)
        errors.append(abs(one(encoded,TRUE_PARAMETER,19,361)-one(encoded,TRUE_PARAMETER,20,892)))
    assert max(errors)<2e-4,errors
    e=case(3,"port_density")
    e["pulses"]=[]
    e["port"]=4
    R,c=thermal(e,TRUE_PARAMETER)
    output=predict([e],TRUE_PARAMETER)[0]
    N=330
    charges=np.rint(N*c).astype(int)/N
    cap=np.floor(R*N)/N
    r=np.arange(int((cap-np.sum(abs(charges)))*N/2)+1)
    ix,iy=np.meshgrid(r,r)
    ids=(ix+iy)<=r[-1]
    pair=np.array([ix[ids],iy[ids]]).T*(2/N)+abs(charges)
    occupations=configurations(charges,pair,cap)
    exact=np.average(occupations[:,-1],weights=aggregate(e,occupations))
    assert abs(exact-output)<.006,(exact,output)
    return dict(refined_phase_and_population_integration=max(errors),
                finite_canonical_neutral_population=exact,limiting_neutral_population=output)
