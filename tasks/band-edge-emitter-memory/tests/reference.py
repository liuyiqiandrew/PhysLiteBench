from functools import lru_cache
import json

import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import expm_multiply

PARAMETER="gain"
TRUE_PARAMETER=.932
CALIBRATION_SEED=28212896
NOISE_FRACTION=.0029


@lru_cache(maxsize=170)
def inverse(delta,length,t,g,count):
    period=3.7*t
    gamma=6.3/t
    omega=2*np.pi*np.arange(-count,count+1)/period
    z=1j*gamma-omega
    root=np.sqrt(z)
    q=g*g/(2*root)
    row=z-delta-q
    connection=q*np.exp(-length*root)
    inverse=np.vstack([(row-connection)**-1,(row+connection)**-1])
    subtract=1/(z-delta)
    contour=np.exp(1j*omega*t)
    return np.exp(-1j*delta*t)+1j*np.exp(gamma*t)/period*((inverse-subtract)@contour)


def settings(e):
    d=e["separation"]
    gap=-(np.pi*e["locked_order"]/d)**2 if "locked_order" in e else e["detuning"]
    vectors=[np.array(e[name])@np.array([1.,1j]) for name in ("initial","probe")]
    return d,gap,*vectors


def predict_at(e,g,count=12288):
    if e["observable"]=="isolated_transfer":
        return float((1-np.cos(2*g*e["setting"]))/2)
    d,gap,p,z=settings(e)
    response=inverse(gap,d,e["scaled_time"],g,count)
    change=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
    return float(abs(np.vdot(change@z,response*(change@p)))**2)


def predict(experiments,gain):
    return np.array([predict_at(e,gain) for e in experiments])


def v(values):
    x=np.array(values)
    x=x/np.linalg.norm(x)
    return np.round(np.column_stack((x.real,x.imag)),13).tolist()


def setup(i,t,lock=False):
    v0=[1.,.0] if i%2==0 else [1.,.65j]
    v1=[1.,1.j] if i%3==0 else ([.0,1.] if i%3==1 else [1.,-1.])
    base=dict(observable="registered",separation=[.83,1.22,1.64,1.96,2.62,3.22][i],
              scaled_time=t,initial=v(v0),probe=v(v1))
    if lock:
        base["locked_order"]=1 if i%2==0 else 2
    else:
        base["detuning"]=[-.54,1.07,-2.17,-1.52,.43,-.74][i]
    return base


def calibration_inputs():
    return [dict(observable="isolated_transfer",setting=.18+.033*i) for i in range(27)]


def hidden_inputs():
    groups={"retained_band_edge_weight":[], "locked_embedded_poles":[],
            "spatial_memory_and_coherent_detection":[], "isolated_anchors":[]}
    for i in range(6):
        for t in (.24+.049*i,1.47+.21*i,3.31-.095*i):
            groups["retained_band_edge_weight"].append(setup(i,t))
            groups["locked_embedded_poles"].append(setup(i,t,True))
        for z in ([1,1],[1,-1j]):
            groups["spatial_memory_and_coherent_detection"].append(dict(
                setup(i,.78+.23*i,(i%2)==0),initial=v([1,1j]),probe=v(z)))
        groups["isolated_anchors"].append(calibration_inputs()[2*i])
    return groups


def lattice(e,g,distance,pad=170):
    d,gap,init,probe=settings(e)
    lam=(d/distance)**1.5
    if e["scaled_time"]/lam**(4/3) > pad/2.6:
        pad=int(2.9*e["scaled_time"]/lam**(4/3))+40
    modes=2*pad+distance+1
    K=lil_matrix((modes+2,modes+2))
    K[:modes,:modes]=diags([np.ones(modes-1),-2*np.ones(modes),np.ones(modes-1)],[-1,0,1])
    for i,j in enumerate((pad,pad+distance)):
        K[modes+i,modes+i]=gap*lam**(4/3)
        K[modes+i,j]=g*lam
        K[j,modes+i]=g*lam
    a=np.r_[np.zeros(modes),init]
    result=expm_multiply(-1j*e["scaled_time"]/lam**(4/3)*K.tocsr(),a)
    return float(abs(np.vdot(probe,result[-2:]))**2)


def checks():
    rows=[]
    for e in [setup(2,.89),setup(4,1.92,True)]:
        ref=predict_at(e,TRUE_PARAMETER)
        delta=abs(ref-predict_at(e,TRUE_PARAMETER,26720))
        assert delta<2e-7,delta
        a,b=[lattice(e,TRUE_PARAMETER,cut) for cut in (8,20)]
        assert abs(ref-b)<.021,(a,b,ref)
        rows.append(dict(finite_lattice_8sites=a,finite_lattice_20sites=b,
                         weak_limit=ref,inversion_refinement=delta))
    return dict(finite_hopping_controls=rows)
