"""Independent joint mass balances and electrochemical-potential equations."""
from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares

TRUE_PARTITION=.72


@lru_cache(maxsize=8192)
def equilibrium(partition, vg, vb, charge, amount_a, amount_b):
    z=np.array([1., 2., -1.])
    amounts=np.array([amount_a, amount_b, amount_a+2*amount_b-charge*vg])
    active=amounts>0
    za=z[active]; na=amounts[active]
    n=len(na)
    mean=na/(vg+vb)
    def residual(unknown):
        gel=np.exp(unknown[:n]); bath=np.exp(unknown[n:2*n]); potential=unknown[-1]
        balance=(vg*gel+vb*bath-na)/na
        chemical=np.log(gel/bath)-np.log(partition)-za*potential
        electric=np.array([(gel@za-charge)/(1+charge+sum(mean))])
        return np.r_[balance,chemical,electric]
    initial=np.r_[np.log(mean),np.log(mean),0.]
    result=least_squares(residual,initial,ftol=2e-13,xtol=2e-13,gtol=2e-13,max_nfev=400)
    assert result.success and np.max(abs(residual(result.x)))<1e-10
    concentrations=np.zeros((2,3))
    concentrations[0,active]=np.exp(result.x[:n]);concentrations[1,active]=np.exp(result.x[n:2*n])
    return concentrations


def predict(experiments, partition=TRUE_PARTITION):
    out=[]
    for e in experiments:
        values=equilibrium(partition,e['gel_volume'],e['bath_volume'],e['fixed_charge'],e['amount_a'],e['amount_b'])
        out.append(values[0 if e['compartment']=='gel' else 1,e['species']])
    return np.array(out)


def readings(vg,vb,q,na,nb):
    return [dict(gel_volume=vg,bath_volume=vb,fixed_charge=q,amount_a=na,amount_b=nb,
                 compartment=compartment,species=s)
            for compartment in ['gel','bath'] for s in range(3) if s==2 or [na,nb][s]>0]


def calibration_inputs():
    out=[]
    for vg,vb in [(.4,1.2),(1.1,.6)]:
        for q in [.25,.7,1.2]:
            for species in [0,1]:
                for excess in [.4,1.3]:
                    n=(q*vg+excess)/(species+1)
                    out.extend(readings(vg,vb,q,n if species==0 else 0.,n if species==1 else 0.))
    return out


def hidden_inputs():
    return {name:sum([readings(vg,vb,q,na,nb) for vg,vb,q,na,nb in values],[])
            for name,values in dict(
                mixed_charge=[(.4,1.2,1.2,.5,.6),(.4,1.2,1.8,.4,.7),(.7,1.,1.5,.9,.6)],
                finite_bath=[(1.4,.3,1.2,1.2,1.1),(1.1,.25,1.6,.8,1.4),(.9,.3,1.8,1.2,1.0)],
                unequal_salts=[(.7,.8,1.7,.2,1.2),(.6,1.1,1.5,1.8,.15),(1.2,.6,1.4,.4,1.8)]).items()}
