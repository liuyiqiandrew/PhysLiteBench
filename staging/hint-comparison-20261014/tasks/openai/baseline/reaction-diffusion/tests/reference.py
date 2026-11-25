from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

D=1e-9
L=.001

def load_data():
    return dict(np.load(Path(__file__).parent/"data/calibration.npz"))

def associated_initial(total):
    # Independent scalar root of the ideal mass-action equation.
    from scipy.optimize import brentq
    prepared=[]
    for total_a,total_b in total:
        if min(total_a,total_b)==0:bound=0.
        else:bound=brentq(lambda p:p-.5*(total_a-p)*(total_b-p)*(total_a+total_b-2*p)**2,0.,min(total_a,total_b),xtol=1e-14)
        prepared.append([total_a-bound,total_b-bound,total_a+total_b-2*bound,bound])
    return np.array(prepared)

def finite_reaction(t,x,initial,diffusivity=D,refinement=4,dissociation_rate=40.):
    # Four independently transported species with a reversible finite-rate
    # reaction; no equilibrium elimination during the evolution.
    dense=np.linspace(x[0],x[-1],refinement*(len(x)-1)+1)
    totals=np.column_stack([np.interp(dense,x,initial[:,i]) for i in range(2)])
    prepared=associated_initial(totals)
    count=len(dense);dx=dense[1]-dense[0];volume=np.full(count,dx);volume[[0,-1]]*=.5
    valence=np.array([1.,1.,-1.,0.]);mobility=diffusivity*np.array([1.,4.,2.,.15])
    stoichiometry=np.array([-1.,-1.,-2.,1.])
    def rhs(time,flat):
        concentration=flat.reshape(count,4)
        midpoint=(concentration[:-1]+concentration[1:])/2
        gradient=np.diff(concentration,axis=0)/dx
        field=-(gradient@(valence*mobility))/(midpoint@(valence**2*mobility))
        flux=np.zeros((count+1,4))
        flux[1:-1]=-mobility*(gradient+midpoint*valence*field[:,None])
        reaction=dissociation_rate*(.5*concentration[:,0]*concentration[:,1]*concentration[:,2]**2-concentration[:,3])
        return (-np.diff(flux,axis=0)/volume[:,None]+reaction[:,None]*stoichiometry).ravel()
    sparsity=diags([np.ones(4*count-abs(i)) for i in range(-7,8)],range(-7,8),format='csr')
    solution=solve_ivp(rhs,(0,t[-1]),prepared.ravel(),method='BDF',t_eval=t,rtol=2e-9,atol=2e-11,jac_sparsity=sparsity)
    assert solution.success
    concentration=solution.y.T.reshape(len(t),count,4)
    assert np.max(abs(concentration[:,:,0]+concentration[:,:,1]-concentration[:,:,2]))<1e-7
    assert concentration.min()>-1e-9
    totals=concentration[:,:,:2]+concentration[:,:,3:4]
    return totals[:,::refinement,:]

def trajectory(t,x,initial,diffusivity=D,refinement=4,rate=20.):
    slow=finite_reaction(t,x,initial,diffusivity,refinement,rate)
    middle=finite_reaction(t,x,initial,diffusivity,refinement,2*rate)
    fast=finite_reaction(t,x,initial,diffusivity,refinement,4*rate)
    return (slow-6*middle+8*fast)/3

@lru_cache(maxsize=1)
def hidden_cases():
    x=np.linspace(0,L,65);z=np.pi*x/L;t=np.linspace(0,250,51)
    profiles=[np.column_stack((1+.7*np.cos(z),np.full(len(x),.8))),
              np.column_stack((np.full(len(x),.6),1+.7*np.cos(z))),
              np.column_stack((1+.7*np.cos(2*z),np.full(len(x),.6)))]
    return [(t,x,initial,trajectory(t,x,initial)) for initial in profiles]

def metrics(model,data):
    residual=np.array([model.predict(data["t"],data["x"],initial) for initial in data["initial"]])-data["concentration"]
    assert np.isfinite(residual).all()
    errors=[];species_errors=[];minimum=[];mass_error=[]
    for t,x,initial,expected in hidden_cases():
        prediction=np.asarray(model.predict(t,x,initial))
        assert prediction.shape==expected.shape and np.isfinite(prediction).all()
        mean=np.trapezoid(initial,x,axis=0)/(x[-1]-x[0])
        values=np.linalg.norm(prediction-expected,axis=(0,1))/np.linalg.norm(expected-mean,axis=(0,1))
        species_errors.append(values.tolist());errors.append(float(max(values)))
        minimum.append(float(prediction.min()))
        mass_error.append(float(np.max(abs(np.trapezoid(prediction,x,axis=1)/(x[-1]-x[0])-mean))))
    return {"diffusivity_m2_s":float(model.diffusivity),"relative_diffusivity_error":float(abs(model.diffusivity/D-1)),
            "calibration_reduced_chi2":float(np.sum((residual/data["sigma"])**2)/(residual.size-1)),
            "hidden_nrmse":errors,"hidden_species_nrmse":species_errors,
            "minimum_hidden_concentration":minimum,"maximum_mean_concentration_error":mass_error}


from scipy.linalg import eigh

def finite_fluctuations(t,totals,diffusivity=D,harmonic=1,separation=10000.):
    # Four-species linear chemical Langevin dynamics with finite electrostatic
    # compressibility and finite association. No slow-species projection.
    concentration=associated_initial(np.asarray(totals)[None,:])[0]
    charge=np.array([1.,1.,-1.,0.]);reaction=np.array([-1.,-1.,-2.,1.]);readout=np.array([[1.,0.,0.,1.],[0.,1.,0.,1.]])
    k2=(harmonic*np.pi/L)**2
    hessian=np.diag(1/concentration)+separation/np.sum(concentration)*np.outer(charge,charge)
    kinetic=k2*np.diag(diffusivity*np.array([1.,4.,2.,.15])*concentration)+separation*diffusivity*k2*concentration[3]*np.outer(reaction,reaction)
    value,basis=eigh(hessian);root=(basis*np.sqrt(value))@basis.T;inverse_root=(basis/np.sqrt(value))@basis.T
    rates,modes=eigh(root@kinetic@root);projection=readout@inverse_root@modes
    assert rates.min()>0
    return np.array([(projection*np.exp(-rates*time))@projection.T for time in t])

def correlations(t,totals,diffusivity=D,harmonic=1,separation=10000.):
    values=[finite_fluctuations(t,totals,diffusivity,harmonic,separation*s) for s in [1,2,4]]
    return (values[0]-6*values[1]+8*values[2])/3

def fluctuation_cases():
    return {name:[dict(t=np.array([0.,.5,2.,10.,40.,120.,300.,600.]),totals=np.array(total),harmonic=h)
                  for total,h in entries] for name,entries in {
        'dilute_charge_correlations':[([.2,.2],1),([.2,.4],3),([.4,.2],7)],
        'bound_cluster_correlations':[([2.,2.],1),([1.5,2.],4),([2.,1.5],8)],
        'asymmetric_mixture_correlations':[([.3,1.8],2),([1.9,.4],5),([.6,1.6],8)],
    }.items()}

mean_metrics=metrics
def metrics(model,data):
    result=mean_metrics(model,data);groups={}
    for name,experiments in fluctuation_cases().items():
        measured=[];expected=[]
        for e in experiments:
            value=model.correlate(e['t'],e['totals'],e['harmonic'])
            assert isinstance(value,np.ndarray) and value.shape==(len(e['t']),2,2) and np.isfinite(value).all()
            measured.extend(value.ravel());expected.extend(correlations(**e).ravel())
        groups[name]=float(np.linalg.norm(np.array(measured)-expected)/np.linalg.norm(expected))
    result['fluctuation_nrmse']=groups
    return result
