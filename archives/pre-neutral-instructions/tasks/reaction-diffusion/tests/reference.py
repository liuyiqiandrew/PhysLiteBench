from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

D=1e-9
L=.001

def load_data():
    return dict(np.load(Path(__file__).parent/"data/calibration.npz"))

def trajectory(t,x,initial,diffusivity=D,refinement=4):
    # Evolve all three ions independently on a finer grid, enforcing zero current.
    dense=np.linspace(x[0],x[-1],refinement*(len(x)-1)+1)
    prepared=np.column_stack([np.interp(dense,x,initial[:,i]) for i in range(2)])
    prepared=np.column_stack((prepared,prepared.sum(axis=1)))
    count=len(dense);dx=dense[1]-dense[0]
    volume=np.full(count,dx);volume[[0,-1]]*=.5
    valence=np.array([1.,1.,-1.]);mobility=diffusivity*np.array([1.,4.,2.])
    def rhs(time,flat):
        concentration=flat.reshape(count,3)
        midpoint=(concentration[:-1]+concentration[1:])/2
        gradient=np.diff(concentration,axis=0)/dx
        field=-(gradient@(valence*mobility))/(midpoint@(valence**2*mobility))
        flux=np.zeros((count+1,3))
        flux[1:-1]=-mobility*(gradient+midpoint*valence*field[:,None])
        return (-np.diff(flux,axis=0)/volume[:,None]).ravel()
    sparsity=diags([np.ones(3*count-abs(i)) for i in range(-5,6)],range(-5,6),format="csr")
    solution=solve_ivp(rhs,(0,t[-1]),prepared.ravel(),method="BDF",t_eval=t,rtol=1e-9,atol=1e-11,jac_sparsity=sparsity)
    assert solution.success
    concentration=solution.y.T.reshape(len(t),count,3)
    assert np.max(abs(concentration[:,:,0]+concentration[:,:,1]-concentration[:,:,2]))<1e-8
    return concentration[:,::refinement,:2]

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
