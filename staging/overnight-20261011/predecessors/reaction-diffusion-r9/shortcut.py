import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp
from scipy.sparse import diags

# Neutral four-ion association A+ + B+ + 2 C- <-> P.
ASSOCIATION=0.5

def equilibrium(total):
    a,b=np.asarray(total).T
    bound=np.zeros_like(a)
    for _ in range(24):
        free_a=a-bound;free_b=b-bound;anion=free_a+free_b
        value=bound-ASSOCIATION*free_a*free_b*anion**2
        derivative=1+ASSOCIATION*(anion**3+4*free_a*free_b*anion)
        change=value/derivative
        bound=np.clip(bound-change,0.,np.minimum(a,b))
        if np.max(abs(change))<1e-13:break
    return np.column_stack((a-bound,b-bound,a+b-2*bound,bound))

class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        def objective(log_d):
            self.diffusivity = np.exp(log_d)
            predicted = np.array([self._binary(data["t"],data["x"],initial) for initial in data["initial"]])
            return np.sum(((predicted-data["concentration"])/data["sigma"])**2)
        result = minimize_scalar(objective,bounds=(np.log(1e-11),np.log(1e-7)),method="bounded",options={"xatol":1e-11})
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self,t,x,initial):
        t,x,initial=np.asarray(t),np.asarray(x),np.asarray(initial)
        count=len(x)-1
        spatial_rate=4/(x[1]-x[0])**2*np.sin(np.arange(count+1)*np.pi/(2*count))**2
        coefficients=dct(initial,type=1,axis=0)
        rates=spatial_rate[:,None]*self.diffusivity*np.array([4/3,8/3])[None,:]
        return idct(np.exp(-t[:,None,None]*rates[None,:,:])*coefficients[None,:,:],type=1,axis=1)

    def predict(self,t,x,initial):
        t,x,initial=np.asarray(t),np.asarray(x),np.asarray(initial)
        if t[-1]==0:
            return initial[None,:,:].copy()
        if np.all(initial[:,0]==0) or np.all(initial[:,1]==0):
            return self._binary(t,x,initial)
        count=len(x);dx=x[1]-x[0]
        width=np.full(count,dx);width[[0,-1]]*=.5
        da,db,dc=self.diffusivity*np.array([1.,4.,2.])
        def derivative(time,flat):
            c=equilibrium(flat.reshape(count,2))
            face=(c[:-1]+c[1:])/2
            gradient=np.diff(c,axis=0)/dx
            potential_gradient=-(da*gradient[:,0]+db*gradient[:,1]-dc*gradient[:,2])/(da*face[:,0]+db*face[:,1]+dc*face[:,2])
            free_flux=-np.array([da,db])*(gradient[:,:2]+face[:,:2]*potential_gradient[:,None])
            flux=np.zeros((count+1,2))
            flux[1:-1]=free_flux
            return (-np.diff(flux,axis=0)/width[:,None]).ravel()
        sparsity=diags([np.ones(2*count-abs(i)) for i in range(-3,4)],range(-3,4),format="csr")
        result=solve_ivp(derivative,(0,t[-1]),initial.ravel(),method="BDF",t_eval=t,rtol=1e-8,atol=1e-10,jac_sparsity=sparsity)
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T.reshape(len(t),count,2)
