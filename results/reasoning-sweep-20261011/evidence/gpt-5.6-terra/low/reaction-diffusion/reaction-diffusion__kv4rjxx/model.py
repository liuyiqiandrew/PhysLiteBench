"""Electroneutral, locally equilibrated transport model for the cell."""
import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.sparse import diags

ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
LENGTH = 1e-3

def equilibrium(totals):
    """Free A, free B, free C, P at equilibrium for total A/B."""
    z = np.asarray(totals, float)
    if z.shape[-1] != 2: raise ValueError("totals need final dimension 2")
    a, b = z[..., 0], z[..., 1]
    p = np.zeros_like(a); upper = np.minimum(a, b)
    for _ in range(50):
        af, bf = a-p, b-p; c = af+bf
        step = (p-ASSOCIATION*af*bf*c*c)/(1+ASSOCIATION*(c**3+4*af*bf*c))
        new = np.clip(p-step, 0., upper)
        if np.max(np.abs(new-p), initial=0.) < 2e-14:
            p = new; break
        p = new
    return np.stack((a-p, b-p, a+b-2*p, p), axis=-1)

class TransportModel:
    def __init__(self): self.diffusivity = None

    @staticmethod
    def _binary(t, x, initial, diffusivity):
        n = len(x)-1; dx = x[1]-x[0]
        lam = 4/dx**2*np.sin(np.arange(n+1)*np.pi/(2*n))**2
        co = dct(initial, type=1, axis=0)
        rate = lam[:,None]*diffusivity*np.array([4/3, 8/3])
        return idct(np.exp(-t[:,None,None]*rate[None])*co[None], type=1, axis=1)

    def fit(self, data):
        x=np.asarray(data['x'],float); t=np.asarray(data['t'],float)
        initial=np.asarray(data['initial'],float); observed=np.asarray(data['concentration'],float)
        sigma=np.asarray(data['sigma'],float)
        if initial.ndim != 3 or observed.shape[:1] != initial.shape[:1] or sigma.shape != observed.shape:
            raise ValueError('calibration arrays have incompatible shapes')
        if len(x)<2 or np.any(sigma<=0): raise ValueError('invalid calibration mesh or uncertainty')
        def objective(log_d):
            pred=np.array([self._binary(t,x,v,np.exp(log_d)) for v in initial])
            return float(np.sum(((pred-observed)/sigma)**2))
        result=minimize_scalar(objective,bounds=(np.log(1e-13),np.log(1e-6)),method='bounded',options={'xatol':1e-12})
        if not result.success or not np.isfinite(result.fun): raise RuntimeError('could not fit diffusivity')
        self.diffusivity=float(np.exp(result.x)); return self

    def predict(self,t,x,initial):
        if self.diffusivity is None: raise RuntimeError('fit must be called before predict')
        t=np.asarray(t,float); x=np.asarray(x,float); initial=np.asarray(initial,float)
        if initial.shape != (len(x),2) or len(x)<2 or np.any(np.diff(x)<=0): raise ValueError('invalid mesh or initial shape')
        if t.ndim!=1 or not len(t) or np.any(t<0) or np.any(np.diff(t)<0): raise ValueError('t must be increasing and nonnegative')
        if np.any(initial<0): raise ValueError('negative concentrations are invalid')
        if np.all(initial[:,0]==0) or np.all(initial[:,1]==0): return self._binary(t,x,initial,self.diffusivity)
        n=len(x); dx=x[1]-x[0]; volume=np.full(n,dx); volume[[0,-1]]*=.5
        da,db,dc=self.diffusivity*np.array([1.,4.,2.])
        def rhs(_,flat):
            s=equilibrium(flat.reshape(n,2)); face=(s[:-1]+s[1:])/2; grad=np.diff(s,axis=0)/dx
            field=-(da*grad[:,0]+db*grad[:,1]-dc*grad[:,2])/(da*face[:,0]+db*face[:,1]+dc*face[:,2])
            ionic=-np.array([da,db])*(grad[:,:2]+face[:,:2]*field[:,None])
            pflux=-self.diffusivity*NEUTRAL_MOBILITY*grad[:,3]
            flux=np.zeros((n+1,2)); flux[1:-1]=ionic+pflux[:,None]
            return (-np.diff(flux,axis=0)/volume[:,None]).ravel()
        sparse=diags([np.ones(2*n-abs(k)) for k in range(-3,4)],range(-3,4),format='csr')
        out=solve_ivp(rhs,(0,float(t[-1])),initial.ravel(),t_eval=t,method='BDF',rtol=1e-8,atol=1e-10,jac_sparsity=sparse)
        if not out.success: raise RuntimeError(out.message)
        return out.y.T.reshape(len(t),n,2)

    def correlate(self,t,totals,harmonic=1):
        if self.diffusivity is None: raise RuntimeError('fit must be called before correlate')
        t=np.asarray(t,float); totals=np.asarray(totals,float)
        if t.ndim!=1 or np.any(t<0) or np.any(np.diff(t)<0): raise ValueError('t must be increasing and nonnegative')
        if totals.shape!=(2,) or np.any(totals<.2) or np.any(totals>2): raise ValueError('totals must be in [.2,2]')
        if isinstance(harmonic,bool) or int(harmonic)!=harmonic or not 1<=int(harmonic)<=8: raise ValueError('invalid harmonic')
        c=equilibrium(totals)
        q=np.array([1.,1.,-1.,0.]); P=np.diag(c)
        # Measured totals are free cation plus neutral-cluster content.  The
        # free anion is already constrained by electroneutrality (c_C=a+b).
        S=np.array([[1.,0.],[0.,1.],[0.,0.],[1.,1.]])
        chi_species=P-np.outer(P@q,P@q)/(q@P@q)
        mobile=np.diag(self.diffusivity*np.array([1.,4.,2.,.15])*c)
        mobile-=np.outer(mobile@q,mobile@q)/(q@mobile@q)
        chi=S.T@chi_species@S; mobility=S.T@mobile@S
        rate=(int(harmonic)*np.pi/LENGTH)**2*mobility@np.linalg.inv(chi)
        return np.array([expm(-lag*rate)@np.diag(totals) for lag in t])
