from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar


@lru_cache(maxsize=256)
def diffusion_tensor(a,b,c,d,diffusivity,cutoff=20):
    if (b==0 and d==0) or (a==0 and c==0):
        return np.diag([diffusivity+(a*a+c*c/4)/(2*diffusivity),diffusivity+(b*b+d*d/4)/(2*diffusivity)])
    modes=[(i,j) for i in range(-cutoff,cutoff+1) for j in range(-cutoff,cutoff+1) if (i,j)!=(0,0)]
    lookup={p:i for i,p in enumerate(modes)};n=len(modes)
    rows=list(range(n));cols=list(range(n))
    values=[complex(diffusivity*(i*i+j*j)) for i,j in modes]
    rhs=np.zeros((n,2),complex)
    for component,coordinate,amplitudes in [(0,1,(a,c)),(1,0,(b,d))]:
        for wave,amplitude in enumerate(amplitudes,1):
            for sign in [-1,1]:
                q=[0,0];q[coordinate]=wave*sign;q=tuple(q)
                v=amplitude*sign/(2j)
                rhs[lookup[q],component]+=v
                for col,p in enumerate(modes):
                    row=lookup.get((p[0]+q[0],p[1]+q[1]))
                    if row is not None:
                        rows.append(row);cols.append(col);values.append(-1j*p[component]*v)
    matrix=sparse.coo_matrix((values,(rows,cols)),shape=(n,n)).tocsc()
    chi=spsolve(matrix,rhs)
    k2=np.array([i*i+j*j for i,j in modes])
    return diffusivity*np.eye(2)+diffusivity*np.real(chi.conj().T@(k2[:,None]*chi))


def predict_at(experiments,diffusivity):
    out=[]
    for e in experiments:
        tensor=diffusion_tensor(e['a'],e['b'],e['c'],e['d'],diffusivity)
        direction=np.array([np.cos(e['angle']),np.sin(e['angle'])])
        out.append(direction@tensor@direction)
    return np.asarray(out,dtype=float)


class Model:
    def __init__(self):
        self.diffusivity=None

    def fit(self,records):
        experiments=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def objective(D):
            residual=(predict_at(experiments,D)-values)/sigma
            return float(residual@residual)
        result=minimize_scalar(objective,bounds=(.8,1.2),method='bounded',options={'xatol':1e-13})
        choices=[.8,float(result.x),1.2]
        self.diffusivity=min(choices,key=objective)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.diffusivity)
