"""Author-only common quadratic stability certificate by eigenvector cuts."""
from pathlib import Path
from itertools import product
import json,time
import numpy as np
from scipy.optimize import linprog

BOUNDS={'gamma':[.4,1.1],'kx':[1.,1.6],'ky':[2.2,2.8],'h':[.9/.8,.9/.2],'omega':[-.6,.6],'strain':[-.4,.4]}
B=.9


def drift(gamma,kx,ky,h,omega,strain):
    A=np.zeros((6,6));A[:2,2:4]=np.eye(2)
    A[2:4,:2]=-np.diag([kx,ky])-h*np.eye(2);A[2:4,2:4]=-gamma*np.eye(2);A[2:4,4:]=h*np.eye(2)
    A[4:,:2]=h/B*np.eye(2);A[4:,4:]=np.array([[strain,-omega],[omega,-strain]])-h/B*np.eye(2)
    return A


def solve(bounds,maxiter=250):
    start=time.monotonic();vertices=[dict(zip(bounds,x)) for x in product(*bounds.values())]
    mats=np.array([drift(**v) for v in vertices]);pairs=[(i,j) for i in range(6) for j in range(i,6)]
    basis=[]
    for i,j in pairs:
        e=np.zeros((6,6));e[i,j]=e[j,i]=1;basis.append(e)
    basis=np.array(basis);Q=np.array([-(basis@A+np.einsum('ij,kjl->kil',A.T,basis)) for A in mats])
    # A constraint is t <= v'P v or t <= v'[-A'P-PA]v.
    systems=np.concatenate([basis[None],Q]);cuts=[]
    def cut(system,v):cuts.append(np.r_[-np.einsum('i,kij,j->k',v,system,v),v@v])
    for system in systems:
        for v in np.eye(6):cut(system,v)
    trace=np.r_[np.trace(basis,axis1=1,axis2=2),0.]
    obj=np.r_[np.zeros(len(basis)),-1.];history=[];best=None
    for iteration in range(maxiter):
        res=linprog(obj,A_ub=np.array(cuts),b_ub=np.zeros(len(cuts)),A_eq=trace[None],b_eq=[1.],bounds=[(None,None)]*len(obj),method='highs')
        if not res.success:raise RuntimeError(res.message)
        P=np.einsum('k,kij->ij',res.x[:-1],basis);t=float(res.x[-1]);worst=1e99;added=0
        for system in systems:
            M=np.einsum('k,kij->ij',res.x[:-1],system);eig,V=np.linalg.eigh(M);worst=min(worst,float(eig[0]))
            for k in np.where(eig<t-1e-8)[0]:cut(system,V[:,k]);added+=1
        history.append({'iteration':iteration,'lp_upper_bound':t,'actual_minimum_eigenvalue':worst,'cuts':len(cuts),'added':added})
        best={'P':P.tolist(),'margin':worst,'upper_bound':t,'history':history,'bounds':bounds,'vertices':vertices,'run_seconds':time.monotonic()-start}
        if worst>1e-5 or added==0 or t<0:break
    return best


def main():
    report=solve(BOUNDS)
    Path(__file__).with_name('stability-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['history','vertices','P']},indent=2))
    print('iterations',len(report['history']))
    print(np.array(report['P']))

if __name__=='__main__':main()
