import numpy as np
from scipy.linalg import expm


def generator(stage, kappa, chi):
    energy=np.asarray(stage['energies']); barrier=np.asarray(stage['barriers']); affinity=stage['affinity']
    matrix=np.zeros((3,3))
    for i in range(3):
        j=(i+1)%3
        forward=kappa*np.exp(energy[i]-barrier[i]+affinity/6)
        backward=kappa*np.exp(energy[j]-barrier[i]-affinity/6)
        matrix[j,i]+=forward*np.exp(chi/3)
        matrix[i,j]+=backward*np.exp(-chi/3)
        matrix[i,i]-=forward
        matrix[j,j]-=backward
    return matrix


def cycle(e,kappa,chi):
    result=np.eye(3)
    for s in e['stages']:
        result=expm(s['duration']*generator(s,kappa,chi))@result
    return result


def response(e,kappa,renew=False):
    tilted=cycle(e,kappa,e['field'])
    if not renew:
        return float(np.log(np.max(np.linalg.eigvals(tilted).real)))
    ordinary=cycle(e,kappa,0.)
    equation=ordinary-np.eye(3);equation[-1,:]=1
    p=np.linalg.solve(equation,np.array([0.,0.,1.]))
    return float(np.log(np.sum(tilted@p)))


if __name__=='__main__':
    rng=np.random.default_rng(611101)
    found=[]
    for _ in range(600):
        stages=[dict(energies=rng.uniform(-1.5,1.5,3).tolist(),barriers=rng.uniform(0,3,3).tolist(),affinity=float(rng.uniform(-1,2)),duration=float(rng.uniform(.1,1.2))) for i in range(3)]
        e=dict(stages=stages,field=1.4)
        truth=response(e,1.1);approx=response(e,1.1,True)
        if .04<truth<.5:
            found.append((abs(approx/truth-1),truth,approx,e))
    for row in sorted(found,key=lambda x:x[0],reverse=True)[:8]:
        print(row)
