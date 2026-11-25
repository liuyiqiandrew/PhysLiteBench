import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

ORDER=64
SPEEDS=np.array([1.,.6])
CAPACITIES=np.array([3.,1./.6**3])


def operators(resistive_rate,normal_rate,wavenumber,order=ORDER):
    n=np.arange(order-1);off=(n+1)/np.sqrt((2*n+1)*(2*n+3))
    direction=np.diag(off,1)+np.diag(off,-1)
    streaming=np.kron(np.diag(SPEEDS),direction)
    isotropic=np.zeros((2*order,2*order));normal=np.zeros_like(isotropic)
    for i in range(2):
        isotropic[i*order,i*order]=1.
        for j in range(2):normal[i*order,j*order]=CAPACITIES[j]/CAPACITIES.sum()
        normal[i*order+1,i*order+1]=1.
    identity=np.eye(2*order)
    return -1j*wavenumber*streaming+resistive_rate*(isotropic-identity)+normal_rate*(normal-identity)


def predict_at(experiments,resistive_rate,order=ORDER):
    out=[]
    for e in experiments:
        initial=np.zeros((2,order),dtype=complex);initial[:,:3]=np.array(e['initial'])/np.sqrt([1.,3.,5.])
        if e['wavenumber']==0 and np.all(initial[:,:2]==0):
            final=initial*np.exp(-(resistive_rate+e['normal_rate'])*e['time'])
        else:
            generator=operators(resistive_rate,e['normal_rate'],e['wavenumber'],order)
            final=(expm(generator*e['time'])@initial.ravel()).reshape(2,order)
        values=final[:,e['moment']]/np.sqrt(2*e['moment']+1)
        value=CAPACITIES@values/CAPACITIES.sum() if e['branch']==-1 else values[e['branch']]
        out.append(float(value.real if e['quadrature']=='cosine' else -value.imag))
    return np.array(out)


class Model:
    def __init__(self):
        self.resistive_rate=None

    def fit(self,records):
        time=np.array([r['input']['time'] for r in records]);normal=np.array([r['input']['normal_rate'] for r in records])
        initial=[]
        for r in records:
            e=r['input'];values=np.array(e['initial'])[:,e['moment']]/(2*e['moment']+1)
            initial.append(CAPACITIES@values/CAPACITIES.sum() if e['branch']==-1 else values[e['branch']])
        initial=np.array(initial);value=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        self.resistive_rate=float(minimize_scalar(lambda rate:np.sum(((initial*np.exp(-(rate+normal)*time)-value)/sigma)**2),bounds=(.15,.6),method='bounded',options={'xatol':1e-12}).x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.resistive_rate)
