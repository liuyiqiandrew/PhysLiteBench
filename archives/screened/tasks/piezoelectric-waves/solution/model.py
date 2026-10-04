import numpy as np
from scipy.optimize import minimize_scalar

DENSITY = 6000.
SHEAR_MODULUS = 20e9
DIELECTRIC = np.diag([8e-9,8e-9,1e-8])
PIEZOELECTRIC = np.zeros((3,3,3))
PIEZOELECTRIC[0,0,2] = PIEZOELECTRIC[0,2,0] = 12.
PIEZOELECTRIC[1,1,2] = PIEZOELECTRIC[1,2,1] = 12.
PIEZOELECTRIC[2,0,0] = PIEZOELECTRIC[2,1,1] = -7.
PIEZOELECTRIC[2,2,2] = 18.


def predict_at(experiments,lame_parameter):
    out=[]
    for e in experiments:
        n=np.asarray(e['direction'],dtype=float)
        mechanical=SHEAR_MODULUS*np.eye(3)+(lame_parameter*1e9+SHEAR_MODULUS)*np.outer(n,n)
        polarization=np.einsum('kij,i->kj',PIEZOELECTRIC,n)
        coupling=n@polarization
        acoustic=mechanical+np.outer(coupling,coupling)/(n@DIELECTRIC@n)
        speeds=np.sqrt(np.linalg.eigvalsh(acoustic)/DENSITY)
        out.append(speeds[e['branch']])
    return np.array(out)


class Model:
    def __init__(self):
        self.lame_parameter=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def loss(parameter):
            return np.sum(((predict_at(inputs,parameter)-values)/sigma)**2)
        result=minimize_scalar(loss,bounds=(24.,50.),method='bounded',options={'xatol':1e-11})
        self.lame_parameter=float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.lame_parameter)
