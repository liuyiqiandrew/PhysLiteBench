"""Gaussian phase average independent of the submitted density-matrix code."""
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.spatial.transform import Rotation

TRUE_GAMMA=.43
CHI2_LIMIT=1.5
PARAMETER_LIMIT=.05
PREDICTION_LIMIT=.03
NODES,WEIGHTS=hermgauss(96)
WEIGHTS=WEIGHTS/np.sqrt(np.pi)


def reference(experiments,gamma=TRUE_GAMMA):
    result=[]
    for e in experiments:
        phi=np.sqrt(4*gamma*e['wait'])*NODES
        joint=np.ones(len(phi))
        for (p,a),(rp,ra) in zip(e['preparation'],e['readout']):
            spin=Rotation.from_rotvec(a*np.array([np.cos(p),np.sin(p),0.])).apply([0,0,1])
            noisy=Rotation.from_rotvec(np.column_stack([np.zeros(len(phi)),np.zeros(len(phi)),phi])).apply(spin)
            after=Rotation.from_rotvec(ra*np.array([np.cos(rp),np.sin(rp),0.])).apply(noisy)
            joint*=.5*(1+after[:,2])
        result.append(float(WEIGHTS@joint))
    return np.clip(result,0,1)


def hidden_groups():
    groups=[]
    for readout_phase in (np.pi/2,0.):
        groups.append([{'wait':float(t),'preparation':[[np.pi/2,np.pi/2],[np.pi/2,np.pi/2]],
                        'readout':[[readout_phase,-np.pi/2],[readout_phase,-np.pi/2]]}
                       for t in np.linspace(.1,5.,41)])
    return groups


def checked_prediction(model,experiments):
    p=np.asarray(model.predict(experiments),dtype=float)
    if p.shape!=(len(experiments),) or not np.isfinite(p).all():raise ValueError('invalid prediction')
    if np.any(p< -1e-10) or np.any(p>1+1e-10):raise ValueError('invalid probability')
    return p


def metrics(model,runs,references=None):
    gamma=float(model.gamma)
    if not np.isfinite(gamma) or gamma<=0:raise ValueError('invalid gamma')
    p=checked_prediction(model,[r['experiment'] for r in runs])
    chi2=sum(((value-r['probability'])/r['sigma'])**2 for value,r in zip(p,runs))
    if references is None:references=[reference(g) for g in hidden_groups()]
    err=[float(np.sqrt(np.mean((checked_prediction(model,g)-truth)**2))) for g,truth in zip(hidden_groups(),references)]
    return {'gamma_s_minus_1':gamma,'relative_parameter_error':abs(gamma/TRUE_GAMMA-1),
            'calibration_reduced_chi2':float(chi2/(len(runs)-1)), 'hidden_probability_rmse':err}
