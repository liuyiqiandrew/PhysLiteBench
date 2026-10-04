"""Frozen kinetic cell problems and conservative stationary spatial balance."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np

TRUE_PARAMETER = .8


def experiment(ratio,contrast,potential,winding=0,twist=0.,push=0.,observable='current'):
    return dict(ratio=ratio,contrast=contrast,potential=potential,winding=winding,twist=twist,push=push,observable=observable)


@lru_cache(None)
def cell_coefficients(ratio,contrast,winding,twist,order,offset):
    x = (np.arange(order)+offset)*2*np.pi/order
    a = 1+contrast*np.cos(x)
    ap = -contrast*np.sin(x)
    angle = winding*x+twist*np.sin(x)
    angle_prime = winding+twist*np.cos(x)
    n = len(x)
    coupling = np.zeros((n,2,2))
    coupling[:,0,0] = a*np.cos(angle)
    coupling[:,0,1] = -a*np.sin(angle)
    coupling[:,1,0] = a*np.sin(angle)
    coupling[:,1,1] = a*np.cos(angle)
    derivative = np.zeros_like(coupling)
    derivative[:,0,0] = ap*np.cos(angle)-a*angle_prime*np.sin(angle)
    derivative[:,0,1] = -ap*np.sin(angle)-a*angle_prime*np.cos(angle)
    derivative[:,1,0] = ap*np.sin(angle)+a*angle_prime*np.cos(angle)
    derivative[:,1,1] = ap*np.cos(angle)-a*angle_prime*np.sin(angle)
    relaxation = np.zeros((n,4,4))
    relaxation[:,:2,:2] = np.eye(2)
    relaxation[:,:2,2:] = -coupling
    relaxation[:,2:,2:] = np.eye(2)/ratio
    relaxation_prime = np.zeros_like(relaxation)
    relaxation_prime[:,:2,2:] = -derivative
    driving = np.diag([0.,0.,2/ratio**2,2/ratio**2])
    # Solve A C + C A^T = Q independently at every spatial node.
    operator = np.einsum('nij,kl->nikjl',relaxation,np.eye(4))+np.einsum('ij,nkl->nikjl',np.eye(4),relaxation)
    covariance = np.linalg.solve(operator.reshape(n,16,16),np.broadcast_to(driving.ravel(),(n,16))[...,None])[...,0].reshape(n,4,4)
    selector = np.c_[np.eye(2),np.zeros((2,2))]
    corrector = np.linalg.solve(relaxation.transpose(0,2,1),np.broadcast_to(selector.T,(n,4,2))).transpose(0,2,1)
    corrector_prime = np.linalg.solve(relaxation.transpose(0,2,1),(-corrector@relaxation_prime).transpose(0,2,1)).transpose(0,2,1)
    noise_drift = np.einsum('nja,na->nj',corrector_prime,covariance[:,0,:])
    diffusion = covariance[:,:2,:]@corrector.transpose(0,2,1)
    diffusion = .5*(diffusion+diffusion.transpose(0,2,1))
    # Differentiate the Lyapunov equation, not an assumed diffusion formula.
    rhs_prime = -relaxation_prime@covariance-covariance@relaxation_prime.transpose(0,2,1)
    covariance_prime = np.linalg.solve(operator.reshape(n,16,16),rhs_prime.reshape(n,16,1))[...,0].reshape(n,4,4)
    diffusion_prime = covariance_prime[:,:2,:]@corrector.transpose(0,2,1)+covariance[:,:2,:]@corrector_prime.transpose(0,2,1)
    return x,noise_drift,diffusion,diffusion_prime


def grid_prediction(e,strength,order):
    args=(e['ratio'],e['contrast'],e['winding'],e['twist'],order)
    edge,bd,k,kp=cell_coefficients(*args,.5)
    bx=e['potential']*np.sin(edge)+strength*bd[:,0]
    transport=(bx-strength*kp[:,0,0])/(strength*k[:,0,0])
    # These even amplitude/potential preparations have zero periodic x flux.
    # Integrate the local fitted-flux relation across each cell face.
    jump=2*np.pi/order*transport
    assert abs(np.sum(jump))<1e-10
    log_density=np.r_[0.,np.cumsum(jump[:-1])]
    density=np.exp(log_density-np.max(log_density));density/=np.sum(density)
    x,bd,_,_=cell_coefficients(*args,0.)
    if e['observable']=='cosine':return float(density@np.cos(x))
    return float(density@(e['push']+strength*bd[:,1]))


def predict(experiments,strength,order=256):
    return np.array([(4*grid_prediction(e,strength,2*order)-grid_prediction(e,strength,order))/3 for e in experiments])


def hidden_inputs():
    return {
        'winding':[experiment(.5,.4,.7,1,.2,.1),experiment(1.2,.5,1.2,-1,-.5,-.2),experiment(3.,.25,.4,2,.8,.2),experiment(4.,.6,1.4,-2,.5,-.1)],
        'periodic_frame':[experiment(2.,.6,.2,0,1.,.1),experiment(3.,.45,1.4,0,-1.,.2),experiment(4.,0.,1.5,0,1.,-.1),experiment(1.,.2,.8,0,-.8,.15)],
        'mixed_controls':[experiment(.25,.65,1.5,2,-1.,.3),experiment(.8,.1,.3,-2,1.,-.25),experiment(2.5,.55,.9,1,-.7,.1),experiment(4.,.3,0.,-1,.4,-.2)],
        'position_anchors':[experiment(.3,.6,.4,1,.8,.2,'cosine'),experiment(1.4,.2,1.2,-2,-.6,-.1,'cosine'),experiment(3.7,0.,.9,0,.7,.3,'cosine')]
    }


def calibration_inputs():
    inputs=[experiment(r,c,p,observable='cosine') for r in [.5,2.5] for c in [.2,.55] for p in [.3,.8,1.3]]
    inputs += [experiment(1.,0.,p,observable='cosine') for p in [.6,1.4]]
    return inputs*12


def generate_data():
    inputs=calibration_inputs();values=predict(inputs,TRUE_PARAMETER)
    rng=np.random.default_rng(927131)
    records=[dict(input=e,value=float(v+.0006*z),sigma=.0006) for e,v,z in zip(inputs,values,rng.normal(size=len(inputs)))]
    task=Path(__file__).resolve().parents[1]
    for name in ['environment/data/calibration.json','tests/data/calibration.json']:
        (task/name).write_text(json.dumps(records,indent=2)+'\n')
