"""Direct lab-frame interface/current linear equations, without homogenized-tensor formulas."""
import numpy as np

PARAMETER = 'mobility'
TRUE_PARAMETER = .8


def experiment(field,fraction=.4,angle=.6,observable='longitudinal_resistivity'):
    return dict(field=float(field),fraction=float(fraction),normal_angle=float(angle),observable=observable)


def calibration_inputs():
    return [experiment(b,fraction=f) for f in [0.,1.] for b in np.linspace(-1.5,1.5,50)]


def hidden_inputs():
    return {'oblique_layers':[experiment(b) for b in np.r_[np.linspace(-2.5,-.3,12),np.linspace(.3,2.5,12)]],
            'changed_fraction':[experiment(1.8,fraction=f,angle=.9) for f in np.linspace(.15,.85,24)],
            'rotated_layers':[experiment(-2.,fraction=.65,angle=a) for a in np.linspace(.15,1.4,24)]}


def local_resistance(mu,B,sign):
    return np.array([[1.,-sign*mu*B],[sign*mu*B,1.]])/(1e4*mu)


def fields(e,mobility):
    sigma=[np.linalg.inv(local_resistance(mobility,e['field'],1)),np.linalg.inv(local_resistance(2*mobility,e['field'],-1))]
    f=e['fraction'];a=e['normal_angle'];n=np.array([np.cos(a),np.sin(a)]);t=np.array([-np.sin(a),np.cos(a)])
    # Unknowns are E_Ax,E_Ay,E_Bx,E_By. Equations enforce averaged imposed
    # current, normal-current continuity and tangential-field continuity.
    matrix=np.vstack([np.c_[f*sigma[0],(1-f)*sigma[1]],np.r_[n@sigma[0],-n@sigma[1]],np.r_[t,-t]])
    answer=np.linalg.solve(matrix,np.array([1.,0.,0.,0.]))
    return answer[:2],answer[2:],sigma,n,t


def predict(experiments,mobility):
    out=[]
    for e in experiments:
        first,second,_,_,_=fields(e,mobility)
        mean=e['fraction']*first+(1-e['fraction'])*second
        out.append(mean[0 if e['observable']=='longitudinal_resistivity' else 1])
    return np.array(out)
