"""Four complex interface equations for electric and magnetic field continuity."""
import numpy as np

PARAMETER='thickness'
TRUE_PARAMETER=.083


def experiment(wavelength,angle=.3,exit_index=1.,observable='transmittance'):
    return dict(wavelength=float(wavelength),angle=float(angle),exit_index=float(exit_index),observable=observable)


def calibration_inputs():
    return [experiment(wavelength,angle=angle) for angle in [0.,.3,.6,.9] for wavelength in np.linspace(.55,1.1,25)]


def hidden_inputs():
    return {'substrate_1_3':[experiment(w,exit_index=1.3) for w in np.linspace(.5,1.,24)],
            'substrate_1_8':[experiment(w,angle=.8,exit_index=1.8) for w in np.linspace(.5,1.,24)],
            'angular_scan':[experiment(.7,angle=a,exit_index=1.6) for a in np.linspace(0.,1.,24)]}


def fields(e,thickness):
    q=np.sin(e['angle']);left=np.sqrt(1-q*q);film=np.sqrt(2.1**2-q*q);right=np.sqrt(e['exit_index']**2-q*q)
    forward=np.exp(2j*np.pi*film*thickness/e['wavelength']);backward=1/forward
    matrix=np.array([[1,-1,-1,0],[-left,-film,film,0],[0,forward,backward,-1],[0,film*forward,-film*backward,-right]],complex)
    r,a,b,t=np.linalg.solve(matrix,np.array([-1,-left,0,0],complex))
    return r,t,left,right,matrix@np.array([r,a,b,t])-np.array([-1,-left,0,0])


def predict(experiments,thickness):
    out=[]
    for e in experiments:
        r,t,left,right,_=fields(e,thickness)
        reflected=float(abs(r)**2)
        transmitted=float(right*abs(t)**2/left)
        out.append(reflected if e['observable']=='reflectance' else transmitted)
    return np.array(out)
