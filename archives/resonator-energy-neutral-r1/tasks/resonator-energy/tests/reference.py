"""Coupled host/resonator finite elements and their mechanical energy."""
import numpy as np
from scipy.sparse import diags, bmat
from scipy.sparse.linalg import spsolve
from functools import lru_cache

TRUE_PARAMETER = 1.13


def experiment(length, frequency, mass=0., resonance=2., amplitude=.03):
    return dict(length=length, frequency=frequency, resonator_mass=mass,
                resonance_frequency=resonance, incident_amplitude=amplitude)


def calibration_inputs():
    return [experiment(length, frequency, amplitude=amplitude)
            for repeat in range(2) for length in [.6,1.1,1.8]
            for frequency in [.35,.65,1.,1.4,1.8,2.]
            for amplitude in [.02,.03,.04]]


def hidden_inputs():
    result = {}
    for mass,name in [(.25,'light_attachments'),(.65,'moderate_attachments'),(1.1,'heavy_attachments')]:
        result[name] = [experiment(length,ratio*resonance,mass,resonance,amplitude)
                       for length,resonance,ratio in [(.7,1.6,.60),(1.1,1.9,.72),(1.7,2.2,.82)]
                       for amplitude in [.02,.03,.04]]
    return result


@lru_cache(None)
def matrices(length, cells):
    spacing=length/cells; nodes=cells+1
    diagonal=np.r_[1.,np.repeat(2.,nodes-2),1.]
    mass=diags([np.ones(nodes-1),2*diagonal,np.ones(nodes-1)],[-1,0,1],format='csc')*spacing/6
    stiffness=diags([-np.ones(nodes-1),diagonal,-np.ones(nodes-1)],[-1,0,1],format='csc')/spacing
    boundary=diags(np.r_[1.,np.zeros(nodes-2),1.],format='csc')
    return mass,stiffness,boundary


def solve(experiment, host_stiffness=TRUE_PARAMETER, cells=1024):
    length=experiment['length']; frequency=experiment['frequency']; attached=experiment['resonator_mass']
    resonance=experiment['resonance_frequency']; amplitude=experiment['incident_amplitude']
    mass,stiffness,boundary=matrices(length,cells)
    spring=attached*resonance**2; impedance=1j*frequency*np.sqrt(2.)
    host=host_stiffness*stiffness+(spring-frequency**2)*mass-impedance*boundary
    rhs=np.zeros(cells+1,complex);rhs[0]=-2*impedance*amplitude
    if attached:
        operator=bmat([[host,-spring*mass],[-spring*mass,(spring-attached*frequency**2)*mass]],format='csc')
        state=spsolve(operator,np.r_[rhs,np.zeros(cells+1)])
        displacement=state[:cells+1];resonator=state[cells+1:]
    else:
        displacement=spsolve(host,rhs);resonator=np.zeros_like(displacement)
    def quadratic(v, matrix):return float(np.vdot(v,matrix@v).real)
    energy=(frequency**2*quadratic(displacement,mass)+host_stiffness*quadratic(displacement,stiffness)
            +attached*frequency**2*quadratic(resonator,mass)+spring*quadratic(resonator-displacement,mass))/4
    return energy,displacement,resonator


def predict(experiments, host_stiffness=TRUE_PARAMETER, cells=1024):
    return np.array([solve(e,host_stiffness,cells)[0] for e in experiments])
