"""Independent complex state-vector evolution and adaptive Gaussian integration."""
import numpy as np
from scipy.integrate import quad_vec

PARAMETER = 'noise_width'
TRUE_PARAMETER = .95
PAULI = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]], dtype=complex)


def rotation(field, duration):
    field = np.array(field, dtype=float)
    norm = np.linalg.norm(field)
    if norm == 0:
        return np.eye(2, dtype=complex)
    return np.cos(norm*duration/2)*np.eye(2)-1j*np.sin(norm*duration/2)*np.einsum('i,ijk->jk', field/norm, PAULI)


def pulse(spec):
    phase, angle = spec
    return rotation([np.cos(phase),np.sin(phase),0.],angle)


def calibration_inputs():
    return [dict(preparation=[[np.pi/2,np.pi/2],[0.,0.]],readout=[[np.pi/2,-np.pi/2],[0.,0.]],
                 segments=[dict(duration=float(t),fields=[[0.,0.,0.],[0.,0.,0.]])])
            for t in np.linspace(.05,2.,100)]


def hidden_inputs():
    return {
        'parallel_drives': [dict(preparation=[[np.pi/2,1.1],[np.pi/2,1.1]],
                                readout=[[np.pi/2,-1.1],[np.pi/2,-1.1]],
                                segments=[dict(duration=float(t),fields=[[.7,0.,.25],[.7,0.,.25]]),
                                          dict(duration=.7,fields=[[0.,.9,-.3],[0.,.9,-.3]])])
                            for t in np.linspace(.8,2.,24)],
        'opposite_readouts': [dict(preparation=[[np.pi/2,np.pi/2],[np.pi/2,np.pi/2]],
                                  readout=[[np.pi/2,-np.pi/2],[np.pi/2,np.pi/2]],
                                  segments=[dict(duration=float(t),fields=[[.45,0.,.2],[.45,0.,.2]]),
                                            dict(duration=.9,fields=[[0.,.65,0.],[0.,.65,0.]])])
                              for t in np.linspace(.8,2.,24)],
        'unequal_controls': [dict(preparation=[[np.pi/2,1.4],[np.pi/2,1.2]],
                                 readout=[[np.pi/2,-1.4],[1.3,-1.2]],
                                 segments=[dict(duration=float(t),fields=[[.5,.1,.15],[.65,0.,.2]]),
                                           dict(duration=.8,fields=[[-.15,.6,-.2],[0.,.7,-.1]])])
                             for t in np.linspace(.8,2.,24)]}


def predict(experiments, noise_width):
    preparations = [[pulse(e['preparation'][i])@np.array([1.,0.],dtype=complex) for i in [0,1]] for e in experiments]
    readouts = [[pulse(e['readout'][i]) for i in [0,1]] for e in experiments]
    def integrand(z):
        out = []
        for j,e in enumerate(experiments):
            product = 1.
            for i in [0,1]:
                state = preparations[j][i].copy()
                for segment in e['segments']:
                    field = np.array(segment['fields'][i],dtype=float)
                    field[2] += noise_width*z
                    state = rotation(field,segment['duration'])@state
                state = readouts[j][i]@state
                product *= abs(state[0])**2
            out.append(product)
        return np.array(out)*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    return quad_vec(integrand,-10.,10.,epsabs=2e-11,epsrel=2e-11)[0]
