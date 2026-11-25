"""Interface-matched interior wave density, independent of scattering derivatives."""
import numpy as np
from numpy.polynomial.legendre import leggauss

TRUE_PARAMETER=.9
NODES,WEIGHTS=leggauss(48)


def amplitudes(energy,potential,width):
    k=np.sqrt(2*energy);q=np.sqrt(2*(energy-potential)+0j)
    cosine=np.cos(q*width);sine_over_q=width*np.sinc(q*width/np.pi)
    q_sine=q*np.sin(q*width)
    # Unknowns: reflected amplitude, transmitted amplitude, interior cosine
    # amplitude, and interior sine-over-q amplitude.
    matrix=np.array([[-1,0,1,0],[1j*k,0,0,1],
                     [0,-1,cosine,sine_over_q],[0,-1j*k,-q_sine,cosine]],complex)
    result=np.linalg.solve(matrix,np.array([1,1j*k,0,0]))
    return result,q


def population(energy,potential,width,nodes=NODES,weights=WEIGHTS):
    coefficients,q=amplitudes(energy,potential,width)
    x=(nodes+1)*width/2
    wave=coefficients[2]*np.cos(q*x)+coefficients[3]*x*np.sinc(q*x/np.pi)
    return float(width/2*np.dot(weights,abs(wave)**2)/np.sqrt(2*energy))


def weak_absorption(energy,potential,width,eta):
    coefficients,_=amplitudes(energy,potential-1j*eta,width)
    reflection,transmission=coefficients[:2]
    return float((1-abs(reflection)**2-abs(transmission)**2)/(2*eta))


def predict(experiments,width):
    return np.array([population(e['energy'],e['potential'],width) for e in experiments])


def experiment(energy,potential):return dict(energy=float(energy),potential=float(potential))


def calibration_inputs():return [experiment(e,0.) for _ in range(2) for e in np.linspace(.6,3.2,72)]


def hidden_inputs():
    return {'evanescent':[experiment(e,v) for e in [.55,.85,1.15] for v in [2.2,3.6,4.4]],
            'barrier_top':[experiment(v*factor,v) for v in [1.1,1.7,2.5] for factor in [.85,1.,1.15]],
            'propagating':[experiment(e,v) for e in [1.5,2.2,3.2] for v in [.5,1.,1.4]]}
