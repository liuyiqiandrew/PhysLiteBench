"""Direct current-operator moments and adaptive spectral quadrature."""
import numpy as np
from scipy.integrate import quad

TRUE_PARAMETER = .55
CONTACT = np.diag(np.sqrt([.7,1.1]))
RIGHT = np.diag([0.,1.])


def scattering(omega, shift, coupling):
    hamiltonian = np.array([[1.4+shift,coupling],[coupling,1.9+shift]])
    response = np.linalg.solve(1j*(hamiltonian-omega*np.eye(2))+.5*CONTACT@CONTACT, CONTACT)
    return np.eye(2)-CONTACT@response


def spectral_moments(omega, experiment, coupling):
    scattering_matrix = scattering(omega,experiment['shift'],coupling)
    energy_change = scattering_matrix.conj().T@RIGHT@scattering_matrix-RIGHT
    occupation = 1/np.expm1(omega/np.array([experiment['left_temperature'],experiment['right_temperature']]))
    mean = np.sum(np.diag(energy_change)*occupation).real
    variance = 0.
    for i in range(2):
        for j in range(2):
            variance += (energy_change[i,j]*energy_change[j,i]).real*occupation[i]*(1+occupation[j])
    return omega*mean, omega**2*variance


def predict(experiments, coupling):
    return np.array([quad(lambda omega:spectral_moments(omega,e,coupling)[0 if e['readout']=='current' else 1],*e['band'],epsabs=2e-11,epsrel=2e-11)[0]/(2*np.pi) for e in experiments])


def experiment(shift, left, right, band, readout='noise'):
    return dict(shift=shift,left_temperature=left,right_temperature=right,band=list(band),readout=readout)


def calibration_inputs():
    inputs=[]
    for repeat in range(4):
        for shift in [-.2,0.,.2]:
            for band in [[.4,3.4],[.8,2.4]]:
                for temperature in [.6,1.,1.8]:
                    inputs.append(experiment(shift,temperature,temperature,band))
                for left,right in [(1.8,.4),(.5,1.5),(2.4,.8)]:
                    inputs.append(experiment(shift,left,right,band,'current'))
    return inputs


def hidden_inputs():
    return {
        'hot_left':[experiment(s,tl,tr,band) for s,tl,tr,band in [(-.2,2.8,.2,[.4,3.4]),(0.,3.,.15,[.6,2.5]),(.2,2.5,.3,[.8,3.1]),(.1,2.8,.4,[1.,2.8])]],
        'hot_right':[experiment(s,tl,tr,band) for s,tl,tr,band in [(.2,.2,2.8,[.4,3.4]),(-.2,.3,3.,[.5,2.7]),(0.,.15,2.6,[.8,2.4]),(-.1,.4,2.8,[1.,2.8])]],
        'resolved_bands':[experiment(s,tl,tr,band) for s,tl,tr,band in [(-.2,3.,.2,[.7,1.3]),(0.,.15,2.8,[1.,1.6]),(.2,2.8,.15,[1.8,2.6]),(0.,.2,3.,[1.7,2.4])]]}
