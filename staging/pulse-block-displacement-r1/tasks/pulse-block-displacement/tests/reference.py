"""Independent Maxwell boundary solve and temporal energy-flux centroids."""
from functools import lru_cache
import numpy as np

TRUE_PARAMETER = .31


def transmission(omega, strength, length):
    eps = 1+strength/(1-omega*omega)
    mu = 1+strength/(1-omega*omega)
    k = omega*np.sqrt(eps*mu)
    impedance = np.sqrt(mu/eps)
    ep, em = np.exp(1j*k*length), np.exp(-1j*k*length)
    matrix = np.zeros((len(omega),4,4), dtype=complex)
    matrix[:,0,:] = [-1,1,1,0]
    matrix[:,1,0] = 1
    matrix[:,1,1] = 1/impedance
    matrix[:,1,2] = -1/impedance
    matrix[:,2,1] = ep; matrix[:,2,2] = em; matrix[:,2,3] = -1
    matrix[:,3,1] = ep/impedance; matrix[:,3,2] = -em/impedance; matrix[:,3,3] = -1
    rhs = np.zeros((len(omega),4,1), dtype=complex)
    rhs[:,0,0] = rhs[:,1,0] = 1
    solution = np.linalg.solve(matrix, rhs)[...,0]
    return solution[:,3], solution[:,0]


@lru_cache(maxsize=2048)
def temporal_readings(strength, center, width, length, samples=16384, step=.5):
    offsets = 2*np.pi*np.fft.fftfreq(samples, d=step)
    active = np.abs(offsets)<width
    incident = np.zeros(samples, dtype=complex)
    incident[active] = (1-(offsets[active]/width)**2)**4
    amplitude, reflected = transmission(center+offsets[active], strength, length)
    outgoing = incident.copy()
    outgoing[active] *= amplitude
    vacuum = incident*np.exp(1j*offsets*length)
    times = (np.arange(samples)-samples//2)*step

    def moment(spectrum):
        flux = np.abs(np.fft.fftshift(np.fft.fft(spectrum)))**2
        return float(np.dot(times,flux)/flux.sum())

    start, finish, free = moment(incident), moment(outgoing), moment(vacuum)
    # The final isolated-system center-of-energy balance converts the outgoing
    # energy centroid's lag relative to vacuum into Mc^2*DeltaX/E, with c=1.
    return finish-start, finish-free


def predict(experiments, strength, samples=16384, step=.5):
    return np.array([temporal_readings(strength,e['center'],e['width'],e['length'],samples,step)
                     [0 if e['mode']=='transit' else 1] for e in experiments])


def calibration_inputs():
    return [{'mode':'transit','center':c,'width':w,'length':length}
            for c in [1.5,1.6,1.7] for w in [.04,.1] for length in [.5,1.,2.]]


def hidden_inputs():
    groups = {}
    for name, center in [('lower_band',1.51),('middle_band',1.60),('upper_band',1.69)]:
        groups[name] = [{'mode':'displacement','center':center,'width':w,'length':length}
                        for w in [.04,.065,.095] for length in [.6,1.2,1.8]]
    groups['transit_anchor'] = [{'mode':'transit','center':c,'width':w,'length':length}
                               for c,w,length in [(1.54,.055,.7),(1.63,.085,1.3),(1.66,.075,1.7)]]
    return groups
