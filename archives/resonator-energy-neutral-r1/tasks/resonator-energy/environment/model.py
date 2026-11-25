import numpy as np


LEAD_IMPEDANCE = np.sqrt(2.)


def wave_data(experiment, host_stiffness):
    length = experiment['length']
    frequency = experiment['frequency']
    mass = experiment['resonator_mass']
    resonance = experiment['resonance_frequency']
    amplitude = experiment['incident_amplitude']
    density = 1+mass*resonance**2/(resonance**2-frequency**2) if mass else 1.
    wave_number = frequency*np.sqrt(density/host_stiffness)
    cosine, sine = np.cos(wave_number*length), np.sin(wave_number*length)
    impedance = 1j*frequency*LEAD_IMPEDANCE
    matrix = np.array([[impedance,host_stiffness*wave_number],
        [-host_stiffness*wave_number*sine-impedance*cosine,
         host_stiffness*wave_number*cosine-impedance*sine]])
    a,b = np.linalg.solve(matrix, [2*impedance*amplitude,0.])
    first = length/2+np.sin(2*wave_number*length)/(4*wave_number)
    second = length-first
    mixed = np.real(a.conjugate()*b)*sine**2/wave_number
    displacement = abs(a)**2*first+abs(b)**2*second+mixed
    strain = wave_number**2*(abs(a)**2*second+abs(b)**2*first-mixed)
    return density, displacement, strain


def predict_at(experiments, host_stiffness):
    result = []
    for experiment in experiments:
        density, displacement, strain = wave_data(experiment, host_stiffness)
        frequency = experiment['frequency']
        result.append((density*frequency**2*displacement+host_stiffness*strain)/4)
    return np.array(result)


class Model:
    def __init__(self):
        self.host_stiffness = None

    def fit(self, records):
        raise NotImplementedError("Fit the host stiffness from calibration records.")

    def predict(self, experiments):
        return predict_at(experiments, self.host_stiffness)
