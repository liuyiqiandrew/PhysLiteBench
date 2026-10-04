import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize_scalar

NODES, WEIGHTS = hermgauss(96)
WEIGHTS = WEIGHTS/np.sqrt(np.pi)


def rotate(vectors, fields, duration):
    """Apply a rotation about each field, with angular frequency |field|."""
    norm = np.linalg.norm(fields, axis=-1, keepdims=True)
    axis = np.divide(fields, norm, out=np.zeros_like(fields), where=norm>0)
    angle = norm*duration
    cosine, sine = np.cos(angle), np.sin(angle)
    return vectors*cosine+np.cross(axis, vectors)*sine+axis*np.sum(axis*vectors, axis=-1, keepdims=True)*(1-cosine)


def pulse(vectors, pulse_spec):
    phase, angle = pulse_spec
    return rotate(vectors, np.array([np.cos(phase), np.sin(phase), 0.]), angle)


class Model:
    def __init__(self):
        self.noise_width = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        detunings = np.sqrt(2)*self.noise_width*NODES
        for e in experiments:
            probability = []
            for bead in [0, 1]:
                vector = np.tile(pulse(np.array([0., 0., 1.]), e['preparation'][bead]), (len(NODES), 1))
                for segment in e['segments']:
                    field = np.tile(np.array(segment['fields'][bead], dtype=float), (len(NODES), 1))
                    field[:, 2] += detunings
                    vector = rotate(vector, field, segment['duration'])
                vector = pulse(vector, e['readout'][bead])
                probability.append((1+vector[:, 2])/2)
            out.append(np.dot(WEIGHTS, probability[0])*np.dot(WEIGHTS, probability[1]))
        return np.array(out)
