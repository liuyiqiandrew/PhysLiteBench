"""Prediction model for the gyroscopic thermal-motion apparatus."""
import numpy as np

J = np.array([[0., 1.], [-1., 0.]])


def stiffness(experiment):
    angle = experiment['trap_angle']
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return rotation @ np.diag([experiment['stiffness_1'],
                               experiment['stiffness_2']]) @ rotation.T


def susceptibility(experiment, drag):
    omega = experiment['frequency']
    inverse = (stiffness(experiment)-omega**2*np.eye(2)
               -1j*omega*(drag*np.eye(2)-experiment['field']*J))
    return np.linalg.inv(inverse)


def spectrum(experiment, drag):
    omega = experiment['frequency']
    temperature = experiment['temperature']
    if omega == 0:
        compliance = np.linalg.inv(stiffness(experiment))
        return 2*drag*temperature*compliance@compliance
    response = susceptibility(experiment, drag)
    response = (response+response.T)/2
    return (2*temperature/omega)*response.imag


def predict_at(experiments, drag):
    values = []
    for experiment in experiments:
        omega = experiment['frequency']
        sensor = np.array([1., experiment['weight']*
                           np.exp(1j*omega*experiment['delay'])])
        values.append(float(np.real(sensor@spectrum(experiment, drag)@
                                    sensor.conj())))
    return np.asarray(values)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        from scipy.optimize import minimize_scalar
        records = list(records)
        inputs = [record['input'] for record in records]
        measured = np.array([record['value'] for record in records])
        sigma = np.array([record['sigma'] for record in records])
        def loss(drag):
            residual = (predict_at(inputs, drag)-measured)/sigma
            return float(residual@residual)
        result = minimize_scalar(loss, bounds=(.4, 1.1), method='bounded',
                                 options={'xatol': 1e-12})
        self.drag = float(result.x)
        return self

    def predict(self, experiments):
        if self.drag is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.drag)
