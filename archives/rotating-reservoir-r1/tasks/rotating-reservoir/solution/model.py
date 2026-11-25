import numpy as np
from scipy.linalg import solve_continuous_lyapunov

R=np.array([[0.,-1.],[1.,0.]])
B_DRAG=.9


def stationary_covariance(experiment, drag):
    k=np.diag([experiment['stiffness_x'],experiment['stiffness_y']])
    omega=experiment['angular_speed']
    drift=np.block([[np.zeros((2,2)),np.eye(2)],
                    [-k+B_DRAG*omega*R,-(drag+B_DRAG)*np.eye(2)]])
    noise=np.zeros((4,4))
    noise[2:,2:]=2*(drag*experiment['temperature_a']+
                    B_DRAG*experiment['temperature_b'])*np.eye(2)
    return solve_continuous_lyapunov(drift,-noise)


def heat_rate(experiment, drag):
    covariance=stationary_covariance(experiment,drag)
    omega=experiment['angular_speed']
    angular_moment=np.trace(R@covariance[:2,2:])
    speed_squared=np.trace(covariance[2:,2:])
    position_squared=np.trace(covariance[:2,:2])
    return B_DRAG*(speed_squared-2*omega*angular_moment+omega**2*position_squared)-2*B_DRAG*experiment['temperature_b']


def predict_at(experiments, drag):
    return np.array([heat_rate(e,drag) for e in experiments])


class Model:
    def __init__(self):
        self.drag=None

    def fit(self, records):
        contrast=np.array([(r['input']['temperature_a']-r['input']['temperature_b'])/r['sigma'] for r in records])
        observed=np.array([r['value']/r['sigma'] for r in records])
        conductance=float(contrast@observed/(contrast@contrast))
        self.drag=conductance*B_DRAG/(2*B_DRAG-conductance)
        return self

    def predict(self, experiments):
        return predict_at(experiments,self.drag)
