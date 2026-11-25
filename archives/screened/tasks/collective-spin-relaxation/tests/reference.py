"""Stationary probabilities from the connected spin-one transition graph."""
import numpy as np

PARAMETER = 'temperature'
TRUE_PARAMETER = .7


def experiment(field,observable='magnetization',selection='all'):
    return dict(field=float(field),observable=observable,selection=selection)


def calibration_inputs():
    return [experiment(h,selection='triplet') for h in np.r_[np.linspace(-1.5,-.15,50),np.linspace(.15,1.5,50)]]


def hidden_inputs():
    fields = np.r_[np.linspace(-1.4,-.2,16),np.linspace(.2,1.4,16)]
    return {observable:[experiment(h,observable) for h in fields]
            for observable in ['magnetization','parallel_probability','spin_correlation']}


def predict(experiments,temperature):
    out = []
    for e in experiments:
        m = np.array([1.,0.,-1.])
        weight = np.exp((e['field']*m-abs(e['field']))/temperature)
        p = weight/weight.sum()
        values = {'magnetization':m,'parallel_probability':m*m,'spin_correlation':np.full(3,.25)}
        out.append(float(p@values[e['observable']]))
    return np.array(out)
