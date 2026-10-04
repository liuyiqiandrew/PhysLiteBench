import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm

PARAMETER = 'relaxation_time'
TRUE_PARAMETER = .8


def calibration_inputs():
    return [dict(initial_covariance=[[1.5,0.],[0.,.5]],segments=[dict(duration=float(t),gradient=[[0.,-float(w)],[float(w),0.]])],observable='normal_stress')
            for t,w in zip(np.geomspace(.05,2.4,100),np.linspace(.15,1.05,100))]


def hidden_inputs():
    identity = [[1.,0.],[0.,1.]]
    return {
        'shear_normal_stress':[dict(initial_covariance=identity,segments=[dict(duration=float(t),gradient=[[0.,.65],[0.,0.]])],observable='normal_stress') for t in np.linspace(1.,6.,9)],
        'planar_extension':[dict(initial_covariance=identity,segments=[dict(duration=float(t),gradient=[[.34,0.],[0.,-.34]])],observable='normal_stress') for t in np.linspace(.5,6.,9)],
        'changed_axis':[dict(initial_covariance=identity,segments=[dict(duration=2.5,gradient=[[0.,.65],[0.,0.]]),dict(duration=float(t),gradient=[[0.,0.],[.65,0.]])],observable='shear_stress') for t in np.linspace(.3,3.5,9)],
    }


def connector_covariance(experiment, relaxation_time, quadrature=32):
    nodes,weights = leggauss(quadrature)
    covariance = np.array(experiment['initial_covariance'],dtype=float)
    for segment in experiment['segments']:
        duration = segment['duration']
        drift = np.array(segment['gradient'])-np.eye(2)/(2*relaxation_time)
        fundamental = expm(drift*duration)
        thermal = np.zeros((2,2))
        # Independent Gaussian increments of the connector are transported
        # by the affine bead flow from their injection time to the readout.
        for node,weight in zip(nodes,weights):
            propagator = expm(drift*(duration*(node+1)/2))
            thermal += weight*(propagator@propagator.T)*duration/(2*relaxation_time)
        covariance = fundamental@covariance@fundamental.T+thermal
    return covariance


def predict(experiments, relaxation_time, quadrature=32):
    out = []
    for e in experiments:
        c = connector_covariance(e,relaxation_time,quadrature)
        if e['observable'] == 'shear_stress':
            value = c[0,1]
        elif e['observable'] == 'normal_stress':
            value = c[0,0]-c[1,1]
        else:
            value = c[0,0]+c[1,1]-2
        out.append(float(value))
    return np.array(out)
