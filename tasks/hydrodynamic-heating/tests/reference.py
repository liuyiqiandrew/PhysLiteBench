"""First-order Maxwell–fluid boundary problem and total energy flux."""
import numpy as np
from scipy.linalg import expm

TRUE_PARAMETER = 1.


def boundary_state(experiment, plasma):
    omega=experiment['frequency']
    lateral=omega*np.sin(experiment['angle'])
    cosine=np.cos(experiment['angle'])
    damping=.06
    pressure_speed=.1
    def derivative(y):
        ex,magnetic,current_normal,charge=y
        ez=(current_normal-1j*lateral*magnetic)/(1j*omega)
        current_parallel=(plasma**2*ex-1j*pressure_speed**2*lateral*charge)/(damping-1j*omega)
        return np.array([1j*omega*magnetic+1j*lateral*ez,
                         1j*omega*ex-current_parallel,
                         1j*omega*charge-1j*lateral*current_parallel,
                         (plasma**2*ez-(damping-1j*omega)*current_normal)/pressure_speed**2])
    generator=np.column_stack([derivative(column) for column in np.eye(4)])
    transition=expm(generator*experiment['thickness'])
    boundary=np.array([[1,cosine,0,0],np.array([1,-cosine,0,0])@transition,[0,0,1,0],transition[2]])
    initial=np.linalg.solve(boundary,[2*cosine,0,0,0])
    return generator,initial


def absorbed_fraction(experiment, plasma):
    generator,initial=boundary_state(experiment,plasma)
    def total_flux(z):
        ex,magnetic,current_normal,charge=expm(generator*z)@initial
        electromagnetic=ex*np.conj(magnetic)
        mechanical=.1**2/plasma**2*charge*np.conj(current_normal)
        return np.real(electromagnetic+mechanical)/np.cos(experiment['angle'])
    low,high=np.array(experiment['window'])*experiment['thickness']
    return total_flux(low)-total_flux(high)


def predict(experiments, plasma):
    cache={}
    values=[]
    for experiment in experiments:
        key=(experiment['frequency'],experiment['thickness'],experiment['angle'],tuple(experiment['window']))
        if key not in cache:cache[key]=absorbed_fraction(experiment,plasma)
        values.append(cache[key])
    return np.array(values)


def experiment(thickness,frequency,angle,window):
    return dict(thickness=thickness,frequency=frequency,angle=angle,window=list(window))


def calibration_inputs():
    return [experiment(d,w,0.,window) for repeat in range(4)
            for d in [.25,.45,.75] for w in [.75,.95,1.15,1.4]
            for window in [[0.,.3],[.3,.7],[.7,1.],[0.,1.]]]


def hidden_inputs():
    return {
        'front_region':[experiment(d,w,a,[0.,.3]) for d,w,a in [(.25,.8,.9),(.25,1.4,.9),(.5,1.05,.9),(.8,1.2,.75)]],
        'interior_region':[experiment(d,w,a,[.3,.7]) for d,w,a in [(.25,1.2,.9),(.25,1.4,.9),(.5,1.05,.8),(.8,1.2,.75)]],
        'rear_region':[experiment(d,w,a,[.7,1.]) for d,w,a in [(.25,.8,-.9),(.25,1.4,-.9),(.5,1.05,-.9),(.8,1.2,-.75)]]}
