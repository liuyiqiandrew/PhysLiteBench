"""Initial-Maxwellian velocity quadrature propagated by gyro and bounce actions."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss

PARAMETER='pressure'
TRUE_PARAMETER=.055


@lru_cache(16)
def particles(pressure,order):
    nodes,weights=hermgauss(order)
    speeds=np.sqrt(2*pressure)*nodes
    velocity=np.array(np.meshgrid(speeds,speeds,speeds,indexing='ij')).reshape(3,-1).T
    probability=np.einsum('i,j,k->ijk',weights,weights,weights).ravel()/np.pi**1.5
    return velocity,probability


def predict(experiments,pressure=TRUE_PARAMETER,order=10):
    velocity,probability=particles(pressure,order)
    gyro_action=(velocity[:,0]**2+velocity[:,1]**2)/2
    gyro_phase=np.arctan2(velocity[:,1],velocity[:,0])
    initial_length=100.
    bounce_action=2*abs(velocity[:,2])*initial_length
    out=[]
    for e in experiments:
        radius_ratio=e['radius_ratio'];length=initial_length*e['length_ratio']
        field=1/radius_ratio**2
        perpendicular_speed=np.sqrt(2*gyro_action*field)
        parallel_velocity=np.sign(velocity[:,2])*bounce_action/(2*length)
        detector_velocity=(np.sin(e['theta'])*perpendicular_speed*np.cos(gyro_phase)
                           +np.cos(e['theta'])*parallel_velocity)
        number_density=initial_length/(radius_ratio**2*length)
        particle_flux=number_density*np.dot(probability,detector_velocity**2)
        field_vector=np.array([0.,0.,field]);normal=np.array([np.sin(e['theta']),0.,np.cos(e['theta'])])
        magnetic_flux=.5*np.dot(field_vector,field_vector)-np.dot(normal,field_vector)**2
        out.append(particle_flux+magnetic_flux)
    return np.array(out)


def reading(a,b,theta):
    return dict(radius_ratio=float(a),length_ratio=float(b),theta=float(theta))


def calibration_inputs():
    return [reading(s,s,theta) for s in np.linspace(.75,1.25,20) for theta in [0.,np.pi/4,np.pi/2]]


def hidden_inputs():
    return {name:[reading(a,b,theta) for a,b in geometries for theta in [.71,np.pi/4,.86]]
            for name,geometries in [
                ('radial_compression',[(a,b) for a in [.7,.8,.9] for b in [1.1,1.3]]),
                ('parallel_compression',[(a,b) for a in [1.1,1.3] for b in [.7,.8,.9]]),
                ('volume_preserving',[(a,1/a**2) for a in [.89,.94,1.1,1.18]])]}
