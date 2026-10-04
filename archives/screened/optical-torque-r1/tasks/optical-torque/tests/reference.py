from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial.transform import Rotation

TRUE_PARAMETER = 1.1


def experiment(orientation, direction, field, axis, observable):
    return dict(orientation=np.asarray(orientation).tolist(), direction=np.asarray(direction).tolist(),
                field_real=np.real(field).tolist(), field_imag=np.imag(field).tolist(),
                axis=np.asarray(axis).tolist(), observable=observable)


def calibration_inputs():
    inputs = []
    for _ in range(4):
        for euler in [[0.,0.,0.],[.2,.4,.6],[.4,-.3,.7],[-.5,.1,.9]]:
            R = Rotation.from_euler('xyz', euler).as_matrix()
            for j in range(3):
                for amplitude in [.5,.75,1.]:
                    inputs.append(experiment(R, R[:,(j+1)%3], amplitude*R[:,j], R[:,(j+1)%3], 'force'))
    return inputs


def hidden_inputs():
    groups = {}
    for pair, name in [((0,1),'xy_polarization'),((0,2),'xz_polarization'),((1,2),'yz_polarization')]:
        group = []
        for angle, hand, euler in [(.55,1,[.2,.4,.6]),(.75,-1,[.4,-.3,.7]),(.95,1,[-.5,.1,.9])]:
            R = Rotation.from_euler('xyz', euler).as_matrix()
            i,j = pair
            field = R[:,i]*np.cos(angle)+1j*hand*R[:,j]*np.sin(angle)
            direction = np.cross(R[:,i], R[:,j])
            group.append(experiment(R, direction, field, direction, 'torque'))
        groups[name] = group
    return groups


@lru_cache(None)
def surface(radius, order):
    z,w = leggauss(order)
    phi = np.arange(2*order)*np.pi/order
    zz,pp = np.meshgrid(z,phi,indexing='ij')
    normals = np.stack([np.sqrt(1-zz*zz)*np.cos(pp), np.sqrt(1-zz*zz)*np.sin(pp), zz],axis=-1).reshape(-1,3)
    weights = np.repeat(w,2*order)*np.pi/order*radius**2
    return radius*normals,normals,weights


def stress_vectors(experiment, strength, radius=.4, order=28):
    position,normal,weights = surface(radius,order)
    R = np.array(experiment['orientation'])
    bare = R@np.diag(strength*np.array([1.,1.7,2.6]))@R.T
    alpha = np.linalg.inv(np.linalg.inv(bare)-1j*np.eye(3)/(6*np.pi))
    field = np.array(experiment['field_real'])+1j*np.array(experiment['field_imag'])
    direction = np.array(experiment['direction'])
    dipole = alpha@field
    phase = np.exp(1j*position@direction)
    electric = phase[:,None]*field
    magnetic = phase[:,None]*np.cross(direction,field)
    radial = (normal@dipole)[:,None]*normal
    scale = np.exp(1j*radius)/(4*np.pi)
    scattered_electric = scale*((dipole-radial)/radius+(3*radial-dipole)*(1/radius**3-1j/radius**2))
    scattered_magnetic = scale*(1/radius+1j/radius**2)*np.cross(normal,dipole)
    def traction(E,H):
        return .5*np.real(E*np.sum(E.conj()*normal,axis=1)[:,None]+H*np.sum(H.conj()*normal,axis=1)[:,None]-.5*normal*np.sum(abs(E)**2+abs(H)**2,axis=1)[:,None])
    incident = traction(electric,magnetic)
    scattered = traction(scattered_electric,scattered_magnetic)
    total = traction(electric+scattered_electric,magnetic+scattered_magnetic)-incident
    return weights@total, weights@np.cross(position,total), weights@np.cross(position,scattered)


def predict(experiments, strength, radius=.4, order=28):
    result = []
    for e in experiments:
        force,torque,_ = stress_vectors(e,strength,radius,order)
        result.append(np.array(e['axis'])@(force if e['observable']=='force' else torque))
    return np.array(result)
