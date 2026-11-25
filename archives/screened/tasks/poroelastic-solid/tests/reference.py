"""Solve the stress/pressure constitutive equations without eliminating pressure."""
import numpy as np

PARAMETER = 'young_modulus'
TRUE_PARAMETER = 1450.
BOUNDS = (800., 2200.)
COMPONENTS = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']


def state(strain, young_modulus, alpha=.8, modulus=2400.):
    strain = np.asarray(strain)
    nu = .25
    shear = young_modulus/(2*(1+nu))
    lame = young_modulus*nu/((1+nu)*(1-2*nu))
    stiffness = np.eye(6)*(2*shear)
    stiffness[:3, :3] += lame
    equations = np.eye(7)
    equations[:3, 6] = alpha
    equations[6, 6] = 1/modulus
    rhs = np.r_[stiffness@strain, -alpha*np.sum(strain[:3])]
    return np.linalg.solve(equations, rhs)


def predict(experiments, young_modulus):
    return np.array([state(e['strain'], young_modulus)[COMPONENTS.index(e['component'])]
                     for e in experiments])


def calibration_inputs():
    result = []
    for amplitude in np.linspace(-.0018, .0018, 32):
        a = float(amplitude)
        for strain, component in [([a, -a, 0., 0., 0., 0.], 'xx'),
                                  ([0., a, -a, 0., 0., 0.], 'zz'),
                                  ([0., 0., 0., a, 0., 0.], 'xy'),
                                  ([0., 0., 0., 0., a, -a/2], 'xz')]:
            result.append(dict(strain=strain, component=component))
    return result


def hidden_inputs():
    hydrostatic = []
    uniaxial = []
    mixed = []
    for a in np.linspace(.0003, .0017, 24):
        a = float(a)
        for component in COMPONENTS[:3]:
            hydrostatic.append(dict(strain=[-a, -a, -a, 0., 0., 0.], component=component))
            uniaxial.append(dict(strain=[a, 0., 0., 0., 0., 0.], component=component))
            mixed.append(dict(strain=[-.7*a, a, .4*a, -.2*a, .3*a, .1*a], component=component))
    return dict(hydrostatic_compression=hydrostatic, constrained_extension=uniaxial, mixed_strain=mixed)
