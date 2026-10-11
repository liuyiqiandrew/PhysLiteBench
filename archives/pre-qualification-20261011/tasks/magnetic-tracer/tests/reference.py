"""Finite-mass four-dimensional OU evolution, extrapolated after measurement."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

TRUE_PARAMETER = .8


@lru_cache(1024)
def finite_mass(friction, field, temperature, kx, ky, time, cxx, cxy, cyy, mass):
    generator = np.zeros((4, 4))
    generator[:2, 2:] = np.eye(2)
    generator[2:, :2] = -np.diag([kx, ky]) / mass
    generator[2:, 2:] = np.array([[-friction, field], [-field, -friction]]) / mass
    # The full equilibrium covariance solves the underdamped covariance equation.
    # Maxwell initial velocities make the nonequilibrium increment purely positional.
    equilibrium = np.diag([temperature / kx, temperature / ky,
                           temperature / mass, temperature / mass])
    initial = np.zeros((4, 4))
    initial[:2, :2] = np.array([[cxx, cxy], [cxy, cyy]]) - equilibrium[:2, :2]
    propagator = expm(time * generator)
    evolved = propagator @ initial @ propagator.T
    covariance = equilibrium + evolved
    mechanical_area = np.zeros((4, 4))
    mechanical_area[0, 3] = mechanical_area[3, 0] = .5
    mechanical_area[1, 2] = mechanical_area[2, 1] = -.5
    primitive = solve_continuous_lyapunov(generator.T, -mechanical_area)
    area = float(np.trace(primitive @ (initial - evolved)))
    return np.array([covariance[0, 0], covariance[0, 1], covariance[1, 1], area])


def predict(experiments, friction=TRUE_PARAMETER, mass_scale=.002):
    out = []
    index = {'xx': 0, 'xy': 1, 'yy': 2, 'area': 3}
    for e in experiments:
        kx, ky = e['stiffness']
        c = e['initial_covariance']
        mass = mass_scale * friction**2 / max(kx, ky)
        values = [finite_mass(float(friction), e['field'], e['temperature'], kx, ky,
                              e['time'], c[0][0], c[0][1], c[1][1], mass / factor)
                  for factor in [1, 2, 4]]
        limiting = (values[0] - 6 * values[1] + 8 * values[2]) / 3
        out.append(limiting[index[e['readout']]])
    return np.array(out)


def experiment(time, field=0., temperature=.8, stiffness=(.8, 1.6),
               initial=((1.5, .3), (.3, .5)), readout='area'):
    return dict(time=float(time), field=float(field), temperature=float(temperature),
                stiffness=list(stiffness), initial_covariance=[list(row) for row in initial], readout=readout)


def calibration_inputs():
    return [experiment(t, 0., temp, trap, initial, readout)
            for trap, initial in [((.8, 1.6), ((1.5, .3), (.3, .5))),
                                  ((1.5, .7), ((.6, -.25), (-.25, 1.4)))]
            for temp in [.5, 1.1] for t in [.1, .25, .5, 1., 2., 3.]
            for readout in ['xx', 'xy', 'yy', 'area']]


def hidden_inputs():
    return {name: [experiment(t, field, temperature, trap, initial, readout)
                   for t in [.2, .5, 1., 2., 4.] for readout in ['xx', 'xy', 'yy', 'area']]
            for name, field, temperature, trap, initial in [
                ('area_with_relaxation', .9, .8, (.8, 1.6), ((1.5, .3), (.3, .5))),
                ('reversed_field', -1.5, 1.1, (1.5, .7), ((.6, -.25), (-.25, 1.4))),
                ('mixed_preparation', 1.8, .5, (1.1, 1.4), ((1.2, -.4), (-.4, 1.1)))
            ]}
