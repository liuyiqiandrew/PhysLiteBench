from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scipy.special import expit


@lru_cache(1)
def _grid():
    nodes, weights = np.polynomial.legendre.leggauss(192)
    angles = np.pi*(nodes+1)/2
    return angles, weights*np.pi/2


@lru_cache(512)
def _occupation(energy, temperature, contact):
    angles, weights = _grid()
    band_energy = -2*np.cos(angles)
    sine = np.sin(angles)
    filling = expit(-band_energy/temperature)
    surface = (band_energy-2j*sine)/2
    spectral = -np.imag(1/(band_energy-energy-contact**2*surface))/np.pi
    result = float(weights@(2*sine*spectral*filling))
    def surface_outside(value):
        return (value-np.sign(value)*np.sqrt(value**2-4))/2
    def denominator(value):
        return value-energy-contact**2*surface_outside(value)
    poles = []
    if contact**2 > 2-energy:
        poles.append(brentq(denominator, 2+1e-12, 2+abs(energy)+2*contact))
    if contact**2 > 2+energy:
        poles.append(brentq(denominator, -2-abs(energy)-2*contact, -2-1e-12))
    for pole in poles:
        derivative = (1-abs(pole)/np.sqrt(pole**2-4))/2
        weight = 1/(1-contact**2*derivative)
        result += weight*expit(-pole/temperature)
    return float(result)


class Model:
    def __init__(self):
        self.coupling_scale = .8

    def fit(self, records):
        """Fit the common coupling scale to calibration observations.

        The calibration uncertainty is supplied with each observation, so the
        natural one-parameter fit is the bounded weighted least-squares
        estimate.  Keeping the parameter bounded is important here: the
        apparatus specification only identifies scales in ``[.65, .95]`` and
        it also makes fitting well behaved for unusually noisy records.
        """
        records = list(records)
        if not records:
            return self

        inputs = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records],
                            dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records],
                            dtype=float)

        if (not np.all(np.isfinite(values)) or
                not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0)):
            raise ValueError('calibration values and sigmas must be finite, '
                             'with positive sigmas')

        def objective(scale):
            prediction = np.asarray([
                _occupation(item['orbital_energy'], item['temperature'],
                            scale * item['contact_multiplier'])
                for item in inputs
            ])
            residual = (prediction - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.65, 0.95),
            method='bounded',
            options={'xatol': 1e-12},
        )
        # ``minimize_scalar`` is reliable for this smooth one-dimensional
        # objective; the clamp also guarantees the public parameter contract
        # in the event of a solver tolerance edge case.
        self.coupling_scale = float(np.clip(result.x, 0.65, 0.95))
        return self

    def predict(self, experiments):
        return np.asarray([_occupation(e['orbital_energy'], e['temperature'],
                           self.coupling_scale*e['contact_multiplier']) for e in experiments])
