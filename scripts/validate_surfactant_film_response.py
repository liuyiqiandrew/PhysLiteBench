"""Regenerate meter readings on request and check the two completed controls."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/surfactant-film-response'


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calibration(reference):
    rng = np.random.default_rng(31873)
    records = []
    for i in range(39):
        depth, mu = rng.uniform(.45, 1.9), rng.uniform(.45, 1.7)
        e = dict(h_lower=depth, h_upper=depth, mu_lower=mu, mu_upper=mu,
                 k=rng.uniform(.65, 2.1), omega=rng.uniform(.25, 4.5),
                 fx=.9, fy=.85, fz=-.95, axis='xyz'[i % 3],
                 part='imag' if i % 6 in (3, 5) else 'real')
        value = reference.component(e, reference.TRUE_PARAMETER)
        records.append(dict(input=e, value=round(float(value), 4), sigma=1e-4 / np.sqrt(12)))
    for i in range(12):
        e = {k: v for k, v in records[i]['input'].items() if k not in ('fx','fy','fz','part','axis')}
        e.update(pair='cc' if i % 2 else 'hh', part='real', temperature_lower=1., temperature_upper=1.)
        value = reference.noise_spectrum(e, reference.TRUE_PARAMETER)
        records.append(dict(input=e, value=round(float(value), 4), sigma=1e-4/np.sqrt(12)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    reference = load(TASK / 'tests/reference.py')
    oracle = load(TASK / 'solution/model.py')
    shortcut = load(ROOT / 'scripts/surfactant_film_response_baseline.py')
    private = TASK / 'tests/data/calibration.json'
    public = TASK / 'environment/data/calibration.json'
    if args.generate:
        data = json.dumps(calibration(reference), indent=2) + '\n'
        private.write_text(data)
        public.write_text(data)
    assert private.read_bytes() == public.read_bytes()
    records = json.loads(private.read_text())
    assert records == calibration(reference)
    cases = reference.hidden_inputs()
    truth = {name: reference.predict(es, reference.TRUE_PARAMETER) for name, es in cases.items()}
    fixture = TASK / 'tests/data/hidden_truth.json'
    if args.generate:
        fixture.write_text(json.dumps({k:v.tolist() for k,v in truth.items()}, indent=2)+'\n')
    stored = json.loads(fixture.read_text())
    assert stored.keys() == truth.keys()
    for group in truth:
        np.testing.assert_allclose(stored[group], truth[group], atol=1e-11, rtol=1e-9)
    for name, module in [('oracle', oracle), ('shortcut', shortcut)]:
        model = module.Model().fit(records)
        cal = model.predict([r['input'] for r in records])
        residual = [(out-r['value'])/r['sigma'] for out, r in zip(cal, records)]
        chi2 = np.dot(residual, residual) / (len(records) - 1)
        assert chi2 < 1.5 and abs(model.surface_viscosity / reference.TRUE_PARAMETER - 1) < .03
        errors = {key: np.linalg.norm(model.predict(es) - truth[key]) / np.linalg.norm(truth[key])
                  for key, es in cases.items()}
        print(name, 'surface_viscosity', model.surface_viscosity, 'reduced_chi2', chi2,
              'relative_rmse', errors)
        if name == 'oracle':
            assert max(errors.values()) < .025
        else:
            assert max(errors[key] for key in ('normal_drive','lateral_drive','mixed_drive')) < .025
            assert max(errors[key] for key in ('cross_lag','height_power','density_power','cross_in_phase')) < .025
            assert min(errors[key] for key in ('grating_cc','grating_ch','grating_hh')) > .025
        for key in ('grating_hh','grating_cc','grating_ch'):
            rms = np.linalg.norm(model.predict(cases[key]) - truth[key])/np.sqrt(len(truth[key]))
            assert (rms < max(.0002, .025*np.sqrt(np.mean(truth[key]**2)))) == (name == 'oracle')

    # At fixed drive the surfactant can only compress longitudinally. Mirror
    # the apparatus: the x response to a normal force must change sign.
    for e in cases['normal_drive']:
        flipped = dict(e, h_lower=e['h_upper'], h_upper=e['h_lower'],
                       mu_lower=e['mu_upper'], mu_upper=e['mu_lower'])
        v = reference.component(e, 1.0)
        assert abs(reference.component(flipped, 1.0) + v) < 1e-12
        same = dict(e, h_upper=e['h_lower'], mu_upper=e['mu_lower'])
        assert abs(reference.component(same, 1.0)) < 1e-12

    # Compare the boundary-value and closed-form calculations, including the
    # largest depth-wavenumber product and both viscosity fit boundaries.
    experiments = [e for group in cases.values() for e in group]
    extremes = [dict(e, k=k, h_lower=1.9, h_upper=.45, mu_lower=1.7, mu_upper=.45)
                for k in (.65, 2.1) for e in experiments[:4]]
    for viscosity in (.75, reference.TRUE_PARAMETER, 1.45):
        inputs = experiments + extremes
        exact = reference.predict(inputs, viscosity)
        closed = np.array([oracle.response(e, viscosity) for e in inputs])
        assert np.max(abs(exact - closed)) < 1e-11
    # Detailed balance makes the equilibrium time-antisymmetric cross spectrum vanish.
    # The driven non-equilibrium pair obeys Hermitian symmetry at each frequency.
    for e in cases['cross_lag']:
        eq = dict(e, temperature_lower=1., temperature_upper=1.)
        assert abs(reference.noise_spectrum(eq, 1.0)) < 1e-11
        reverse = dict(e, pair='hc')
        assert abs(reference.noise_spectrum(reverse, 1.0) + reference.noise_spectrum(e, 1.0)) < 1e-11
        assert np.max(abs(oracle.bulk_noise(e, 24)-oracle.bulk_noise(e, 16))) < 1e-11
    # Interchanging optical probes conjugates both the thermal grating and the measured spectrum.
    for e in cases['grating_ch'][::3]:
        g = e['grating']
        swapped = dict(e, k=g['k2'], omega=g['omega2'], pair='hc',
                       grating=dict(g, k2=e['k'], omega2=e['omega'],
                                    amplitudes=[[a,-b] for a,b in g['amplitudes']]))
        old = oracle.spectrum(e, 1.)
        new = reference.noise_spectrum(swapped, 1.)
        assert abs(old - (new if e['part']=='real' else -new)) < 1e-11
    print('Mirror, detailed balance, Hermitian spectra, quadrature and independent BVP checks passed.')


if __name__ == '__main__':
    main()
