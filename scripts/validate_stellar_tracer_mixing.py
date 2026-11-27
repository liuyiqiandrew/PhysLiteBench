"""Regenerate rounded calibration and frozen counts only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/stellar-tracer-mixing"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load("ref", TASK / "tests/reference.py")
oracle = load("oracle", TASK / "solution/model.py")
control = load("control", ROOT / "scripts/stellar_tracer_mixing_baseline.py")


def inputs(field, steps, radii):
    return [dict(slow=field, steps=steps, peri=rp, apo=ra) for rp, ra in radii]


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def check_aperture(module):
    # For a very weak drive at fixed aperture, the local half-line solution
    # tends to erf(distance / sqrt(2 B tau)) times the unperturbed density.
    from math import erf

    field = [1., 1.]
    q = dict(slow=field, steps=[], peri=1.4, apo=3.)
    e, L, _ = module.orbit_bin(q)
    j = 1 / np.sqrt(-2*e) - module.offset(L, *field)
    initial = (-e / .4)**ref.TRUE_PARAMETER
    train = dict(coupling=[1e-12, 0.], exposure=.1, binding_limit=-e-1e-6)
    assert abs(module.train_density(e,L,field,train,ref.TRUE_PARAMETER)/initial - 1) < 1e-10
    for amplitude in [1e-7, 1e-4, 1e-3]:
        train["coupling"] = [amplitude, 0.]
        width = np.sqrt(.1 * module.variance(np.array([j]), L, field, train["coupling"])[0])
        for z in [.1, 1., 3.]:
            cap = j + z*width
            train["binding_limit"] = .5/(cap + module.offset(L, *field))**2
            observed = module.train_density(e, L, field, train, ref.TRUE_PARAMETER) / initial
            assert abs(observed - erf(z/np.sqrt(2))) < .002


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    cal_path = TASK / "environment/data/calibration.json"
    private = TASK / "tests/data"
    if args.generate:
        cal = inputs([1, 1], [], [(.2, .6), (.3, 1.8), (.8, 2.3), (1.2, 4.8),
                                   (.5, 6.7), (1.6, 3.9), (.7, 1.1), (2, 7.4)])
        rows = [dict(input=q, value=float(format(ref.signal(q), ".9g"))) for q in cal]
        write(cal_path, rows)
        write(private / "calibration.json", rows)
        a, b, c = [1.9, .41], [.64, 1.7], [1.28, .69]
        bins = [(.2, 1.2), (.65, 2.8), (.31, 5.2), (1.1, 4.9)]
        groups = {"slow": inputs(a, [], bins), "one_switch": inputs(c, [a], bins),
                  "mixed_cycle": inputs(a, [b, a], bins),
                  "two_switches": inputs(b, [a, c], bins)}
        for name, field, coupling, exposure, limit in [
                ("mass_train", [1., 1.], [1., 0.], .25, .09),
                ("shape_train", [.85, .44], [0., -1.], .38, .12),
                ("mixed_train", [1.3, .7], [-.72, .88], .21, .16)]:
            train = dict(exposure=exposure, coupling=coupling, binding_limit=limit)
            groups[name] = [dict(q, train=train) for q in inputs(field, [], [
                (.22, .77), (.35, 1.85), (.54, 3.52), (.38, 4.25)])]
        for name, field, coupling, exposure, limit, hold in [
                ("coherent_mass", [1., 1.], [1., 0.], .085, .08, 14.),
                ("coherent_shape", [1.2, .7], [.4, -.7], .15, .1, 6.2)]:
            train = dict(exposure=exposure, coupling=coupling, binding_limit=limit, hold=hold)
            groups[name] = [dict(q, train=train) for q in inputs(field, [], [
                (.3, .8), (.65, 1.5), (.55, 2.2), (.3, 3.5)])]
        write(private / "predictions.json", {g: [dict(input=q, value=ref.train_signal(q) if "train" in q else ref.signal(q, order=80))
                                                 for q in qs] for g, qs in groups.items()})
    assert cal_path.read_bytes() == (private / "calibration.json").read_bytes()
    # An integral number of radial periods cancels the pair's first-order kick.
    q = dict(slow=[1., 1.], steps=[], peri=.4, apo=2.)
    e, angular, _ = oracle.orbit_bin(q)
    period = 2*np.pi/(-2*e)**1.5
    train = dict(coupling=[.8,-.5], exposure=.3, binding_limit=.1, hold=period)
    expected = (-e/.4)**ref.TRUE_PARAMETER
    assert abs(oracle.train_density(e, angular, q["slow"], train, ref.TRUE_PARAMETER)
               / expected - 1) < 1e-10
    rows = json.loads(cal_path.read_text())
    cases = json.loads((private / "predictions.json").read_text())
    # The coherent aperture can lie within a few floating-point energy units
    # of the query. Compute the physical separation without subtracting actions.
    q = dict(slow=[1., 1.], steps=[], peri=1.4, apo=3.)
    e, L, _ = oracle.orbit_bin(q)
    j = 1/np.sqrt(-2*e) - oracle.offset(L, 1, 1)
    train = dict(coupling=[1e-12, 0], exposure=.1, hold=24.)
    width = np.sqrt(.1 * oracle.coherent_variance(np.array([j]), L, [1, 1],
                                                  train["coupling"], 24.)[0])
    from math import erf
    for units in [1, 5, 10]:
        cut = -e - units*np.spacing(-e)
        train["binding_limit"] = cut
        distance = (-e-cut)/(np.sqrt(2*cut*-e)*(np.sqrt(cut)+np.sqrt(-e)))
        expected = (-e/.4)**ref.TRUE_PARAMETER * erf(distance / (np.sqrt(2)*width))
        measured = oracle.train_density(e, L, [1., 1.], train, ref.TRUE_PARAMETER)
        assert abs(measured/expected-1) < 1e-6

    for impl in (oracle, control):
        check_aperture(impl)
        model = impl.Model().fit(rows)
        assert abs(model.exponent - ref.TRUE_PARAMETER) < 1e-7
        assert np.allclose(model.predict([r["input"] for r in rows]),
                           [r["value"] for r in rows], rtol=1e-7)
        errors = {}
        for group, sample in cases.items():
            qs = [r["input"] for r in sample]
            truth = np.array([r["value"] for r in sample])
            if impl is oracle:
                check = [ref.train_signal(q, level=12, cells_per_level=70)
                         if "train" in q else ref.signal(q, order=56) for q in qs]
                assert np.max(abs(check / truth - 1)) < .004
            errors[group] = float(np.max(abs(model.predict(qs) / truth - 1)))
        if impl is oracle:
            assert max(errors.values()) < 0.004
        else:
            assert max(v for k, v in errors.items() if not k.startswith("coherent_")) < .001
            assert min(v for k, v in errors.items() if k.startswith("coherent_")) > .02
        print(impl.__name__, "exponent", model.exponent, "max relative errors", errors)


if __name__ == "__main__":
    main()
