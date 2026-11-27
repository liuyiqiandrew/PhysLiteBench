"""Reproduce both data files and check the range contraction against an endpoint optimization."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/pore-echo-memory"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load(TASK / "tests/reference.py")


def records():
    calibration = []
    for a in [.17, .85, 1.9, 2.4]:
        for d in [.18, .94, 2.23]:
            x = dict(mode="single", a=a, D_A=d)
            calibration.append(dict(input=x, value=round(ref.predict(x), 8)))
    cases = {}
    for name, spacings in [("near", [0., 1.8, 3.2]), ("separated", [7.1, 10.4, 14.6]),
                            ("unequal", [4.7, 8.7, 17.5])]:
        rows = []
        for i, s in enumerate(spacings):
            for a,b,d,g in [(.25,1.78,2.19,.24), (1.65,1.21,.31,.89), (.67,.36,.63,1.8)]:
                x = dict(mode="pair", a=a, b=b, D_A=d, D_B=g, s=s)
                rows.append(dict(input=x, value=ref.predict(x)))
        cases[name] = rows
    return calibration, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    regenerate = parser.parse_args().regenerate
    calibration, cases = records()
    for path in [TASK / "environment/data/calibration.json", TASK / "tests/data/calibration.json"]:
        if regenerate:
            path.write_text(json.dumps(calibration, indent=2) + "\n")
        assert json.loads(path.read_text()) == calibration
    path = TASK / "tests/data/predictions.json"
    if regenerate:
        path.write_text(json.dumps(cases, indent=2) + "\n")
    stored = json.loads(path.read_text())
    assert stored.keys() == cases.keys()
    for name, rows in cases.items():
        assert [r["input"] for r in rows] == [r["input"] for r in stored[name]]
        np.testing.assert_allclose([r["value"] for r in rows], [r["value"] for r in stored[name]], atol=1e-8, rtol=0)
    for role, source in [("oracle", TASK / "solution/model.py"),
                         ("shortcut", ROOT / "scripts/pore_echo_memory_baseline.py")]:
        model = load(source).Model().fit(calibration)
        assert abs(model.intensity - ref.TRUE_PARAMETER) < 1e-7
        calibration_error = max(abs(model.predict([r["input"] for r in calibration]) - [r["value"] for r in calibration]))
        assert calibration_error < 1e-6
        errors = {}
        for name, rows in cases.items():
            actual = model.predict([r["input"] for r in rows])
            truth = np.array([r["value"] for r in rows])
            errors[name] = float(max(abs(actual/truth - 1)))
        print(role, model.intensity, calibration_error, errors)
        assert errors["near"] < 1e-6
        if role == "oracle":
            assert max(errors.values()) < 1e-6
        else:
            assert min(errors["separated"], errors["unequal"]) > .2

    # Identical private defects and dynamics: symmetry forces equal widths.
    # At zero spacing the ranges coincide; at large spacing each spans s/2.
    for s in [0, 18]:
        x = dict(mode="pair", a=1, b=1, D_A=.5, D_B=.5, s=s)
        w = (2*np.pi**2/(2+ref.TRUE_PARAMETER))**(1/3) if s == 0 else s/2
        expected = 2*w + np.pi**2/w**2 + ref.TRUE_PARAMETER*max(w,s)
        np.testing.assert_allclose(ref.predict(x), expected, atol=1e-8, rtol=0)


if __name__ == "__main__":
    main()
