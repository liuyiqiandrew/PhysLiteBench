"""Check the two controls, frozen data, symmetries and impulse convergence."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/slab-flux-pump"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def query(bias, low, high, readout):
    return dict(bias=bias, low=low, high=high, readout=readout)


def records(inputs, reference, n, increments=1):
    return [{"input": q, "value": reference.signal(q, n=n, increments=increments)}
            for q in inputs]


def validate(generate):
    ref = load(TASK / "tests/reference.py")
    oracle = load(TASK / "solution/model.py")
    baseline = load(ROOT / "scripts/slab_flux_pump_baseline.py")
    cal_inputs = [query(0., lo, hi, r)
                  for lo, hi in [(-1.45, 1.32), (-1.20, .42), (-.65, 1.4),
                                 (.13, 1.28), (-1.54, -.25), (-.3, .24)]
                  for r in ["crest", "trough"]]
    calibration = records(cal_inputs, ref, 16384)
    for row in calibration:
        row["value"] = round(row["value"], 8)
    settings = [(.26, -1.40, 1.30), (-.26, -1.40, 1.30),
                (.23, -.65, 1.50), (-.21, -1.45, .81),
                (.25, .10, 1.55), (-.27, -1.55, -.30),
                (.18, -.82, .77), (-.12, -.66, .88), (.24, -.2, .3),
                (0., -1.26, 1.4)]
    truth = {}
    for r in ["crest", "trough", "impulse"]:
        inputs = [query(*params, r) for params in settings]
        truth[r] = records(inputs, ref, 4800 if r == "impulse" else 16384, 1280)
    for path in [TASK / "environment/data/calibration.json", TASK / "tests/data/calibration.json"]:
        if generate:
            path.write_text(json.dumps(calibration, indent=2) + "\n")
        assert json.loads(path.read_text()) == calibration
    path = TASK / "tests/data/predictions.json"
    if generate:
        path.write_text(json.dumps(truth, indent=2) + "\n")
    shipped = json.loads(path.read_text())
    assert set(shipped) == set(truth)
    for key, rows in truth.items():
        assert [r["input"] for r in rows] == [r["input"] for r in shipped[key]]
        np.testing.assert_allclose([r["value"] for r in rows], [r["value"] for r in shipped[key]],
                                   atol=1e-10, rtol=0)

    # Separate space and time refinements on all ten impulse preparations.
    convergence = {}
    for label, n, increments in [("halve spatial resolution", 2400, 1280),
                                  ("halve temporal resolution", 4800, 640)]:
        convergence[label] = max(abs(ref.signal(r["input"], n=n, increments=increments) - r["value"])
                                 for r in truth["impulse"])
    print("finite-volume refinement differences", convergence)
    assert max(convergence.values()) < 0.004
    # Closed cycles with a flux-free core have zero integrated voltage.
    # Reflection reverses current; the cycle voltages at both faces are equal.
    impulse = [r["value"] for r in truth["impulse"]]
    assert max(abs(impulse[-2]), abs(impulse[-1]), abs(impulse[0] + impulse[1])) < 0.002
    # Fully penetrated crest at zero bias: average sqrt((1+h)^2 - 2*j*d) - 1 over 0<d<1.
    h, j = 1.5, ref.TRUE_PARAMETER
    exact = ((1 + h)**3 - ((1 + h)**2 - 2*j)**1.5) / (3*j) - 1
    q = query(0., -1.5, h, "crest")
    assert abs(ref.signal(q, n=4096) - exact) < 1e-7
    assert abs(oracle.signal(q, j) - exact) < 1e-7

    for name, module in [("oracle", oracle), ("shortcut", baseline)]:
        model = module.Model().fit(calibration)
        errors = {}
        for key, rows in [("calibration", calibration), *truth.items()]:
            calculated = model.predict([r["input"] for r in rows])
            errors[key] = float(np.max(abs(calculated - [r["value"] for r in rows])))
        print(name, "pinning", model.pinning, "maximum absolute errors", errors)
        assert abs(model.pinning - j) < 1e-4 and errors["calibration"] < 1e-5
        if name == "oracle":
            assert max(errors["crest"], errors["trough"]) < 0.003 and errors["impulse"] < 0.008
        else:
            assert min(errors["crest"], errors["trough"]) > 0.003 and errors["impulse"] > 0.008


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
