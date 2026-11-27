"""Validate frozen spin fringes and both controls. Pass --generate to rebuild the data."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/spin-ring-fringe"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load(TASK / "tests/reference.py")
oracle = load(TASK / "solution/model.py")
shortcut = load(ROOT / "scripts/spin_ring_fringe_baseline.py")


def calibration():
    rng = np.random.default_rng(7319)
    records = []
    for i in range(24):
        n, beta = 10 + i % 6, 1.1 + .15 * (i % 5)
        h = .4 + .14 * (i % 4)
        x = ref.experiment(n, beta, [h, 0, 0, 0], [[.72, [h + .11, 0, 0, 0]]], i * .27)
        signal = ref.predict([x], ref.TRUE_PARAMETER)[0]
        records.append(dict(input=x, value=signal + rng.normal(0, .002), sigma=.002))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    hidden = {}
    for family, inputs in ref.hidden_inputs().items():
        truth = ref.predict(inputs, ref.TRUE_PARAMETER)
        hidden[family] = [dict(input=x, value=y) for x, y in zip(inputs, truth)]
    records = calibration()
    fixtures = [(TASK / "environment/data/calibration.json", records),
                (TASK / "tests/data/calibration.json", records),
                (TASK / "tests/data/hidden.json", hidden)]
    for path, data in fixtures:
        if args.generate:
            path.write_text(json.dumps(data, indent=1) + "\n")
        checked = json.loads(path.read_text())
        if isinstance(data, list):
            data, checked = {"calibration": data}, {"calibration": checked}
        assert data.keys() == checked.keys()
        for name in data:
            assert [{k:v for k,v in r.items() if k != "value"} for r in data[name]] == [
                {k:v for k,v in r.items() if k != "value"} for r in checked[name]]
            np.testing.assert_allclose([r["value"] for r in data[name]], [r["value"] for r in checked[name]], atol=1e-10)
    x = ref.hidden_inputs()["pulse-echo"][0] | {"n": 6}
    np.testing.assert_allclose(ref.spatial_trace(x), ref.spin_trace(x), atol=1e-11)
    for module in [oracle, shortcut]:
        model = module.Model().fit(records)
        predictors = np.array([module.fringe(r["input"]) for r in records])
        rng = np.random.default_rng(7320)
        noise = rng.normal(0, .002, (256, len(records)))
        measurements = ref.predict([r["input"] for r in records], ref.TRUE_PARAMETER) + noise
        fitted = measurements @ predictors / (predictors @ predictors)
        assert np.max(abs(fitted - ref.TRUE_PARAMETER)) < .005
        observed = model.predict([r["input"] for r in records])
        chi2 = np.sum(((observed - [r["value"] for r in records]) / .002)**2) / (len(observed) - 1)
        assert chi2 < 2.5
        print(module.__name__, "visibility", model.visibility, "reduced chi2", chi2)
        assert abs(model.visibility - ref.TRUE_PARAMETER) < .005
        for family, data in hidden.items():
            truth = np.array([r["value"] for r in data])
            inputs = [r["input"] for r in data]
            rmse = np.sqrt(np.mean((model.predict(inputs) - truth)**2))
            print(" ", family, rmse)
            if module is oracle:
                np.testing.assert_allclose([module.fringe(x) * ref.TRUE_PARAMETER for x in inputs], truth, atol=1e-10)
                assert rmse < .008
            else:
                assert rmse > .008


if __name__ == "__main__":
    main()
