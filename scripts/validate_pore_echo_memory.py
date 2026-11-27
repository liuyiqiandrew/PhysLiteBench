"""Compare the independent Fourier reference to both controls and the shipped data."""
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


def sample_scan(rng, size):
    return {"start": int(rng.integers(4)),
            "signs": rng.choice([-1, 1], size=size).tolist(),
            "hops": rng.choice([0.15, 0.4, 0.65, 0.85, 1.0], size=size-1).tolist()}


def samples():
    rng = np.random.default_rng(426007)
    calibration = []
    for size in [1, 2, 3, 4, 5, 6] * 2:
        scan = sample_scan(rng, size)
        curvature = ref.evaluate(scan, ref.TRUE_WIDTH, curvature=True)
        calibration.append({"scan": scan, "curvature": round(curvature, 7)})
    truth = []
    for group in ["moving_echo", "same_scan", "two_scans"]:
        for size in range(3, 10):
            scan = sample_scan(rng, size)
            gain = float(rng.choice([0.85, 1.15, 1.45, 1.8]))
            query = {"mode": "echo", "gain": gain, "scan": scan}
            if group != "moving_echo":
                query["mode"] = "covariance"
                query["partner"] = scan if group == "same_scan" else sample_scan(rng, size - 1)
            target = ref.evaluate(scan, gain * ref.TRUE_WIDTH, query.get("partner"))
            truth.append({"group": group, "input": query, "value": target})
    return calibration, truth


def validate(regenerate):
    calibration, truth = samples()
    for path, expected in [(TASK / "tests/data/calibration.json", calibration),
                           (TASK / "environment/data/calibration.json", calibration),
                           (TASK / "tests/data/predictions.json", truth)]:
        if regenerate:
            path.write_text(json.dumps(expected, indent=2) + "\n")
        actual = json.loads(path.read_text())
        if expected is truth:
            assert [r["group"] for r in actual] == [r["group"] for r in expected]
            assert [r["input"] for r in actual] == [r["input"] for r in expected]
            np.testing.assert_allclose([r["value"] for r in actual], [r["value"] for r in expected], atol=2e-12, rtol=0)
        else:
            assert actual == expected
    # A sealed pore gives a Brownian-free single-site phase; different sealed sites factor.
    pinned = {"start": 1, "signs": [1, -1, 1], "hops": [0, 0]}
    echo = np.exp(-ref.TRUE_WIDTH**2 / 2)
    assert abs(ref.evaluate(pinned, ref.TRUE_WIDTH) - echo) < 1e-12
    assert abs(ref.evaluate(pinned, ref.TRUE_WIDTH, pinned) - (1 - echo**2)) < 1e-12
    assert abs(ref.evaluate(pinned, ref.TRUE_WIDTH, dict(pinned, start=2))) < 1e-12

    for name, path in [("oracle", TASK / "solution/model.py"),
                       ("shortcut", ROOT / "scripts/pore_echo_memory_baseline.py")]:
        model = load(path).Model().fit(calibration)
        assert abs(model.width - ref.TRUE_WIDTH) < 1e-5
        measured = [r["curvature"] for r in calibration]
        np.testing.assert_allclose(model.predict_calibration(calibration), measured, atol=1e-5, rtol=0)
        results = {}
        for group in ["moving_echo", "same_scan", "two_scans"]:
            cases = [r for r in truth if r["group"] == group]
            actual = model.predict([r["input"] for r in cases])
            results[group] = float(np.max(abs(actual - [r["value"] for r in cases])))
        if name == "oracle":
            assert max(results.values()) < 1e-7
        else:
            assert min(results.values()) > 0.002
        print(name, "width", model.width, "maximum absolute errors", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    validate(parser.parse_args().regenerate)
