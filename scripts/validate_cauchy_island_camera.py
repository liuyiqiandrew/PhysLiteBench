"""Compare the time-resolved strip flux with the conformal mixed-boundary Green solution."""
import argparse
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/cauchy-island-camera"
sys.path[:0] = [str(TASK / "environment"), str(TASK / "solution"), str(TASK / "tests")]
from arch import reading as solve
from reference import reading as renewal
from test_public import examples


def calibration():
    pairs = [
        (-1.6, 0.43),
        (-1.2, 1.07),
        (-0.62, 0.58),
        (-0.43, 1.61),
        (0.34, 0.88),
        (0.73, 0.52),
        (0.95, 1.2),
        (1.17, 0.44),
        (1.55, 0.81),
        (1.72, 0.61),
    ]
    return [
        dict(
            input=dict(mode="free", wave_number=k, lag=t),
            value=round(float(np.exp(-0.89 * k**2 * t)), 8),
        )
        for k, t in pairs
    ]


def science():
    rng = np.random.default_rng(248139)

    def draw(a, b):
        return round(float(rng.uniform(a, b)), 6)

    rows = []
    for i in range(27):
        rows.append(
            dict(
                launch=draw(-0.64, 0.64),
                landing=(-1 if i % 2 == 0 else 1) * (1 + draw(0.26, 0.86)),
                heights=[0.0 if i % 6 == 0 else draw(0.23, 0.62), draw(0.82, 1.61)],
                prior=None if i % 8 == 0 and i // 9 != 1 else draw(0.46, 1.32),
                window=[draw(0.055, 0.23), draw(0.46, 1.2)],
            )
        )
    return {
        k: [dict(q, observable=k) for q in rows[i * 9 : (i + 1) * 9]]
        for i, k in enumerate(("final", "history", "duration"))
    }


def analytic(x, y, h):
    a = np.pi / h
    l = np.exp(-a)
    u = np.exp(a)
    s = np.exp(a * x)
    t = np.exp(a * y)
    at = (
        np.sqrt((s - l) * (u - s))
        / (np.pi * np.sqrt(abs((t - l) * (t - u))) * abs(t - s))
        * a
        * t
    )
    total = np.sqrt(1 - x * x) / (np.pi * np.sqrt(y * y - 1) * abs(y - x))
    return at / total


def validate(generate):
    cal = calibration()
    data = {}
    for i, (k, shots) in enumerate(science().items()):
        diff = np.zeros(3)
        rows = []
        for q in shots:
            v = renewal(q, 0.89, 100)
            diff = np.maximum(diff, abs(v - solve(q, 0.89, 60)))
            rows.append(dict(input=q, value=round(float(v[i]), 8)))
            same = dict(q, heights=[0.0, 0.96], prior=0.96)
            z = solve(same, 0.89)
            assert abs(z[0] * z[1] - analytic(q["launch"], q["landing"], 0.96)) < 1.0e-7
        assert np.max(diff) < 0.00018
        data[k] = rows
        print(k, "independent weak errors", diff)
    for q, expected in examples():
        np.testing.assert_allclose(solve(q, 0.89, 96), expected, atol=0.00006)
        np.testing.assert_allclose(renewal(q, 0.89, 100), expected, atol=0.00012)
    for path, value in [
        (TASK / "environment/data/calibration.json", cal),
        (TASK / "tests/data/calibration.json", cal),
        (TASK / "tests/data/predictions.json", data),
    ]:
        if generate:
            path.write_text(json.dumps(value, indent=1) + "\n")
        frozen = json.loads(path.read_text())
        if isinstance(value, dict):
            assert frozen.keys() == value.keys()
            for k, rows in value.items():
                assert len(frozen[k]) == len(rows)
                for a, b in zip(frozen[k], rows):
                    assert (
                        a["input"] == b["input"] and abs(a["value"] - b["value"]) < 1e-7
                    )
        else:
            assert frozen == value
    for path, control in [
        (TASK / "solution/model.py", "oracle"),
        (TASK / "environment/model.py", "one arch"),
        (ROOT / "scripts/cauchy_island_camera_baseline.py", "independent duration"),
    ]:
        model = runpy.run_path(str(path))["Model"]().fit(cal)
        errors = {}
        for k, rows in data.items():
            vals = model.predict([r["input"] for r in rows])
            assert np.isfinite(vals).all()
            errors[k] = float(max(abs(vals - [r["value"] for r in rows])))
        print(control, errors)
        assert abs(model.diffusivity - 0.89) < 2.0e-7
        if control == "oracle":
            assert max(errors.values()) < 0.00018
        else:
            assert errors["duration"] > 0.1
            if control == "independent duration":
                assert max(errors["history"], errors["final"]) < 1.0e-6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
