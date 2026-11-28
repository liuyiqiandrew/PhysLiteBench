"""Regenerate family count readings with an independent complex generating-function solver."""
import argparse
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/cauchy-island-camera"
sys.path[:0] = [str(TASK / "environment"), str(TASK / "solution"), str(TASK / "tests")]
from tree import solve
from reference import solve as renewal
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
            value=round(float(np.exp(-0.89 * abs(k) * t)), 8),
        )
        for k, t in pairs
    ]


def science():
    rng = np.random.default_rng(274282)

    def draw(low, high):
        return round(float(rng.uniform(low, high)), 6)

    groups = {}
    for i in range(27):
        q = dict(
            rate=draw(0.15, 0.35),
            recoil=draw(0.2, 0.44),
            thresholds=[draw(0.13, 0.7), draw(0.13, 0.7)],
            efficiencies=[draw(0.39, 0.95), draw(0.39, 0.95)],
            source=["single", "pair"][i % 2],
            launch=draw(-0.44, 0.44),
        )
        label = ["red", "coincidence", "clean"][i // 9]
        groups.setdefault(label, []).append(dict(q, observable=label))
    return groups


def validate(generate):
    cal = calibration()
    data = {}
    for i, (k, shots) in enumerate(science().items()):
        diff = np.zeros(3)
        rows = []
        for q in shots:
            v = renewal(q, 0.89, 138, 6)
            diff = np.maximum(diff, abs(v - solve(q, 0.89, 88)))
            rows.append(dict(input=q, value=round(float(v[i]), 8)))
        assert np.max(diff) < 0.00018
        data[k] = rows
        print(k, "independent weak errors", diff)
    for q, expected in examples():
        np.testing.assert_allclose(solve(q, 0.89, 96), expected, atol=0.00006)
        np.testing.assert_allclose(renewal(q, 0.89, 120, 6), expected, atol=0.00012)
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
        (TASK / "environment/model.py", "one fission"),
        (ROOT / "scripts/cauchy_island_camera_baseline.py", "same recoil"),
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
            assert errors["clean"] > 0.013


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
