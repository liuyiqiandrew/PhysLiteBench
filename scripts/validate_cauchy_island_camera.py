"""Regenerate detector labels and compare an independent first-exit law with the spectral generator."""
import argparse
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/cauchy-island-camera"
sys.path[:0] = [str(TASK / "environment"), str(TASK / "solution"), str(TASK / "tests")]
from spectral import reading
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
            value=round(float(np.exp(-0.89 * abs(k) * t)), 8),
        )
        for k, t in pairs
    ]


def science():
    rng = np.random.default_rng(401873)
    groups = {}

    def d(a, b):
        return round(float(rng.uniform(a, b)), 6)

    for k in ("landing", "donor", "clock"):
        rows = []
        for i in range(9):
            h = d(0.52, 1.22)
            g = d(0.36, 0.88)
            a = 1 + g
            bars = [[-1.0, 1.0], [a, a + 2 * h]]
            start = i % 2
            c0 = (bars[start][0] + bars[start][1]) / 2
            hw = (bars[start][1] - bars[start][0]) / 2
            x = c0 + hw * d(-0.64, 0.64)
            if i % 3 == 0:
                y = 1 + g * d(0.26, 0.74)
            elif i % 3 == 1:
                y = -1 - d(0.26, 1.0)
            else:
                y = bars[1][1] + d(0.26, 1.0)
            mark = dict(
                island=(i % 2 if i % 3 == 1 else 1 - i % 2),
                profile=[d(0.11, 0.38), d(0.25, 0.75), d(0.78, 0.99)],
            )
            if i % 2:
                mark["profile"].reverse()
            q = dict(
                islands=[[round(v, 6) for v in side] for side in bars],
                launch=round(x, 6),
                landing=round(y, 6),
                donor=None if k == "landing" else mark,
                rates=[d(0.2, 1.2), d(0.1, 1.1)],
                relay=d(0.31, 0.88),
                observable=k,
            )
            if i % 3 == 1:
                q["relay"] = 1.0
            if i % 3 == 2:
                q["rates"][i % 2] = 0.0
            rows.append(q)
        groups[k] = rows
    return groups


def validate(generate):
    cal = calibration()
    data = {}
    for k, qs in science().items():
        rows = []
        error = 0.0
        for q in qs:
            val = renewal(q, 0.89, 165)
            error = max(error, abs(val - reading(q, 0.89)))
            rows.append(dict(input=q, value=round(float(val), 8)))
        assert error < 0.00015
        print(k, "independent weak discrepancy", error)
        data[k] = rows
    for q, expected in examples():
        for k, val in zip(("landing", "donor", "clock"), expected):
            shot = dict(q, observable=k)
            assert abs(renewal(shot, 0.91, 116) - val) < 0.00015
            assert abs(reading(shot, 0.91) - val) < 0.000001
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
            for group, rows in value.items():
                assert len(frozen[group]) == len(rows)
                for record, actual in zip(frozen[group], rows):
                    assert (
                        record["input"] == actual["input"]
                        and abs(record["value"] - actual["value"]) < 1e-7
                    )
        else:
            assert frozen == value
    for path, control in [
        (TASK / "solution/model.py", "oracle"),
        (TASK / "environment/model.py", "midpoint"),
        (ROOT / "scripts/cauchy_island_camera_baseline.py", "independent gate"),
    ]:
        model = runpy.run_path(str(path))["Model"]().fit(cal)
        assert abs(model.diffusivity - 0.89) < 2e-7
        errors = {}
        for k, rows in data.items():
            values = model.predict([r["input"] for r in rows])
            assert np.isfinite(values).all()
            errors[k] = float(max(abs(values - [r["value"] for r in rows])))
        if control == "oracle":
            assert max(errors.values()) < 0.00015
        elif control == "midpoint":
            assert min(errors.values()) > 0.02
        else:
            assert (
                max(errors["donor"], errors["landing"]) < 0.00015
                and errors["clock"] > 0.6
            )
        print(control, errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
