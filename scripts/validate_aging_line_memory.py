import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/aging-line-memory"
sys.path[:0] = [str(TASK / d) for d in ("environment", "solution", "tests")]
from pump import pump
from reference import truth


def cases():
    rng = np.random.default_rng(4062571)

    def tr(lo, hi):
        return round(float(rng.uniform(lo, hi)), 4)

    def launch(positive):
        return {
            "temperature": tr(0.535, 0.699),
            "clock": tr(0.6, 1.15),
            "offsets": [tr(-0.60, 0.22), tr(-0.58, 0.17)],
            "levels": [tr(0.28, 0.80), tr(0.24, 0.61) if positive else 0.0],
        }

    data = {}
    for name in ["burned_line", "reporter_transfer", "host_background"]:
        arr = []
        for k in range(8):
            while True:
                a = launch(k % 4 in (1, 3))
                b = launch(k % 4 in (2, 3))
                lo = tr(0.15, 0.90)
                hi = round(lo + tr(0.85, 2.0), 4)
                row = {
                    "exposures": [a, b],
                    "pump_band": [lo, hi],
                    "shift": tr(0.65, 1.25) * (-1 if k % 2 else 1),
                    "read": {
                        "burned_line": "tagged",
                        "reporter_transfer": "B"
                        if b["levels"][1] > 0 and k % 3
                        else "A",
                        "host_background": "host",
                    }[name],
                }
                val, den = pump(row, 0.9863)
                if den > 0.0007 and 0.018 < val < 0.982:
                    break
            arr.append(row)
        data[name] = arr
    return data


def main(generate):
    tg = 0.9863
    cal = [
        dict(input=dict(threshold=x), value=round(float(np.exp(-x / tg)), 8))
        for x in (0.22, 0.35, 0.52, 0.66, 0.80, 1.05, 1.24, 1.46, 1.68, 1.92)
    ]
    data = {}
    worst = 0
    weights = []
    for g, qs in cases().items():
        rows = []
        for q in qs:
            v, m = truth(q, tg)
            s, om = pump(q, tg)
            np.testing.assert_allclose([v, m], [s, om], rtol=0, atol=2e-7)
            assert 0 < v < 1 and m > 0.0007
            weights.append(m)
            worst = max(worst, abs(v - s), abs(m - om))
            rows.append(dict(input=q, value=round(v, 9)))
        data[g] = rows
    for name, obj in [
        ("tests/data/calibration.json", cal),
        ("environment/data/calibration.json", cal),
        ("tests/data/predictions.json", data),
    ]:
        p = TASK / name
        if generate:
            p.write_text(json.dumps(obj, indent=1) + "\n")
        actual = json.loads(p.read_text())
        if isinstance(obj, list):
            assert actual == obj
        else:
            assert obj.keys() == actual.keys()
            for k, rows in obj.items():
                assert [r["input"] for r in rows] == [r["input"] for r in actual[k]]
                np.testing.assert_allclose(
                    [r["value"] for r in rows],
                    [r["value"] for r in actual[k]],
                    atol=1.1e-9,
                    rtol=0,
                )
    print(
        "Pooled joint densities",
        min(weights),
        max(weights),
        "independent discrepancy",
        worst,
    )
    for role, path in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("shortcut", ROOT / "scripts/aging_line_memory_baseline.py"),
    ]:
        m = runpy.run_path(str(path))["Model"]().fit(cal)
        assert abs(m.depth_scale - tg) < 1e-8
        err = {
            k: float(
                max(abs(m.predict([r["input"] for r in rs]) - [r["value"] for r in rs]))
            )
            for k, rs in data.items()
        }
        print(role, err)
        assert (
            max(err.values()) < 1e-6 if role == "oracle" else min(err.values()) > 0.004
        )
    a = dict(temperature=0.55, clock=0.85, offsets=[-0.48, 0.11], levels=[0.75, 0.4])
    b = dict(temperature=0.67, clock=1.2, offsets=[0.2, -0.49], levels=[1.05, 0.6])
    q = dict(exposures=[a, b], pump_band=[0.24, 1.72], shift=0.7, read="tagged")
    for method in [pump, truth]:
        for shift, target in [(0.7, 0.12735679), (0.0, 0.10110947), (-0.7, 0.06719938)]:
            assert abs(method(dict(q, shift=shift), 1.0)[0] - target) < 2e-7
        assert (
            abs(sum(method(dict(q, read=k), 1.0)[0] for k in ("A", "B", "host")) - 1)
            < 2e-12
        )
        b["levels"][1] = 0.0
        assert method(dict(q, read="B"), 1.0)[0] == 0.0
        b["levels"][1] = 0.6


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    main(p.parse_args().generate)
