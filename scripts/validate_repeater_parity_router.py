"""Compare the calibrated repeater, independent herald/phase enumeration, physical controls and worked shots."""
import argparse
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/repeater-parity-router"
sys.path[:0] = [str(TASK / "environment"), str(TASK / "solution"), str(TASK / "tests")]
from reference import explicit
from fringe import ensemble


def calibration():
    return [
        dict(input=dict(exposure=t), value=round(float(np.exp(-0.87 * t)), 8))
        for t in (0.16, 0.27, 0.42, 0.58, 0.69, 0.94, 1.13, 1.24, 1.45, 1.58)
    ]


def science():
    rng = np.random.default_rng(113709)

    def draw(l, h):
        return round(float(rng.uniform(l, h)), 6)

    groups = {}
    for family in ("delivery", "contention", "fringe"):
        rows = []
        for i in range(7):
            L = (3 if i % 2 else 4) if family != "fringe" else (3 if i != 5 else 2)
            ends = []
            for k in range(L):
                if k:
                    ends.extend([(2 * k - 2, 2 * k), (2 * k - 1, 2 * k + 1)])
                ends.append((2 * k, 2 * k + 1))
            if i % 3 == 0:
                ends.insert(3, (0, 3))
            if i % 3 == 1:
                ends.insert(5, (3, 4))
            if i % 3 == 2:
                ends.pop(3 if L > 2 else 2)
            if len(ends) > 10:
                ends.remove((4, 5))
            edge = [
                [
                    u,
                    v,
                    draw(0.25, 1.12),
                    draw(0.09, 0.8),
                    draw(0.20, 0.94) * (-1 if rng.random() < 0.24 else 1),
                ]
                for u, v in ends
            ]
            n = 2 * L
            pairs = (
                [[0, n - 1], [1, n - 2]]
                if i % 3 == 0
                else [[1, n - 1], [0, n - 1]]
                if i % 3 == 1
                else [[0, n - 1], [0, n - 1]]
            )
            if family == "fringe" and i == 4:
                pairs = [[0, n - 1], [0, n - 1]]
            parity = [0] * n
            for u, v, *_ in edge:
                if rng.random() < 0.27:
                    parity[u] ^= 1
                    parity[v] ^= 1
            if i % 2:
                parity[int(rng.integers(0, n))] = None
            if i % 3 == 1:
                for j in rng.choice(n, 2, replace=False):
                    parity[int(j)] = None
            rows.append(
                dict(links=edge, parities=parity, requests=pairs, observable=family)
            )
        groups[family] = rows
    return groups


def examples():
    t = np.log(2.0)
    single = dict(
        links=[[0, 1, t, t, -0.35]], parities=[1, 1], requests=[[0, 1], [0, 1]]
    )
    first = [3 / 5 - 8 * t / 15, 8 * t / 9, 1.0]
    # Capacities of the direct path D and the two-link bypass L, clipped at two.
    p = np.array([0.5, t / 2, (1 - t) / 2])
    l = np.array([0.75, 0.25 - p[2] ** 2, p[2] ** 2])
    joint = fringe = 0.0
    w = -0.24
    for d in range(3):
        for b in range(3):
            count = int(d == 2) + int(b == 2) + 2 * int(d > 0 and b > 0)
            if count:
                joint += p[d] * l[b]
                fringe += (
                    p[d]
                    * l[b]
                    * (int(d == 2) + int(b == 2) + 2 * w * int(d > 0 and b > 0))
                    / count
                )
    second = [joint, 1 - joint / (5 / 8), fringe / joint]
    tri = dict(
        links=[[0, 2, t, 0, 0.5], [0, 1, t, 0, 0.6], [1, 2, t, 0, -0.8]],
        parities=[None] * 3,
        requests=[[0, 2], [0, 2]],
    )
    return [
        (dict(q, observable=k), v)
        for q, truth in [(single, first), (tri, second)]
        for k, v in zip(("delivery", "contention", "fringe"), truth)
    ]


def validate(generate):
    a = 0.87
    cal = calibration()
    data = {}
    oracle = runpy.run_path(str(TASK / "solution/model.py"))["Model"]()
    oracle.activity = a
    for group, qs in science().items():
        rows = []
        errors = []
        for q in qs:
            v = explicit(q, a)[group]
            assert np.isfinite(v) and -1 - 1.0e-12 <= v <= 1 + 1.0e-12
            errors.append(abs(oracle.read(q) - v))
            rows.append(dict(input=q, value=round(float(v), 8)))
        assert max(errors) < 1.0e-10
        data[group] = rows
        print(group, "independent maximum error", max(errors))
    for p, values in [
        (TASK / "environment/data/calibration.json", cal),
        (TASK / "tests/data/calibration.json", cal),
        (TASK / "tests/data/predictions.json", data),
    ]:
        if generate:
            p.write_text(json.dumps(values, indent=1) + "\n")
        assert json.loads(p.read_text()) == values
    for p, label in [
        (TASK / "solution/model.py", "oracle"),
        (TASK / "environment/model.py", "independent loads"),
        (ROOT / "scripts/repeater_parity_router_baseline.py", "fresh phases"),
    ]:
        model = runpy.run_path(str(p))["Model"]().fit(cal)
        assert abs(model.activity - a) < 2.0e-7
        err = {}
        for family, rows in data.items():
            prediction = model.predict([r["input"] for r in rows])
            assert np.isfinite(prediction).all()
            err[family] = float(max(abs(prediction - [r["value"] for r in rows])))
        if label == "oracle":
            assert max(err.values()) < 2.0e-7
        if label == "independent loads":
            assert min(err.values()) > 0.01
        if label == "fresh phases":
            assert (
                max(err["delivery"], err["contention"]) < 2.0e-7 and err["fringe"] > 0.2
            )
        print(label, err)
    pooled = max(abs(ensemble(r["input"], a)[1] - r["value"]) for r in data["fringe"])
    assert pooled > 0.02
    print("weighting shots by the number of path pairs", pooled)
    oracle.activity = 1.0
    for q, value in examples():
        np.testing.assert_allclose(oracle.read(q), value, atol=1.0e-10)
        np.testing.assert_allclose(
            explicit(q, 1.0)[q["observable"]], value, atol=1.0e-10
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
