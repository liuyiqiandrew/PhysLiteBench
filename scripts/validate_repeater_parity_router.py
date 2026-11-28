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
from reference import conditional
from tagged import value as tagged_reading


def calibration():
    return [
        dict(input=dict(exposure=t), value=round(float(np.exp(-0.87 * t)), 8))
        for t in (0.16, 0.27, 0.42, 0.58, 0.69, 0.94, 1.13, 1.24, 1.45, 1.58)
    ]


def science():
    rng = np.random.default_rng(773142)
    family = {}
    for k in ("delivery", "contention", "fringe"):
        qs = []
        for i in range(7):
            L = 2 if i in (0, 5) else 3
            edges = [(0, 1)]
            for n in range(1, L):
                edges.extend(
                    [(2 * n - 2, 2 * n), (2 * n - 1, 2 * n + 1), (2 * n, 2 * n + 1)]
                )
            if i % 3 == 0:
                edges[0] = (0, 3)
            if i % 3 == 1:
                edges[3] = (1, 2)
            if i == 4:
                edges.pop(-1)
            if L == 2:
                edges.append((0, 3)) if (0, 3) not in edges else edges.append((0, 1))
            m = len(edges)
            N = 2 * L
            vis = rng.uniform(0.22, 0.90, m) * rng.choice([-1, 1], m)
            links = [
                [
                    u,
                    v,
                    round(float(rng.uniform(0.46, 1.13)), 6),
                    round(float(rng.uniform(0.16, 0.75)), 6),
                    round(float(h), 6),
                ]
                for (u, v), h in zip(edges, vis)
            ]
            current = [0] * N
            for u, v, *_ in links:
                if rng.random() < 0.35:
                    current[u] ^= 1
                    current[v] ^= 1
            for j in rng.choice(N, min(2, N), replace=False):
                current[int(j)] = None
            routes = (
                [[0, N - 1], [1, N - 2]]
                if i % 3 == 0
                else [[0, N - 1], [1, N - 1]]
                if i % 3 == 1
                else [[0, N - 1], [0, N - 1]]
            )
            watch = sorted(rng.choice(m, min(m, 2 + i % 3), replace=False).tolist())
            qs.append(
                dict(
                    links=links,
                    requests=routes,
                    parities=current,
                    pilot=dict(links=watch, count=[0, 1, 2, 1, 2, 1, 2][i]),
                    observable=k,
                )
            )
        family[k] = qs
    return family


def examples():
    a = np.log(2.0)
    single = dict(
        links=[[0, 1, a, a, -0.35]],
        parities=[1, 1],
        requests=[[0, 1], [0, 1]],
        pilot=dict(links=[0], count=3),
    )
    tri = dict(
        links=[[0, 2, a, 0, 0.5], [0, 1, a, 0, 0.6], [1, 2, a, 0, -0.8]],
        parities=[None] * 3,
        requests=[[0, 2], [0, 2]],
        pilot=dict(links=[0], count=1),
    )
    return [
        (dict(q, observable=k), v)
        for q, vs in [
            (single, [0.875, 0.0, 1.0]),
            (tri, [0.25, 0.75, 0.48 + 0.52 / 3 * (1 - a) ** 2]),
        ]
        for k, v in zip(("delivery", "contention", "fringe"), vs)
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
            v = conditional(q, a)[group]
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
        (TASK / "environment/model.py", "counter disconnected"),
        (
            ROOT / "scripts/repeater_parity_router_baseline.py",
            "separate readout phases",
        ),
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
        if label == "counter disconnected":
            assert min(err.values()) > 0.01
        if label == "separate readout phases":
            assert (
                max(err["delivery"], err["contention"]) < 2.0e-7
                and err["fringe"] > 0.04
            )
        print(label, err)
    saturated = {
        k: max(abs(tagged_reading(r["input"], a, saturate=True) - r["value"]) for r in rows)
        for k, rows in data.items()
    }
    print("one count per occupied negative link", saturated)
    assert max(saturated.values()) > 0.04
    oracle.activity = 1.0
    for q, value in examples():
        np.testing.assert_allclose(oracle.read(q), value, atol=1.0e-10)
        np.testing.assert_allclose(
            conditional(q, 1.0)[q["observable"]], value, atol=1.0e-10
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
