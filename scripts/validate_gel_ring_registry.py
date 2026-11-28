import argparse
import json
from pathlib import Path
import sys, runpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/gel-ring-registry"
sys.path[:0] = [str(TASK / d) for d in ("environment", "solution", "tests")]
from registry import shot, Printer
from gel import tightened
from reference import reference
from finite_tree import check as finite_tree


def configs():
    rng = np.random.default_rng(5157401)
    out = {}

    def ring():
        pk = rng.integers(0, 4, size=2)
        biases = np.exp(rng.uniform(-1.7, 0.3, (2, 4)))
        biases[np.arange(2), pk] *= rng.uniform(3.5, 8.0, size=2)
        return dict(
            feed=np.round(rng.uniform(2.0, 12.0, 2), 3).tolist(),
            routes=np.round(biases, 3).tolist(),
            switch=np.round(rng.uniform(0.035, 0.84, (4, 2)), 3).tolist(),
            blue=round(float(rng.uniform(0.08, 0.92)), 3),
        )

    for g in ["seam", "windings", "mixed_readouts"]:
        rows = []
        for i in range(8):
            n = (
                (2, 3, 4, 5, 6, 7, 8, 9)[i]
                if g == "seam"
                else (3, 4, 5, 6, 6, 7, 8, 10)[i]
            )
            for _ in range(100):
                q = dict(
                    length=n,
                    rings=[ring(), ring()],
                    gate=sorted(
                        rng.choice(4, size=(i % 2) + 1, replace=False).tolist()
                    ),
                )
                if g == "seam" or (g == "mixed_readouts" and i % 2):
                    q["probe"] = sorted(
                        rng.choice(4, size=(i + 1) % 2 + 1, replace=False).tolist()
                    )
                else:
                    q.update(
                        flux=np.round(rng.uniform(-2.4, 2.4, 2), 3).tolist(),
                        phase=round(float(rng.uniform(-2.7, 2.7)), 3),
                    )
                v, d = shot(q, 0.9387, tightened)
                if d > 0.00075 and 0.015 < v < 0.985:
                    break
            else:
                raise RuntimeError(n)
            rows.append(q)
        out[g] = rows
    return out


def main(generate):
    z = 0.9387
    cal = [
        dict(input=dict(dose=d), value=round(1 / (1 + z * d), 8))
        for d in (0.32, 0.55, 0.72, 0.95, 1.2, 1.74, 2.15, 2.80, 3.54, 4.85)
    ]
    data = {}
    worst = 0
    least = 1.0
    for g, qs in configs().items():
        rows = []
        for q in qs:
            value, gate = shot(q, z, tightened)
            independent = reference(q, z)
            assert 0 < value < 1 and gate > 0.00075
            worst = max(worst, abs(value - independent))
            least = min(least, gate)
            rows.append(dict(input=q, value=round(independent, 9)))
        data[g] = rows
    q = data["seam"][0]["input"]
    exact = shot(q, z, tightened)[0]
    limits = [finite_tree(q, z, d) for d in (5, 7, 9)]
    assert abs(limits[-1] - exact) < 2e-5 and abs(limits[-1] - exact) < abs(
        limits[1] - exact
    ) < abs(limits[0] - exact)
    print("Explicit finite tree", limits, "infinite tree", exact)
    assert worst < 1e-11
    print(
        "Independent methods",
        worst,
        "minimum registration probability after sorting and the first seam gate",
        least,
    )
    for name, obj in [
        ("tests/data/calibration.json", cal),
        ("environment/data/calibration.json", cal),
        ("tests/data/predictions.json", data),
    ]:
        p = TASK / name
        if generate:
            p.write_text(json.dumps(obj, indent=1) + "\n")
        assert json.loads(p.read_text()) == obj
    for role, file in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("shortcut", ROOT / "scripts/gel_ring_registry_baseline.py"),
    ]:
        m = runpy.run_path(str(file))["Model"]().fit(cal)
        assert abs(m.activity - z) < 1e-7
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
    ring = dict(feed=[4, 6], routes=[[1] * 4] * 2, switch=[[0.2, 0.3]] * 4, blue=0.45)
    q = dict(length=2, rings=[ring, ring], gate=[0])
    for a, w in [
        (dict(q, probe=[0]), 0.6),
        (dict(q, flux=[np.pi / 2, 0], phase=0.0), 0.4),
    ]:
        assert (
            abs(shot(a, z, tightened)[0] - w) < 1e-12
            and abs(reference(a, z) - w) < 1e-12
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    main(p.parse_args().generate)
