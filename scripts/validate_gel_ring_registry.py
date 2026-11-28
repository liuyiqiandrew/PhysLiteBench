import argparse, json, runpy, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/gel-ring-registry"
sys.path[:0] = [str(TASK / d) for d in ("environment", "solution", "tests")]
from registry import observable, Printer
from gel import joint
from reference import truth


def shots():
    rng = np.random.default_rng(751016)

    def feed():
        return dict(
            feed=round(float(rng.uniform(2.1, 10.7)), 3),
            routes=np.round(np.exp(rng.uniform(-1.0, 1.35, 4)), 3).tolist(),
        )

    groups = {}
    for g in ("seam", "shifted", "winding"):
        rows = []
        for j in range(8):
            n = (
                [3, 4, 5, 3, 4, 5, 4, 6][j]
                if g != "shifted"
                else [3, 3, 4, 3, 4, 5, 4, 5][j]
            )
            for _ in range(80):
                q = dict(
                    length=n,
                    trunk=feed(),
                    arms=[feed(), feed()],
                    shifted=g == "shifted" or (g == "winding" and j % 2 == 1),
                    gate=sorted(rng.choice(4, size=1 + j % 2, replace=False).tolist()),
                )
                if g == "winding":
                    q.update(
                        flux=np.round(rng.uniform(-2.6, 2.6, 2), 3).tolist(),
                        phase=round(float(rng.uniform(-2.7, 2.7)), 3),
                    )
                else:
                    q["probe"] = sorted(
                        rng.choice(4, size=1 + (j + 1) % 2, replace=False).tolist()
                    )
                value, P = truth(q, 0.9387)
                if P > 2.5e-5 and 0.025 < value < 0.975:
                    break
            else:
                raise RuntimeError((g, j))
            rows.append(q)
        groups[g] = rows
    return groups


def main(generate):
    z = 0.9387
    cal = [
        dict(input=dict(dose=d), value=round(1 / (1 + z * d), 8))
        for d in (0.32, 0.55, 0.72, 0.95, 1.2, 1.74, 2.15, 2.80, 3.54, 4.85)
    ]
    data = {}
    worst = 0.0
    minimum = 1.0
    for g, qs in shots().items():
        rows = []
        for q in qs:
            value, P = truth(q, z)
            exact = observable(q, z, joint)
            worst = max(worst, abs(value - exact))
            minimum = min(minimum, P)
            assert 0.025 < value < 0.975 and P > 2.5e-5
            rows.append(dict(input=q, value=round(value, 9)))
        data[g] = rows
    print("Independent methods", worst, "minimum absolute event probability", minimum)
    assert worst < 1e-11
    for name, obj in [
        ("tests/data/calibration.json", cal),
        ("environment/data/calibration.json", cal),
        ("tests/data/predictions.json", data),
    ]:
        p = TASK / name
        if generate:
            p.write_text(json.dumps(obj, indent=1) + "\n")
        assert json.loads(p.read_text()) == obj
    for role, where in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("shortcut", ROOT / "scripts/gel_ring_registry_baseline.py"),
    ]:
        m = runpy.run_path(str(where))["Model"]().fit(cal)
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
    q = dict(
        length=3,
        trunk=dict(routes=[4, 1, 2, 1], feed=6),
        arms=[dict(routes=[5, 2, 1, 1], feed=4), dict(routes=[1, 2, 5, 2], feed=7)],
        gate=[0, 2],
        probe=[1, 3],
        shifted=True,
    )
    for flag, target in [(True, 0.15036470), (False, 0.10805815)]:
        row = dict(q, shifted=flag)
        assert (
            abs(observable(row, 1.0, joint) - target) < 1e-8
            and abs(truth(row, 1.0)[0] - target) < 1e-8
        )
    larger = [truth(q, 1.0, k)[0] for k in (3, 5, 7, 9)]
    print("Microscopic tree cutoffs", larger)
    assert abs(larger[-1] - 0.150364705) < 1e-8


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    main(p.parse_args().generate)
