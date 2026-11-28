"""Compare neutral parity sectors with extrapolated microscopic Slater matrix elements."""
from pathlib import Path
import json, argparse, runpy, sys
import numpy as np

root = Path(__file__).resolve().parents[1]
t = root / "tasks/fermi-arc-interferometer"
sys.path[:0] = [str(t / k) for k in ["environment", "solution", "tests"]]
from edge import reading, operator
from reference import extrapolate, minor_matrix

V = 0.81473


def example(sign):
    return dict(
        arcs=[[0.0, 0.25, 0.0, 0.0]],
        detuning=0,
        state=[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0 if sign > 0 else np.pi]],
    )


def generate():
    rng = np.random.default_rng(774008)

    def u(a, b):
        return round(float(rng.uniform(a, b)), 4)

    cal = []
    for phase in np.linspace(-2.6, 2.8, 12).round(4):
        cal.append(dict(input=dict(blank=True, phase=phase), value=round(V * np.cos(phase), 8)))
    groups = {}
    for group in ["filled_droplet", "edge_doublet", "coherent_packet"]:
        rows = []
        for i in range(6):
            count = 2 + i % 2 if group == "filled_droplet" else 1 + i % 3
            grid = np.arange(1, 32)
            for _ in range(1000):
                pts = np.sort(rng.choice(grid, 2 * count, replace=False)) / 32
                if min(np.diff(pts)) >= 0.0625:
                    break
            arcs = [[p1, p2, u(-0.65, 0.65), u(-0.65, 0.65)] for p1, p2 in pts.reshape(-1, 2)]
            if group == "filled_droplet":
                v = [[1.0, 0.0]] + [[0.0, 0.0]] * 3
            elif group == "edge_doublet":
                v = [[0.0, 0.0]] * 2 + [[u(0.4, 1.0), u(-3.1, 3.1)], [u(0.4, 1.0), u(-3.1, 3.1)]]
            else:
                v = [[u(0.2, 1.0), u(-3.1, 3.1)] for j in range(4)]
            q = dict(arcs=arcs, detuning=int(rng.integers(-2, 3)), state=v)
            pred = reading(q)
            raw = extrapolate(q)
            np.testing.assert_allclose(pred, raw, atol=3e-5, rtol=0)
            rows.append(dict(input=q, value=round(V * pred, 9)))
        groups[group] = rows
    return cal, groups


def main(write):
    for sign, expected in [(1, 0.125), (-1, 0.5)]:
        q = example(sign)
        np.testing.assert_allclose(reading(q), expected, atol=1e-13)
        np.testing.assert_allclose(extrapolate(q), expected, rtol=0, atol=2e-7)
        np.testing.assert_allclose(
            minor_matrix(q, 32), minor_matrix(q, 32, direct=True), atol=1e-12
        )
    cal, groups = generate()
    for p, v in [
        (t / "tests/data/hidden.json", groups),
        (t / "tests/data/calibration.json", cal),
        (t / "environment/data/calibration.json", cal),
    ]:
        if write:
            p.write_text(json.dumps(v, separators=(",", ":")) + "\n")
        assert json.loads(p.read_text()) == v
    for label, path in [
        ("oracle", t / "solution/model.py"),
        ("starter", t / "environment/model.py"),
        ("dephased", root / "scripts/fermi_arc_interferometer_baseline.py"),
    ]:
        m = runpy.run_path(str(path))["Model"]().fit(cal)
        e = [
            max(abs(m.predict([r["input"] for r in rows]) - [r["value"] for r in rows]))
            for rows in groups.values()
        ]
        print(label, e, flush=True)
        if label == "oracle":
            assert max(e) < 1e-7
        elif label == "starter":
            assert min(e) > 0.003
        else:
            assert e[0] < 1e-7 and min(e[1:]) > 0.003


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--generate", action="store_true")
    main(a.parse_args().generate)
