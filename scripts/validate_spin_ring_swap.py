"""Compare physical spin-basis swaps, finite free-fermion replica traces and their limits."""
from pathlib import Path
import argparse, json, runpy, sys
import numpy as np

root = Path(__file__).resolve().parents[1]
t = root / "tasks/spin-ring-swap"
sys.path[:0] = [str(t / p) for p in ("environment", "solution", "tests")]
from swap import continuum
from reference import finite, wave, extrapolate

V = 0.83457


def example(flag):
    return dict(
        arcs=[[0.0, 0.125], [0.375, 0.75]],
        phase_plates=[1, 1],
        records=[] if flag is None else [[[0], flag]],
    )


def generate():
    rng = np.random.default_rng(183718)
    cal = [
        dict(input=dict(blank=True, phase=p), value=round(V * np.cos(p), 8))
        for p in np.linspace(-2.5, 2.6, 12).round(4)
    ]
    groups = {}
    for group in ["single_monitor", "coarse_monitor", "crossed_monitors"]:
        rows = []
        for i in range(6):
            m = 2 + i % 2
            for _ in range(800):
                pts = np.sort(rng.choice(np.arange(16), 2 * m, replace=False)) / 16
                if min(np.diff(pts)) >= 1 / 16 and 1 - pts[-1] + pts[0] >= 1 / 8:
                    break
            aa = pts.reshape(-1, 2).tolist()
            rot = [1] * m
            if i % 3:
                rot[i % m] = -1
            if group == "single_monitor":
                rec = [[[i % m], (-1) ** i]]
            elif group == "coarse_monitor":
                rec = [[[0, 1] if m == 2 or i != 5 else [1, 2], 1 if m == 2 else (-1) ** i]]
            else:
                rec = (
                    [[[0, 1], 1], [[1], (-1) ** i]]
                    if m == 2
                    else [[[0, 2], (-1) ** i], [[0, 1], (-1) ** (i // 2)]]
                )
                if i == 5:
                    rec = [[[0, 1, 2], -1], [[1], 1]]
            q = dict(arcs=aa, records=rec, phase_plates=rot)
            v = continuum(q)
            np.testing.assert_allclose(extrapolate(q), v, rtol=0, atol=3e-5)
            rows.append(dict(input=q, value=round(V * v, 9)))
        groups[group] = rows
    return cal, groups


def main(write):
    a, b, c, d = [0.0, 0.125, 0.375, 0.75]
    s = lambda x: np.sin(np.pi * x)
    x = s(b - a) * s(d - c) / (s(c - a) * s(d - b))
    base = (1 - x) ** (-0.25)
    vals = [
        base * (1 + np.sqrt(x) + np.sqrt(1 - x)) / 2,
        base * (1 + np.sqrt(x)),
        base * np.sqrt(1 - x),
    ]
    for flag, num in zip([None, 1, -1], vals):
        q = example(flag)
        np.testing.assert_allclose(continuum(q), num, rtol=0, atol=1e-12)
        print("two-block example", flag, num, flush=True)
    for records in ([], [[[0, 1], 1]], [[[1, 2], -1], [[0], 1]], [[[0, 1, 2], -1]]):
        for plates in ([1, 1, 1], [1, -1, 1]):
            q = dict(
                arcs=[[0.0, 0.125], [0.25, 0.375], [0.5, 0.625]],
                records=records,
                phase_plates=plates,
            )
            np.testing.assert_allclose(wave(q, 8), finite(q, 8), atol=8e-13, rtol=0)
    cal, groups = generate()
    for f, v in [
        (t / "tests/data/hidden.json", groups),
        (t / "environment/data/calibration.json", cal),
        (t / "tests/data/calibration.json", cal),
    ]:
        if write:
            f.write_text(json.dumps(v, separators=(",", ":")) + "\n")
        assert json.loads(f.read_text()) == v
    for label, path in [
        ("oracle", t / "solution/model.py"),
        ("starter", t / "environment/model.py"),
        ("independent", root / "scripts/spin_ring_swap_baseline.py"),
    ]:
        m = runpy.run_path(str(path))["Model"]().fit(cal)
        e = [
            max(abs(m.predict([r["input"] for r in rs]) - [r["value"] for r in rs]))
            for rs in groups.values()
        ]
        print(label, e, flush=True)
        if label == "oracle":
            assert max(e) < 1e-7
        elif label == "starter":
            assert min(e) > 0.003
        else:
            assert e[0] < 1e-7 and min(e[1:]) > 0.003


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generate", action="store_true")
    main(p.parse_args().generate)
