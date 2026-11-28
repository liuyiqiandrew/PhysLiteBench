"""Compare acoustic heralding with explicit finite-occupancy flip labels."""
from math import factorial
from pathlib import Path
import argparse, json, runpy, sys
import numpy as np

root = Path(__file__).resolve().parents[1]
t = root / "tasks/defect-spectral-hole"
sys.path[:0] = [str(t / k) for k in ["environment", "solution", "tests"]]
from echo import reading
from reference import truth

V = 0.83642


def example(colors=("R", "R")):
    return dict(
        phase=0.1,
        mix=[0.0, 0.0],
        gate=[0.9, 0.9],
        unsigned_b=False,
        kicks=[[1.1, 1.2, 0.8]],
        domains=[
            dict(weight=1.0, ratio=1.0, pockets=[[0.7, 0.9, 0.5], [0.8, 0.8, 0.4], [0.5, 1.0, 0.9]])
        ],
        microphones=[[0.0, 0.4], [0.0, 0.8], [1.0, 0.6]],
        listen_until=1.2,
        clicks=[[0.25, colors[0]], [0.85, colors[1]]],
    )


def counted(same):
    x = 0.9
    h = 2.0
    phi = 0.1
    total = 0.0
    num = 0.0
    for n in range(1, 23):
        # At fixed n, the flip rates and Poisson ordering densities are explicit.
        g = (
            np.exp(-0.5 - n * 0.9 * 1.2 - x * x / (2 * n))
            * 0.5**n
            / (factorial(n) * np.sqrt(2 * np.pi * n))
        )
        one = n * 0.6 if same else 0.0
        two = n * (n - 1) * 0.6 * (0.6 if same else 0.4)
        gain = np.cos(phi + h * x * (1 - 4 / n)) * np.exp(-2 * h * h * max(2 - 4 / n, 0.0))
        total += g * (one + two)
        num += g * (one * np.cos(phi + h * x) + two * gain)
    return num / total


def generate():
    rng = np.random.default_rng(118541)

    def s(a, b):
        return round(rng.uniform(a, b), 4)

    def pocket():
        return [s(0.35, 1.05), s(0.66, 1.0), s(0.45, 1.22)]

    cal = [
        dict(input=dict(blank=True, phase=p), value=round(V * np.cos(p), 8))
        for p in np.linspace(-2.4, 2.5, 12).round(4)
    ]
    groups = {}
    for group in ["quiet_window", "tagged_flips", "repeated_emission"]:
        rows = []
        for i in range(7):
            u, v = s(-0.25, 0.25), s(-0.25, 0.25)
            r = s(0.7, 1.28) * (-1) ** i
            w = s(0.23, 0.77)
            domains = [
                dict(weight=w, ratio=r, pockets=[pocket() for j in range(3)]),
                dict(weight=1 - w, ratio=-r, pockets=[pocket() for j in range(3)]),
            ]
            mics = [[s(0.55, 0.98), s(0.13, 0.87)] for j in range(3)]
            end = s(1.05, 1.48)
            kicks = [[s(0.42, 0.65), s(1.4, 2.7) * (-1) ** i, s(0.8, 2.0) * (-1) ** (i // 2)]]
            if i % 3 != 1:
                kicks.append([s(1.12, 1.65), s(1.1, 2.5), s(1.3, 2.7) * (-1) ** i])
            taps = [] if group == "quiet_window" else [[s(0.14, 0.39), "R" if i % 2 else "B"]]
            if group == "repeated_emission" or (group == "tagged_flips" and i >= 4):
                taps.append(
                    [
                        s(0.7, 0.94),
                        taps[0][1]
                        if group == "repeated_emission"
                        else ("R" if taps[0][1] == "B" else "B"),
                    ]
                )
            x = s(0.4, 1.19) * (-1) ** i
            y = s(0.42, 1.31) * (-1) ** (i // 2)
            if i % 3 == 0:
                y = r * x
            if i == 1:
                y = 0.0
            if i == 3:
                x = y = 0.0
            if i == 5:
                x = 0.0
                if group == "tagged_flips":
                    # Two different acoustic colors force two different private pockets.
                    mics = [[1.0, 1.0], [1.0, 0.0], [0.0, 0.5]]
                    x = y = 0.0
            a, b = x + u * y, v * x + y
            folded = i >= 4
            q = dict(
                phase=s(-2.7, 2.7),
                mix=[u, v],
                gate=[a, abs(b) if folded else b],
                unsigned_b=folded,
                kicks=kicks,
                domains=domains,
                microphones=mics,
                listen_until=end,
                clicks=taps,
            )
            val = truth(q)
            np.testing.assert_allclose(reading(q), val, atol=4e-7, rtol=0)
            rows.append(dict(input=q, value=round(V * val, 9)))
        groups[group] = rows
    return cal, groups


def main(write):
    for same in (True, False):
        q = example(("R", "R" if same else "B"))
        e = counted(same)
        print("worked, same color" if same else "worked, different colors", e, flush=True)
        for method in (truth, reading):
            np.testing.assert_allclose(method(q), e, rtol=0, atol=2e-8)
    cal, groups = generate()
    for f, data in [
        (t / "tests/data/hidden.json", groups),
        (t / "environment/data/calibration.json", cal),
        (t / "tests/data/calibration.json", cal),
    ]:
        if write:
            f.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        assert json.loads(f.read_text()) == data
    for label, path in [
        ("oracle", t / "solution/model.py"),
        ("starter", t / "environment/model.py"),
        ("distinct", root / "scripts/defect_spectral_hole_baseline.py"),
    ]:
        m = runpy.run_path(str(path))["Model"]().fit(cal)
        values = [
            max(abs(m.predict([r["input"] for r in rows]) - [r["value"] for r in rows]))
            for rows in groups.values()
        ]
        print(label, values, flush=True)
        if label == "oracle":
            assert max(values) < 4e-7
        elif label == "starter":
            assert min(values) > 0.003
        else:
            assert max(values[:2]) < 4e-7 and values[2] > 0.003


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    main(parser.parse_args().generate)
