"""Compare coherent pair amplitudes with direct spin operators and Gaussian quadrature."""
from pathlib import Path
import argparse, json, runpy, sys
import numpy as np

root = Path(__file__).resolve().parents[1]
t = root / "tasks/defect-spectral-hole"
sys.path[:0] = [str(t / s) for s in ["environment", "solution", "tests"]]
from echo import reading
from reference import truth

V = 0.83642


def example():
    flat = [[1.0, 0.0]] * 3
    return dict(
        phase=0.32,
        mix=[0.0, 0.0],
        gate=[0.7, 0.0],
        unsigned_b=False,
        count=3,
        domains=[
            dict(
                weight=1.0, ratio=-1.0, pockets=[[0.7, 0.8, 0.7], [0.8, 0.8, 0.5], [0.9, 0.9, 0.7]]
            )
        ],
        pulses=[[0.65, 0.0, 0.0, 1.8, 0.0]],
        write=[flat, flat],
        read=[[flat, flat], [flat, flat]],
    )


def counted():
    # One pocket on a line with three sites, all optical modes uniform.
    # Two pairs either coincide or share exactly one site. The reference arm has no phase kicks.
    from itertools import product

    x = 0.7
    sig = 0.8
    h = 1.8
    theta = 0.32
    m = np.exp(-2 * 0.7 * 0.65)

    def char(c):
        value = 0j
        for s in product((1, -1), repeat=3):
            phase = h * np.array(c) * s
            w = np.prod((1 + m * np.array(s)) / 2)
            value += w * np.exp(
                -1j * x * phase.mean() - 0.5 * sig**2 * sum((phase - phase.mean()) ** 2)
            )
        return value

    one = char([1, 1, 0])
    shared = char([0, 1, -1])
    return 2 * (np.exp(1j * theta) * one).real / (1 + (3 + 6 * shared.real) / 9)


def generate():
    rng = np.random.default_rng(779341)

    def s(a, b):
        return round(rng.uniform(a, b), 4)

    def pop():
        return [s(0.35, 1.15), s(0.62, 0.93), s(0.3, 0.93)]

    def beam():
        return [[s(0.48, 1.0), s(-2.7, 2.7)] for k in range(3)]

    cal = [
        dict(input=dict(blank=True, phase=p), value=round(V * np.cos(p), 8))
        for p in np.linspace(-2.4, 2.5, 12).round(4)
    ]
    groups = {}
    for name in ["two_sites", "pair_interference", "unresolved_domains"]:
        rows = []
        for i in range(5):
            n = 2 if name == "two_sites" else 3 + i % 2
            u, v = s(-0.21, 0.22), s(-0.2, 0.24)
            r = (-1) ** i * s(0.67, 1.34)
            w = s(0.25, 0.77)
            domains = [
                dict(weight=w, ratio=r, pockets=[pop() for j in range(3)]),
                dict(weight=1 - w, ratio=-r, pockets=[pop() for j in range(3)]),
            ]
            if name != "unresolved_domains":
                domains = domains[:1]
                domains[0]["weight"] = 1.0
            x = s(0.45, 1.18) * (-1) ** i
            y = s(0.42, 1.17) * (-1) ** (i // 2)
            if i == 0:
                y = r * x
            if i == 2:
                y = 0.0
            if i == 3 and name == "unresolved_domains":
                u = v = 0.0
                y = r * x
            a = x + u * y
            b = v * x + y
            kicks = [[s(0.13, 0.78), s(-0.9, 1.3), s(-1.14, 1.3), s(-1.2, 1.4), s(-1.3, 1.32)]]
            if i in (1, 2, 4):
                kicks.append(
                    [s(0.88, 1.58), s(-1.3, 1.15), s(-1.1, 1.35), s(-1.25, 1.1), s(-1.2, 1.3)]
                )
            q = dict(
                count=n,
                phase=s(-2.6, 2.6),
                mix=[u, v],
                gate=[a, abs(b) if name == "unresolved_domains" else b],
                unsigned_b=name == "unresolved_domains",
                pulses=kicks,
                domains=domains,
                write=[beam(), beam()],
                read=[[beam(), beam()], [beam(), beam()]],
            )
            expected = truth(q)
            oracle = reading(q)
            np.testing.assert_allclose(oracle, expected, rtol=0, atol=2e-6)
            rows.append(dict(input=q, value=round(V * expected, 9)))
        groups[name] = rows
    return cal, groups


def main(write):
    ans = counted()
    print("unlabeled pair example", ans, flush=True)
    for m in (truth, reading):
        np.testing.assert_allclose(m(example()), ans, rtol=0, atol=5e-8)
    cal, groups = generate()
    for p, value in [
        (t / "tests/data/hidden.json", groups),
        (t / "tests/data/calibration.json", cal),
        (t / "environment/data/calibration.json", cal),
    ]:
        if write:
            p.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        assert json.loads(p.read_text()) == value
    for label, path in [
        ("oracle", t / "solution/model.py"),
        ("starter", t / "environment/model.py"),
        ("pair record", root / "scripts/defect_spectral_hole_baseline.py"),
    ]:
        m = runpy.run_path(str(path))["Model"]().fit(cal)
        err = [
            max(abs(m.predict([r["input"] for r in rows]) - [r["value"] for r in rows]))
            for rows in groups.values()
        ]
        print(label, err, flush=True)
        if label == "oracle":
            assert max(err) < 2e-6
        elif label == "starter":
            assert min(err) > 0.003
        else:
            assert err[0] < 2e-6 and min(err[1:]) > 0.003


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generate", action="store_true")
    main(p.parse_args().generate)
