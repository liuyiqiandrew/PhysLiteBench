"""Check the two residence recorders against the distribution of a frozen ordinary glass."""
import argparse
import json
from pathlib import Path
import runpy, sys
import numpy as np
from scipy.special import roots_laguerre, i0, i1

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/aging-line-memory"
sys.path[:0] = [str(TASK / p) for p in ("environment", "solution", "tests")]
from clocks import readings
from reference import truth


def preparations():
    rng = np.random.default_rng(562974)

    def draw(a, b):
        return round(float(rng.uniform(a, b)), 4)

    out = {}
    for g in ("unvisited", "two_recorders", "host_spectrum"):
        rows = []
        for i in range(8):
            while True:
                q = dict(
                    temperature=draw(0.535, 0.705),
                    clock=draw(0.6, 1.4),
                    offsets=[draw(-0.69, 0.28), draw(-0.65, 0.31)],
                    levels=[
                        draw(0.43, 1.63),
                        0.0
                        if g == "unvisited" or (g == "host_spectrum" and i % 4 == 0)
                        else draw(0.25, 1.07),
                    ],
                )
                if g == "host_spectrum":
                    q["read"] = "host"
                    lo = draw(0.12, 0.78)
                    q["band"] = [lo, round(lo + draw(0.55, 2.12), 4)]
                elif g == "unvisited":
                    q["read"] = "host" if i % 3 == 0 else "A"
                else:
                    q["read"] = ("A", "B", "host")[i % 3]
                if truth(q, 0.9863)[1] > 0.008:
                    break
            rows.append(q)
        out[g] = rows
    return out


def finite_chain_check():
    # Keep every ordinary site and both reporters as explicit Markov states.
    rates = np.array([1.6, 0.57, 0.2, 0.9, 2.2, 0.46, 3.8, 9.0])
    s = 0.83
    na = len(rates)
    H = np.sum(s / (s + rates[2:]))
    exits = np.array(
        [[s, 0.0, rates[0] * H / (2 + H)], [0.0, s, rates[1] * H / (2 + H)]]
    )
    pA, pB = 0.68, 1.27
    dep = rates[:2] * (1 - 1 / (2 + H)) + s
    aa, bb = rates[:2] / (2 + H)
    z, w = roots_laguerre(42)
    x = z[:, None] / (pA + dep[0])
    y = z[None, :] / (pB + dep[1])
    bxy = np.sqrt(aa * bb * x * y)
    # Integrate the B=0 mass and the positive two-clock density with two penalties.
    base = np.array([1 / (pA + dep[0]), 0.0])
    A = aa * bb * x * i1(2 * bxy) / bxy
    B = aa * i0(2 * bxy)
    base += np.array([w @ A @ w, w @ B @ w]) / np.prod([pA + dep[0], pB + dep[1]])
    M = (
        np.diag(s + rates + np.r_[pA, pB, np.zeros(na - 2)])
        - rates[:, None] * np.ones((na, na)) / na
    )
    mask = np.column_stack((np.arange(na) == 0, np.arange(na) == 1, np.arange(na) >= 2))
    direct = s * np.linalg.solve(M, mask)[0]
    np.testing.assert_allclose(base @ exits, direct, rtol=0, atol=2e-13)


def main(generate):
    tg = 0.9863
    cal = [
        dict(input=dict(threshold=x), value=round(float(np.exp(-x / tg)), 8))
        for x in (0.22, 0.35, 0.52, 0.66, 0.80, 1.05, 1.24, 1.46, 1.68, 1.92)
    ]
    data = {}
    masses = []
    worst = 0.0
    for g, qs in preparations().items():
        rows = []
        for q in qs:
            y, p = truth(q, tg)
            v = readings(q, tg)
            worst = max(worst, float(max(abs(np.array(v) - [y, p]))))
            np.testing.assert_allclose(v, [y, p], atol=3e-7, rtol=0)
            assert p > 0.008 and 0 < y < 1
            masses.append(p)
            rows.append(dict(input=q, value=round(y, 9)))
        data[g] = rows
    for name, obj in [
        ("tests/data/calibration.json", cal),
        ("environment/data/calibration.json", cal),
        ("tests/data/predictions.json", data),
    ]:
        f = TASK / name
        if generate:
            f.write_text(json.dumps(obj, indent=1) + "\n")
        frozen = json.loads(f.read_text())
        if isinstance(obj, list):
            assert obj == frozen
        else:
            assert obj.keys() == frozen.keys()
            for k, rows in obj.items():
                assert [r["input"] for r in rows] == [r["input"] for r in frozen[k]]
                np.testing.assert_allclose(
                    [r["value"] for r in rows],
                    [r["value"] for r in frozen[k]],
                    atol=1.1e-9,
                    rtol=0,
                )
    print(
        "Range of pooled mass/density",
        min(masses),
        max(masses),
        "independent discrepancy",
        worst,
    )
    for role, path in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("shortcut", ROOT / "scripts/aging_line_memory_baseline.py"),
    ]:
        model = runpy.run_path(str(path))["Model"]().fit(cal)
        assert abs(model.depth_scale - tg) < 1e-8
        errors = {}
        for k, rs in data.items():
            y = model.predict([r["input"] for r in rs])
            assert np.isfinite(y).all()
            errors[k] = float(max(abs(y - [r["value"] for r in rs])))
        print(role, errors)
        assert (
            max(errors.values()) < 1e-6
            if role == "oracle"
            else min(errors.values()) > 0.004
        )
    finite_chain_check()
    q = dict(
        temperature=0.6, clock=1.0, offsets=[0.0, 0.0], levels=[0.8, 0.0], read="A"
    )
    for method in (truth, readings):
        assert abs(method(q, 1.0)[0] - 0.68095082) < 1e-7
        assert method(dict(q, read="B"), 1.0)[0] == 0.0
        for k, v in [("A", 0.12559963), ("B", 0.57100954), ("host", 0.10029672)]:
            a = dict(q, levels=[0.8, 0.55], read=k)
            if k == "host":
                a["band"] = [0.25, 1.65]
            assert abs(method(a, 1.0)[0] - v) < 1e-7
    q = data["two_recorders"][0]["input"]
    a, b, c = (readings(dict(q, read=i), tg)[0] for i in ("A", "B", "host"))
    assert abs(a + b + c - 1) < 1e-13


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    main(parser.parse_args().generate)
