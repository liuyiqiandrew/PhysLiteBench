import argparse, json, runpy, sys
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/critical-seam-magnet"
sys.path[:0] = [str(TASK / d) for d in ("environment", "solution", "tests")]
from sectors import blocks, M
from film import device
from lattice import fermions
from spin_enumeration import explicit


@lru_cache(None)
def substrate(nx, ny, points):
    return fermions(nx, ny, points)


def calc(q, scale):
    def measure(f):
        nx = 16 * scale
        ny = round(nx * f["aspect"])
        p = tuple(map(tuple, np.rint(np.array(f["marks"]) * [nx, ny]).astype(int)))
        return M @ substrate(nx, ny, p), M @ substrate(nx, ny, ())

    return device(q, 0.9268, measure)


def choices():
    rng = np.random.default_rng(827361)
    library = []
    for k in range(12):
        n = 2 + 2 * (k % 3)
        alpha = [0.75, 1.0, 1.25][(k + k // 3) % 3]
        while True:
            p = rng.integers(1, 8, size=(n, 2))
            dif = abs(p[:, None, :] - p[None, :, :])
            d = np.minimum(dif, 8 - dif).astype(float)
            if (np.triu((d * d).sum(2) < 4, 1)).any() or len(set(p[:, 1])) == 1:
                continue
            break
        library.append(dict(aspect=alpha, marks=(p / 8).tolist()))
    groups = {}
    for g in ["overlap", "connected", "response"]:
        rows = []
        for i in range(8):
            I, J = rng.choice(len(library), size=2, replace=False)
            q = dict(
                films=[library[I], library[J]],
                bias=np.round(rng.uniform(-0.9, 0.9, 2), 3).tolist(),
                coupling=round(float(rng.uniform(-1, 1)), 3),
                read=g,
            )
            if g == "response":
                q["dial"] = int(i % 2)
            rows.append(q)
        groups[g] = rows
    return groups


def main(generate):
    gain = 0.9268
    cal = [
        dict(input=dict(bias=h), value=round(float(np.tanh(gain * h)), 8))
        for h in (-1.30, -1, -0.76, -0.45, -0.22, 0.18, 0.39, 0.70, 0.94, 1.24)
    ]
    data = {}
    worst = 0.0
    finite = 0.0
    for g, qs in choices().items():
        rows = []
        for q in qs:
            exact = device(q, gain, blocks)
            short, long = (calc(q, l) for l in (2, 4))
            worst = max(worst, abs((4 * long - short) / 3 - exact))
            finite = max(finite, abs(long - exact))
            assert abs((4 * long - short) / 3 - exact) < 2e-6
            if q["read"] == "response":
                axis = q["dial"]
                h = 0.00001
                vec = np.eye(2)[axis]

                def g0(shift):
                    return device(
                        dict(q, read="overlap", bias=np.array(q["bias"]) + shift * vec),
                        gain,
                        blocks,
                    )

                assert abs((g0(h) - g0(-h)) / (2 * h) - exact) < 1e-8
            rows.append(dict(input=q, value=round(exact, 9)))
        data[g] = rows
    print(
        "Maximum finite-film error at 64 columns and after extrapolation", finite, worst
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
    for role, where in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("shortcut", ROOT / "scripts/critical_seam_magnet_baseline.py"),
    ]:
        m = runpy.run_path(str(where))["Model"]().fit(cal)
        assert abs(m.gain - gain) < 1e-7
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
    for pts in [
        [(0, 0), (2, 2)],
        [(0, 0), (2, 1), (1, 2), (0, 3)],
        [(1, 0), (2, 1), (0, 2), (1, 3)],
    ]:
        obs, Z = explicit(3, 4, pts)
        np.testing.assert_allclose(obs, M @ fermions(3, 4, pts), rtol=0, atol=2e-12)
        np.testing.assert_allclose(Z, M @ fermions(3, 4, []), rtol=0, atol=2e-12)
    q = dict(
        films=[
            dict(aspect=0.75, marks=[[0.25, 0.25], [0.625, 0.75]]),
            dict(
                aspect=1.0,
                marks=[[0.125, 0.25], [0.25, 0.5], [0.625, 0.25], [0.75, 0.75]],
            ),
        ],
        bias=[0.35, -0.2],
        coupling=0.4,
        dial=1,
    )
    for read, y in [
        ("connected", 0.09834891),
        ("overlap", 0.86751282),
        ("response", 0.17310032),
    ]:
        assert abs(device(dict(q, read=read), 1.0, blocks) - y) < 1e-8


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    main(p.parse_args().generate)
