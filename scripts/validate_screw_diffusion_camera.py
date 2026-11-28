"""Check the balanced rigid stroke, gradient pickup and physical zero-pose-shift approximation."""
from pathlib import Path
import importlib.util
import sys, json, argparse
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/screw-diffusion-camera"
sys.path[:0] = [str(TASK / "solution"), str(TASK / "environment"), str(TASK / "tests")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from gradient import pickup
from gradient_reference import reference_gradient
from reference import measurement, TRUE_PARAMETER

oracle = load("oracle", TASK / "solution/model.py")
starter = load("starter", TASK / "environment/model.py")
baseline = load("baseline", ROOT / "scripts/screw_diffusion_camera_baseline.py")


def moments(q, nu):
    s = q["pickup"]
    t1, t2 = q["times"]
    T = t1 + t2
    end = s["end"]
    mid = s["middle"]
    pieces = [(0, T, end)] if mid is None else [(0, t1, mid), (t1, T, end - mid)]
    mean = nu * sum(d * ((a + b) / 2 - T / 3) for a, b, d in pieces)

    def corr(t, v):
        # E[(X_t²/2-X_T²/6)(X_v²/2-X_T²/6)] for one free planar driver.
        return nu**2 * (
            t * v + 2 * np.minimum(t, v) ** 2 - (T * (t + v) + 2 * (t * t + v * v)) / 3 + T * T / 3
        )

    nodes, weights = leggauss(3)

    def integral(a, b, fn):
        return sum(w * (b - a) / 2 * fn(a + (b - a) * (x + 1) / 2) for x, w in zip(nodes, weights))

    second = 0.0
    for a, b, d in pieces:
        selfint = 2 * integral(a, b, lambda t: integral(a, t, lambda v: corr(t, v)))
        second += 2 * nu * (integral(a, b, lambda t: corr(t, t)) - selfint / (b - a))
        for c, e, h in pieces:
            area = selfint if a == c else integral(a, b, lambda t: integral(c, e, lambda v: corr(t, v)))
            second += d * h * area / ((b - a) * (e - c))
    drift = s["stroke"] * T
    second += 2 * drift * mean + drift**2
    mean += drift
    scale = (nu * T) ** 1.5
    return mean / scale, second / scale**2


def experiments():
    rng = np.random.default_rng(305791)

    def axis(h):
        x = rng.normal(size=3)
        return np.round(h * x / np.linalg.norm(x), 5).tolist()

    groups = {}
    for name in ["origin", "tagged", "quadrature", "reversal"]:
        qs = []
        for i in range(7):
            k = np.round(rng.uniform(0.7, 2.35, 2), 4)
            if name == "reversal" or i % 2:
                k[1] = -k[1]
            q = dict(
                mode="fringe",
                times=np.round(rng.uniform(0.16, 1.25, 2), 4).tolist(),
                wave=k.tolist(),
                hand=(-1) ** i,
                tag=axis(rng.uniform(0.4, 1.26)),
                gates=[axis(rng.uniform(0.7, 0.98)), axis(rng.uniform(0.7, 0.98))],
                phase=round(rng.uniform(-0.9, 0.9), 5),
            )
            if name == "origin":
                q["tag"] = [0.0, 0.0, 0.0]
            if name == "quadrature":
                q["phase"] = 1.57079633
            qs.append(q)
        groups[name] = qs
    rng = np.random.default_rng(3376101)
    for family in ("gradient_slope", "gradient_noise", "waypoint"):
        qs = []
        for i in range(6):
            d = round(rng.uniform(0.47, 1.19) * (-1) ** i, 4)
            m = round(d * rng.uniform(-0.2, 0.83), 4) if family == "waypoint" else None
            s = dict(
                stroke=round((-1) ** (i // 2) * rng.uniform(0.38, 0.95), 5),
                axes=rng.choice(3, 2, replace=False).tolist(),
                end=d,
                middle=m,
                read="slope"
                if family == "gradient_slope" or family == "waypoint" and i % 2
                else "curvature",
            )
            qs.append(
                dict(
                    mode="fringe",
                    times=np.round(rng.uniform(0.43, 1.04, 2), 4).tolist(),
                    hand=(-1) ** i,
                    wave=np.round(rng.uniform(0.71, 1.93, 2) * [1.0, (-1) ** i], 4).tolist(),
                    gates=[axis(0.94), axis(0.93)],
                    tag=axis(rng.uniform(0.7, 1.25)),
                    phase=float([0.15, np.pi / 2, -0.35][i % 3]),
                    pickup=s,
                )
            )
        groups[family] = qs
    return groups


def sample():
    return dict(
        mode="fringe",
        hand=1,
        times=[0.52, 0.73],
        wave=[1.58, -1.18],
        tag=[0.75, -0.12, 0.64],
        gates=[[0.87, -0.23, 0.12], [-0.1, 0.77, 0.39]],
        phase=0.3,
        pickup=dict(axes=[1, 2], end=0.81, middle=-0.53, read="slope", stroke=0.65),
    )


def main(generate):
    gamma = TRUE_PARAMETER
    data = TASK / "tests/data"
    cal = [
        dict(input=dict(mode="rotation", axis=j, t=t), value=round(np.exp(-gamma * (2.49 - r) * t), 8))
        for j, r in enumerate([0.55, 0.82, 1.12])
        for t in [0.15, 0.41, 0.76, 1.23]
    ]
    assert (
        cal
        == json.loads((data / "calibration.json").read_text())
        == json.loads((TASK / "environment/data/calibration.json").read_text())
    )
    groups = experiments()
    records = {}
    for name, qs in groups.items():
        records[name] = [
            dict(
                input=q,
                value=round(reference_gradient(q, gamma) if "pickup" in q else measurement(q), 8),
            )
            for q in qs
        ]
    labels = data / "hidden.json"
    if generate:
        labels.write_text(json.dumps(records, separators=(",", ":")) + "\n")
    stored = json.loads(labels.read_text())
    assert stored.keys() == records.keys()
    for g, rs in records.items():
        assert [r["input"] for r in rs] == [r["input"] for r in stored[g]]
        np.testing.assert_allclose(
            [r["value"] for r in rs], [r["value"] for r in stored[g]], atol=8e-8, rtol=0
        )
    for module in [oracle, baseline, starter]:
        m = module.Model().fit(cal)
        errors = {
            g: max(abs(m.predict(qs) - [r["value"] for r in records[g]])) for g, qs in groups.items()
        }
        print(module.__name__, m.mobility, errors, flush=True)
        assert abs(m.mobility - gamma) < 1e-7
        if module is oracle:
            assert max(errors.values()) < 4e-4
        else:
            assert max(errors[g] for g in ("origin", "tagged", "quadrature", "reversal")) < 5e-5
            assert min(errors[g] for g in ("gradient_slope", "gradient_noise", "waypoint")) > 0.003
    q = sample()
    for mid in (None, -0.53):
        for r, read in enumerate(("slope", "curvature"), 1):
            shot = dict(q, pickup=dict(q["pickup"], middle=mid, read=read))
            bare = dict(
                shot, wave=[0.0, 0.0], gates=[[0.0, 0.0, 0.0]] * 2, phase=np.pi / 2 if r == 1 else 0.0
            )
            target = moments(bare, gamma)[r - 1]
            assert abs(pickup(bare, gamma) - target) < 3e-7
            assert abs(reference_gradient(bare, gamma) - target) < 1e-4
            assert abs(pickup(shot, gamma) - reference_gradient(shot, gamma)) < 5e-5
    for speed, target in ((-0.65, 0.01169334), (0.65, 0.20588342)):
        shot = dict(q, pickup=dict(q["pickup"], read="slope", stroke=speed))
        assert abs(pickup(shot, gamma) - target) < 2e-6
        assert abs(reference_gradient(shot, gamma) - target) < 3e-6
        assert abs(pickup(shot, gamma, body_drive=False) - target) > 0.015
    for name in ("gradient_slope", "gradient_noise", "waypoint"):
        q = groups[name][2]
        assert (
            abs(
                pickup(q, gamma, order=8, nodes=24)
                - reference_gradient(q, gamma, order=8, h=0.001, nodes=41)
            )
            < 1e-4
        )


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--generate", action="store_true")
    main(cli.parse_args().generate)
