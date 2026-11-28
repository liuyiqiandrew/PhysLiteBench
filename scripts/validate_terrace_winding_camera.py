"""Validate tagged reflected pairs and their joint contact-clock densities."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, roots_legendre

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/terrace-winding-camera"
sys.path.insert(0, str(TASK / "solution"))
from camera import camera, gap_kernel


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def groups():
    rng = np.random.default_rng(529408)
    result = {}
    for branch in ("avoiding", "one_recorder", "two_recorders", "tagged"):
        rows = []
        for i in range(9):
            L = [7.0, 8.0, 9.0][i % 3]

            def pins():
                gap = round(float(rng.uniform(0.75, L - 1)), 3)
                x = round(float(rng.uniform(0.1, L - gap - 0.1)), 3)
                return [x, round(x + gap, 3)]

            bot, t = pins(), pins()
            if branch == "tagged":
                lo, hi = sorted(rng.choice(np.arange(1, 19) * L / 20, 2, replace=False))
                win = [round(float(lo), 3), round(float(hi), 3)]
            else:
                win = (
                    [0.0, L] if i % 2 == 0 else [round(0.14 * L, 3), round(0.78 * L, 3)]
                )
            clock = [round(float(rng.uniform(0.03, 0.68)), 3) for _ in range(2)]
            if branch == "avoiding":
                clock = [0.0, 0.0]
            if branch == "one_recorder":
                clock[i % 2] = 0.0
            if branch == "tagged" and i % 3 < 2:
                clock[i % 3] = 0.0
            q = dict(
                mode="contact",
                circumference=L,
                height=round(float(rng.uniform(9, 18)), 3),
                level=round(float(rng.uniform(0.35, 0.85)), 3),
                bottom=bot,
                top=t,
                tag=i % 2,
                winding=(-1, 0, 1)[i % 3],
                window=win,
                clocks=clock,
            )
            rows.append(q)
        result[branch] = rows
    return result


def robin(q, rate, penalty):
    gamma = rate
    D = 2 * gamma
    L = q["circumference"]
    T = q["height"]
    t = T * q["level"]
    g = q["bottom"][1] - q["bottom"][0]
    end = [
        q["top"][(i + q["winding"]) % 2] + ((i + q["winding"]) // 2) * L
        for i in range(2)
    ]
    e = end[1] - end[0]
    mean = ((1 - t / T) * sum(q["bottom"]) + (t / T) * sum(end)) / 2
    ka, kb = penalty
    ka /= L
    kb /= L

    def phase(k):
        return k * L - np.arctan(ka / k) - np.arctan(kb / k)

    ks = np.array(
        [
            brentq(
                lambda k: phase(k) - j * np.pi,
                max(1e-10, j * np.pi / L),
                (j + 1) * np.pi / L,
            )
            for j in range(44)
        ]
    )
    theta = np.arctan(ka / ks)
    norm = L / 2 + (np.sin(2 * ks * L - 2 * theta) + np.sin(2 * theta)) / (4 * ks)
    x, w = roots_legendre(64)
    x = (1 + x) * L / 2
    w *= L / 2
    first = np.cos(ks * g - theta) * np.exp(-D * t * ks**2) / norm
    prop = np.cos(x[:, None] * ks - theta) @ first
    sig = np.sqrt(gamma * t * (1 - t / T))
    loc = mean + (q["tag"] - 0.5) * x[:, None]
    s = L * np.arange(-6, 7)
    lo, hi = q["window"]
    capture = np.sum(ndtr((hi + s - loc) / sig) - ndtr((lo + s - loc) / sig), axis=1)
    return (
        w @ (capture * prop * gap_kernel(x, e, T - t, L, D)) / gap_kernel(g, e, T, L, D)
    )


def reweighted(q, rate, k):
    x, w = roots_legendre(22)
    x = np.r_[0, (1 + x) * 2.5]
    w = np.r_[1, w * 2.5]
    total = 0.0
    for i, a in enumerate(x):
        for j, b in enumerate(x):
            value = camera(dict(q, clocks=[a, b]), rate)
            total += w[i] * w[j] * np.exp(-k[0] * a - k[1] * b) * value
    return total, robin(q, rate, k)


def main(generate):
    ref = load(TASK / "tests/reference.py")
    rate = 0.927
    calibrations = [
        dict(input=dict(mode="wandering", height=h), value=round(2 * rate * h, 8))
        for h in [0.35, 0.707, 0.92, 1.643, 2.21, 3.735, 4.996, 6.207]
    ]
    predicted = {}
    for name, qs in groups().items():
        predicted[name] = [
            dict(input=q, value=round(ref.image(q, rate), 8)) for q in qs
        ]
    for path, data in [
        (TASK / "tests/data/calibration.json", calibrations),
        (TASK / "environment/data/calibration.json", calibrations),
        (TASK / "tests/data/predictions.json", predicted),
    ]:
        if generate:
            path.write_text(json.dumps(data, indent=1) + "\n")
        frozen = json.loads(path.read_text())
        if isinstance(data, list):
            assert frozen == data
        else:
            assert frozen.keys() == data.keys()
            for family, rows in data.items():
                actual = frozen[family]
                assert [row["input"] for row in actual] == [
                    row["input"] for row in rows
                ]
                np.testing.assert_allclose(
                    [row["value"] for row in actual],
                    [row["value"] for row in rows],
                    atol=1e-7,
                    rtol=0,
                )
    models = {
        "oracle": TASK / "solution/model.py",
        "shortcut": ROOT / "scripts/terrace_winding_camera_baseline.py",
        "starter": TASK / "environment/model.py",
    }
    for name, path in models.items():
        model = load(path).Model().fit(calibrations)
        assert abs(model.rate - rate) < 1e-8
        errors = {}
        for family, rows in predicted.items():
            signals = model.predict([r["input"] for r in rows])
            assert np.isfinite(signals).all()
            errors[family] = float(max(abs(signals - [r["value"] for r in rows])))
        print(name, errors)
        if name == "oracle":
            assert max(errors.values()) < 3e-7
        else:
            assert min(errors.values()) > 0.004
    q = dict(
        circumference=8.0,
        height=17.0,
        level=0.65,
        bottom=[1.0, 3.5],
        top=[0.7, 5.0],
        winding=1,
        tag=1,
        window=[0.4, 6.7],
    )
    for levels, expected in [([0.15, 0.25], 0.49221), ([0.0, 0.0], 0.03089)]:
        assert abs(camera(dict(q, clocks=levels), rate) - expected) < 5e-6
    # Moving the origin between the bottom pins reverses both recorder names.
    # Use a window that stays within one periodic image after the shift.
    q0 = dict(q, winding=0, window=[2.5, 6.0])
    cut = 2.0
    length = q0["circumference"]
    shifted = dict(
        q0,
        bottom=sorted((x - cut) % length for x in q0["bottom"]),
        top=sorted((x - cut) % length for x in q0["top"]),
        winding=1 - sum(x < cut for x in q0["top"]),
        tag=1 - q0["tag"],
        window=[x - cut for x in q0["window"]],
    )
    for clocks in ([0.0, 0.27], [0.15, 0.25], [0.0, 0.0]):
        signal = camera(dict(q0, clocks=clocks), rate)
        translated = camera(dict(shifted, clocks=clocks[::-1]), rate)
        assert abs(signal - translated) < 1e-10
    for penalty in [[0.7, 1.2], [1.9, 0.4]]:
        reconstructed, direct = reweighted(q, rate, penalty)
        print("Robin heat kernel", reconstructed, direct)
        assert abs(reconstructed - direct) < 1e-10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    main(parser.parse_args().generate)
