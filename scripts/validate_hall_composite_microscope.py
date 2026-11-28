"""Regenerate Hall microscope data with the constructive laminate reference."""
from pathlib import Path
import sys, json, argparse, importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/hall-composite-microscope"
sys.path[:0] = [str(TASK / "solution"), str(TASK / "environment"), str(TASK / "tests")]
from meter import homogeneous
from laminate import probe
from duality import signal

GAIN = 0.8736


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample():
    return dict(
        mode="local",
        materials=[
            dict(
                longitudinal=[[1.1, 0.38], [0.38, 2.1]],
                hall=[[0.55, 0.43], [0.43, -0.2]],
            ),
            dict(
                longitudinal=[[3.2, -0.50], [-0.50, 0.7]],
                hall=[[-0.61, 0.27], [0.27, 1.05]],
            ),
        ],
        bias=[1.1, -0.85],
        drives=[[[0.8, 0.1], [-0.2, 0.5]], [[-0.4, 0.7], [0.6, 0.15]]],
        contacts=[[0.75, -0.4], [0.3, 0.8]],
        phase=0,
        angle=0.0,
    )


def shots():
    rng = np.random.default_rng(583271)

    def unit(n, mag):
        v = rng.normal(size=n)
        return np.round(mag * v / np.linalg.norm(v), 5).tolist()

    def scale_matrix(values):
        t = rng.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        m = (R * values) @ R.T
        return np.round((m + m.T) / 2, 5).tolist()

    def material():
        return dict(
            longitudinal=scale_matrix(
                [rng.uniform(0.32, 1.46), rng.uniform(1.61, 4.6)]
            ),
            hall=scale_matrix([rng.uniform(-1.9, -0.13), rng.uniform(0.12, 1.95)]),
        )

    def drive():
        return [unit(2, rng.uniform(0.4, 0.98)), unit(2, rng.uniform(0.4, 0.98))]

    def bulk():
        return dict(
            bias=round(rng.uniform(0.5, 1.44) * rng.choice([-1, 1]), 5),
            drive=drive(),
            contact=unit(2, 0.95),
            angle=round(rng.uniform(-2.8, 2.8), 5),
        )

    training = [
        dict(mode="calibration", material=material(), **bulk()) for _ in range(12)
    ]
    groups = {
        "bulk": [
            dict(mode="bulk", materials=[material(), material()], **bulk())
            for _ in range(8)
        ]
    }
    for family in ("field_power", "cross_bias", "circulation"):
        group = []
        for i in range(8):
            a = bulk()
            e = a["drive"]
            u = a["contact"]
            ident = family == "field_power"
            d = e if ident else drive()
            v = u if ident else unit(2, 0.93)
            f = a["bias"]
            g = (
                f
                if ident
                else round(
                    rng.uniform(0.57, 1.42) * np.sign(f) * (-1 if i % 3 else 1), 5
                )
            )
            group.append(
                dict(
                    mode="local",
                    materials=[material(), material()],
                    bias=[f, g],
                    drives=[e, d],
                    contacts=[u, v],
                    phase=i % 2,
                    angle=0.0
                    if ident
                    else float(
                        np.pi / 2
                        if family == "circulation"
                        else [-0.34, 0.0, 0.49][i % 3]
                    ),
                )
            )
        groups[family] = group
    return training, groups


def checks():
    from checkerboard import probe as cell_probe

    q = sample()
    a, b = q["materials"]
    scalar0 = dict(longitudinal=[[1, 0], [0, 2]], hall=[[0, 0], [0, 0]])
    scalar1 = dict(scalar0, longitudinal=[[4.5, 0], [0, 2]])
    e = [[1, 0], [0, 0]]
    f = [[0, 1], [0, 0]]
    g = np.sqrt(4.5)
    scalar = dict(q, materials=[scalar0, scalar1], contacts=[[1, 0], [1, 0]])
    for shot, wanted in [
        (dict(scalar, mode="bulk", bias=0, drive=e, contact=[1, 0]), g),
        (dict(scalar, drives=[e, e], phase=0), g / 2),
        (dict(scalar, drives=[e, e], phase=1), 1 / (2 * g)),
        (dict(scalar, drives=[e, f], angle=np.pi / 2, phase=0), g * g / (1 + g * g)),
        (dict(scalar, drives=[e, f], angle=np.pi / 2, phase=1), 1 / (1 + g * g)),
    ]:
        assert abs(signal(shot, GAIN) - wanted) < 1e-11
        assert abs(probe(shot, GAIN) - wanted) < 1e-11

    for materials in (
        [a, b],
        [a, a],
        [scalar0, scalar1],
        [scalar0, scalar0],
        [scalar0, dict(scalar0, longitudinal=[[2, 0], [0, 4]])],
        [
            dict(scalar0, longitudinal=[[0.25, 0], [0, 5]], hall=[[0, 2], [2, 0]]),
            dict(scalar0, longitudinal=[[5, 0], [0, 0.25]], hall=[[-2, 0], [0, 2]]),
        ],
    ):
        for bias in ([1.5, -1.5], [0, 0], [0.7, 0.7], [1e-13, -1e-13]):
            shot = dict(q, materials=materials, bias=bias, angle=0.72)
            exact = signal(shot, GAIN)
            assert abs(probe(shot, GAIN, steps=14) - exact) < 1e-9
            assert (
                abs(
                    signal(dict(shot, materials=materials[::-1], phase=1), GAIN) - exact
                )
                < 1e-10
            )
            rev = dict(
                shot,
                bias=bias[::-1],
                drives=q["drives"][::-1],
                contacts=q["contacts"][::-1],
                angle=-0.72,
            )
            assert abs(signal(rev, GAIN) - exact) < 1e-10
            wedges = sum(
                signal(dict(shot, angle=np.pi / 2, phase=p), GAIN) for p in (0, 1)
            )
            left, right = [
                np.asarray(c) @ np.asarray(d)
                for c, d in zip(q["contacts"], q["drives"])
            ]
            assert abs(wedges - np.linalg.det([left, right])) < 1e-10

    # A conforming periodic mesh resolves the specified checkerboard directly.
    exact = signal(q, GAIN)
    errors = [abs(cell_probe(q, GAIN, n) - exact) for n in (12, 24, 48, 96)]
    assert all(y < 0.56 * x for x, y in zip(errors, errors[1:])) and errors[-1] < 0.0008
    print("checkerboard triangular mesh errors", errors, flush=True)


def main(generate):
    checks()
    training, groups = shots()
    cal = [dict(input=q, value=round(homogeneous(q, GAIN), 8)) for q in training]
    truth = {
        g: [dict(input=q, value=round(probe(q, GAIN), 8)) for q in qs]
        for g, qs in groups.items()
    }
    for path, rows in [
        (TASK / "environment/data/calibration.json", cal),
        (TASK / "tests/data/calibration.json", cal),
        (TASK / "tests/data/hidden.json", truth),
    ]:
        if generate:
            path.write_text(json.dumps(rows, separators=(",", ":")) + "\n")
        assert json.loads(path.read_text()) == rows
    for name, path in [
        ("oracle", TASK / "solution/model.py"),
        ("control", ROOT / "scripts/hall_composite_microscope_baseline.py"),
        ("starter", TASK / "environment/model.py"),
    ]:
        model = load(name, path).Model().fit(cal)
        errors = {
            g: max(abs(model.predict(qs) - [r["value"] for r in truth[g]]))
            for g, qs in groups.items()
        }
        print(name, model.gain, errors, flush=True)
        assert abs(model.gain - GAIN) < 1e-7
        if name == "oracle":
            assert max(errors.values()) < 2e-7
        else:
            assert (
                min(errors[g] for g in ("field_power", "cross_bias", "circulation"))
                > 0.002
            )
        if name == "control":
            assert errors["bulk"] < 2e-7
    q = sample()
    for angle in (0.0, np.pi / 2):
        for p in (0, 1):
            shot = dict(q, angle=angle, phase=p)
            value = signal(shot, 0.94)
            print("sample", angle, p, value, flush=True)
            assert abs(value - probe(shot, 0.94, steps=14)) < 1e-11
    for family, qs in groups.items():
        for q in qs:
            assert abs(probe(q, GAIN, steps=12) - signal(q, GAIN)) < 1e-10


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--generate", action="store_true")
    main(cli.parse_args().generate)
