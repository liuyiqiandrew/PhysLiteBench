"""Regenerate the pinned stochastic area readouts and check the independent camera and gyro control."""
from pathlib import Path
import importlib.util
import sys, json, argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/screw-diffusion-camera"
sys.path[:0] = [str(TASK / "solution"), str(TASK / "environment"), str(TASK / "tests")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from area import rotate_area
from area_reference import reference_area
from reference import measurement, TRUE_PARAMETER

oracle = load("oracle", TASK / "solution/model.py")
starter = load("starter", TASK / "environment/model.py")
baseline = load("baseline", ROOT / "scripts/screw_diffusion_camera_baseline.py")


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
    rng = np.random.default_rng(7503891)

    def pin(rad):
        v = rng.normal(size=2)
        return np.round(rad * v / np.linalg.norm(v), 5).tolist()

    for family in ("final_pin", "checkpoint", "oriented_area"):
        group = []
        for i in range(6):
            end = pin(rng.uniform(0.65, 1.18))
            middle = None
            if family == "checkpoint" or (family == "oriented_area" and i % 2):
                # Reflect the midpoint to the near side of the endpoint bisector.
                v = np.array(pin(rng.uniform(0.43, 0.85)))
                if np.dot(v, end) < 0:
                    v = -v
                middle = v.tolist()
            plane = rng.choice(3, size=2, replace=False).tolist()
            s = dict(
                axes=plane, field=round((-1) ** i * rng.uniform(1.30, 2.19), 4), middle=middle, end=end
            )
            group.append(
                dict(
                    mode="fringe",
                    hand=(-1) ** i,
                    times=np.round(rng.uniform(0.58, 1.04, 2), 4).tolist(),
                    wave=np.round(rng.uniform(0.70, 1.90, 2) * [1, (-1) ** i], 4).tolist(),
                    tag=axis(rng.uniform(0.62, 1.23)),
                    gates=[axis(0.94), axis(0.96)],
                    phase=float(np.pi / 2 if family == "oriented_area" else [0.15, -0.31, 1.12][i % 3]),
                    area=s,
                )
            )
        groups[family] = group
    return groups


def sample():
    return dict(
        mode="fringe",
        hand=1,
        times=[0.59, 0.73],
        wave=[1.58, -1.2],
        tag=[0.75, -0.12, 0.64],
        gates=[[0.87, -0.23, 0.12], [-0.1, 0.77, 0.39]],
        phase=0.3,
        area=dict(axes=[1, 2], field=1.4, middle=[0.69, -0.41], end=[0.43, 0.7]),
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
            dict(input=q, value=round(reference_area(q, gamma) if "area" in q else measurement(q), 8))
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
            assert min(errors[g] for g in ("final_pin", "checkpoint", "oriented_area")) > 0.003
    q = sample()
    s = q["area"]
    for p, target in ((None, 0.1380375), (s["middle"], 0.1164458)):
        prepared = dict(q, area=dict(s, middle=p))
        assert abs(rotate_area(prepared, gamma) - target) < 1.0e-5
        assert abs(reference_area(prepared, gamma) - target) < 1.0e-5
        bare = dict(prepared, wave=[0, 0], gates=[[0, 0, 0]] * 2, phase=0.87)
        physical = baseline.Model().fit(cal).predict([bare])[0]
        assert abs(rotate_area(bare, gamma) - physical) < 1.0e-6
        assert abs(reference_area(bare, gamma) - physical) < 1.0e-5
    # Opposite gyro endpoints have the same area marginal. Their coupling
    # to this tagged scattering phase is directly observable.
    lever = np.array([-0.38, 1.37, 0.29])
    endpoint = dict(
        q,
        gates=[(lever / np.linalg.norm(lever)).tolist(), [0.0, 0.0, 0.0]],
        tag=[1.2, 0.0, 0.0],
        wave=[1.4, 1.4],
        phase=float(np.pi / 2),
    )
    for sign, expected in ((1, 0.07903115), (-1, -0.07038281)):
        shot = dict(endpoint, area=dict(s, axes=[2, 0], end=[sign * 1.1, 0.0], middle=None))
        assert abs(rotate_area(shot, gamma) - expected) < 1.0e-6
        assert abs(reference_area(shot, gamma) - expected) < 1.0e-5
    for name in ("final_pin", "checkpoint", "oriented_area"):
        prepared = groups[name][0]
        assert (
            abs(
                rotate_area(prepared, gamma, order=7, levels=28, hermite=15)
                - reference_area(prepared, gamma, order=6, points=32, box=12.5, h=0.001)
            )
            < 3.0e-4
        )


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--generate", action="store_true")
    main(cli.parse_args().generate)
