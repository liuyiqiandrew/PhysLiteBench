"""Validate continuum feedback against independently propagated two-photon wavefunctions."""
import argparse
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/photon-transfer-fringe"
sys.path[:0] = [str(TASK / "environment"), str(TASK / "solution"), str(TASK / "tests")]
from reference import reading
from interferometer import delay_signal


def packed(z):
    return np.stack((z.real, z.imag), axis=-1).tolist()


def square():
    shape = np.ones((2, 1), complex) / np.sqrt(2)
    return dict(
        rates=[1.0, 0.0],
        duration=0.5,
        atom=[0.0, 0.0],
        analyzer=None,
        control=[0.0, 0.0, 0.0],
        gap=[0.0, 0.0],
        input_modes=packed(np.stack((shape, shape))),
        input_rotation=[0.0, 0.0],
        output_mode=packed(shape),
        output_rotation=[0.0, 0.0],
        detuning=[[0.0], [0.0]],
        return_detuning=[[0.0], [0.0]],
        arms=[[0.0, 0.0, np.pi], [0.0, 0.0, np.pi]],
        input_reverse=[False, False],
        output_reverse=[True, True],
        temporal=packed(np.eye(2, dtype=complex)[::-1]),
        phase=0.0,
        conditional=False,
        **{"lambda": 0.0}
    )


def shots():
    rng = np.random.default_rng(3855391)

    def draw(lo, hi, n=None):
        return np.round(rng.uniform(lo, hi, n), 6).tolist()

    def rotation():
        return [draw(0.25, 1.32), draw(-2.65, 2.65)]

    def mode(n):
        v = rng.uniform(0.62, 1.3, (2, n)) * np.exp(
            1j * rng.uniform(-1.65, 1.65, (2, n))
        )
        return v / np.linalg.norm(v)

    sets = {}
    for group in ("routed_return", "temporal_interference", "atomic_echo"):
        rows = []
        for i in range(8):
            n = 1 if i % 4 == 0 else 2
            h = [mode(n) for _ in range(3)]
            t, p = rotation()
            u = np.array(
                [
                    [np.cos(t), -np.exp(-1j * p) * np.sin(t)],
                    [np.exp(1j * p) * np.sin(t), np.cos(t)],
                ]
            )
            if group == "routed_return":
                u = np.eye(2, dtype=complex)[:: (1 if i % 2 else -1)]
            flags = lambda: [bool(x) for x in rng.integers(0, 2, size=2)]
            q = dict(
                rates=draw(0.52, 1.62, 2),
                duration=draw(0.77, 1.65),
                atom=rotation(),
                analyzer=rotation() if group == "atomic_echo" or i % 4 == 1 else None,
                control=[*rotation(), draw(-2.5, 2.5)],
                gap=[draw(0.04, 0.48), draw(-0.88, 0.88)],
                input_modes=packed(np.stack(h[:2])),
                input_rotation=rotation(),
                output_mode=packed(h[2]),
                output_rotation=rotation(),
                detuning=draw(-1, 1, (2, n)),
                return_detuning=draw(-1, 1, (2, n)),
                arms=[[*rotation(), draw(-2.85, 2.85)] for _ in range(2)],
                input_reverse=flags(),
                output_reverse=flags(),
                temporal=packed(u),
                phase=draw(-1.0, 1.0),
                conditional=i % 3 != 0,
                **{"lambda": draw(1.6, 3.1)}
            )
            rows.append(q)
        sets[group] = rows
    return sets


def validate(generate=False):
    vis = 0.8436
    calibration = json.loads((TASK / "environment/data/calibration.json").read_text())
    assert len(calibration) == 12 and calibration == json.loads(
        (TASK / "tests/data/calibration.json").read_text()
    )
    groups = {}
    max_disagreement = 0.0
    for family, qs in shots().items():
        rows = []
        for q in qs:
            exact = delay_signal(q)
            assert 0.005 < exact[1] <= 1
            a, b, c = [np.array(reading(q, m)) for m in (16, 32, 64)]
            extrapolated = (8 * c - 6 * b + a) / 3
            max_disagreement = max(
                max_disagreement, float(np.max(abs(extrapolated - exact)))
            )
            np.testing.assert_allclose(extrapolated, exact, atol=3e-6, rtol=0)
            rows.append(dict(input=q, value=round(vis * exact[0], 8)))
        groups[family] = rows
    print("maximum independent wavefunction disagreement", max_disagreement)
    p = TASK / "tests/data/predictions.json"
    if generate:
        p.write_text(json.dumps(groups, indent=1) + "\n")
    frozen = json.loads(p.read_text())
    assert frozen.keys() == groups.keys()
    for family, rows in groups.items():
        for x, y in zip(rows, frozen[family], strict=True):
            assert x["input"] == y["input"] and abs(x["value"] - y["value"]) < 1e-7
    for name, file in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("which-level", ROOT / "scripts/photon_transfer_fringe_baseline.py"),
    ]:
        model = runpy.run_path(str(file))["Model"]().fit(calibration)
        assert abs(model.visibility - vis) < 1e-8
        errors = {}
        for family, rows in frozen.items():
            value = model.predict([r["input"] for r in rows])
            assert np.isfinite(value).all()
            errors[family] = float(np.max(np.abs(value - [r["value"] for r in rows])))
        assert (
            max(errors.values()) < 1e-7
            if name == "oracle"
            else min(errors.values()) > 0.01
        )
        print(name, errors)
    q = square()
    B = 2 * (1 - np.exp(-0.5))
    forward = (B * B + 12 * np.exp(-0.5) - 7) ** 2
    for reverse, truth in [(True, 1.0), (False, forward)]:
        q["output_reverse"] = [reverse] * 2
        q["temporal"] = packed(np.eye(2, dtype=complex)[:: (-1 if reverse else 1)])
        np.testing.assert_allclose(delay_signal(q), (truth, truth), atol=2e-12)
        a, b, c = [np.array(reading(q, m)) for m in (16, 32, 64)]
        np.testing.assert_allclose(
            (8 * c - 6 * b + a) / 3, (truth, truth), rtol=0, atol=2e-7
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generate", action="store_true")
    validate(p.parse_args().generate)
