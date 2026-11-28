"""Validate waveguide echoes using microscopic spin-cell and ordered scattering calculations."""
import argparse
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/photon-transfer-fringe"
sys.path[:0] = [str(TASK / "environment"), str(TASK / "solution"), str(TASK / "tests")]
from returning import echo, spatial_rotation
from reference import echo_ref, memory


def as_pairs(z):
    return np.stack((z.real, z.imag), axis=-1).tolist()


def square():
    # Both polarizations occupy the whole square pulse. Polarization 1 is a spectator.
    flat = [[1, 0]]
    return dict(
        rates=[1.0, 0.0],
        atom=[0, 0],
        analyzer=None,
        control=[0, 0, 0],
        gap=[0, 0],
        first=dict(
            durations=[1.0],
            detuning=[0.0],
            input_modes=[flat, flat],
            input_rotation=[0, 0],
            write_mode=flat,
        ),
        second=dict(
            durations=[1.0],
            detuning=[0.0],
            read_mode=flat,
            output_mode=flat,
            output_rotation=[0, 0],
        ),
        store=dict(
            cells=2,
            write_k=[0.0, 0.0],
            read_k=[0.0, 0.0],
            write=[np.pi / (2 * np.sqrt(2))] * 2,
            read=[np.pi / (2 * np.sqrt(2))] * 2,
            rotation=[0, 0],
            phase=0.0,
        ),
        handoff="unmeasured",
        conditional=False,
        phase=0.0,
        **{"lambda": 0.0}
    )


def families():
    rng = np.random.default_rng(853324)

    def uni(a, b, size=None):
        return np.round(rng.uniform(a, b, size), 5).tolist()

    def spin():
        return [uni(0.2, 1.35), uni(-2.5, 2.5)]

    def make(d):
        f = rng.uniform(0.65, 1.25, d) * np.exp(1j * rng.uniform(-0.55, 0.55, d))
        return f / np.linalg.norm(f)

    groups = {}
    for family in ("spin_storage", "return_interference", "retrieval_eraser"):
        items = []
        for i in range(9):
            n = 2 + i % 2
            k = 2 + (i // 2) % 2
            dt1 = uni(0.4, 1.3, n)
            dt2 = uni(0.4, 1.3, k)
            phases = [make(n), make(n), make(n), make(k), make(k)]
            one = dict(
                durations=dt1,
                detuning=uni(-0.95, 0.95, n),
                input_modes=[as_pairs(x) for x in phases[:2]],
                input_rotation=spin(),
                write_mode=as_pairs(phases[2]),
            )
            two = dict(
                durations=dt2,
                detuning=uni(-0.95, 0.95, k),
                read_mode=as_pairs(phases[3]),
                output_mode=as_pairs(phases[4]),
                output_rotation=spin(),
            )
            cells = 2 + i % 2
            store = dict(
                cells=cells,
                write=uni(0.48, 1.58, 2),
                read=uni(0.49, 1.62, 2),
                rotation=spin(),
                phase=uni(-2.95, 2.95),
            )
            store["write_k"] = (
                [0.0, 0.0] if family == "return_interference" else uni(-1.45, 1.45, 2)
            )
            store["read_k"] = (
                [0.0, 0.0] if family == "return_interference" else uni(-1.07, 1.07, 2)
            )
            q = dict(
                handoff="ground" if family == "spin_storage" else "unmeasured",
                rates=uni(0.42, 1.65, 2),
                atom=spin(),
                analyzer=None
                if family == "spin_storage"
                or (family == "return_interference" and i % 3 == 0)
                else spin(),
                control=[*spin(), uni(-2.6, 2.6)],
                gap=[uni(0.05, 0.6), uni(-1.0, 1.0)],
                first=one,
                second=two,
                store=store,
                conditional=(i % 4 != 0),
                phase=uni(-1, 1),
                **{"lambda": uni(1.55, 3.0)}
            )
            items.append(q)
        groups[family] = items
    return groups


def validate(generate=False):
    vis = 0.8436
    calibration = json.loads((TASK / "environment/data/calibration.json").read_text())
    assert len(calibration) == 12 and calibration == json.loads(
        (TASK / "tests/data/calibration.json").read_text()
    )
    groups = {}
    for family, qs in families().items():
        rows = []
        for q in qs:
            result = echo_ref(q)
            np.testing.assert_allclose(echo(q), result, atol=2e-12, rtol=0)
            assert 0.01 < result[1] <= 1.0
            rows.append(dict(input=q, value=round(vis * result[0], 8)))
        groups[family] = rows
    path = TASK / "tests/data/predictions.json"
    if generate:
        path.write_text(json.dumps(groups, indent=1) + "\n")
    frozen = json.loads(path.read_text())
    assert frozen.keys() == groups.keys()
    for family, rows in groups.items():
        for x, y in zip(rows, frozen[family], strict=True):
            assert x["input"] == y["input"] and abs(x["value"] - y["value"]) < 1e-7
    for name, file in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("which-number", ROOT / "scripts/photon_transfer_fringe_baseline.py"),
    ]:
        model = runpy.run_path(str(file))["Model"]().fit(calibration)
        assert abs(model.visibility - vis) < 1e-8
        errors = {}
        for family, rows in frozen.items():
            value = model.predict([r["input"] for r in rows])
            assert np.isfinite(value).all()
            errors[family] = float(np.max(np.abs(value - [r["value"] for r in rows])))
        if name == "oracle":
            assert max(errors.values()) < 1e-7
        if name == "starter":
            assert min(errors.values()) > 0.1
        if name == "which-number":
            assert (
                errors["spin_storage"] < 1e-7
                and min(errors["retrieval_eraser"], errors["return_interference"]) > 0.1
            )
        print(name, errors)
    q = square()
    B = 2 * (1 - np.exp(-0.5))
    S = 1 - 2 * (1 - B)
    w2 = np.sqrt(2) * (np.cos(np.pi * np.sqrt(3) / 2) - 1) / 3
    for phase in (0.0, np.pi):
        q["store"]["phase"] = phase
        exact = (
            abs(S * S * w2 * w2 * np.exp(2j * phase) + B * B * np.exp(1j * phase)) ** 2
        )
        for method in (echo, echo_ref):
            np.testing.assert_allclose(method(q), (exact, exact), atol=2e-13)
    # A singly occupied two-cell spin wave is dark at π mismatch; the doubly occupied spin wave is bright.
    s = square()["store"]
    s["write_k"] = [np.pi, 0.0]
    np.testing.assert_allclose(memory(1, s)[0, 0], 0.0, atol=1e-13)
    np.testing.assert_allclose(memory(2, s)[0, 0], w2**2, atol=1e-13)
    for n, form in [(1, 0.0), (2, 1.0)]:
        np.testing.assert_allclose(
            spatial_rotation((0,) * n, (0,) * n, np.eye(2), s), form, atol=1e-13
        )
    for family, rows in groups.items():
        err = max(
            abs(echo(q["input"], dilute=True)[0] * vis - q["value"]) for q in rows
        )
        assert err > 0.02
        print("dilute Rabi pulses", family, err)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--generate", action="store_true")
    validate(a.parse_args().generate)
