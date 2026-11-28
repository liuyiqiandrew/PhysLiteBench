"""Compare the calibrated repeater, independent herald/phase enumeration, physical controls and worked shots."""
import argparse
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/repeater-parity-router"
sys.path[:0] = [str(TASK / "environment"), str(TASK / "solution"), str(TASK / "tests")]
from reference import reference
from swap import values


def calibration():
    return [
        dict(input=dict(exposure=t), value=round(float(np.exp(-0.87 * t)), 8))
        for t in (0.16, 0.27, 0.42, 0.58, 0.69, 0.94, 1.13, 1.24, 1.45, 1.58)
    ]


def science():
    rng = np.random.default_rng(68294)
    gs = {}

    def draw(lo, hi):
        return round(float(rng.uniform(lo, hi)), 6)

    def state():
        return [draw(0.14, 1.37), draw(-2.0, 2.0), draw(-2.5, 2.5)]

    for k in ("marker", "herald", "fringe"):
        rows = []
        for j in range(9):
            first = state()
            second = state()
            if j in (2, 6):
                second = [
                    round(first[0] + draw(-0.07, 0.07), 6),
                    round(first[1] + draw(-0.27, 0.27), 6),
                    round(first[2] + draw(-0.24, 0.24), 6),
                ]
            if j in (0, 4):
                first = [draw(0.06, 0.2), draw(-0.45, 0.45), draw(-1.4, 1.4)]
                second = [draw(1.27, 1.46), draw(-0.65, 0.65), draw(-1.4, 1.4)]
            words = (
                ["XX", "XX"],
                ["ZX", "XY"],
                ["XZ", "ZX"],
                ["YY", "XX"],
                ["ZZ", "XZ"],
                ["XY", "ZZ"],
                ["ZX", "ZX"],
                ["ZY", "YY"],
                ["XX", "ZZ"],
            )
            rows.append(
                dict(
                    loading=draw(0.6, 2.85),
                    parity=[None, 0, 1][j % 3],
                    marker=dict(phase=draw(-2.7, 2.7), sign=(-1) ** (j % 2)),
                    outer=[state(), state()],
                    bank=[first, second],
                    analyzers=[draw(-1.8, 1.8), draw(-1.8, 1.8)],
                    report=[1, None, 0][j % 3],
                    spin=words[j],
                    observable=k,
                )
            )
        gs[k] = rows
    return gs


def examples():
    lam = np.log(2.0)
    bell = [np.pi / 4, 0.0, 0.0]
    q = dict(
        loading=lam,
        parity=0,
        marker=dict(sign=1, phase=0.0),
        outer=[bell, bell],
        bank=[[0, 0, 0], [np.pi / 2, 0, 0]],
        analyzers=[0, 0],
        report=0,
        spin=["XX", "XX"],
    )
    product = dict(q, bank=[[0.0, np.pi / 2, 0.0]] * 2, spin=["ZX", "XZ"])
    return [
        (q, [0.5, 0.125, 2 * lam**2]),
        (dict(q, parity=1), [0.5, 0.125, 0.0]),
        (product, [1.0, 0.125, 1.0]),
        (dict(product, report=1), [1.0, 0.125, -1.0]),
    ]


def validate(generate):
    a = 0.87
    cal = calibration()
    data = {}
    for k, qs in science().items():
        rows = []
        errors = []
        for q in qs:
            pred = values(q, a)
            ref = reference(q, a)
            assert ref["marker"] > 0 and ref["herald"] > 0
            assert all(np.isfinite(list(ref.values())))
            errors.append(max(abs(pred[t] - ref[t]) for t in ref))
            rows.append(dict(input=q, value=round(float(ref[k]), 8)))
        assert max(errors) < 1.0e-10
        data[k] = rows
        print(k, "independent quantum-state difference", max(errors))
    for p, rows in [
        (TASK / "environment/data/calibration.json", cal),
        (TASK / "tests/data/calibration.json", cal),
        (TASK / "tests/data/predictions.json", data),
    ]:
        if generate:
            p.write_text(json.dumps(rows, indent=1) + "\n")
        assert json.loads(p.read_text()) == rows
    for p, name in [
        (TASK / "solution/model.py", "oracle"),
        (TASK / "environment/model.py", "dephased source"),
        (
            ROOT / "scripts/repeater_parity_router_baseline.py",
            "independent remote outputs",
        ),
    ]:
        m = runpy.run_path(str(p))["Model"]().fit(cal)
        assert abs(m.activity - a) < 2.0e-7
        err = {}
        for k, rows in data.items():
            v = m.predict([r["input"] for r in rows])
            assert np.isfinite(v).all()
            err[k] = float(max(abs(v - [r["value"] for r in rows])))
        if name == "oracle":
            assert max(err.values()) < 3.0e-7
        if name == "dephased source":
            assert min(err.values()) > 0.01
        if name == "independent remote outputs":
            assert max(err["marker"], err["herald"]) < 3.0e-7 and err["fringe"] > 0.3
        print(name, err)
    for alt in ["erase_all", "no_feedforward"]:
        print(
            alt,
            {
                k: max(
                    abs(values(r["input"], a, shortcut=alt)[k] - r["value"])
                    for r in rows
                )
                for k, rows in data.items()
            },
        )
    for q, expected in examples():
        for fn in (values, reference):
            np.testing.assert_allclose(
                list(fn(q, 1.0).values()), expected, atol=1.0e-12
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
