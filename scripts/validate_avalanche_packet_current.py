"""Audit avalanche gain, packet timing, induced charge, controls and frozen data."""
import argparse
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/avalanche-packet-current"
sys.path[:0] = [str(TASK / "environment"), str(TASK / "solution"), str(TASK / "tests")]
from geometry import reading
from reference import predict


def calibration():
    return [
        dict(input=dict(active_length=t), value=round(float(np.exp(-0.92 * t)), 8))
        for t in [0.21, 0.38, 0.47, 0.64, 0.76, 0.89, 1.05, 1.17, 1.32, 1.43]
    ]


def square():
    return dict(
        depth=0.2,
        velocities=[1.0, 1.0],
        dead_space=[0.45, 0.12],
        ionization=[1.0, 0.0],
        feedback=0.0,
        gain=2,
        last_delay=0.0,
        multiplicity=2,
        tag=None,
        read=[0.0, 0.8],
    )


def queries():
    rng = np.random.default_rng(9203241)

    def uniform(lo, hi, size=None):
        return np.round(rng.uniform(lo, hi, size), 5).tolist()

    out = {}
    for family in ("prompt_cluster", "delayed_packet", "tagged_waveform"):
        rows = []
        for i in range(8):
            x = uniform(0.27, 0.63)
            v = [uniform(0.85, 1.3), uniform(0.69, 1.13)]
            prompt = family == "prompt_cluster"
            g = (2 if i % 3 == 0 else 3) if prompt else (3 if i % 3 == 0 else 4)
            earliest = (1 - x) / v[0]
            tau = (
                0.0
                if prompt
                else round(x * (1 / v[0] + 1 / v[1]) * uniform(0.33, 0.79), 5)
            )
            m = g if prompt and i % 2 else (None if prompt or i % 3 == 0 else 1 + i % 2)
            early = uniform(0.04, 0.34) * earliest
            late = uniform(0.74, 1.46) * (earliest + 0.5 * tau)
            read = [round(early, 5), round(late, 5)]
            tag = (
                None
                if family != "tagged_waveform"
                else [
                    round(0.04 * earliest, 5),
                    round((0.65 + 0.1 * (i % 4)) * earliest, 5),
                    uniform(0.62, 1.32),
                    uniform(0.19, 0.33),
                ]
            )
            q = dict(
                depth=x,
                velocities=v,
                dead_space=[uniform(0.045, 0.115), uniform(0.039, 0.102)],
                ionization=uniform(1.62, 2.96, 2),
                feedback=uniform(0.42, 1.16),
                gain=g,
                last_delay=tau,
                multiplicity=m,
                tag=tag,
                read=read,
            )
            rows.append(q)
        out[family] = rows
    return out


def validate(generate=False):
    a = 0.92
    cal = calibration()
    expect = {}
    different = {}
    convergence = {}
    for key, qs in queries().items():
        rows = []
        earlier = []
        ref = []
        for q in qs:
            full = reading(q, a, 24)
            lower = reading(q, a, 18)
            alternative = predict(q, a, 128)
            assert full[1] > 0 and 0 <= full[0] <= 1
            ref.append(abs(full[0] - alternative[0]))
            earlier.append(abs(full[0] - lower[0]))
            rows.append(dict(input=q, value=round(full[0], 8)))
        expect[key] = rows
        different[key] = max(ref)
        convergence[key] = max(earlier)
    assert max(different.values()) < 0.00045 and max(convergence.values()) < 0.00004
    print("independent waveform differences", different)
    print("polygon quadrature refinement", convergence)
    for p, data in [
        (TASK / "environment/data/calibration.json", cal),
        (TASK / "tests/data/calibration.json", cal),
        (TASK / "tests/data/predictions.json", expect),
    ]:
        if generate:
            p.write_text(json.dumps(data, indent=1) + "\n")
        stored = json.loads(p.read_text())
        assert stored == cal if isinstance(data, list) else stored.keys() == data.keys()
        if isinstance(data, dict):
            for k, rows in data.items():
                for x, y in zip(rows, stored[k], strict=True):
                    assert (
                        x["input"] == y["input"] and abs(x["value"] - y["value"]) < 1e-7
                    )
    for name, p in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("clamped", ROOT / "scripts/avalanche_packet_current_baseline.py"),
    ]:
        model = runpy.run_path(str(p))["Model"]().fit(cal)
        assert abs(model.activity - a) < 1e-7
        err = {}
        for k, rows in expect.items():
            v = model.predict([r["input"] for r in rows])
            assert np.isfinite(v).all()
            err[k] = float(np.max(abs(v - [r["value"] for r in rows])))
        assert (
            max(err.values()) < 0.00004
            if name == "oracle"
            else min(err.values()) > 0.003
        )
        print(name, err)
    for method in (lambda q: reading(q, a, 20), lambda q: predict(q, a, 64)):
        q = square()
        m = 0.45 + 1 / a - 0.35 / np.expm1(0.35 * a)
        np.testing.assert_allclose(
            method(q), ((1 + 2 * (0.8 - m)) / 2, 1 - np.exp(-0.35 * a)), atol=1e-10
        )
        q = dict(
            q,
            depth=0.4,
            dead_space=[0.05, 0.05],
            ionization=[2.0, 2.0],
            feedback=0.6,
            last_delay=0.3,
            multiplicity=1,
            read=[0.22, 0.7],
        )
        assert abs(method(q)[0] - 0.61) < 1e-12
        q = queries()["delayed_packet"][2]
        y, p = method(dict(q, multiplicity=None))
        sums = np.zeros(2)
        for m in range(1, q["gain"]):
            z, w = method(dict(q, multiplicity=m))
            sums += np.array([z * w, w])
        np.testing.assert_allclose(sums, [y * p, p], atol=1e-12)
    print(
        "terminal-electrode control",
        {
            k: max(
                abs(reading(r["input"], a, 10, terminal=True)[0] - r["value"])
                for r in rows
            )
            for k, rows in expect.items()
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generate", action="store_true")
    validate(p.parse_args().generate)
