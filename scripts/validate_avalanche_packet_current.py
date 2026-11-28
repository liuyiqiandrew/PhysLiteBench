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
from geometry import packet_law as physical, observable
from reference import packet_law as alternate


def calibration():
    return [
        dict(input=dict(active_length=t), value=round(float(np.exp(-0.92 * t)), 8))
        for t in [0.21, 0.38, 0.47, 0.64, 0.76, 0.89, 1.05, 1.17, 1.32, 1.43]
    ]


def queries():
    rng = np.random.default_rng(521478)

    def draw(a, b, size=None):
        return np.round(rng.uniform(a, b, size), 6).tolist()

    groups = {}
    for k in ("one_delayed_contact", "joint_waveform", "packet_fractions"):
        rows = []
        for i in range(8):
            x = draw(0.33, 0.64)
            v = [draw(0.88, 1.28), draw(0.73, 1.08)]
            te, th = (1 - x) / v[0], x / v[1]
            tmax = 1 / v[0] + 1 / v[1]
            de = x * tmax * draw(0.37, 0.77)
            dh = (1 - x) * tmax * draw(0.36, 0.82)
            if k == "one_delayed_contact":
                # An impact by the species opposite the prompt contact would have delayed that contact.
                g = 3 if i % 3 else 4
                delays = [0.0, dh] if i % 2 else [de, 0.0]
                m = [g, 1] if i % 2 else [1, g]
            else:
                g = 5 if i in (1, 5) else (3 if i % 3 == 0 else 4)
                delays = [de, dh]
                m = (
                    [1, 1],
                    [1, 2],
                    [2, 1],
                    [1, None],
                    [None, 2],
                    [2, 2],
                    [2, 1],
                    [None, None],
                )[i]
                if k == "packet_fractions":
                    m = (
                        [1, 1],
                        [2, 2],
                        [2, 1],
                        [1, 2],
                        [1, 1],
                        [1, 2],
                        [2, 1],
                        [1, 1],
                    )[i]
            LO = draw(0.15, 0.74) * min(te, th)
            HI = draw(0.77, 1.21) * max(te + delays[0], th + delays[1])
            tag = (
                [
                    round(0.05 * te, 6),
                    round(0.85 * te, 6),
                    draw(0.7, 1.35),
                    draw(0.18, 0.34),
                ]
                if i % 2
                else None
            )
            rows.append(
                dict(
                    depth=x,
                    velocities=v,
                    dead_space=[draw(0.046, 0.095), draw(0.041, 0.092)],
                    ionization=draw(1.72, 2.85, 2),
                    feedback=draw(0.46, 1.05),
                    gain=g,
                    last_delays=np.round(delays, 6).tolist(),
                    multiplicities=m,
                    tag=tag,
                    read=np.round([LO, HI], 6).tolist(),
                    observable="fraction" if k == "packet_fractions" else "waveform",
                )
            )
        groups[k] = rows
    return groups


def worked():
    return dict(
        depth=0.4,
        velocities=[1.0, 1.0],
        ionization=[2.0, 2.0],
        dead_space=[0.0, 0.0],
        feedback=0.0,
        gain=3,
        tag=None,
        read=[0.25, 0.7],
        last_delays=[0.3, 0.4],
        multiplicities=[2, 1],
        observable="waveform",
    )


def validate(generate=False):
    a = 0.92
    cal = calibration()
    expect = {}
    agree = {}
    repeat = {}
    for key, qs in queries().items():
        rows = []
        external = []
        refine = []
        for q in qs:
            truth = physical(q, a, 18)
            test = alternate(q, a, 10)
            lo = physical(q, a, 12)
            y = observable(q, truth)
            assert np.isfinite(y) and 0 <= y <= 1 and truth[0].sum() > 0
            external.append(abs(y - observable(q, test)))
            refine.append(abs(y - observable(q, lo)))
            rows.append(dict(input=q, value=round(float(y), 8)))
            assert np.allclose(truth[0].sum(), test[0].sum(), rtol=0.001, atol=0.000025)
        expect[key] = rows
        agree[key] = max(external)
        repeat[key] = max(refine)
    assert max(agree.values()) < 0.0003 and max(repeat.values()) < 0.00015
    print("independent waveform and probability differences", agree)
    print("quadrature refinement", repeat)
    for p, data in [
        (TASK / "environment/data/calibration.json", cal),
        (TASK / "tests/data/calibration.json", cal),
        (TASK / "tests/data/predictions.json", expect),
    ]:
        if generate:
            p.write_text(json.dumps(data, indent=1) + "\n")
        frozen = json.loads(p.read_text())
        assert frozen == cal if isinstance(data, list) else frozen.keys() == data.keys()
        if isinstance(data, dict):
            for k, rows in data.items():
                for x, y in zip(rows, frozen[k], strict=True):
                    assert (
                        x["input"] == y["input"] and abs(x["value"] - y["value"]) < 1e-7
                    )
    for name, p in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("independent heights", ROOT / "scripts/avalanche_packet_current_baseline.py"),
    ]:
        m = runpy.run_path(str(p))["Model"]().fit(cal)
        assert abs(m.activity - a) < 1e-7
        err = {}
        for k, rows in expect.items():
            vals = m.predict([r["input"] for r in rows])
            assert np.isfinite(vals).all()
            err[k] = float(np.max(abs(vals - [r["value"] for r in rows])))
        if name == "oracle":
            assert max(err.values()) < 0.00015
        if name == "starter":
            assert min(err.values()) > 0.004
        if name == "independent heights":
            assert (
                err["packet_fractions"] > 0.04
                and max(err["joint_waveform"], err["one_delayed_contact"]) < 0.00015
            )
        print(name, err)
    q = worked()
    for method in (physical, alternate):
        law = method(q, a, 9)
        assert np.count_nonzero(law[0] > 1e-12) == 3
        for pair, val in [([2, 1], 0.6), ([1, 2], 2 / 3), ([1, 1], 19 / 30)]:
            obs = dict(q, multiplicities=pair)
            assert abs(observable(obs, law) - val) < 1e-12
            assert (
                abs(observable(dict(obs, observable="fraction"), law) - 1 / 3) < 1e-12
            )
        wildcard = dict(q, multiplicities=[None, None])
        assert abs(observable(dict(wildcard, observable="fraction"), law) - 1) < 1e-12
        np.testing.assert_allclose(
            observable(wildcard, law), np.mean([0.6, 2 / 3, 19 / 30]), atol=1e-12
        )
    for k, rows in expect.items():
        err = max(
            abs(
                observable(r["input"], physical(dict(r["input"], feedback=0.0), a, 9))
                - r["value"]
            )
            for r in rows
        )
        print("clamped bias", k, err)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generate", action="store_true")
    validate(p.parse_args().generate)
