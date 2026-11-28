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
from geometry import values
from reference import finite


def calibration():
    return [
        dict(input=dict(active_length=t), value=round(float(np.exp(-0.92 * t)), 8))
        for t in [0.21, 0.38, 0.47, 0.64, 0.76, 0.89, 1.05, 1.17, 1.32, 1.43]
    ]


def science():
    rng = np.random.default_rng(194628)

    def draw(a, b):
        return round(float(rng.uniform(a, b)), 6)

    groups = {}
    for group in (
        "point_weights",
        "ridge_density",
        "area_density",
        "conditioned_waveform",
    ):
        qs = []
        for i in range(6):
            part = {
                "point_weights": "point",
                "ridge_density": "ridge",
                "area_density": "area",
                "conditioned_waveform": ("point", "ridge", "area")[i % 3],
            }[group]
            g = 4 if part == "area" or i % 2 else 3
            x = draw(0.37, 0.57)
            ve = draw(0.88, 1.18)
            vh = draw(0.83, 1.09)
            p = i % 2
            vs = [ve, vh]
            s = ve + vh
            prompt = [(1 - x) / ve, x / vh]
            t = round(draw(0.65, 0.82) * min(prompt), 6)
            delays = [0.0, 0.0]
            delays[1 - p] = round(t * (1 + vs[p] / vs[1 - p]), 10)
            ua = round(draw(0.07, 0.13) * t, 8)
            va = round(draw(0.45, 0.54) * t, 8)
            ub = round(draw(0.13, 0.23) * t, 8)
            vb = round(draw(0.68, 0.75) * t, 8)
            ga, gb = [ua, va], [ub, vb]
            if part == "point":
                # Both gates can lie entirely before the first free impact.
                free = [t * draw(0.8, 0.9)] * (g - 2)
                if i in (2, 5):
                    ga[0] = round(draw(0.29, 0.37) * t, 8)
                    gb[0] = round(draw(0.30, 0.36) * t, 8)
                    free[0] = draw(0.14, 0.22) * t
            elif part == "ridge":
                # One free impact lies in both gates or only in the longer gate.
                star = draw(0.32, 0.41) * t if i % 3 else draw(0.56, 0.65) * t
                free = [star] + [t * draw(0.83, 0.90)] * (g - 3)
            else:
                free = [draw(0.31, 0.4) * t, draw(0.55, 0.65) * t]

            def F(gate):
                l, h = gate
                return round(
                    s * (h - l + sum(max(0.0, h - max(l, birth)) for birth in free)), 12
                )

            gates = [ga, gb]
            z = list(map(F, gates))
            if i % 3 == 1:
                gates.reverse()
                z.reverse()
            read = [round(t * draw(0.12, 0.28), 8), round(t * draw(0.62, 0.92), 8)]
            qs.append(
                dict(
                    depth=x,
                    velocities=vs,
                    dead_space=[draw(0.026, 0.041), draw(0.026, 0.041)],
                    ionization=[draw(1.65, 2.65), draw(1.65, 2.65)],
                    feedback=draw(0.45, 0.93),
                    gain=g,
                    tag=None,
                    last_delays=delays,
                    multiplicities=[None, None],
                    read=read,
                    gates=gates,
                    charges=z,
                    part=part,
                    observable="waveform"
                    if group == "conditioned_waveform"
                    else "intensity",
                )
            )
        groups[group] = qs
    return groups


def examples():
    q = dict(
        depth=0.4,
        velocities=[1.0, 1.0],
        ionization=[2.0, 2.0],
        dead_space=[0.0, 0.0],
        feedback=0.0,
        gain=3,
        tag=None,
        last_delays=[0.0, 0.6],
        multiplicities=[None, None],
        read=[0.0, 0.25],
        gates=[[0.0, 0.15], [0.0, 0.23]],
    )
    results = []
    for part, z, intensity, current in [
        ("point", [0.3, 0.46], 7 / 30, 59 / 350),
        ("ridge", [0.3, 0.54], 5 / 3, 31 / 150),
        ("ridge", [0.45, 0.77], 5 / (3 * np.sqrt(2)), 17 / 60),
        ("area", [0.31, 0.83], 50 / 9, 113 / 400),
    ]:
        if part == "area":
            q.update(gain=4, gates=[[0.0, 0.1], [0.0, 0.2]])
        results.append((dict(q, part=part, charges=z), intensity, current))
    return results


def validate(generate):
    cal = calibration()
    a = 0.92
    data = {}
    for name, qs in science().items():
        delta = []
        rows = []
        for q in qs:
            intensity, wave, _ = values(q, a, 10)
            other = finite(q, a, 8, 0.0004)
            assert intensity > 0.007 and 0 <= wave <= 1 and np.isfinite(other).all()
            delta.append(np.abs(np.array([intensity, wave]) - other))
            value = wave if q["observable"] == "waveform" else intensity
            rows.append(dict(input=q, value=round(float(value), 8)))
        diff = np.max(delta, axis=0)
        assert np.max(diff) < 0.0005
        data[name] = rows
        print(
            name, "max independent intensity and waveform differences", diff, flush=True
        )
    for p, rows in [
        (TASK / "environment/data/calibration.json", cal),
        (TASK / "tests/data/calibration.json", cal),
        (TASK / "tests/data/predictions.json", data),
    ]:
        if generate:
            p.write_text(json.dumps(rows, indent=1) + "\n")
        assert json.loads(p.read_text()) == rows
    for p, label in [
        (TASK / "solution/model.py", "oracle"),
        (TASK / "environment/model.py", "independent channels"),
        (ROOT / "scripts/avalanche_packet_current_baseline.py", "independent read"),
    ]:
        m = runpy.run_path(str(p))["Model"]().fit(cal)
        err = {}
        for name, rows in data.items():
            raw = m.predict([r["input"] for r in rows])
            assert np.isfinite(raw).all()
            err[name] = float(max(abs(raw - [r["value"] for r in rows])))
        if label == "oracle":
            assert max(err.values()) < 0.0001
        if label == "independent channels":
            assert min(err.values()) > 0.01
        if label == "independent read":
            assert (
                err["conditioned_waveform"] > 0.05
                and max(v for k, v in err.items() if k != "conditioned_waveform")
                < 0.0001
            )
        print(label, err, flush=True)
    for q, intensity, wave in examples():
        np.testing.assert_allclose(values(q, a, 9)[:2], [intensity, wave], atol=1.0e-10)
        np.testing.assert_allclose(
            finite(q, a, 8, 0.0004), [intensity, wave], atol=1.0e-6
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
