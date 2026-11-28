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
from reference import finite


def calibration():
    return [
        dict(input=dict(active_length=t), value=round(float(np.exp(-0.92 * t)), 8))
        for t in [0.21, 0.38, 0.47, 0.64, 0.76, 0.89, 1.05, 1.17, 1.32, 1.43]
    ]


def science():
    rng = np.random.default_rng(784213)

    def draw(a, b, size=None):
        return np.round(rng.uniform(a, b, size), 6).tolist()

    groups = {}
    for group in ("charge_lines", "charge_density", "gated_current"):
        rows = []
        for i in range(8):
            joint = i % 3 != 0
            g = (5 if i in (2, 5) else 4) if joint else (3 if i % 2 == 0 else 4)
            x = draw(0.37, 0.56)
            ve = draw(0.86, 1.18)
            vh = draw(0.78, 1.08)
            te = (1 - x) / ve
            th = x / vh
            em = te * draw(0.48, 0.71)
            hm = th * draw(0.47, 0.72)
            e = round(hm * (1 + vh / ve), 6)
            h = round(em * (1 + ve / vh), 6)
            if not joint:
                if i % 2:
                    e = 0.0
                else:
                    h = 0.0
            tmark = min(em, hm) if joint else (em if e == 0 else hm)
            lo = round(tmark * draw(0.39, 0.53), 6)
            hi = round(tmark * draw(0.77, 0.92), 6)
            step = (ve + vh) * (hi - lo)
            line = group == "charge_lines" or group == "gated_current" and i % 2 == 0
            charge = (
                (1 + (i % 2 if group == "charge_lines" else int(i in (2, 6)))) * step
                if line
                else (1.0 + draw(0.24, 0.75)) * step
            )
            if line and i == 0:
                charge = 2 * step
            read = [
                round(max(0, lo - tmark * draw(0.08, 0.18)), 6),
                round(hi + tmark * draw(0.09, 0.33), 6),
            ]
            mult = [None, None]
            if joint and i in (1, 5):
                mult = [1, None] if i == 1 else [None, 2]
            rows.append(
                dict(
                    depth=x,
                    velocities=[ve, vh],
                    ionization=draw(1.65, 2.75, 2),
                    dead_space=draw(0.035, 0.057, 2),
                    feedback=draw(0.42, 1.05),
                    gain=g,
                    tag=None,
                    last_delays=[e, h],
                    multiplicities=mult,
                    read=read,
                    spectrum=dict(
                        window=[lo, hi],
                        charge=round(charge, 14),
                        part="line" if line else "density",
                    ),
                    observable="waveform" if group == "gated_current" else "spectrum",
                )
            )
        groups[group] = rows
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
        read=[0.0, 0.25],
        last_delays=[0.0, 0.6],
        multiplicities=[None, None],
        spectrum=dict(window=[0.0, 0.15], charge=0.3, part="line"),
    )


def validate(generate):
    import time

    start = time.monotonic()
    cal = calibration()
    nominal = 0.92
    expected = {}
    for family, qs in science().items():
        rows = []
        differences = []
        for q in qs:
            v = reading(q, nominal, 12)
            intensity, current = finite(q, nominal, 9, 0.001)
            assert intensity > 0.04 and current is not None and 0 <= current <= 1
            alternate = intensity if q["observable"] == "spectrum" else current
            differences.append(abs(v - alternate))
            rows.append(dict(input=q, value=round(float(v), 8)))
        assert max(differences) < 0.00055
        print(family, "independent maximum difference", max(differences), flush=True)
        expected[family] = rows
    for path, values in [
        (TASK / "environment/data/calibration.json", cal),
        (TASK / "tests/data/calibration.json", cal),
        (TASK / "tests/data/predictions.json", expected),
    ]:
        if generate:
            path.write_text(json.dumps(values, indent=1) + "\n")
        assert json.loads(path.read_text()) == values
    for location, name in [
        (TASK / "solution/model.py", "oracle"),
        (TASK / "environment/model.py", "finite resolution"),
        (
            ROOT / "scripts/avalanche_packet_current_baseline.py",
            "independent avalanche",
        ),
    ]:
        model = runpy.run_path(str(location))["Model"]().fit(cal)
        errors = {}
        assert abs(model.activity - nominal) < 1.0e-7
        for family, rows in expected.items():
            prediction = model.predict([r["input"] for r in rows])
            assert np.isfinite(prediction).all()
            errors[family] = float(max(abs(prediction - [r["value"] for r in rows])))
        if name == "oracle":
            assert max(errors.values()) < 0.00002
        if name == "finite resolution":
            assert min(errors.values()) > 0.004 and errors["charge_density"] > 0.01
        if name == "independent avalanche":
            assert (
                errors["gated_current"] > 0.02
                and max(errors["charge_density"], errors["charge_lines"]) < 0.00002
            )
        print(name, errors, flush=True)
    q = worked()
    for part, z, intensity, current in [
        ("line", 0.3, 0.5, 17 / 90),
        ("density", 0.45, 5 / 3, 17 / 60),
    ]:
        q["spectrum"].update(part=part, charge=z)
        np.testing.assert_allclose(
            finite(q, nominal, 8), [intensity, current], atol=1.0e-8
        )
        for field, value in [("spectrum", intensity), ("waveform", current)]:
            np.testing.assert_allclose(
                reading(dict(q, observable=field), nominal), value, atol=1.0e-10
            )
    print("Elapsed seconds", round(time.monotonic() - start, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
