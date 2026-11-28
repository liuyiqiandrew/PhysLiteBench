"""Regenerate the reference camera readings with independent Gaussian bridges."""
from pathlib import Path
import sys, json, importlib.util, argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/kinetic-excursion-aperture"
sys.path[:0] = [str(TASK / "tests"), str(TASK / "environment"), str(TASK / "solution")]
from bridges import sample
from excursion import query

D = 0.9273


def shots():
    rng = np.random.default_rng(618240)
    cal = [dict(mode="free", lag=round(t, 5)) for t in rng.uniform(0.12, 1.2, 10)]
    groups = {}
    for mode in ("clear", "direction", "pulse"):
        rows = []
        for j in range(7):
            q = dict(
                mode=mode,
                launch=round(rng.uniform(0.40, 0.90), 5),
                incoming=round(rng.uniform(0.38, 0.85), 5),
                times=np.round(rng.uniform(0.53, 1.1, 2), 5).tolist(),
                position=round(rng.uniform(0.12, 0.28), 5),
            )
            if mode == "clear":
                q["velocity"] = round(rng.uniform(-0.55, 0.50), 5)
            else:
                q["cut"] = round(rng.uniform(-0.21, 0.23), 5)
            if mode == "pulse":
                q["kick"] = round(rng.uniform(0.45, 0.98), 5)
            rows.append(q)
        groups[mode] = rows
    return cal, groups


def write(p, x):
    p.write_text(json.dumps(x, separators=(",", ":")) + "\n")


def regenerate():
    cal, groups = shots()
    cal = [dict(input=q, value=round(2 * D * q["lag"], 8)) for q in cal]
    write(TASK / "environment/data/calibration.json", cal)
    write(TASK / "tests/data/calibration.json", cal)
    truth = {}
    for i, (g, qs) in enumerate(groups.items()):
        truth[g] = []
        for j, q in enumerate(qs):
            draws = [
                sample(q, D, 2**17, depth=7, seed=12581 + 1000 * i + 10 * j + k)
                for k in range(4)
            ]
            reading = np.mean(draws)
            err = abs(reading - query(q, D, 64))
            print(
                g,
                j,
                reading,
                "standard error",
                np.std(draws, ddof=1) / 2,
                "vs killed kernel",
                err,
                flush=True,
            )
            assert err < 0.0024
            truth[g].append(dict(input=q, value=round(reading, 6)))
    write(TASK / "tests/data/hidden.json", truth)


def load(n, p):
    spec = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Model()


def validate():
    cal = json.loads((TASK / "environment/data/calibration.json").read_text())
    truth = json.loads((TASK / "tests/data/hidden.json").read_text())
    for name, p in [
        ("oracle", TASK / "solution/model.py"),
        ("outbound only", ROOT / "scripts/kinetic_excursion_aperture_baseline.py"),
        ("starter", TASK / "environment/model.py"),
    ]:
        m = load(name, p).fit(cal)
        errors = {
            g: float(
                max(abs(m.predict([r["input"] for r in rs]) - [r["value"] for r in rs]))
            )
            for g, rs in truth.items()
        }
        print(name, m.diffusivity, errors, flush=True)
        assert (
            max(errors.values()) < 0.0025
            if name == "oracle"
            else min(errors.values()) > 0.006
        )
    example = dict(
        launch=0.48,
        incoming=0.55,
        times=[0.85, 0.76],
        position=0.16,
        velocity=0.2,
        cut=0.04,
        kick=0.85,
    )
    for mode in ("clear", "direction", "pulse"):
        q = dict(example, mode=mode)
        print("worked", mode, query(q, 0.83, 64), flush=True)
    # In the symmetric, unkicked excursion the velocity at a fixed positive
    # midpoint location has an even distribution by time reversal.
    q = dict(
        example,
        mode="direction",
        launch=0.66,
        incoming=0.66,
        times=[0.71, 0.71],
        cut=0.0,
    )
    assert abs(query(q, D, 60) - 0.5) < 0.00004
    q = dict(q, mode="pulse", kick=0.0)
    assert abs(query(q, D) - 0.5) < 1e-12
    for rs in truth.values():
        for r in rs:
            q = r["input"]
            assert abs(query(q, D, 40) - query(q, D, 80)) < 0.00015


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    if parser.parse_args().generate:
        regenerate()
    validate()
