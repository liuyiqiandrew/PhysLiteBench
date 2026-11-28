"""Reproduce and audit the independent collision-sampler measurements."""
from pathlib import Path
import argparse, importlib.util, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/inelastic-collapse-calorimeter"
sys.path[:0] = [str(TASK / "tests"), str(TASK / "environment"), str(TASK / "solution")]
from renewal import signal, log_drift, slow
from sampler import slow_fraction, ensemble
from physics_checks import renewal_check, tail_bound

GAIN = 0.8347
PROFILES = [
    (0.37, [0.365, 0.020]),
    (0.46, [0.31, 0.016]),
    (2.40, [0.022, 0.47]),
    (2.76, [0.029, 0.38]),
]


def measurements():
    rng = np.random.default_rng(913022)
    t = [dict(mode="free", lag=round(v, 5)) for v in rng.uniform(0.11, 1.38, 12)]
    hidden = {}
    hidden["selector"] = [
        dict(mode="selector", threshold=v)
        for v in [0.24, 0.39, 0.58, 0.83, 1.1, 1.56, 2.02, 2.86]
    ]
    switches = [
        (0.25, [0.024, 0.27]),
        (0.25, [0.62, 0.11]),
        (2.84, [0.105, 0.56]),
        (2.84, [0.29, 0.046]),
        (0.53, [0.32, 0.045]),
        (1.31, [0.071, 0.47]),
        (0.48, [0.31, 0.036]),
        (1.7, [0.122, 0.53]),
    ]
    hidden["sticking"] = [
        dict(mode="sticking", threshold=v, restitution=r) for v, r in switches
    ]
    for group in ("timer", "wall_heat", "joint"):
        rows = []
        for threshold, r in PROFILES:
            for j in range(2):
                rows.append(
                    dict(
                        mode="timer" if group == "timer" else "calorimeter",
                        threshold=threshold,
                        restitution=r,
                        launch=round(rng.uniform(0.45, 0.91), 5),
                        clock=0
                        if group == "wall_heat"
                        else round(rng.uniform(0.27, 1.15), 5),
                        dose=0
                        if group == "timer"
                        else round(rng.uniform(0.18, 1.10), 5),
                    )
                )
        hidden[group] = rows
    return t, hidden


def store(path, content):
    path.write_text(json.dumps(content, separators=(",", ":")) + "\n")


def generate():
    cal, hidden = measurements()
    cal = [dict(input=q, value=round(2 * GAIN * q["lag"] ** 3 / 3, 8)) for q in cal]
    store(TASK / "environment/data/calibration.json", cal)
    store(TASK / "tests/data/calibration.json", cal)
    truth = {}
    for g in ("selector", "sticking"):
        truth[g] = []
        for q in hidden[g]:
            p = slow_fraction(q["threshold"], GAIN, 2**22, 10183)
            if g == "selector":
                value = p
            else:
                f, r = q["restitution"]
                estimated = np.pi / np.sqrt(3) + (1 - p) * np.log(f) + p * np.log(r)
                assert abs(estimated) > 0.08
                value = float(estimated < 0)
            truth[g].append(dict(input=q, value=round(value, 6)))
    for g in ("timer", "wall_heat", "joint"):
        truth[g] = []
    for index, (thresh, _) in enumerate(PROFILES):
        chosen = [
            (g, q)
            for g in ("timer", "wall_heat", "joint")
            for q in hidden[g]
            if q["threshold"] == thresh
        ]
        qs = [q for g, q in chosen]
        means, errors = ensemble(qs, GAIN, 2**21, 12191 + index, depth=-65)
        assert max(errors) < 0.00035
        print(
            "reference",
            thresh,
            "largest sampling standard error",
            max(errors),
            flush=True,
        )
        for (g, q), value in zip(chosen, means):
            truth[g].append(dict(input=q, value=round(value, 6)))
    store(TASK / "tests/data/hidden.json", truth)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model()


def main(rebuild):
    if rebuild:
        generate()
    for t, u, v in [(1.64, 0.51, 0.70), (0.62, 0.83, 0.29), (2.4, 0.34, 1.1)]:
        full, first, other = renewal_check(t, u, v, GAIN, 128)
        assert abs(full - first - other) < 1e-9, (full, first, other)
        print("boundary flux", full, first, other, flush=True)
    cal = json.loads((TASK / "environment/data/calibration.json").read_text())
    rows = json.loads((TASK / "tests/data/hidden.json").read_text())
    assert cal == json.loads((TASK / "tests/data/calibration.json").read_text())
    for name, path in [
        ("oracle", TASK / "solution/model.py"),
        ("control", ROOT / "scripts/inelastic_collapse_calorimeter_baseline.py"),
        ("starter", TASK / "environment/model.py"),
    ]:
        model = load(name, path).fit(cal)
        errors = {
            g: max(
                abs(model.predict([r["input"] for r in rs]) - [r["value"] for r in rs])
            )
            for g, rs in rows.items()
        }
        print(name, model.diffusivity, errors, flush=True)
        assert abs(model.diffusivity - GAIN) < 1e-7
        if name == "oracle":
            assert max(errors.values()) < 0.001
        elif name == "control":
            assert max(errors[g] for g in ["selector", "sticking", "wall_heat"]) < 0.001
            assert min(errors[g] for g in ["timer", "joint"]) > 0.007
        else:
            assert max(errors.values()) > 0.04
    for q in [
        dict(mode="selector", threshold=0.5),
        dict(mode="sticking", threshold=0.25, restitution=[0.024, 0.27]),
        dict(mode="sticking", threshold=0.25, restitution=[0.62, 0.11]),
        dict(
            mode="timer",
            threshold=0.37,
            restitution=[0.365, 0.020],
            launch=0.67,
            clock=0.46,
            dose=0,
        ),
        dict(
            mode="calorimeter",
            threshold=0.37,
            restitution=[0.365, 0.020],
            launch=0.67,
            clock=0.46,
            dose=0.47,
        ),
    ]:
        print("worked", q, signal(q, 0.9), flush=True)
    for g in ("timer", "wall_heat", "joint"):
        for r in rows[g][::2]:
            q = r["input"]
            v = signal(q, GAIN)
            assert tail_bound(q, GAIN) < 0.000006
            assert abs(v - signal(q, GAIN, 0.105, -80, 30)) < 0.00008
            scaled = dict(
                q,
                launch=q["launch"] * 0.89,
                clock=q["clock"] / 0.89**2,
                dose=q["dose"] / 0.89**2,
            )
            assert abs(v - signal(scaled, GAIN)) < 0.00005


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    main(p.parse_args().generate)
