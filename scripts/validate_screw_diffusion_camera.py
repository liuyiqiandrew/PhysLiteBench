"""Regenerate and cross-check the spin transition counter and its gated fringe."""
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


from monitor import monitored
from monitor_reference import ref_monitor
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
    rng = np.random.default_rng(3205719)
    for family in ("count_weight", "one_change", "two_changes"):
        shots = []
        for i in range(8):
            instrument = dict(
                filters=[
                    dict(
                        band=(["central", "upper"][(i + j // 2) % 2]), turn=axis(rng.uniform(0.06, 2.9))
                    )
                    for j in (0, 2)
                ],
                changes=(i % 3 if family == "count_weight" else 1 if family == "one_change" else 2),
                start=i % 2,
                initial=axis(rng.uniform(0.72, 0.98)),
                analyzer=None if i in (0, 5) else axis(0.96),
                read="weight" if family == "count_weight" else "selected",
            )
            shots.append(
                dict(
                    mode="fringe",
                    hand=(-1) ** i,
                    tag=axis(rng.uniform(0.65, 1.25)),
                    gates=[axis(0.93), axis(0.97)],
                    times=np.round(rng.uniform(0.2, 1.05, 2), 4).tolist(),
                    wave=np.round(rng.uniform(0.62, 1.96, 2) * [1.0, (-1) ** i], 4).tolist(),
                    phase=float([0.15, np.pi / 2, -0.45, 1.19][i % 4]),
                    monitor=instrument,
                )
            )
        if family == "count_weight":
            # Complementary equatorial analyzers separate two-level coherence
            # from full resolution of the magnetic levels within each band.
            for i, sign in enumerate((1, -1)):
                settings = [{"band": "upper", "turn": [0.0, 0.0, 0.0]}] * 2
                spin = dict(
                    filters=settings,
                    changes=0,
                    start=0,
                    initial=[1.0, 0.0, 0.0],
                    analyzer=[sign, 0.0, 0.0],
                    read="weight",
                )
                shots[i] = dict(
                    mode="fringe",
                    hand=1,
                    tag=[0.0, 0.0, 0.0],
                    times=[0.15, 0.21],
                    wave=[0.0, 0.0],
                    phase=0.0,
                    gates=[[0.0, 0.0, 0.0]] * 2,
                    monitor=spin,
                )
        groups[family] = shots
    return groups


def sample():
    return dict(
        mode="fringe",
        hand=1,
        times=[0.52, 0.81],
        wave=[1.53, -1.08],
        tag=[0.55, -0.22, 0.71],
        gates=[[0.88, 0.15, -0.05], [-0.18, 0.69, 0.57]],
        phase=-0.24,
        monitor=dict(
            filters=[
                dict(band="upper", turn=[0.24, -0.51, 0.32]),
                dict(band="central", turn=[0.87, 0.25, -0.71]),
            ],
            start=0,
            initial=[0.6, -0.32, 0.41],
            analyzer=[-0.04, 0.78, 0.59],
            changes=1,
            read="selected",
        ),
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
            dict(input=q, value=round(ref_monitor(q, gamma) if "monitor" in q else measurement(q), 8))
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
            assert max(errors.values()) < 5e-5
        else:
            assert max(errors[g] for g in ("origin", "tagged", "quadrature", "reversal")) < 5e-5
            assert min(errors[g] for g in ("count_weight", "one_change", "two_changes")) > 0.003
    q = sample()
    for n, target in enumerate([0.12973269, 0.15987080, 0.14032937]):
        prepared = dict(q, monitor=dict(q["monitor"], changes=n))
        assert abs(monitored(prepared, gamma) - target) < 1e-6
        assert abs(ref_monitor(prepared, gamma) - target) < 1.0e-6
    rate = 1.5 * gamma * (0.55 + 0.82) * sum(q["times"])
    from math import factorial

    fixed = {"band": "central", "turn": [0, 0, 0]}
    for n in (0, 1, 2):
        s = dict(q["monitor"], filters=[fixed, fixed], changes=n, read="weight", analyzer=None)
        prepared = dict(q, monitor=s, phase=0.0, wave=[0.0, 0.0], gates=[[0.0, 0.0, 0.0]] * 2)
        exact = np.exp(-rate) * rate**n / factorial(n)
        assert abs(monitored(prepared, gamma) - exact) < 1.0e-10
        assert abs(ref_monitor(prepared, gamma) - exact) < 1.0e-6
    from scipy.linalg import expm

    # With aligned upper filters the no-change populations obey a two-rate
    # absorbing chain. Its real coherence obeys one scalar decay equation.
    a = 1.5 * gamma * (0.55 + 0.82)
    b = 2 * gamma * (0.55 + 0.82)
    survival = np.sum(expm(0.36 * np.array([[-a, a], [a, -a - b]])) @ np.array([0.5, 0.5]))
    coherence = np.exp(-0.36 * gamma * (0.55 + 4 * 0.82 + 1.12))
    for i, sign in enumerate((1, -1)):
        expected = (survival + sign * coherence) / 2
        prepared = groups["count_weight"][i]
        assert abs(monitored(prepared, gamma) - expected) < 1.0e-10
        assert abs(ref_monitor(prepared, gamma) - expected) < 1.0e-6
    # Reversing an upper filter axis swaps the two labels while preserving
    # both physical subspaces. Exactly one extra change occurs at the switch.
    same = {"band": "upper", "turn": [0.0, 0.0, 0.0]}
    flipped = {"band": "upper", "turn": [np.pi, 0.0, 0.0]}
    for read in ("weight", "selected"):
        plain = dict(
            q, monitor=dict(q["monitor"], changes=1, read=read, analyzer=None, filters=[same, same])
        )
        swapped = dict(q, monitor=dict(plain["monitor"], changes=2, filters=[same, flipped]))
        for solver in (monitored, ref_monitor):
            assert abs(solver(plain, gamma) - solver(swapped, gamma)) < 1.0e-7
    for label in ("one_change", "two_changes"):
        for prepared in groups[label]:
            probability = monitored(
                dict(
                    prepared,
                    wave=[0.0, 0.0],
                    phase=0.0,
                    monitor=dict(prepared["monitor"], analyzer=None, read="weight"),
                ),
                gamma,
            )
            assert probability >= 0.01
    for g in ("count_weight", "one_change", "two_changes"):
        prepared = groups[g][1]
        assert (
            abs(monitored(prepared, gamma, order=8) - ref_monitor(prepared, gamma, order=8, step=0.001))
            < 2e-5
        )


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--generate", action="store_true")
    main(cli.parse_args().generate)
