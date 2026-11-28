"""Validate the chronology of a gyro-driven spin echo and its tagged-particle fringe."""
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


from replay import replay
from replay_reference import ref_replay
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
    rng = np.random.default_rng(91355027)
    for family in ("stored_record", "reversed_sense", "analyzed_echo"):
        shots = []
        for i in range(8):
            s = dict(
                coils=[axis(rng.uniform(0.72, 1.72)) for j in range(3)],
                sense=1
                if family == "stored_record"
                else -1
                if family == "reversed_sense"
                else (-1) ** i,
                initial=axis(rng.uniform(0.55, 0.98)),
                analyzer=None if family != "analyzed_echo" and i % 2 == 0 else axis(0.95),
                forward_pulse=axis(rng.uniform(0.2, 2.8)),
                replay_pulse=axis(rng.uniform(0.2, 2.8)),
                control_phase=[0.38, -1.24, 2.29, -2.58][i % 4],
            )
            shots.append(
                dict(
                    mode="fringe",
                    hand=(-1) ** i,
                    times=np.round(rng.uniform(0.24, 1.1, 2), 4).tolist(),
                    wave=np.round(rng.uniform(0.75, 2.05, 2) * [1.0, (-1) ** i], 4).tolist(),
                    tag=axis(rng.uniform(0.45, 1.2)),
                    gates=[axis(0.91), axis(0.94)],
                    phase=float([0.25, np.pi / 2, -0.55, 1.12][i % 4]),
                    replay=s,
                )
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
        replay=dict(
            coils=[[1.3, -0.2, 0.1], [0.1, -0.3, 1.2], [-0.2, 1.05, 0.6]],
            sense=1,
            initial=[0.6, -0.32, 0.41],
            analyzer=[-0.04, 0.78, 0.59],
            replay_pulse=[0.72, -0.34, 0.39],
            forward_pulse=[0.31, 1.21, -0.11],
            control_phase=0.75,
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
            dict(input=q, value=round(ref_replay(q, gamma) if "replay" in q else measurement(q), 8))
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
            assert min(errors[g] for g in ("stored_record", "reversed_sense", "analyzed_echo")) > 0.003
    q = sample()
    s = q["replay"]
    for sense, target in [(1, 0.07820919), (-1, 0.04193766)]:
        prepared = dict(q, replay=dict(s, sense=sense))
        assert (
            abs(replay(prepared, gamma) - target) < 1e-6
            and abs(ref_replay(prepared, gamma) - target) < 1e-6
        )
    iso = dict(
        q,
        wave=[0, 0],
        phase=0,
        gates=[[0, 0, 0]] * 2,
        replay=dict(
            s,
            coils=np.diag(1 / np.sqrt([0.55, 0.82, 1.12])).tolist(),
            sense=-1,
            control_phase=0.0,
            analyzer=None,
            forward_pulse=[0, 0, 0],
            replay_pulse=[0, 0, 0],
        ),
    )
    exact = 0.25 + 0.75 * np.exp(-2 * gamma * sum(q["times"]))
    assert abs(replay(iso, gamma) - exact) < 1e-10 and abs(ref_replay(iso, gamma) - exact) < 1e-6
    aligned = dict(q, replay=dict(iso["replay"], sense=1, coils=[[1, 0, 0], [-0.3, 0, 0], [0.7, 0, 0]]))
    target = measurement(q)
    assert abs(replay(aligned, gamma) - target) < 2e-6
    assert abs(ref_replay(aligned, gamma) - target) < 2e-6
    # Summing the two complementary path ports cancels interference. With no
    # spin analyzer the electronic outcomes are exhaustive on each optical shot.
    bright = dict(q, replay=dict(s, analyzer=None))
    dark = dict(q, replay=dict(s, analyzer=None, control_phase=s["control_phase"] + np.pi))
    assert abs(replay(bright, gamma) + replay(dark, gamma) - target) < 2e-6
    assert abs(ref_replay(bright, gamma) + ref_replay(dark, gamma) - target) < 2e-6
    for g in ("stored_record", "reversed_sense", "analyzed_echo"):
        q = groups[g][0]
        assert abs(replay(q, gamma, order=8) - ref_replay(q, gamma, order=8, step=0.001)) < 5e-5


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--generate", action="store_true")
    main(cli.parse_args().generate)
