import argparse, json, runpy, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/critical-seam-magnet"
sys.path[:0] = [str(TASK / d) for d in ("environment", "solution", "tests")]
from sectors import normalized
from chain import channel
from spin_basis import literal


def settings():
    rng = np.random.default_rng(27414003)
    groups = {}
    for g in ("clamped", "annealed", "response"):
        rs = []
        for j in range(8):
            count = 2 * (1 + j % 4)
            while True:
                t = sorted(rng.choice(np.arange(1, 64), size=count, replace=False))
                if min(np.diff(t + [t[0] + 64])) >= 3:
                    break
            q = dict(
                sites=[float(i / 64) for i in t],
                aspect=round(float(rng.uniform(0.43, 1.38)), 3),
                bias=round(float(rng.uniform(-0.95, 0.95)), 3),
                seam=(-1 if j % 3 else 1) if g == "clamped" else "free",
                parity=(-1 if j % 2 else 1),
                read="response" if g == "response" else "spin",
            )
            rs.append(q)
        groups[g] = rs
    return groups


def main(generate):
    gain = 0.9268
    cal = [
        dict(input=dict(bias=h), value=round(float(np.tanh(gain * h)), 8))
        for h in (-1.30, -1, -0.76, -0.45, -0.22, 0.18, 0.39, 0.70, 0.94, 1.24)
    ]
    data = {}
    worst = 0.0
    finite = 0.0
    for g, qs in settings().items():
        rows = []
        for q in qs:
            exact = normalized(q, gain)
            short, long = (channel(q, gain, N) for N in (256, 512))
            limit = (4 * long - short) / 3
            worst = max(worst, abs(limit - exact))
            finite = max(finite, abs(long - exact))
            assert abs(limit - exact) < 2e-7
            if q["read"] == "response":
                h = 0.00002

                def spin(shift):
                    return normalized(
                        dict(q, read="spin", bias=q["bias"] + shift), gain
                    )

                assert abs((spin(h) - spin(-h)) / (2 * h) - exact) < 1e-8
            rows.append(dict(input=q, value=round(exact, 9)))
        data[g] = rows
    print("Finite chain maximum at 512 sites", finite, "after extrapolation", worst)
    for name, obj in [
        ("tests/data/calibration.json", cal),
        ("environment/data/calibration.json", cal),
        ("tests/data/predictions.json", data),
    ]:
        p = TASK / name
        if generate:
            p.write_text(json.dumps(obj, indent=1) + "\n")
        assert json.loads(p.read_text()) == obj
    for role, where in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("shortcut", ROOT / "scripts/critical_seam_magnet_baseline.py"),
    ]:
        m = runpy.run_path(str(where))["Model"]().fit(cal)
        assert abs(m.gain - gain) < 1e-7
        err = {
            k: float(
                max(abs(m.predict([r["input"] for r in rs]) - [r["value"] for r in rs]))
            )
            for k, rs in data.items()
        }
        print(role, err)
        assert (
            max(err.values()) < 1e-6 if role == "oracle" else min(err.values()) > 0.004
        )
    q = dict(
        sites=[0.125, 0.4375],
        aspect=0.85,
        bias=0.32,
        read="spin",
        seam="free",
        parity=-1,
    )
    for t, read, target in [
        ("free", "spin", 1.11506357),
        ("free", "response", 0.15321969),
        (-1, "spin", 0.49761299),
        (1, "spin", 1.20252516),
    ]:
        assert abs(normalized(dict(q, seam=t, read=read), 1.0) - target) < 1e-8
    err = 0.0
    for p in [-1, 0, 1]:
        for t in [-1, 1, "free"]:
            for read in ["spin", "response"]:
                row = dict(
                    q, sites=[0.125, 0.375, 0.625, 0.875], seam=t, parity=p, read=read
                )
                err = max(err, abs(literal(row, 1.0, 8) - channel(row, 1.0, 8)))
    print("Explicit eight-spin thermal traces", err)
    assert err < 1e-11


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    main(p.parse_args().generate)
