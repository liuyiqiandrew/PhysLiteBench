"""Regenerate the disordered-glass measurements from the joint sample transform."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/aging-line-memory"
sys.path[:0] = [str(TASK / p) for p in ("environment", "solution", "tests")]
from landscape import marked
from reference import exact


def preparations():
    rng = np.random.default_rng(95882394)

    def draw(lo, hi):
        return round(float(rng.uniform(lo, hi)), 4)

    out = {}
    for fam in ("thermal_tags", "line_selection", "quenched_memory"):
        rows = []
        for i in range(8):
            while True:
                if fam == "thermal_tags":
                    prep = [0, 1, i % 2]
                    gate = [[0, 1, i % 2 == 0]]
                    tag = 0
                    mark = i % 2
                elif fam == "line_selection":
                    prep = [0, 1, 0, i % 2]
                    gate = [[0, 1, i % 2 == 0]]
                    tag = 1 if i % 3 else -1
                    mark = (i + 1) % 3
                else:
                    prep = [i % 2, 1 - i % 2, 1, 0]
                    gate = [[0, 1, i % 3 == 0], [1, 2, i % 4 == 1]]
                    tag = 0 if i % 3 else 1
                    mark = i % 3
                ts = [draw(0.48, 0.58), draw(0.63, 0.73)]
                if i % 2:
                    ts.reverse()
                h = [draw(-0.44, -0.13), draw(0.13, 0.44)]
                if i % 3:
                    h.reverse()
                q = dict(
                    settings=[[ts[0], h[0]], [ts[1], h[1]]],
                    preparations=prep,
                    gate=gate,
                    tag=tag,
                    mark=mark,
                    exposure=draw(0.25, 2.5),
                )
                if exact(q, 0.9863)[1] > 0.015:
                    break
            rows.append(q)
        out[fam] = rows
    return out


def main(generate):
    tg = 0.9863
    cal = [
        dict(input=dict(threshold=x), value=round(float(np.exp(-x / tg)), 8))
        for x in (0.22, 0.35, 0.52, 0.66, 0.80, 1.05, 1.24, 1.46, 1.68, 1.92)
    ]
    data = {}
    rates = []
    worst = 0.0
    for group, qs in preparations().items():
        rows = []
        for q in qs:
            y, p = exact(q, tg)
            subject = marked(q, tg)
            worst = max(worst, max(abs(np.array(subject) - [y, p])))
            np.testing.assert_allclose(subject, (y, p), atol=5e-7, rtol=0)
            assert p > 0.01 and 0 < y < 1
            rates.append(p)
            rows.append(dict(input=q, value=round(y, 9)))
        data[group] = rows
    for name, obj in [
        ("tests/data/calibration.json", cal),
        ("environment/data/calibration.json", cal),
        ("tests/data/predictions.json", data),
    ]:
        f = TASK / name
        if generate:
            f.write_text(json.dumps(obj, indent=1) + "\n")
        frozen = json.loads(f.read_text())
        if isinstance(obj, list):
            assert obj == frozen
        else:
            assert obj.keys() == frozen.keys()
            for k, rows in obj.items():
                assert [r["input"] for r in rows] == [r["input"] for r in frozen[k]]
                np.testing.assert_allclose(
                    [r["value"] for r in rows],
                    [r["value"] for r in frozen[k]],
                    atol=1.1e-9,
                    rtol=0,
                )
    print("min joint acceptance", min(rates), "max independent discrepancy", worst)
    for role, path in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("shortcut", ROOT / "scripts/aging_line_memory_baseline.py"),
    ]:
        m = runpy.run_path(str(path))["Model"]().fit(cal)
        assert abs(m.depth_scale - tg) < 1e-8
        errors = {}
        for k, rs in data.items():
            y = m.predict([r["input"] for r in rs])
            assert np.isfinite(y).all()
            errors[k] = float(max(abs(y - [r["value"] for r in rs])))
        print(role, errors)
        assert (
            max(errors.values()) < 1e-6
            if role == "oracle"
            else min(errors.values()) > 0.004
        )
    q = dict(
        settings=[[0.5, 0.0], [0.5, 0.0]],
        preparations=[0, 1, 1],
        exposure=0.0,
        mark=0,
        tag=0,
    )
    for same in (False, True):
        q["gate"] = [[0, 1, same]]
        for method in (marked, exact):
            np.testing.assert_allclose(
                method(q, 1.0), (0.75 if same else 0.25, 0.5), atol=2e-8, rtol=0
            )
    q = preparations()["quenched_memory"][0]
    flipped = dict(q, settings=[[t, -h] for t, h in q["settings"]], tag=-q["tag"])
    np.testing.assert_allclose(marked(q, tg), marked(flipped, tg), atol=2e-12, rtol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    main(parser.parse_args().generate)
